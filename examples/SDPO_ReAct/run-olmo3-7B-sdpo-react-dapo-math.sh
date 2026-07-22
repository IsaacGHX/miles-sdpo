#!/bin/bash
# SDPO_ReAct (base version) -- Olmo-3-7B-Instruct on DAPO math, multi-turn
# ReAct rollout with ONE tool (code_interpreter, isolated Docker sandbox),
# single 8x H200 (141GB) node, COLOCATE variant.
#
# Model swap of run-qwen2.5-7B-sdpo-react-dapo-math.sh -- every piece of the
# pipeline downstream of --hf-checkpoint is UNCHANGED and model-agnostic by
# construction:
#   - react_prompt.py's build_react_messages() returns a plain message list;
#     --apply-chat-template renders it ONCE via Olmo-3's OWN chat_template.jinja
#     (loaded from --hf-checkpoint, not hardcoded anywhere in this pipeline).
#   - generate_with_tools.py never touches chat-template control tokens after
#     that first render -- the multi-turn loop is plain string concatenation
#     ("prompt_text + response"), so it works identically regardless of which
#     model produced prompt_text.
#   - sdpo.py's KD teacher-prefix splice (_gen_prompt_suffix) derives the
#     "<|im_end|>\n<|im_start|>assistant\n"-equivalent suffix FROM THE
#     TOKENIZER, not from a hardcoded ChatML assumption.
#
# What genuinely differs for Olmo-3-7B-Instruct, verified via
# model_template_check.py before this script trusts it (see data-prep below):
#   - tokenizer.eos_token is the generic <|endoftext|>, NOT <|im_end|> (unlike
#     Qwen2.5, where eos_token IS <|im_end|>) -- but generation_config.json's
#     eos_token_id list is [<|im_end|>, <|endoftext|>], and SGLang reads that
#     full list from the HF checkpoint, so <|im_end|> (the token that actually
#     closes each rendered turn) is still a real stop condition. Confirmed by
#     model_template_check.py, not assumed.
#   - Olmo-3's rendered system+user+assistant-marker structure is otherwise
#     structurally identical to Qwen2.5's ChatML (<|im_start|>role\n...\n
#     <|im_end|>\n...<|im_start|>assistant\n) -- confirmed by the same check.
#
# Turn budget: 5 turns during TRAINING, 20 during EVAL (per spec) -- see
# eval_aime24.yaml's metadata_overrides + the per-sample override in
# generate_with_tools.py.
#
# Env overrides (same "swap without editing the script" flexibility as
# examples/EPO/enroot-run-epo.sh's EPO_MODEL switch):
#   SDPO_REACT_EVAL_MAX_TURNS    (default: 20, read by eval_aime24.yaml)
#   SDPO_REACT_EVAL_N_SAMPLES    (default: 8, read by eval_aime24.yaml)
#
# usage: bash examples/SDPO_ReAct/run-olmo3-7B-sdpo-react-dapo-math.sh
set -exf

export PYTHONBUFFERED=16
export SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-20}"
export SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "$REPO_ROOT/scripts/models/olmo3-7B.sh"

# --- 0. sandbox sidecar (idempotent, ONE container / ONE port for the whole job) ---
bash "$SCRIPT_DIR/tools/run_sandbox.sh"

# --- 0a. chat-template sanity check -- FAIL LOUDLY before training if Olmo-3's
# template doesn't satisfy the invariants generate_with_tools.py / sdpo.py's
# KD-prefix splice depend on, rather than silently training on a corrupted
# prompt. See model_template_check.py's module docstring for exactly what
# this verifies. Dumps the full rendered one-shot prompt for a human to
# eyeball alongside the automated checks.
(cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.model_template_check \
    --hf-checkpoint /root/Olmo-3-7B-Instruct \
    --dump-path "/root/miles/sdpo_dumps/olmo3_template_check.json")

# --- 0b. data prep: DAPO math + AIME24 eval + ReAct system prompt -----
# Both run as `-m` MODULES from REPO_ROOT (not as bare file paths) --
# build_aime24_eval.py imports react_prompt.build_react_messages so its eval
# prompts carry the SAME <code>/<answer> system prompt as training; that
# cross-module import needs examples.SDPO_ReAct on the import path, which a
# bare `python .../build_aime24_eval.py` invocation does not provide (no repo
# root on sys.path) -- `python -m` from REPO_ROOT does.
mkdir -p /root/dapo-math-17k /root/math_eval
[ -f /root/dapo-math-17k/dapo-math-17k.jsonl ] || \
    hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
[ -f /root/dapo-math-17k/dapo-math-17k-react.jsonl ] || \
    (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.react_prompt \
        --in /root/dapo-math-17k/dapo-math-17k.jsonl \
        --out /root/dapo-math-17k/dapo-math-17k-react.jsonl)
[ -f /root/math_eval/aime24.jsonl ] || \
    (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.build_aime24_eval --out-dir /root/math_eval)

SDPO_REACT_EXP="${SDPO_REACT_EXP:-olmo3-7B-sdpo-react-dapo-math_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/miles/sdpo_dumps/${SDPO_REACT_EXP}"
echo "SDPO_ReAct dump dir: ${DUMP_DIR}"

# Checkpoint dir is STABLE across restarts (independent of SDPO_REACT_EXP's
# timestamp) so a debug restart (e.g. to fix a launcher arg) resumes from
# where it left off instead of re-training from step 0 -- --load automatically
# resumes if this dir already has a checkpoint, and is a no-op on a fresh dir.
# Under /root/data (-> $DATA_DIR, default /fsx/data/$USER -- shared, durable
# network storage with plenty of headroom), NOT /root/miles (-> $REPO_ROOT,
# the /fsx/home git checkout, much smaller/quota-limited: a checkpoint write
# there once genuinely failed mid-save from disk pressure on the Qwen run --
# torch.distributed.checkpoint CheckpointException / "unexpected pos" writer
# corruption), and NOT /root/assets (-> $ASSETS, local nvme -- fast but
# ephemeral/tied to this instance, wrong for something meant to outlive a
# single training session). Different filename than the Qwen run's ckpt so
# the two experiments never collide.
CKPT_DIR="${SDPO_REACT_CKPT_DIR:-/root/data/sdpo_ckpts/olmo3-7B-sdpo-react-dapo-math_ckpt}"

CKPT_ARGS=(
   --hf-checkpoint /root/Olmo-3-7B-Instruct
   --ref-load /root/Olmo-3-7B-Instruct_torch_dist
   --save "${CKPT_DIR}"
   --load "${CKPT_DIR}"
   # Interval 5 (not every step): checkpointing a 7B model's distributed
   # optimizer state has real per-save I/O cost -- this caps a restart's lost
   # progress at <=4 steps without paying that cost every single step.
   --save-interval 5
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k-react.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
   --over-sampling-batch-size 64
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

CUSTOM_GENERATE_ARGS=(
   --custom-generate-function-path examples.SDPO_ReAct.generate_with_tools.generate
)

RM_ARGS=(
   --group-rm
   --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
   --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
   --sdpo-grader dapo
   --sdpo-teacher-backend megatron
   # --sdpo-pure-distill defaults to True (sdpo_group_reward zeroes every
   # trace's task reward so the KD term is the ENTIRE training signal -- see
   # examples/SDPO/run-olmo3-7B-sdpo-math-colocate.sh, which uses this mode
   # but deliberately has NO --dynamic-sampling-filter-path). We want the base
   # version's math-accuracy comparison (this experiment's whole point) to be
   # driven by the real GRPO(task reward), with KD as an added-on distillation
   # term -- so --no-sdpo-pure-distill. This ALSO matters for
   # --dynamic-sampling-filter-path check_reward_nonzero_std above: under pure
   # distill every sample.reward is 0.0, so EVERY group has zero reward
   # variance and gets dropped forever, starving rollout.
   --no-sdpo-pure-distill
   --sdpo-kd-loss
   --sdpo-kd-coef 1.0
   --sdpo-kd-max-tokens 8192
   --sdpo-divergence jsd
   --sdpo-logprob-mode topk
   --opd-log-prob-top-k 100
   --sdpo-is-clip 2.0
   --sdpo-self-teacher
   # Uniform-random correct-peer selection was observed (on the Qwen2.5 run)
   # to drift the KD teacher pool toward no-tool traces over training
   # (no-tool traces are often correct MORE often -- no risk of a mid-trace
   # tool error -- so they dominate the random pool), which then teaches the
   # student away from tool use even though task reward never penalizes it.
   # Prefer a tool-using correct peer when one exists.
   --sdpo-prefer-tool-use-peer
   # NO --sdpo-self-skill / --sdpo-skill-source in this base version: skill
   # generation is orthogonal to getting tool-calling itself working, and adds
   # extra rollout-engine round-trips this pass doesn't need.
)

# --eval-interval 5 matches the "train 5 steps, infer 20 steps" spec (fast
# enough to see a trend without paying a ~4-5min/eval cost every single
# step). --n-samples-per-eval-prompt MUST also be set globally (not just
# inside eval_aime24.yaml's per-dataset config): miles.ray.rollout.metrics.py
# ::log_eval_rollout_data's compute_pass_rate call uses
# args.n_samples_per_eval_prompt (the GLOBAL arg), not
# dataset_cfg.n_samples_per_eval_prompt -- leaving the global at its default
# of 1 silently produces group_size=1 -> compute_pass_rate returns {} -> no
# val-core/aime24_pass@1 panel at all.
EVAL_ARGS=(
   --eval-interval 5
   --eval-config "$SCRIPT_DIR/eval_aime24.yaml"
   --n-samples-per-eval-prompt "${SDPO_REACT_EVAL_N_SAMPLES}"
   --log-passrate
)

PERF_ARGS=(
   # TP=1: Olmo3's QK-norm spans the FULL q/k dim (num_heads*head_dim), so it
   # cannot be split across TP ranks without a cross-TP allreduce of the RMS
   # (TP would normalize over only the local head shard -> wrong). TP=1 keeps
   # each RMSNorm over the full dim -- see scripts/models/olmo3-7B.sh and
   # examples/SDPO/run-olmo3-7B-sdpo-math-colocate.sh, which make the same
   # choice for the same reason.
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 24576
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-grpo-std-normalization
   --sdpo-ema-teacher
   --sdpo-ema-teacher-rate 0.05
   --entropy-coef 0.00
   --observe-training-entropy
   --use-kl-loss
   --kl-loss-coef 0.001
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-warmup-iters 10
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-sdpo
   --wandb-group olmo3-7B-sdpo-react-dapo-math
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.85
   --sglang-router-policy round_robin
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
sleep 2

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# --- checkpoint pruner (background) -----------------------------------------
# Megatron's --save-interval has NO built-in retention limit -- every save
# keeps its OWN full ~100GB iter_NNNNNNN/ directory forever (--load only ever
# reads the newest one via latest_checkpointed_iteration.txt, so older ones
# have zero restart value). Left unpruned, a long run silently fills the
# shared $DATA_DIR volume and the NEXT save fails mid-write with a
# torch.distributed.checkpoint CheckpointException ("unexpected pos") --
# exactly what killed this job once already (disk hit 99% full after ~50
# unpruned iterations). Poll every 60s and delete every iteration dir except
# the one latest_checkpointed_iteration.txt currently points to; killed
# alongside the training job below.
(
    # set -f (noglob) is active for the whole script (see `set -exf` above) --
    # without re-enabling globbing HERE, "${CKPT_DIR}"/iter_* never expands
    # (it's treated as a literal string, so the -d test always fails and
    # NOTHING ever gets pruned -- confirmed live: iter_0000244 and
    # iter_0000249 both sat on disk through multiple pruner cycles before this
    # fix).
    set +f
    while true; do
        sleep 60
        latest_file="${CKPT_DIR}/latest_checkpointed_iteration.txt"
        [ -f "$latest_file" ] || continue
        latest_iter=$(printf 'iter_%07d' "$(cat "$latest_file")")
        for d in "${CKPT_DIR}"/iter_*; do
            [ -d "$d" ] || continue
            [ "$(basename "$d")" = "$latest_iter" ] || rm -rf "$d"
        done
    done
) &
PRUNER_PID=$!
trap 'kill "$PRUNER_PID" 2>/dev/null || true' EXIT

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"/root/Megatron-LM/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
        \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
        \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\",
        \"SDPO_REACT_EVAL_MAX_TURNS\": \"${SDPO_REACT_EVAL_MAX_TURNS}\",
        \"SDPO_REACT_EVAL_N_SAMPLES\": \"${SDPO_REACT_EVAL_N_SAMPLES}\"
     }
   }" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   --colocate \
   --update-weights-interval 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${CUSTOM_GENERATE_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
