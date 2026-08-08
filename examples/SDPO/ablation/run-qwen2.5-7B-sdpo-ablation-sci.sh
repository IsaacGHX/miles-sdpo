#!/bin/bash
# SDPO/RLSD/GRPO ablation matrix -- Qwen2.5-7B-Instruct on SciKnowEval (MCQ,
# single-step, non-agentic), single 8x A100 (80GB) node, COLOCATE variant.
#
# Sibling of ../run-qwen2.5-7B-sdpo-sci-colocate.sh: SAME rollout/eval/skill
# machinery, but with TWO orthogonal ablation axes instead of one fixed arm:
#   SDPO_ABLATION_ALGO  grpo | sdpo | rlsd   -- how the teacher-vs-student
#                        divergence (if any) is consumed
#   SDPO_ABLATION_ARM   a | b | c | d | e | f -- which skill-prefix/skill-KD
#                        machinery is layered on top (see table below)
# GRPO only supports arms a/e/f (skill-prefix-only arms b/c/d have no
# GRPO-relevant training signal without a distillation term to differ from
# the baseline). SDPO/RLSD support all 6 arms. Identical mapping to
# ../../SDPO_ReAct/ablation/run-qwen3.5-4B-sdpo-react-ablation-mathcodesearch.sh's
# own arm/algo table -- see that script's header for the full flag-derivation
# rationale (self-skill's teacher-forward dispatch, GRPO+skill-KD's required
# --sdpo-logprob-mode sampled, etc.); reproduced here for this domain's own
# reward wiring:
#
#   a  Baseline               (no self-skill; --sdpo-response-prefix defaults to trace)
#   b  +correct skill prefix  --sdpo-self-skill --sdpo-skill-source correct --sdpo-response-prefix skill
#   c  +pitfall skill prefix  --sdpo-self-skill --sdpo-skill-source incorrect --sdpo-pitfall-summary-backend self
#                              (NO --sdpo-response-prefix skill -- a correct peer never has
#                              a skill under skill-source=incorrect, see doc/DESIGN_self_skill.md)
#   d  +all skill prefix      --sdpo-self-skill --sdpo-skill-source all --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill
#   e  +skill-sd both (self-success + pitfall-condense)  = d + --sdpo-skill-kd --sdpo-skill-kd-mode both
#   f  +skill-sd both-blind (blind-correct + pitfall-condense)  = d + --sdpo-skill-kd --sdpo-skill-kd-mode both-blind
#
#   grpo  no --sdpo-rlsd, no --sdpo-kd-loss, no --use-opd. Arm a uses
#         plain_grpo_reward (single-sample, no --group-rm) exactly like
#         run-qwen2.5-7B-sdpo-sci-colocate.sh's own arm 1. Arms e/f use
#         sdpo_group_reward + --no-sdpo-pure-distill + --sdpo-logprob-mode
#         sampled (REQUIRED for skill-KD's teacher forward to run at all even
#         though --sdpo-rlsd stays off -- see the mathcodesearch sibling's
#         header for the actor.py::_compute_sdpo_teacher_log_probs finding).
#   sdpo  --sdpo-kd-loss --sdpo-divergence jsd --sdpo-logprob-mode topk
#         --opd-log-prob-top-k 100 --sdpo-is-clip 2.0 --sdpo-kd-coef 1.0
#         --sdpo-kd-max-tokens 8192 -- exact flags from
#         run-qwen2.5-7B-sdpo-sci-colocate.sh's arm 1.2.
#   rlsd  --sdpo-rlsd --sdpo-rlsd-clip-eps 0.2 --sdpo-rlsd-lambda-init 1.0
#         --sdpo-rlsd-lambda-warmup-steps 0 --use-tis --sdpo-logprob-mode
#         sampled -- exact flags from run-qwen2.5-7B-sdpo-sci-rl-colocate.sh's
#         arm 1.1/3/4/5 (lambda=1.0/no-decay variant, not arm 2's paper-default
#         decay-to-inert schedule). No --sdpo-prefer-tool-use-peer here (that's
#         a native-tool-calling-domain-only escape hatch; sci has no tools).
# All three (when not plain-GRPO arm a) share --sdpo-teacher-backend megatron
# --sdpo-ema-teacher --sdpo-ema-teacher-rate 0.05 --sdpo-self-teacher
# --calculate-per-token-loss --entropy-coef 0.00 --observe-training-entropy.
#
# Grading: MCQ letter-match (sdpo.py's _is_correct default grader -- NO
# --sdpo-grader flag needed anywhere in this script, unlike the react
# scripts' --sdpo-grader dapo).
#
# Qwen2.5-7B-Instruct is NON-THINKING (matches run-qwen2.5-7B-sdpo-sci-colocate.sh's
# own precedent): NO --apply-chat-template-kwargs '{"enable_thinking":...}',
# NO --sdpo-remove-thinking-from-demonstration anywhere (no <think> block to
# strip from a peer's response).
#
# A100-80G retune (NOT copied verbatim from the H200-141GB sci-colocate
# scripts, which explicitly size for 141GB): MAX_TOKENS_PER_GPU default 8192
# (vs H200's 24576), arms e/f (heaviest -- extra skill forward/backward) drop
# further to 4096 (vs H200's 16384), matching the SAME per-arm memory-tuning
# pattern the H200 sci scripts already established for their own heaviest
# arm. --sglang-mem-fraction-static default 0.6 (vs H200's 0.75). TP=1 kept
# (Qwen2.5-7B dense, ~14GB bf16 weights alone comfortably fits one A100; the
# lower token cap buys back the activation/optimizer/KV headroom under
# colocate that TP=2 would otherwise be needed for). --optimizer-cpu-offload
# added (not in the H200 script) since 80G has much less slack than 141G.
# THESE ARE STARTING POINTS, not validated against real A100 hardware (none
# available in this session, confirmed via nvidia-smi showing H200s only) --
# retune further on the first real OOM.
#
# Path parameterization (every local save/load path overridable from
# outside, declared here at the top instead of hardcoded deep in the body):
#   SDPO_ABLATION_DATA_ROOT      (default /root)   sci data dir
#   SDPO_ABLATION_MODEL_ROOT     (default /root)    HF checkpoint + torch_dist dirs
#   SDPO_ABLATION_DUMP_ROOT      (default /root/data/sdpo_dumps)
#   SDPO_ABLATION_CKPT_ROOT      (default /root/data/sdpo_ckpts)
#   SDPO_ABLATION_MEGATRON_PATH  (default /root/Megatron-LM)
#
# Other env overrides:
#   SDPO_ABLATION_ALGO           (required) grpo | sdpo | rlsd
#   SDPO_ABLATION_ARM            (required) a | b | c | d | e | f
#   SDPO_ABLATION_NUM_ROLLOUT    (default 101 -- per spec, single-step/sci uses 101)
#   SDPO_ABLATION_MAX_TOKENS_PER_GPU  (default 8192, 4096 for arms e/f)
#   SDPO_ABLATION_SGLANG_MEM_FRACTION (default 0.6)
#
# usage:
#   SDPO_ABLATION_ALGO=sdpo SDPO_ABLATION_ARM=d \
#     bash examples/SDPO/ablation/run-qwen2.5-7B-sdpo-ablation-sci.sh
set -exf

# --- path knobs (override from outside; defaults match every existing SDPO script) ---
SDPO_ABLATION_DATA_ROOT="${SDPO_ABLATION_DATA_ROOT:-/root}"
SDPO_ABLATION_MODEL_ROOT="${SDPO_ABLATION_MODEL_ROOT:-/root}"
SDPO_ABLATION_DUMP_ROOT="${SDPO_ABLATION_DUMP_ROOT:-/root/data/sdpo_dumps}"
SDPO_ABLATION_CKPT_ROOT="${SDPO_ABLATION_CKPT_ROOT:-/root/data/sdpo_ckpts}"
SDPO_ABLATION_MEGATRON_PATH="${SDPO_ABLATION_MEGATRON_PATH:-/root/Megatron-LM}"

export PYTHONBUFFERED=16
SDPO_ABLATION_ALGO="${SDPO_ABLATION_ALGO:?Set SDPO_ABLATION_ALGO to one of: grpo sdpo rlsd}"
SDPO_ABLATION_ARM="${SDPO_ABLATION_ARM:?Set SDPO_ABLATION_ARM to one of: a b c d e f}"
if [ "$SDPO_ABLATION_ALGO" = "grpo" ]; then
    case "$SDPO_ABLATION_ARM" in
        a|e|f) ;;
        *) echo "GRPO only supports arms a/e/f (got '${SDPO_ABLATION_ARM}')" >&2; exit 1 ;;
    esac
fi
SDPO_ABLATION_NUM_ROLLOUT="${SDPO_ABLATION_NUM_ROLLOUT:-101}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
echo "SDPO_ABLATION_ALGO=${SDPO_ABLATION_ALGO} SDPO_ABLATION_ARM=${SDPO_ABLATION_ARM}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_NAME=Qwen2.5-7B-Instruct
MAX_TOKENS_PER_GPU="${SDPO_ABLATION_MAX_TOKENS_PER_GPU:-8192}"
if [ "$SDPO_ABLATION_ARM" = "e" ] || [ "$SDPO_ABLATION_ARM" = "f" ]; then
    MAX_TOKENS_PER_GPU="${SDPO_ABLATION_MAX_TOKENS_PER_GPU:-4096}"
fi
echo "MODEL: ${MODEL_NAME} | max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"
source "$REPO_ROOT/scripts/models/qwen2.5-7B.sh"

# SciKnowEval MCQ dataset, identical across every arm/algo combo -- the whole
# point of this ablation is to hold data/rollout/eval fixed and vary only the
# algo/skill knobs. Build with: python examples/SDPO/build_sci_dataset.py
SCI_DIR="${SDPO_ABLATION_DATA_ROOT}/sci"
TRAIN_DATA="$SCI_DIR/train.jsonl"
mkdir -p "$SCI_DIR"
[ -f "$TRAIN_DATA" ] || \
    (cd "$REPO_ROOT" && python examples/SDPO/build_sci_dataset.py --out-dir "$SCI_DIR" --val-ratio 0.1)

CFG_TAG="${MODEL_NAME}-sci-${SDPO_ABLATION_ALGO}-${SDPO_ABLATION_ARM}"
SDPO_EXP="${SDPO_EXP:-sdpo-ablation-${CFG_TAG}_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="${SDPO_ABLATION_DUMP_ROOT}/${SDPO_EXP}"
echo "SDPO dump dir: ${DUMP_DIR}"
CKPT_DIR="${SDPO_CKPT_DIR:-${SDPO_ABLATION_CKPT_ROOT}/sdpo-ablation-${CFG_TAG}_ckpt}"

EXPLOG="${SDPO_EXPLOG:-$REPO_ROOT/examples/SDPO/ablation/explog.jsonl}"
GIT_COMMIT="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(cd "$REPO_ROOT" && [ -n "$(git status --porcelain 2>/dev/null)" ] && echo dirty || echo clean)"
python - <<PYLOG || true
import json, time, os
row = {
    "exp": "${SDPO_EXP}", "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "algo": "${SDPO_ABLATION_ALGO}", "arm": "${SDPO_ABLATION_ARM}", "domain": "sci",
    "model": "${MODEL_NAME}", "train_data": "${TRAIN_DATA}", "num_rollout": "${SDPO_ABLATION_NUM_ROLLOUT}",
    "dump_dir": "${DUMP_DIR}", "ckpt_dir": "${CKPT_DIR}", "git_commit": "${GIT_COMMIT}", "git_state": "${GIT_DIRTY}",
    "note": os.environ.get("SDPO_NOTE", ""),
}
with open("${EXPLOG}", "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("explog appended ->", "${EXPLOG}")
print(json.dumps(row, ensure_ascii=False, indent=2))
PYLOG

CKPT_ARGS=(
   --hf-checkpoint "${SDPO_ABLATION_MODEL_ROOT}/${MODEL_NAME}"
   --ref-load "${SDPO_ABLATION_MODEL_ROOT}/${MODEL_NAME}_torch_dist"
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)
if [ "${SDPO_SAVE_CKPT:-0}" = "1" ]; then
    CKPT_ARGS+=(--save "${CKPT_DIR}" --load "${CKPT_DIR}" --save-interval "${SDPO_SAVE_INTERVAL:-10}" --override-opt-param-scheduler)
fi

ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_DATA"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout "${SDPO_ABLATION_NUM_ROLLOUT}"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)
if [ "$SDPO_ABLATION_ARM" = "a" ] && [ "$SDPO_ABLATION_ALGO" = "grpo" ]; then
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
else
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_sdpo_group_has_prefix)
    ROLLOUT_ARGS+=(--sdpo-dynamic-filter-min-correct "${SDPO_MIN_CORRECT:-1}")
fi

# ============================================================================ #
# ARM -> skill-related RM_ARGS -- see the SDPO_ReAct mathcodesearch sibling
# script for the full arm->flag mapping rationale (identical here).
# ============================================================================ #
if [ "$SDPO_ABLATION_ARM" = "a" ] && [ "$SDPO_ABLATION_ALGO" = "grpo" ]; then
    RM_ARGS=(
       --custom-rm-path examples.SDPO.sdpo.plain_grpo_reward
    )
else
    RM_ARGS=(
       --group-rm
       --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
       --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
    )
    case "$SDPO_ABLATION_ARM" in
        a) : ;;  # baseline -- no skill flags, --sdpo-response-prefix defaults to trace
        b)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source correct --sdpo-skill-max-new-tokens 2048 --sdpo-response-prefix skill)
            ;;
        c)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source incorrect --sdpo-skill-max-new-tokens 2048 --sdpo-pitfall-summary-backend self)
            ;;
        d)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source all --sdpo-skill-max-new-tokens 2048 \
                      --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill)
            ;;
        e)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source all --sdpo-skill-max-new-tokens 2048 \
                      --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill \
                      --sdpo-skill-kd --sdpo-skill-kd-coef 0.01 --sdpo-skill-kd-mode both)
            ;;
        f)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source all --sdpo-skill-max-new-tokens 2048 \
                      --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill \
                      --sdpo-skill-kd --sdpo-skill-kd-coef 0.01 --sdpo-skill-kd-mode both-blind)
            ;;
    esac
fi

# ============================================================================ #
# ALGORITHM -> GRPO_ARGS. No --sdpo-prefer-tool-use-peer here (tool-calling-
# domain-only escape hatch; sci has no tools).
# ============================================================================ #
case "$SDPO_ABLATION_ALGO" in
    grpo)
        if [ "$SDPO_ABLATION_ARM" = "a" ]; then
            GRPO_ARGS=(
               --advantage-estimator grpo
               --entropy-coef 0.00
               --observe-training-entropy
            )
        else
            RM_ARGS+=(--no-sdpo-pure-distill)
            GRPO_ARGS=(
               --advantage-estimator grpo
               --sdpo-teacher-backend megatron
               --sdpo-ema-teacher
               --sdpo-ema-teacher-rate 0.05
               --sdpo-logprob-mode sampled
               --sdpo-self-teacher
               --entropy-coef 0.00
               --observe-training-entropy
               --calculate-per-token-loss
            )
        fi
        ;;
    sdpo)
        GRPO_ARGS=(
           --advantage-estimator grpo
           --sdpo-teacher-backend megatron
           --sdpo-ema-teacher
           --sdpo-ema-teacher-rate 0.05
           --sdpo-logprob-mode topk
           --opd-log-prob-top-k 100
           --sdpo-divergence jsd
           --sdpo-is-clip 2.0
           --sdpo-kd-loss
           --sdpo-kd-coef 1.0
           --sdpo-kd-max-tokens 8192
           --sdpo-self-teacher
           --entropy-coef 0.00
           --observe-training-entropy
           --calculate-per-token-loss
        )
        ;;
    rlsd)
        RM_ARGS+=(--no-sdpo-pure-distill)
        GRPO_ARGS=(
           --advantage-estimator grpo
           --sdpo-teacher-backend megatron
           --sdpo-ema-teacher
           --sdpo-ema-teacher-rate 0.05
           --sdpo-logprob-mode sampled
           --sdpo-rlsd
           --sdpo-rlsd-clip-eps 0.2
           --sdpo-rlsd-lambda-init 1.0
           --sdpo-rlsd-lambda-warmup-steps 0
           --use-tis
           --sdpo-self-teacher
           --entropy-coef 0.00
           --observe-training-entropy
           --calculate-per-token-loss
        )
        ;;
    *)
        echo "Unknown SDPO_ABLATION_ALGO='${SDPO_ABLATION_ALGO}' (expected: grpo | sdpo | rlsd)" >&2
        exit 1 ;;
esac

# NO --skip-eval-before-train: a genuine step-0 baseline eval is required so
# every arm's "did training help" comparison is against its own real baseline.
EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data
      sci_chem "${SCI_DIR}/val_chemistry.jsonl"
      sci_bio  "${SCI_DIR}/val_biology.jsonl"
      sci_phys "${SCI_DIR}/val_physics.jsonl"
      sci_mat  "${SCI_DIR}/val_material.jsonl"
   --n-samples-per-eval-prompt 8
   --log-passrate
   --eval-max-response-len 16384
   --eval-top-p 1
   --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-warmup-iters 10
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-sdpo
   --wandb-group "sdpo-ablation-${CFG_TAG}"
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SDPO_ABLATION_SGLANG_MEM_FRACTION:-0.6}"
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

if [ "${SDPO_SAVE_CKPT:-0}" = "1" ]; then
(
    set +f
    while true; do
        sleep 60
        latest_file="${CKPT_DIR}/latest_checkpointed_iteration.txt"
        [ -f "$latest_file" ] || continue
        latest_num="$(cat "$latest_file" 2>/dev/null)"
        [ -n "$latest_num" ] || continue
        case "$latest_num" in ''|*[!0-9]*) continue;; esac
        for d in "${CKPT_DIR}"/iter_*; do
            [ -d "$d" ] || continue
            n="$(basename "$d")"; n="${n#iter_}"
            case "$n" in ''|*[!0-9]*) continue;; esac
            if [ "$((10#$n))" -lt "$((10#$latest_num))" ]; then
                rm -rf "$d"
            fi
        done
    done
) &
PRUNER_PID=$!
trap 'kill "$PRUNER_PID" 2>/dev/null || true' EXIT
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${SDPO_ABLATION_MEGATRON_PATH}/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
        \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
        \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\"
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
