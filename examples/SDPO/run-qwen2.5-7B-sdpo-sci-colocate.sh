#!/bin/bash

# SDPO ablation series -- Qwen2.5-7B-Instruct on SciKnowEval (MCQ), single 8x
# H200 (141GB) node, COLOCATE variant. Sibling of
# run-qwen3-4B-sdpo-math-colocate.sh: SAME 6-arm structure and SAME
# $SDPO_ABLATION_ARM switch convention, just swapped to the sci/MCQ domain and
# a non-thinking model:
#
#   SDPO_ABLATION_ARM=1    pure vanilla GRPO, no SDPO at all: single-sample
#                          reward (plain_grpo_reward), no --group-rm, default
#                          std-normalized advantages, no dynamic sampling, no
#                          --calculate-per-token-loss (miles' own default
#                          seq-mean-token-mean aggregation)
#   SDPO_ABLATION_ARM=1.2  SDPO baseline: --group-rm + KD loss (jsd), no skill
#                          anything. PURE distillation for every SDPO arm
#                          below (1.2-5): --sdpo-pure-distill is the DEFAULT
#                          and deliberately left unset (not passed as
#                          --no-sdpo-pure-distill) -- task reward is 0 for
#                          every trace whenever --group-rm's sdpo_group_reward
#                          runs, so the GRPO advantage is exactly 0 and the
#                          entire training signal is the JSD divergence loss
#                          alone: no reward, no advantage, purely KD. Named
#                          "1.2" (not "1.1") to match the math script's
#                          numbering convention, where "1.1" was the FIRST
#                          SDPO-baseline attempt and "1.2" the one with a
#                          grading-bug fix already applied -- this sci script
#                          starts directly from the fixed grader
#                          (examples/SDPO/sdpo.py's _is_correct dapo-path fix
#                          is math-only and irrelevant to MCQ letter-matching
#                          anyway), so there is no separate "1.1" leg here.
#   SDPO_ABLATION_ARM=2    + self-skill, skill-source correct only, NO skill-KD
#   SDPO_ABLATION_ARM=3    + self-skill, skill-source incorrect only, NO skill-KD
#   SDPO_ABLATION_ARM=4    + self-skill, skill-source all, NO skill-KD
#   SDPO_ABLATION_ARM=5    + self-skill, skill-source all, WITH skill-KD (mode=both)
#
# This is an EXPLORATORY ablation, not a full training run:
#   - --num-rollout 100 (not 500) for every arm.
#   - NO checkpointing (no --save/--load) -- nothing here is meant to be resumed
#     or reused; keeping it off also means this can't accidentally repeat the
#     disk-fill crash a full SDPO_ReAct run hit earlier this session.
#
# Qwen2.5-7B-Instruct is NON-THINKING (matches the official lasgroup/SDPO model
# choice, and run-qwen3-8B-sdpo.sh's own precedent -- see that script's
# docstring for why Qwen3's reasoning-collapse risk is avoided by picking
# Qwen2.5 here). Consequently, UNLIKE run-qwen3-4B-sdpo-math-colocate.sh:
#   - NO --apply-chat-template-kwargs '{"enable_thinking":...}' anywhere (the
#     chat template has no such kwarg; passing it would be a silent no-op at
#     best, so it is simply omitted).
#   - NO --sdpo-remove-thinking-from-demonstration on any arm: there is no
#     <think> block in a peer's response to strip, so this flag would be a
#     no-op that just adds a redundant _strip_thinking_blocks() scan per
#     splice. Omitted for clarity, matching run-olmo3-7B-sdpo-sci-colocate.sh's
#     own choice for its (also non-thinking-prefix-stripped) sci runs.
#
# Grading: MCQ letter-match (sdpo.py's _is_correct default grader, "mcq" --
# NO --sdpo-grader flag needed, unlike the math script's --sdpo-grader dapo).
# SciKnowEval's label is the answer LETTER; the model is asked to emit it
# inside <answer>...</answer> (see examples/SDPO/build_sci_dataset.py).
#
# GPU/perf sizing: reused run-olmo3-7B-sdpo-sci-colocate.sh's PROVEN-SAFE
# combo (--max-tokens-per-gpu 24576 + --recompute-granularity full) rather
# than re-deriving a new number -- Qwen2.5-7B (28 layers, hidden=3584) and
# Olmo-3-7B (32 layers, hidden=4096) are close enough in size that the same
# combo should have comparable headroom under colocate; watch arm 1's actual
# backward pass for a cleaner OOM margin signal before trusting this for the
# remaining 5 arms (same caution the math script's own docstring flags).
#
# LR: 1e-6, matching run-qwen3-4B-sdpo-math-colocate.sh's reasoning (the mixed
# GRPO+KD objective needs the smaller, GRPO-safe LR so arm 1's real GRPO
# advantage doesn't blow up) -- kept ONE LR across all 6 arms so the ablation
# isolates the SDPO/skill knobs, not LR. NOT run-qwen3-8B-sdpo.sh's 1e-5
# (that script's arm is pure-KD-only tuning, no arm-1-style real-GRPO leg to
# stay compatible with).
#
# usage:
#   SDPO_ABLATION_ARM=1   bash examples/SDPO/run-qwen2.5-7B-sdpo-sci-colocate.sh
#   SDPO_ABLATION_ARM=1.2 bash examples/SDPO/run-qwen2.5-7B-sdpo-sci-colocate.sh
#   ... etc for 2, 3, 4, 5

set -exf

export PYTHONBUFFERED=16
SDPO_ABLATION_ARM="${SDPO_ABLATION_ARM:?Set SDPO_ABLATION_ARM to one of: 1 1.2 2 3 4 5}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
echo "SDPO_ABLATION_ARM: ${SDPO_ABLATION_ARM}"

source "/root/miles/scripts/models/qwen2.5-7B.sh"

SDPO_EXP="${SDPO_EXP:-qwen2.5-7B-sdpo-sci-ablation-arm${SDPO_ABLATION_ARM}_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/miles/sdpo_dumps/${SDPO_EXP}"
echo "SDPO dump dir: ${DUMP_DIR}"

# NO --save / --load: exploratory ablation only, nothing here should be
# resumed or kept around (see module docstring above).
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-7B-Instruct
   --ref-load /root/Qwen2.5-7B-Instruct_torch_dist
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)

# SciKnowEval MCQ train set, identical across all 6 arms -- the whole point of
# the ablation is to hold data/rollout/eval fixed and vary only the SDPO/skill
# knobs. Build with: python examples/SDPO/build_sci_dataset.py --out-dir /root/sci
ROLLOUT_ARGS=(
   --prompt-data /root/sci/train.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 100
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

# --- ablation-arm-specific RM_ARGS / GRPO_ARGS -------------------------------
case "${SDPO_ABLATION_ARM}" in
    1)
        # Plain GRPO, no SDPO machinery at all: single-sample reward, no
        # --group-rm, no --use-opd/--sdpo-* flags whatsoever.
        # plain_grpo_reward reuses sdpo.py's own _is_correct under the DEFAULT
        # "mcq" grader (no --sdpo-grader flag needed here, unlike the math
        # script's --sdpo-grader dapo) -- the SAME grading criterion arms
        # 1.2-5 use via sdpo_group_reward, so the ablation isolates the
        # SDPO/skill knobs, not a grading-rule difference.
        RM_ARGS=(
            --custom-rm-path examples.SDPO.sdpo.plain_grpo_reward
        )
        GRPO_ARGS=(
            --advantage-estimator grpo
            --entropy-coef 0.00
            --observe-training-entropy
            # NO --calculate-per-token-loss here (unlike every other arm) --
            # this is the "pure vanilla GRPO" baseline: std-normalized
            # advantages (default grpo_std_normalization=True), no dynamic
            # sampling, and miles' own default seq-mean-token-mean loss
            # aggregation (NOT the token-mean every SDPO arm below uses).
        )
        ;;
    1.2)
        # SDPO baseline: group-rm + real KD loss (jsd divergence), self-teacher,
        # EMA teacher -- PURE distillation (--sdpo-pure-distill is the DEFAULT,
        # left unset here rather than passed explicitly, matching every other
        # SDPO arm below): sdpo_group_reward returns task reward 0 for every
        # trace whenever SDPO is active, so the GRPO advantage is exactly 0 and
        # the ENTIRE training signal is -sdpo_kd_coef * JSD(student‖teacher) --
        # no reward, no advantage, purely the divergence loss, per explicit
        # instruction. NO skill generation, NO skill-KD. This isolates "does
        # the base SDPO peer-prefix KD help at all" before layering skill on.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
        )
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
            --sdpo-answer-tag answer
            --entropy-coef 0.00
            --observe-training-entropy
            --calculate-per-token-loss
        )
        ;;
    2)
        # + self-skill, skill-source correct ONLY, no skill-KD. Response-SDPO
        # teacher prefix switches to the peer's SKILL (not the full trace) once
        # self-skill is on, matching run-olmo3-7B-sdpo-sci-colocate.sh's own
        # choice (--sdpo-response-prefix skill), per explicit instruction to
        # use skill-as-prefix whenever self-skill is active.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
        )
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
            --sdpo-answer-tag answer
            --sdpo-self-skill
            --sdpo-skill-source correct
            --sdpo-skill-max-new-tokens 1024
            --sdpo-response-prefix skill
            --entropy-coef 0.00
            --observe-training-entropy
            --calculate-per-token-loss
        )
        ;;
    3)
        # + self-skill, skill-source incorrect ONLY (pitfall warnings from
        # failed traces), no skill-KD.
        # NO --sdpo-response-prefix skill here (unlike arms 2/4/5): the
        # response-SDPO teacher prefix's peer is ALWAYS drawn from
        # correct_indices (sdpo.py's peer-selection pass runs unconditionally,
        # before/independent of --sdpo-skill-source). Under skill-source=
        # incorrect, only FAILED traces ever get a self-generated skill
        # (_skill_eligible returns `not self_ok`) -- a correct peer NEVER has
        # an sdpo_skill. So --sdpo-response-prefix skill's lookup would be
        # None for every sample and silently fall back to the peer's full raw
        # trace every single time (see doc/DESIGN_self_skill.md's own
        # warning). Dropping the flag makes the arm test exactly what its
        # comment says: trace prefix + pitfall-injection-from-failures, no
        # skill-KD (same fix already applied to run-qwen3-4B-sdpo-math-
        # colocate.sh's own arm 3).
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
        )
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
            --sdpo-answer-tag answer
            --sdpo-self-skill
            --sdpo-skill-source incorrect
            --sdpo-skill-max-new-tokens 1024
            --sdpo-pitfall-summary-backend self
            --entropy-coef 0.00
            --observe-training-entropy
            --calculate-per-token-loss
        )
        ;;
    4)
        # + self-skill, skill-source ALL (correct traces -> solution roadmap,
        # incorrect traces -> pitfall warnings), no skill-KD.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
        )
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
            --sdpo-answer-tag answer
            --sdpo-self-skill
            --sdpo-skill-source all
            --sdpo-skill-max-new-tokens 1024
            --sdpo-pitfall-summary-backend self
            --sdpo-response-prefix skill
            --entropy-coef 0.00
            --observe-training-entropy
            --calculate-per-token-loss
        )
        ;;
    5)
        # + self-skill, skill-source ALL, WITH skill-KD (mode=both: correct
        # traces get self-success solution-skill KD, failed traces get
        # pitfall-condense KD -- "both" requires skill-source all, per
        # arguments.py's own assertion in sdpo.py). --sdpo-skill-kd-coef 0.01,
        # matching run-qwen3-4B-sdpo-math-colocate.sh's own arm 5 value.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
        )
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
            --sdpo-answer-tag answer
            --sdpo-self-skill
            --sdpo-skill-source all
            --sdpo-skill-max-new-tokens 1024
            --sdpo-pitfall-summary-backend self
            --sdpo-response-prefix skill
            --sdpo-skill-kd
            --sdpo-skill-kd-coef 0.01
            --sdpo-skill-kd-mode both
            --entropy-coef 0.00
            --observe-training-entropy
            --calculate-per-token-loss
        )
        ;;
    *)
        echo "Unknown SDPO_ABLATION_ARM='${SDPO_ABLATION_ARM}' (expected one of: 1 1.2 2 3 4 5)" >&2
        exit 1
        ;;
esac

# Identical across all 6 arms: all four SciKnowEval domains (already staged
# under /root/sci on this host -- see build_sci_dataset.py). Arm 1 (plain
# GRPO, no --group-rm) still uses sdpo_eval_reward: it works standalone (just
# grades pass@1 via the same MCQ letter-match grader) and keeps the eval
# metric computation identical across every arm regardless of training-side
# reward wiring.
#
# NO --skip-eval-before-train: a genuine step-0 (untrained-checkpoint) eval is
# required so every arm's "did training help" comparison is against its OWN
# real baseline (same reasoning as run-qwen3-4B-sdpo-math-colocate.sh's own
# EVAL_ARGS comment).
EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data
      sci_chem /root/sci/val_chemistry.jsonl
      sci_bio  /root/sci/val_biology.jsonl
      sci_phys /root/sci/val_physics.jsonl
      sci_mat  /root/sci/val_material.jsonl
   --n-samples-per-eval-prompt 8
   --log-passrate
   --eval-max-response-len 16384
   --eval-top-p 1
   --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
)

# See module docstring for the Qwen2.5-7B-vs-Olmo3-7B reasoning behind these
# two numbers (reused run-olmo3-7B-sdpo-sci-colocate.sh's proven-safe combo).
# --max-tokens-per-gpu 24576 OOM'd arm 5's actual backward pass ("CUDA out of
# memory... Tried to allocate 13.20 GiB... 13.02 GiB is free" on GPU 7,
# rollout 14/100) -- confirmed arms 1-4 (all lighter: no skill-KD) complete
# cleanly at 24576, so this is arm-5-specific, not a wrong number for the
# whole script. Arm 5 alone appends EXTRA skill-KD samples to the training
# batch (self-success + pitfall-condense skill sequences each need their own
# forward+backward), pushing peak activation memory higher than the response-
# only arms even though the model/PERF config is otherwise identical -- so
# only arm 5 gets a lower per-GPU token cap, matching this ablation's own
# math-script precedent of tuning memory knobs per the arm that actually
# needs it rather than uniformly discounting every arm.
MAX_TOKENS_PER_GPU=24576
if [ "${SDPO_ABLATION_ARM}" = "5" ]; then
   MAX_TOKENS_PER_GPU=16384
fi
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
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-sdpo
   --wandb-group "qwen2.5-7B-sdpo-sci-ablation-arm${SDPO_ABLATION_ARM}"
   --wandb-key "${WANDB_API_KEY}"
)

# --sglang-mem-fraction-static 0.75 -- reused run-qwen3-4B-sdpo-math-colocate.sh's
# post-OOM-fix value directly rather than re-deriving one for this model/domain;
# watch arm 1's rollout engine for the same "KV cache usage 0.95-1.00 + scratch
# spike OOM" signature that script's own docstring documents before trusting
# this blindly for the remaining 5 arms.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.75
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

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"/root/Megatron-LM/\",
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
