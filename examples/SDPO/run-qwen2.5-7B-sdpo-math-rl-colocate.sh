#!/bin/bash

# RLSD ablation series -- Qwen2.5-7B-Instruct (non-thinking model) on DAPO
# math, single 8x H200 (141GB) node, COLOCATE variant. Sibling of
# run-qwen3-4B-sdpo-math-rl-colocate.sh: SAME arm structure/dataset/eval
# (DAPO math, AIME25+Minerva), SAME RLSD response mechanism (--sdpo-rlsd
# instead of --sdpo-kd-loss), but swapped from Qwen3-4B to Qwen2.5-7B-Instruct.
#
# WHY: three independent arms of the Qwen3-4B math-rl ablation (plain GRPO,
# RLSD-baseline-no-skill, skill-source=correct) each hung at the SAME
# transition point -- right after a weight update, entering the next rollout
# generation batch -- with GPUs at 0% utilization for 5+ minutes and no
# watchdog firing. The hang recurred across totally different SDPO configs at
# different rollout counts (33, 34, ~44), which rules out a config- or
# rollout-count-specific trigger. Since it did NOT recur on
# run-qwen2.5-7B-sdpo-sci-rl-colocate.sh's own arms (same RLSD/skill-KD code
# paths, same colocate architecture, different model + dataset), swapping the
# model to Qwen2.5-7B-Instruct isolates whether Qwen3-4B itself (its chat
# template, no-think config, or something in its weight/KV-cache layout) is
# implicated, while the actual hang root cause is investigated separately.
#
# Qwen2.5-7B-Instruct has no thinking mode, so (unlike the Qwen3-4B script)
# there is no --apply-chat-template-kwargs '{"enable_thinking":...}' to set on
# either rollout or skill generation, and no --sdpo-remove-thinking-from-
# demonstration to drop.
#
#   SDPO_ABLATION_ARM=1    pure vanilla GRPO, no SDPO at all
#   SDPO_ABLATION_ARM=1.1  RLSD baseline, NO self-skill at all -- correct-peer
#                          FULL raw trace as the response teacher prefix.
#   SDPO_ABLATION_ARM=2    + self-skill, skill-source correct only, NO skill-KD
#   SDPO_ABLATION_ARM=3    + self-skill, skill-source incorrect only, NO skill-KD
#   SDPO_ABLATION_ARM=4    + self-skill, skill-source all, NO skill-KD
#   SDPO_ABLATION_ARM=5.1  + self-skill, skill-source all, WITH skill-KD
#                          (mode=both), --sdpo-skill-max-new-tokens 2048.
#   SDPO_ABLATION_ARM=5.2  IDENTICAL to arm 5.1 except --sdpo-skill-kd-mode
#                          both-blind (was both).
#
# lambda: every arm uses lambda=1.0 constant / warmup-steps 0 (RLSD's
# reweighting active for the full run) -- same choice as the Qwen3-4B
# math-rl script, already justified by the sci-rl script's own arm 2 vs 2.2
# comparison.
#
# --sdpo-rlsd requires --sdpo-teacher-backend megatron --sdpo-logprob-mode
# sampled (enforced in miles/utils/arguments.py's validate_args). Mutually
# exclusive with --sdpo-kd-loss / --use-opd.
#
# --use-tis on every SDPO arm: corrects train-vs-rollout-engine log-prob
# drift, orthogonal to RLSD's own delta_t/credit_t.
#
# --no-sdpo-pure-distill on every SDPO arm: RLSD's advantage reweighting has
# nothing to reweight if the GRPO advantage is always 0.
#
# Grading: --sdpo-grader dapo on every SDPO arm.
#
# usage:
#   SDPO_ABLATION_ARM=1   bash examples/SDPO/run-qwen2.5-7B-sdpo-math-rl-colocate.sh
#   SDPO_ABLATION_ARM=1.1 bash examples/SDPO/run-qwen2.5-7B-sdpo-math-rl-colocate.sh
#   ... etc for 2, 3, 4, 5.1, 5.2

set -exf

export PYTHONBUFFERED=16
SDPO_ABLATION_ARM="${SDPO_ABLATION_ARM:?Set SDPO_ABLATION_ARM to one of: 1 1.1 2 3 4 5.1 5.2}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
echo "SDPO_ABLATION_ARM: ${SDPO_ABLATION_ARM}"

source "/root/miles/scripts/models/qwen2.5-7B.sh"

SDPO_EXP="${SDPO_EXP:-qwen2.5-7B-sdpo-math-rl-ablation-arm${SDPO_ABLATION_ARM}_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/miles/sdpo_dumps/${SDPO_EXP}"
echo "SDPO dump dir: ${DUMP_DIR}"

# NO --save / --load: exploratory ablation only, nothing here should be
# resumed or kept around.
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-7B-Instruct
   --ref-load /root/Qwen2.5-7B-Instruct_torch_dist
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)

# DAPO math train set, identical across all arms and to the Qwen3-4B math-rl
# script -- the whole point of this model-swap sibling is to hold
# data/rollout/eval/arm-structure fixed and vary only the base model.
ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
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
        # Plain GRPO, no SDPO machinery at all.
        RM_ARGS=(
            --custom-rm-path examples.SDPO.sdpo.plain_grpo_reward
            --sdpo-grader dapo
        )
        GRPO_ARGS=(
            --advantage-estimator grpo
            --entropy-coef 0.00
            --observe-training-entropy
        )
        ;;
    1.1)
        # RLSD baseline, NO self-skill at all.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
            --sdpo-grader dapo
        )
        GRPO_ARGS=(
            --advantage-estimator grpo
            --no-sdpo-pure-distill
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
            --sdpo-answer-tag answer
            --entropy-coef 0.00
            --observe-training-entropy
            --calculate-per-token-loss
        )
        ;;
    2)
        # + self-skill, skill-source correct ONLY, no skill-KD.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
            --sdpo-grader dapo
        )
        GRPO_ARGS=(
            --advantage-estimator grpo
            --no-sdpo-pure-distill
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
        # + self-skill, skill-source incorrect ONLY (pitfall warnings), no
        # skill-KD. NO --sdpo-response-prefix skill here (skill-source
        # incorrect means no correct peer ever generates a skill).
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
            --sdpo-grader dapo
        )
        GRPO_ARGS=(
            --advantage-estimator grpo
            --no-sdpo-pure-distill
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
        # + self-skill, skill-source ALL, no skill-KD.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
            --sdpo-grader dapo
        )
        GRPO_ARGS=(
            --advantage-estimator grpo
            --no-sdpo-pure-distill
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
    5.1)
        # + self-skill, skill-source ALL, WITH skill-KD (mode=both).
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
            --sdpo-grader dapo
        )
        GRPO_ARGS=(
            --advantage-estimator grpo
            --no-sdpo-pure-distill
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
            --sdpo-answer-tag answer
            --sdpo-self-skill
            --sdpo-skill-source all
            --sdpo-skill-max-new-tokens 2048
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
    5.2)
        # IDENTICAL to arm 5.1 except --sdpo-skill-kd-mode both-blind.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
            --sdpo-grader dapo
        )
        GRPO_ARGS=(
            --advantage-estimator grpo
            --no-sdpo-pure-distill
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
            --sdpo-answer-tag answer
            --sdpo-self-skill
            --sdpo-skill-source all
            --sdpo-skill-max-new-tokens 2048
            --sdpo-pitfall-summary-backend self
            --sdpo-response-prefix skill
            --sdpo-skill-kd
            --sdpo-skill-kd-coef 0.01
            --sdpo-skill-kd-mode both-blind
            --entropy-coef 0.00
            --observe-training-entropy
            --calculate-per-token-loss
        )
        ;;
    *)
        echo "Unknown SDPO_ABLATION_ARM='${SDPO_ABLATION_ARM}' (expected one of: 1 1.1 2 3 4 5.1 5.2)" >&2
        exit 1
        ;;
esac

# Identical across all arms: AIME-2025 + Minerva-Math eval.
EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data
      aime25   /root/math_eval/aime25.jsonl
      minerva  /root/math_eval/minerva_math.jsonl
   --n-samples-per-eval-prompt 8
   --log-passrate
   --eval-max-response-len 16384
   --eval-top-p 1
   --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
)

# Same GPU/perf sizing as run-qwen2.5-7B-sdpo-sci-rl-colocate.sh (same base
# model, same skill-KD memory-pressure OOM fix for arms 5.1/5.2).
MAX_TOKENS_PER_GPU=24576
case "${SDPO_ABLATION_ARM}" in
   5.1|5.2)
      MAX_TOKENS_PER_GPU=16384
      ;;
esac
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
   --wandb-group "qwen2.5-7B-sdpo-math-rl-ablation-arm${SDPO_ABLATION_ARM}"
   --wandb-key "${WANDB_API_KEY}"
)

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
