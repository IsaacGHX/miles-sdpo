？#!/bin/bash

# SDPO (self-distilled policy optimization) — Olmo-3-7B-Instruct on a single
# 8x H200 (141GB) node. Second official lasgroup/SDPO SciKnowEval model.
# usage: bash examples/SDPO/run-olmo3-7B-sdpo.sh

set -exf

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/miles/scripts/models/olmo3-7B.sh"

SDPO_EXP="${SDPO_EXP:-olmo3-7B-sdpo-sci_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/miles/sdpo_dumps/${SDPO_EXP}"
echo "SDPO dump dir: ${DUMP_DIR}"

CKPT_ARGS=(
   --hf-checkpoint /root/Olmo-3-7B-Instruct
   --ref-load /root/Olmo-3-7B-Instruct_torch_dist
   --load /root/Olmo-3-7B-Instruct_miles/
   --save /root/Olmo-3-7B-Instruct_miles/
   --save-interval 50
   --dump-details "${DUMP_DIR}"
   # skip the heavy per-token train_data/*.pt and policy_loss_debug/*.pt dumps;
   # keep rollout_data + sdpo_prompts + skill dumps.
   --no-dump-train-data
   --no-dump-policy-loss-debug
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
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
)

RM_ARGS=(
   --group-rm
   --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
   --sdpo-grader dapo
   # Self-generated skill (SkillOpt, on-policy): the CURRENT policy writes the skill
   # during rollout from a trace's own response. On-policy + trainable (unlike the
   # external --sdpo-trace-condense). Used by skill-KD below and, via
   # --sdpo-response-prefix skill, as the response teacher prefix.
   --sdpo-self-skill
   --sdpo-skill-max-new-tokens 512
   # Which traces get a skill (does NOT change the response teacher prefix):
   # correct | incorrect | env_feedback (placeholder) | all.
   --sdpo-skill-source correct
   # Second SDPO objective on the skill's own tokens: teacher = skill-gen prompt +
   # the sample's OWN correct trace as hint (self-success mode). Independent coef.
   --sdpo-skill-kd
   --sdpo-skill-kd-coef 1.0
   --sdpo-skill-kd-mode self-success
   # Response-SDPO teacher prefix = the correct peer's SKILL (not its full trace).
   # Needs --sdpo-self-skill + --sdpo-skill-source covering correct peers (it does).
   # Falls back to the trace if a peer has no skill.
   --sdpo-response-prefix skill
   # (see examples/SDPO/DESIGN_self_skill.md)
)

EVAL_ARGS=(
   --eval-interval 10
   # --skip-eval-before-train
   # AIME-2025 (integer answers) and
   # Minerva Math (LaTeX answers) as separate eval sets; graded by the general math
   # grader in sdpo_eval_reward (handles both).
   --eval-prompt-data
      aime25   /root/math_eval/aime25.jsonl
      minerva  /root/math_eval/minerva_math.jsonl
   --n-samples-per-eval-prompt 8
   --log-passrate
   # Eval response cap: hard competition math needs long reasoning chains, so give
   # it 16K (double the 8K rollout cap). WITHOUT this flag eval silently falls back
   # to --rollout-max-response-len (8192) via eval_config.pick_from_args, truncating
   # long answers and understating accuracy.
   --eval-max-response-len 16384
   --eval-top-p 1
   --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
)

PERF_ARGS=(
   # TP=1 (DP=4): Olmo3's QK-norm spans the FULL q/k dim (num_heads*head_dim), so
   # it cannot be split across TP ranks without a cross-TP allreduce of the RMS
   # (TP would normalize over only the local head shard -> wrong). TP=1 keeps each
   # RMSNorm over the full dim. A 7B model fits comfortably on one H200 at TP=1.
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
   --use-opd
   --opd-type sglang
   --opd-kl-coef 0.0
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
   --sdpo-pure-distill
   --sdpo-answer-tag answer
   --entropy-coef 0.00
   --observe-training-entropy
   --calculate-per-token-loss
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --lr-warmup-iters 10
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-sdpo
   --wandb-group olmo3-7B-sdpo-sci
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # 0.9 (was 0.8): at 0.8 the rollout GPUs sat at ~82% (118/143 GB), ~25GB idle.
   # 8k generations have a lower logits/activation peak than 16k, so 0.9 gives the
   # KV cache more room (higher throughput) while keeping ~10% headroom. Drop back
   # to 0.8 if the logits_processor OOMs on a long batch.
   --sglang-mem-fraction-static 0.9
   --sglang-chunked-prefill-size 8192
   --sglang-router-policy round_robin
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# 8x H200, DISAGGREGATED 4+4: 4 GPUs train (TP2 x DP2), 4 run SGLang rollout.
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
        \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
        \"OPENAI_API_KEY\": \"${OPENAI_API_KEY}\",
        \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\"
     }
   }" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
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
