#!/bin/bash

# SDPO — NVIDIA-Nemotron-3-Nano-4B-BF16 on SciKnowEval, single 8x H200 node,
# COLOCATE variant of run-nemotron3-4b-sdpo-sci.sh.
# usage: bash examples/SDPO/run-nemotron3-4b-sdpo-sci-colocate.sh
#
# WHY THIS VARIANT — the sibling (disaggregated) script splits the node 4+4: 4
# GPUs train, 4 serve SGLang. The main loop (train.py) is strictly serial
# (generate -> train -> update_weights) with NO rollout/train overlap, so under
# 4+4 each phase leaves half the node idle. Colocate puts BOTH the actor and the
# rollout engines on all 8 GPUs and time-shares them via offload/onload:
#   - rollout uses 8 engines instead of 4  -> ~2x generation throughput
#   - training uses 8 GPUs instead of 4    -> faster actor step
# For a 4B model the per-step offload/onload cost (weights ~8GB) is small, so
# total step time should drop. Compare wall-clock/step against the 4+4 script.
#
# SDPO CORRECTNESS IS UNCHANGED vs the disaggregated script: the EMA teacher,
# self-teacher and teacher forward are all TRAINING-side in-place CPU<->GPU weight
# swaps (TensorBackuper, snapshots live in CPU pinned RAM), fully decoupled from
# how weights reach the rollout engines. Only the parallelism / colocate / memory
# block and the ray launch differ from run-nemotron3-4b-sdpo-sci.sh; every
# SDPO/GRPO/skill/eval knob below is copied verbatim.
#
# MODEL NOTE — Nemotron-3-Nano-4B is DENSE (`nemotron_h` = hybrid Mamba + Attention,
# NO MoE). It loads via AutoBridge (--megatron-to-hf-mode bridge); colocate's
# UpdateWeightFromTensor already uses the AutoBridge HF iterator, so the nemotron_h
# Megatron->HF conversion works out of the box (the hand-written-converter gap that
# only ever affected the distributed transports is irrelevant here).

set -exf

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/miles/scripts/models/nemotron-3-nano-4b.sh"

SDPO_EXP="${SDPO_EXP:-nemotron3-4b-sdpo-sci-colocate_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/miles/sdpo_dumps/${SDPO_EXP}"
echo "SDPO dump dir: ${DUMP_DIR}"

CKPT_ARGS=(
   # AutoBridge builds the full nemotron_h Megatron provider (incl. all Mamba
   # fields) from HF config.json at load time, so ref-load reads the SAME HF
   # checkpoint directly — there is no separate _torch_dist conversion like Olmo3.
   --hf-checkpoint /root/NVIDIA-Nemotron-3-Nano-4B-BF16
   --ref-load /root/NVIDIA-Nemotron-3-Nano-4B-BF16
   --save /root/NVIDIA-Nemotron-3-Nano-4B-BF16_miles/
   --megatron-to-hf-mode bridge
   --save-interval 50
   --dump-details "${DUMP_DIR}"
   # skip the heavy per-token train_data/*.pt and policy_loss_debug/*.pt dumps;
   # keep rollout_data + sdpo_prompts + skill dumps.
   --no-dump-train-data
   --no-dump-policy-loss-debug
)

# SciKnowEval: all four domains' train splits mixed into one jsonl for training;
# per-domain val splits registered for evaluation.
# Build with: python examples/SDPO/build_sci_dataset.py --out-dir /root/sci
ROLLOUT_ARGS=(
   --prompt-data /root/sci/train.jsonl
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
   # SDPO dynamic sampling (DAPO-style): oversample groups and DROP any with < 2
   # correct traces, then re-sample, so every kept group can give EVERY trace a
   # correct-peer prefix (no dead teacher==student, zero-KD samples reaching the
   # batch). check_sdpo_group_has_prefix reads sdpo_correct set by the group RM.
   # over-sampling-batch-size 64 = 2x rollout-batch-size: gives headroom to discard
   # low-correct groups. If the model is too weak early on (few groups reach 2
   # correct), the rollout loop can stall filling the batch — raise oversampling,
   # lower --sdpo-dynamic-filter-min-correct to 1, or warm up without the filter.
   # --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_sdpo_group_has_prefix
   # --sdpo-dynamic-filter-min-correct 2
   # --over-sampling-batch-size 64
)

RM_ARGS=(
   --group-rm
   --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
   # SciKnowEval is well-formed L3 multiple-choice: the label is the answer LETTER,
   # graded by a case-insensitive letter match on the extracted <answer> ('mcq').
   --sdpo-grader mcq
   # -------------------------------------------------------------------------
   # SKILL MECHANISM DISABLED -> base SDPO. (Kept identical to the disaggregated
   # script; re-enable the block below to turn the self-skill setup back on.)
     --sdpo-self-skill
     --sdpo-skill-max-new-tokens 1024
     --sdpo-skill-source correct
   #   --sdpo-skill-kd-coef 1.0
   #   --sdpo-skill-kd-mode self-success
     --sdpo-response-prefix skill
   # -------------------------------------------------------------------------
)

EVAL_ARGS=(
   --eval-interval 10
   # All four SciKnowEval domains as name/path pairs in ONE --eval-prompt-data.
   --eval-prompt-data
      sci_chem /root/sci/val_chemistry.jsonl
      sci_bio  /root/sci/val_biology.jsonl
      sci_phys /root/sci/val_physics.jsonl
      sci_mat  /root/sci/val_material.jsonl
   --n-samples-per-eval-prompt 8
   --log-passrate
   --eval-max-response-len 16384
   --eval-top-p 1
   # SDPO trains with a group RM, which cannot score eval samples (no group step in
   # eval). This per-sample eval RM grades eval pass@1 with the same correctness
   # rule as sdpo_group_reward. Required for eval under --group-rm.
   --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
)

PERF_ARGS=(
   # Nemotron-3-Nano-4B is a Mamba+Attention hybrid. TP=1 (matches the disaggregated
   # script and scripts/run-nemotron-3-nano-4b.sh, the known-working smoke test);
   # the Mamba mixer is handled by the bridge provider. Under colocate all 8 GPUs
   # host the actor (DP=8, TP=1). Dense -> expert parallelism is N/A.
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   # 16384 (was 49152): TP=1 means each GPU holds the full 4B model + Adam state
   # (~80GB); colocate offloads the rollout engine during training, so the training
   # activation budget is the same as the disaggregated script — 49152 tokens/
   # microbatch OOMed the Mamba SSM forward there, so match its 16384. Raise toward
   # 24576 if memory sits idle.
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 0.0
   # Megatron self-teacher. First use on the Mamba hybrid — if the teacher forward
   # errors or drifts, switch to `--sdpo-teacher-backend sglang`.
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
   # Nemotron-3-Nano-4B is a thinking model: its chat template defaults to
   # enable_thinking=True and ends the prompt with '<|im_start|>assistant\n<think>\n',
   # so every rollout response carries a <think>...</think> chain before the
   # requested <reasoning>/<answer> format. Strip it from the peer solution before
   # it becomes the teacher prefix — otherwise the prefix balloons and the teacher
   # teaches the student to echo the reasoning verbatim. Same reason as the Qwen3
   # script; REQUIRED for thinking models. (Grading strips <think> unconditionally,
   # so this only affects the teacher prefix, not scoring.)
   --sdpo-remove-thinking-from-demonstration
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
   --wandb-group nemotron3-4b-sdpo-sci
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   # Colocate: one engine per GPU (TP1), all 8 GPUs. mem-fraction 0.7 (was 0.9 in
   # the disaggregated script) because the actor + optimizer + KV cache now share
   # each GPU; colocate onloads/offloads around each phase, and 0.7 is the value
   # the other 4B colocate examples use (retool/search-r1). Raise cautiously if
   # rollout underutilizes memory; drop if the logits_processor OOMs on a long batch.
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.85
   --sglang-chunked-prefill-size 10240
   --sglang-router-policy round_robin
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# 8x H200, COLOCATE: all 8 GPUs run BOTH the actor (DP8 x TP1) and the SGLang
# rollout engines (8 x TP1), time-shared via offload/onload. --colocate implies
# offload_train/offload_rollout and forces rollout_num_gpus == actor gpus.
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
        \"OPENAI_API_KEY\": \"${OPENAI_API_KEY}\",
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
