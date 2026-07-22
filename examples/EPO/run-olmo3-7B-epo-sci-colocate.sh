#!/bin/bash

# EPO (PMI-credit self-distillation) — Olmo-3-7B-Instruct on SciKnowEval, single
# 8x H200 (141GB) node, COLOCATE variant. Sci counterpart of
# run-olmo3-7B-epo-math-colocate.sh: same colocate / EPO / eval config, only the
# TASK differs (SciKnowEval train set, mcq grader, 4-domain val eval).
# usage: bash examples/EPO/run-olmo3-7B-epo-sci-colocate.sh
#
# Build the dataset first with:
#   python examples/SDPO/build_sci_dataset.py --out-dir /root/sci --val-ratio 0.1

set -exf

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/miles/scripts/models/olmo3-7B.sh"

EPO_EXP="${EPO_EXP:-olmo3-7B-epo-sci-colocate_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/miles/sdpo_dumps/${EPO_EXP}"
echo "EPO dump dir: ${DUMP_DIR}"

CKPT_ARGS=(
   --hf-checkpoint /root/Olmo-3-7B-Instruct
   --ref-load /root/Olmo-3-7B-Instruct_torch_dist
   --load /root/Olmo-3-7B-Instruct_miles/
   --save /root/Olmo-3-7B-Instruct_miles/
   --save-interval 50
   --dump-details "${DUMP_DIR}"
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
   # DAPO-style dynamic sampling: drop degenerate all-same-reward groups
   # (all-correct or all-wrong, e.g. all traces truncated -> reward=0 for
   # everyone, zero GRPO advantage, pure noise) and re-sample instead. See
   # run-olmo3-7B-epo-math-colocate.sh for the fuller rationale.
   --over-sampling-batch-size 64
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

RM_ARGS=(
   --group-rm
   --custom-rm-path examples.EPO.epo.epo_group_reward
   # SciKnowEval is well-formed L3 multiple-choice: label is the answer LETTER,
   # graded by a case-insensitive letter match on the extracted <answer>.
   --sdpo-grader mcq
   --sdpo-answer-tag answer
)

EVAL_ARGS=(
   --eval-interval 10
   --skip-eval-before-train
   --eval-prompt-data
      sci_chem /root/sci/val_chemistry.jsonl
      sci_bio  /root/sci/val_biology.jsonl
      sci_phys /root/sci/val_physics.jsonl
      sci_mat  /root/sci/val_material.jsonl
   --n-samples-per-eval-prompt 8
   --log-passrate
   --eval-max-response-len 16384
   --eval-top-p 1
   --eval-custom-rm-path examples.EPO.epo.epo_eval_reward
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
   --max-tokens-per-gpu 24576
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # EPO's reward is binary 0/1; disable GRPO's /std division (keep only
   # mean-centering) so a low-pass-rate group doesn't blow up the advantage
   # scale independently of credit_t's own reweighting.
   --disable-grpo-std-normalization
   --sdpo-teacher-backend megatron
   --sdpo-ema-teacher
   --sdpo-ema-teacher-rate 0.05
   --epo-credit-loss
   --epo-credit-clip 5.0
   --epo-credit-normalize
   # Ablation knobs (see run-olmo3-7B-epo-math-colocate.sh for details):
   # --epo-credit-mode topk_divergence
   # --epo-credit-divergence jsd
   # --epo-credit-skill
   # Diagnosis plan experiment 1.1 as a free wandb panel -- see
   # run-olmo3-7B-epo-math-colocate.sh for the fuller rationale.
   --epo-credit-token-diagnostics
   --entropy-coef 0.00
   --observe-training-entropy
   # Dr.GRPO (arXiv:2503.20783): divide every sample's pg_loss by a FIXED
   # constant instead of its own/the batch's token count, removing the
   # incentive --calculate-per-token-loss gave longer responses to dominate
   # the gradient. See run-olmo3-7B-epo-math-colocate.sh for the fuller
   # rationale. Requires --calculate-per-token-loss to stay OFF (it's absent
   # here, which is the point -- see examples/DrGRPO/custom_reducer.py).
   --custom-pg-loss-reducer-function-path examples.DrGRPO.custom_reducer.get_pg_loss_reducer
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
   --wandb-group olmo3-7B-epo-sci
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
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
