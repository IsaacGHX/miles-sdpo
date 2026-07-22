#!/bin/bash

# EPO (PMI-credit self-distillation) — Olmo-3-7B-Instruct on DAPO math, single
# 8x H200 (141GB) node, COLOCATE variant. Sibling of
# examples/SDPO/run-olmo3-7B-sdpo-math-colocate.sh: same rollout/colocate/eval
# config and the same correct-peer teacher-prefix machinery, but the SDPO
# KD-loss (direction+density coupled, from the teacher's KL) is replaced by
# EPO's credit_t (density only, from PMI) times the REAL GRPO reward
# (direction only, from the task outcome) — see examples/EPO/epo.py.
# usage: bash examples/EPO/run-olmo3-7B-epo-math-colocate.sh
#
# WHY THIS VARIANT — see the SDPO colocate script's header: colocate puts both
# the actor and the rollout engines on all 8 GPUs and time-shares them via
# offload/onload, roughly doubling both rollout and training throughput vs the
# disaggregated 4+4 split.

set -exf

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/miles/scripts/models/olmo3-7B.sh"

EPO_EXP="${EPO_EXP:-olmo3-7B-epo-math-colocate_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/miles/sdpo_dumps/${EPO_EXP}"
echo "EPO dump dir: ${DUMP_DIR}"

CKPT_ARGS=(
   --hf-checkpoint /root/Olmo-3-7B-Instruct
   --ref-load /root/Olmo-3-7B-Instruct_torch_dist
   # --load /root/Olmo-3-7B-Instruct_miles/
   # --save /root/Olmo-3-7B-Instruct_miles/
   # --save-interval 50
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)

# DAPO math: one train jsonl; AIME-2025 + Minerva-Math as eval sets.
# Build the eval sets with: python examples/SDPO/build_math_eval.py --out-dir /root/math_eval
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
   # DAPO-style dynamic sampling: drop a group where every trace got the SAME
   # reward (all-correct or all-wrong -> GRPO advantage is exactly 0 for
   # everyone, pure noise -- e.g. a group where all 8 traces got truncated
   # before a boxed answer, all reward=0) and re-sample a fresh prompt to fill
   # the batch slot instead. over-sampling-batch-size (2x rollout-batch-size)
   # gives headroom for the extra draws the filter needs.
   --over-sampling-batch-size 64
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

RM_ARGS=(
   --group-rm
   --custom-rm-path examples.EPO.epo.epo_group_reward
   # DAPO math has integer boxed answers, so grade with the DAPO math grader.
   --sdpo-grader dapo
   --sdpo-answer-tag answer
)

EVAL_ARGS=(
   --eval-interval 10
   # --skip-eval-before-train
   --eval-prompt-data
      aime25   /root/math_eval/aime25.jsonl
      minerva  /root/math_eval/minerva_math.jsonl
   --n-samples-per-eval-prompt 8
   --log-passrate
   --eval-max-response-len 16384
   --eval-top-p 1
   --eval-custom-rm-path examples.EPO.epo.epo_eval_reward
)

PERF_ARGS=(
   # TP=1 (colocate -> DP=8): see the SDPO colocate script's PERF_ARGS comment
   # (Olmo3's QK-norm spans the full q/k dim, so TP would normalize a shard).
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
   # EPO's reward is binary 0/1 (task correctness), so the GRPO group std can
   # be tiny (e.g. 1/8 correct -> std~0.35) and blow up the advantage by ~3x --
   # a scale effect that has nothing to do with credit_t's own reweighting and
   # would confound it. Keep GRPO's mean-centering (reward - group_mean) but
   # disable the /std division; re-enable by dropping this flag if you want to
   # compare against std-normalized GRPO.
   --disable-grpo-std-normalization
   # Reuse SDPO's Megatron self-teacher plumbing to get the privileged-context
   # (correct-peer) forward pass: EMA teacher snapshot, updated slowly toward
   # the student each step, so credit_t's "with privileged context" forward
   # does not instantly chase the "without" one into a degenerate collapse.
   --sdpo-teacher-backend megatron
   --sdpo-ema-teacher
   --sdpo-ema-teacher-rate 0.005
   # EPO: credit_t = |logp(y|x,peer,y_<t>) - logp(y|x,y_<t>)| under the SAME
   # (EMA) teacher weights, fused into the GRPO advantage (reward - baseline)
   # in place of SDPO's KD-loss direction+density coupling.
   --epo-credit-loss
   --epo-credit-clip 5.0
   --epo-credit-normalize
   # --epo-credit-mode abs_logp_diff is the default (sampled-token log-prob
   # diff, one extra forward per side). Using topk_divergence instead: a
   # distribution-level JSD over the top-100 next-token distribution (not just
   # the sampled token) -- costs a second top-k forward + O(R*k) lookup per
   # sample, matching SDPO's --opd-log-prob-top-k 100 budget.
   --epo-credit-mode topk_divergence
   --epo-credit-divergence jsd   # reverse_kl | forward_kl | jeffrey | jsd
   --opd-log-prob-top-k 100
   # Ablation: privileged context f = the peer's self-generated SKILL (a
   # condensed solution roadmap) instead of its full raw trace. Also makes
   # sdpo_dumps/<exp>/skill/ non-empty (see --dump-details above).
   # --epo-credit-skill
   # Diagnosis plan experiment 1.1 as a free wandb panel (rollout/
   # epo_credit_{epistemic,strategy,compute,format,other}_mean): does
   # epistemic-token credit collapse first? Costs one convert_ids_to_tokens
   # call per sample per rollout (cheap, non-zero) -- on for this run to watch
   # the response-length runaway investigation.
   --epo-credit-token-diagnostics
   --entropy-coef 0.00
   --observe-training-entropy
   # KL-to-reference anchor (matches examples/search-r1/run_qwen2.5_3B.sh's
   # value): a small, constant pull back toward the SFT checkpoint's output
   # distribution -- the only brake in this config against the response
   # tokens drifting into a degenerate (long/repetitive) mode once credit_t
   # collapses toward 1 and EPO reduces to bare GRPO (see the response-length
   # runaway investigation). --ref-load above already points at the SFT
   # checkpoint; --use-kl-loss triggers backing it up as the ref model.
   --use-kl-loss
   --kl-loss-coef 0.001
   # Dr.GRPO (arXiv:2503.20783): --calculate-per-token-loss made pg_loss's
   # normalizer a GLOBAL batch-wide token count, so a sample's raw contribution
   # to the loss scales with ITS OWN token count -- a 4000-token response
   # injects ~10x the gradient magnitude of a 400-token one at the same
   # per-token advantage. Off (default) + the custom reducer below instead
   # divides every sample's pg_loss by the SAME fixed constant (DIVISOR=1000
   # in examples/DrGRPO/custom_reducer.py), removing the length-scaling
   # incentive that was likely compounding the response-length runaway.
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
   --wandb-group olmo3-7B-epo-math
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # 0.9 OOM'd during eval (--eval-max-response-len 16384, 2x the 8192 training
   # cap, needs more live KV cache headroom): one SGLang engine ran out of
   # memory mid-eval and never recovered, and the framework's next offload()
   # broadcast then hung retrying against the dead engine until the whole job
   # crashed (TimeoutError in release_memory_occupation). Back to 0.85 (the
   # original EPO/SDPO colocate default) for eval-time headroom.
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
