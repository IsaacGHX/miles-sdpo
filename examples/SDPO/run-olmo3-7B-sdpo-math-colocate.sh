#!/bin/bash

# SDPO — Olmo-3-7B-Instruct on DAPO math, single 8x H200 (141GB) node,
# COLOCATE variant. Math counterpart of run-olmo3-7B-sdpo-sci-colocate.sh: same
# colocate / SDPO / skill / pitfall config, only the TASK differs (DAPO math train
# set, dapo grader, AIME-2025 + Minerva-Math eval).
# usage: bash examples/SDPO/run-olmo3-7B-sdpo-math-colocate.sh
#
# WHY THIS VARIANT — the disaggregated main loop (train.py) is strictly serial
# (generate -> train -> update_weights) with NO rollout/train overlap, so a 4+4
# split leaves half the node idle each phase. Colocate puts BOTH the actor and the
# rollout engines on all 8 GPUs and time-shares them via offload/onload:
#   - rollout uses 8 engines instead of 4  -> ~2x generation throughput
#   - training uses 8 GPUs instead of 4    -> faster actor step
# The per-step offload/onload cost (7B weights) is small relative to the step, so
# total step time should drop.
#
# SDPO CORRECTNESS IS UNCHANGED vs the disaggregated script: the EMA teacher,
# self-teacher and teacher forward are all TRAINING-side in-place CPU<->GPU weight
# swaps (TensorBackuper, snapshots live in CPU pinned RAM), fully decoupled from
# how weights reach the rollout engines. Only the parallelism / colocate / memory
# block and the ray launch differ from the disaggregated script; every
# SDPO/GRPO/skill/eval knob below is copied verbatim.

set -exf

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/miles/scripts/models/olmo3-7B.sh"

SDPO_EXP="${SDPO_EXP:-olmo3-7B-sdpo-math-colocate_$(date +%Y%m%d_%H%M%S)}"
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
)

RM_ARGS=(
   --group-rm
   --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
   # DAPO math has integer boxed answers, so grade with the DAPO math grader
   # ('dapo'). (The sci script uses 'mcq' letter-match instead.)
   --sdpo-grader dapo
   # Self-generated skill (SkillOpt, on-policy): the CURRENT policy writes the skill
   # during rollout from a trace's own response. On-policy + trainable (unlike the
   # external --sdpo-trace-condense). Used by skill-KD below and, via
   # --sdpo-response-prefix skill, as the response teacher prefix.
   --sdpo-self-skill
   # 1024 (was 512): the skill is now an instance-grounded 6-10 step solution
   # roadmap (key quantities + intermediate results, stops one step before the
   # answer), ~3-5x longer than the old 3-bullet skill. 512 truncated the tail
   # steps — which is exactly the discriminating info the teacher needs — so give
   # it room to finish. See _SKILL_SYSTEM_PROMPT in examples/SDPO/sdpo.py.
   --sdpo-skill-max-new-tokens 1024
   # Which traces get a skill: correct | incorrect | env_feedback (placeholder) | all.
   #  correct  : correct traces -> instance-grounded solution roadmap (the response
   #             teacher prefix, via --sdpo-response-prefix skill).
   #  incorrect: failed traces -> PITFALL warnings only (mistakes to avoid, never a
   #             solution — a model that failed cannot be trusted to rewrite one).
   #  all      : both -> correct traces give the roadmap prefix AND failed traces feed
   #             the failure-pitfall pipeline below.
   --sdpo-skill-source all
   # Failure-pitfall pipeline (active because skill-source covers incorrect):
   #  1. each failed trace distils its OWN pitfalls, tagged by WHY it failed
   #     (truncated | format | wrong) — see _failure_kind in sdpo.py.
   #  2. the group's per-trace pitfalls are synthesized into ONE short shared
   #     "common mistakes" list by this backend (self = current policy | external =
   #     the --sdpo-condense-* OpenAI endpoint).
   #  That shared list is spliced ONLY into the FAILED traces' teacher prefix (after
   #  the correct-peer roadmap); correct traces keep a clean roadmap-only prefix.
   --sdpo-pitfall-summary-backend self
   # Second SDPO objective on the skill's own tokens (independent coef). Modes:
   #  self-success    : correct traces; teacher hint = the sample's OWN correct trace.
   #  problem-only     : any trace; teacher = skill-gen prompt, no hint.
   #  pitfall-condense : FAILED traces (needs skill-source incorrect|all); student =
   #                     predict pitfalls from the PROBLEM ONLY, teacher = same prompt
   #                     + the group's per-trace failure pitfalls as privileged info.
   # --sdpo-skill-kd
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
   --skip-eval-before-train
   # AIME-2025 (integer answers) and Minerva Math (LaTeX answers) as separate eval
   # sets in ONE --eval-prompt-data (nargs='+', parsed as consecutive name path
   # name path ...; repeating the flag would overwrite, not append). eval_rollout
   # runs them concurrently (asyncio.gather). Graded by the general math grader in
   # sdpo_eval_reward (handles both).
   --eval-prompt-data
      aime25   /root/math_eval/aime25.jsonl
      minerva  /root/math_eval/minerva_math.jsonl
   --n-samples-per-eval-prompt 8
   --log-passrate
   # Full 16k eval length (double the 8K rollout cap). Hard competition math needs
   # long reasoning chains. This no longer OOMs: eval skips the OPD top-k logprob
   # request (see generate()'s `evaluation` gate), which was the real cause of the
   # earlier crash — not the sequence length.
   --eval-max-response-len 16384
   --eval-top-p 1
   # SDPO trains with a group RM, which cannot score eval samples (no group step in
   # eval). This per-sample eval RM grades eval pass@1 with the same correctness
   # rule as sdpo_group_reward. Required for eval under --group-rm.
   --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
)

PERF_ARGS=(
   # TP=1 (colocate -> DP=8): Olmo3's QK-norm spans the FULL q/k dim
   # (num_heads*head_dim), so it cannot be split across TP ranks without a cross-TP
   # allreduce of the RMS (TP would normalize over only the local head shard ->
   # wrong). TP=1 keeps each RMSNorm over the full dim. Under colocate all 8 GPUs
   # host the actor (DP8 x TP1); a 7B model fits comfortably on one H200 at TP=1.
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
   --wandb-group olmo3-7B-sdpo-math
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   # Colocate: one engine per GPU (TP1), all 8 GPUs. mem-fraction 0.7 (was 0.9 in
   # the disaggregated script) because the actor + optimizer + KV cache now share
   # each GPU; colocate onloads/offloads around each phase. Raise cautiously if
   # rollout underutilizes memory; drop if the logits_processor OOMs on a long batch.
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
