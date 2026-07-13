#!/bin/bash

# SDPO (self-distilled policy optimization) — Qwen2.5-7B-Instruct, tuned for a
# single 8x H200 (141GB) node. (Matches the official lasgroup/SDPO model choice;
# Qwen2.5 is non-thinking, avoiding the Qwen3 reasoning-collapse we hit.)
# usage: bash examples/SDPO/run-qwen3-8B-sdpo.sh
#
# Rollout is identical to GRPO. After each group is generated, the custom group
# reward model (examples.SDPO.sdpo.sdpo_group_reward) picks a random correct
# peer trace as a prefix, scores every trace on the *self-teacher* (the rollout
# engine, i.e. the current policy re-synced every rollout) with that prefix, and
# stores a per-token divergence in sample.opd_reverse_kl. The framework then
# subtracts opd_kl_coef * opd_reverse_kl from the GRPO advantages.
#
# Because SDPO is self-distillation, NO separate teacher server is needed:
# --sdpo-self-teacher (default) scores against the rollout engine. To use a
# fixed external teacher instead, see the block at the bottom of this file.

set -exf

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/miles/scripts/models/qwen3-8B.sh"

# Experiment name for the dump dir. Use $SDPO_EXP if set (export it before the
# run), else fall back to <wandb-group>_<timestamp>. Dumps land in
# /root/miles/sdpo_dumps/<EXP>/ (== <repo>/sdpo_dumps/<EXP>/ on the host).
SDPO_EXP="${SDPO_EXP:-qwen3-8B-sdpo-sci_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/miles/sdpo_dumps/${SDPO_EXP}"
echo "SDPO dump dir: ${DUMP_DIR}"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-8B
   --ref-load /root/Qwen3-8B_torch_dist
   --load /root/Qwen3-8B_miles/
   --save /root/Qwen3-8B_miles/
   --save-interval 50
   # Dump every rollout + eval sample (prompt, response, tokens, reward, metadata
   # incl. sdpo_prefix_tokens/correct/ppl) and per-rank train data for post-hoc
   # analysis. Also enables the SDPO teacher/student full-sequence dump below.
   # /root/miles is this repo mounted into the container (== $REPO_ROOT on host),
   # so dumps land in <repo>/sdpo_dumps/<exp>. Set $SDPO_EXP to name the run.
   --dump-details "${DUMP_DIR}"
)

# SciKnowEval + LCBv6: all domains' train splits mixed into one jsonl for
# training; per-domain val splits registered for evaluation.
# Build with: python examples/SDPO/build_sci_dataset.py
ROLLOUT_ARGS=(
   --prompt-data /root/sci/train.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   # 16384: with the megatron self-teacher, scoring is a batched CUDA-graph'd
   # training forward (not sglang eager prefill), so long sequences are no longer
   # the rollout bottleneck (rollout dropped 5286s -> ~60s). Restore full length
   # to minimize truncation and preserve long reasoning chains.
   # Align rollout length with the KD window (--sdpo-kd-max-tokens 4096): under pure
   # distillation the tokens beyond the KD cap get NO training signal, so generating
   # to 16384 wasted compute and left an un-supervised tail. 4096 matches the window.
   --rollout-max-response-len 4096
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

# SDPO reward model: group-level, needs the whole prompt group to pick a prefix.
RM_ARGS=(
   --group-rm
   --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
)

EVAL_ARGS=(
   --eval-interval 10
   # Skip the step-0 baseline eval (14384 samples ~10min) to get into training
   # fast while debugging. Remove this to restore the pre-train baseline.
   --skip-eval-before-train
   # All four SciKnowEval domains as name/path pairs in ONE --eval-prompt-data
   # (it is nargs='+', parsed as consecutive name path name path ...; repeating
   # the flag would overwrite, not append). eval_rollout runs the datasets
   # concurrently (asyncio.gather), so they share the rollout engines' throughput
   # rather than running back-to-back — total time grows sublinearly, not 4x.
   --eval-prompt-data
      sci_chem /root/sci/val_chemistry.jsonl
      sci_bio  /root/sci/val_biology.jsonl
      sci_phys /root/sci/val_physics.jsonl
      sci_mat  /root/sci/val_material.jsonl
   # 4 samples/prompt: eval is ~4x faster than 16. Gives pass@1/2/4. The val sets
   # are already small (<1000 each), so all prompts are used regardless.
   --n-samples-per-eval-prompt 8
   # Report pass@1/2/4/8/16 (via compute_pass_rate) alongside the mean reward
   # (eval/<name> = avg@16). Without this only the mean is logged.
   --log-passrate
   # Full 16k eval length. This no longer OOMs: eval now skips the OPD top-k
   # logprob request (see generate()'s `evaluation` gate), which was the real
   # cause of the earlier crash — not the sequence length itself.
   --eval-max-response-len 16384
   --eval-top-p 1
   # SDPO trains with a group RM, which cannot score eval samples (no group step
   # in eval). This per-sample eval RM grades eval pass@1 with the same
   # correctness rule as sdpo_group_reward. Required for eval under --group-rm.
   --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
)

# H200 (141GB), TP=2 (DP=2 over 4 training GPUs). The megatron self-teacher adds
# a teacher forward over prompt+prefix+response (prefix can be long, ~9k tokens),
# on top of student fwd/bwd + ref fwd. TP=2 shards weights+optimizer+activations
# across 2 cards so a single card no longer holds everything (TP=1 OOM'd in the
# backward). Sequence parallel is re-enabled (it needs TP>1).
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   # full recompute: at 16k sequence length the activations are the dominant
   # memory cost (observed: model+optimizer ~38GB, but forward/backward spiked to
   # ~140GB and OOM'd with selective recompute + 65536 tokens). Full recompute
   # keeps only layer inputs, drastically shrinking the activation footprint.
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   # 24576 (not 65536): caps the per-GPU microbatch token count so the peak
   # activation (even with full recompute) stays well under budget at 16k len.
   --max-tokens-per-gpu 24576
)

# SDPO configuration.
GRPO_ARGS=(
   --advantage-estimator grpo
   # SDPO as a real distribution KD LOSS (not advantage/REINFORCE).
   # --use-opd is enabled only so the reward pipeline stays SDPO's group RM; but
   # opd_kl_coef=0 means the advantage hook subtracts nothing — all the gradient
   # comes from --sdpo-kd-loss below.
   --use-opd
   --opd-type sglang
   --opd-kl-coef 0.0
   # Megatron self-teacher: the rollout reward only picks the correct-peer prefix;
   # the training actor computes the teacher top-k distribution (with prefix) as a
   # detached target, and policy_loss_function pulls the grad-enabled student
   # distribution (no prefix) toward it. Batched CUDA-graph'd forward, ~50x faster
   # than sglang HTTP eager scoring.
   --sdpo-teacher-backend megatron
   # EMA teacher (matches lasgroup/SDPO teacher_regularization='ema'):
   # teacher = (1 - rate) * teacher + rate * student each step, rate 0.05 matches
   # the paper exactly. The EMA copy lags behind the live policy, so the teacher
   # keeps a "reason first" distribution rather than collapsing to "emit <answer>
   # immediately" — combined with lr warmup (10 steps) this prevents round-1 collapse.
   --sdpo-ema-teacher
   --sdpo-ema-teacher-rate 0.05
   --sdpo-logprob-mode topk
   --opd-log-prob-top-k 100      # k for the teacher top-k distribution (+ tail bucket); matches lasgroup/SDPO distillation_topk=100
   --sdpo-divergence jsd         # alpha=0.5 generalized JSD (symmetric, bounded). forward_kl(alpha=0)|reverse_kl(alpha=1)|jeffrey also available
   --sdpo-is-clip 2.0            # IS-ratio clip for off-policy/async (original SDPO is_clip=2.0)
   # THE knob: distribution-level KD loss. loss += sdpo_kd_coef * D(student‖teacher).
   --sdpo-kd-loss
   --sdpo-kd-coef 1.0
   --sdpo-kd-max-tokens 4096     # only distill the first 4k response tokens (caps full-vocab log_softmax memory)
   # KD Clip-Cov (arXiv:2505.22617 idea, adapted to distillation): detach the grad
   # of the top-0.2% response tokens by KD divergence. Those few tokens (the
   # '<answer>'/letter positions the answer-in-prefix teacher spikes on) drive the
   # entropy collapse; freezing their update keeps the distillation signal on the
   # rest while preserving policy entropy. Loss value unchanged (logged), grad cut.
   # --sdpo-kd-clip-cov-frac 0.002
   --sdpo-self-teacher           # self-distillation: teacher = current policy
   --sdpo-pure-distill           # task reward = 0 (advantage=0); KD loss is the whole objective
   # Qwen3 emits <think>...</think> before the requested <reasoning>/<answer> format
   # (confirmed: 256/256 rollouts contain <think>). Strip it from the peer solution
   # before it becomes the teacher prefix — otherwise the prefix is huge and the
   # teacher teaches the student to echo the reasoning verbatim. Matches official
   # SDPO remove_thinking_from_demonstration. REQUIRED for Qwen3 (thinking model).
   --sdpo-remove-thinking-from-demonstration
   # Grading: deterministic MCQ letter-match on the extracted <answer> (aligned with
   # lasgroup/SDPO). The dataset is now filtered to well-formed L3 multiple-choice
   # questions WITH options listed, so the answer is a letter — no LLM judge needed.
   # LLM-as-judge (--sdpo-judge, gpt-5.4-mini via OpenAI) is kept in the code but
   # OFF for now; re-enable it if we move to open-ended (no-options) questions.
   --sdpo-answer-tag answer
   # KL: OFF, aligned with lasgroup/SDPO. The official run_sdpo_all.sh sets NO
   # kl_loss (base actor.yaml default use_kl_loss=false) — SDPO relies on the EMA
   # teacher + is_clip + lr_warmup for stability, not a reference-KL anchor. Our
   # earlier 0.001 anchor was too weak to matter anyway (0.001*kl_ref stayed ~1e-3).
   # (We simply omit --use-kl-loss; the EMA teacher's weight backuper is kept alive
   #  by --sdpo-ema-teacher, which no longer depends on a ref model.)
   --entropy-coef 0.00
   --observe-training-entropy   # log entropy as a metric without adding it to the loss
   # token-mean loss aggregation, aligned with lasgroup/SDPO (verl actor.yaml default
   # loss_agg_mode='token-mean'). miles' default is seq-mean-token-mean (each sequence
   # equal weight), which UP-weights short sequences: a collapsed 9-token response gets
   # the same gradient weight as a 500-token one, feeding the collapse. token-mean
   # (sum over all response tokens / total token count) makes a 9-token response
   # contribute ~9/N — negligible — so a stray collapse no longer dominates the update.
   # Safe here because this is pure distillation (only the KD term contributes to loss).
   --calculate-per-token-loss
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5                     # original SDPO uses 1e-5; 1e-6 (10x smaller) was likely too weak for KD to move val acc
   --lr-decay-style constant
   # LR warmup over the first 10 training iters (matches lasgroup/SDPO
   # lr_warmup_steps=10). lr ramps 0 -> 1e-5, so the very first KD step no longer
   # takes a full-size stride into the teacher's "emit <answer> immediately"
   # distribution and blow entropy to ~0 (the round-1 collapse). Combined with the
   # EMA teacher this gives the policy a gentle start before distillation bites.
   --lr-warmup-iters 10
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-sdpo
   --wandb-group qwen3-8B-sdpo-sci
   --wandb-key "${WANDB_API_KEY}"   # from the host env, forwarded into the container + ray workers
)

# H200: 141GB lets the rollout engine hold a big KV cache -> higher throughput.
# Any native SGLang server arg is passed through with the --sglang- prefix
# (miles auto-exposes ServerArgs), so --sglang-chunked-prefill-size maps to
# SGLang's --chunked-prefill-size on the miles-managed rollout engine.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # 0.8 (was 0.9): 0.9 static left too little for the logits buffer / activations
   # on long 16k generations and the SGLang engine OOM'd (tried to alloc 8.7GB with
   # 4.5GB free, in logits_processor._copy_logits_to_buffer). 0.8 leaves headroom.
   --sglang-mem-fraction-static 0.8
   --sglang-chunked-prefill-size 8192    # smaller prefill chunk -> lower activation/logits peak on long seqs
   # round_robin instead of the router's default cache-aware policy. SDPO teacher
   # scoring sends many similar-prefix requests (prompt+prefix+response); cache-
   # aware piles them onto one engine (seen: 28 running-req on one, 0 on others,
   # only 2/6 GPUs busy). Round-robin spreads them evenly. KV-cache reuse is
   # negligible here anyway (token usage ~0.06), so balancing wins.
   --sglang-router-policy round_robin
   # NOTE: piecewise (prefill) CUDA graph is NOT disabled here. It was disabled
   # only under colocate (where its per-token-bucket torch.compile hangs). In
   # this disaggregated run, disabling it forced prefill into eager mode, which
   # crippled SDPO teacher scoring (a pure-prefill workload) to ~360 tok/s. With
   # piecewise re-enabled, prefill uses CUDA graphs and is far faster.
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ---- Launch Ray + ASYNC training -------------------------------------------
# 8x H200, DISAGGREGATED 4+4: 4 GPUs train (TP2 x DP2), 4 run SGLang rollout.
# train_async.py overlaps rollout N+1 generation with training step N's
# forward/backward, so the training GPUs are no longer idle during rollout.
# Async requires disaggregated (no colocate) — perfect here since rollout
# (GPU 4-7) and training (GPU 0-3) never share memory. 1-step off-policy is the
# standard async tradeoff; the EMA teacher + lr warmup handle the distillation
# stability, so the async weight-version skew is acceptable for the throughput win.
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Clean up any Ray left over from a previous run so `ray start` gets a fresh
# cluster (stale head/workers otherwise cause port-in-use or GPU-busy errors).
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
   `# ^ sync policy weights to the rollout engines every training step, so async` \
   `# stays strictly 1-step off-policy (rollout N+1 uses step-N weights, never older).` \
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

# ---- Cleanup ---------------------------------------------------------------
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

# ---- Optional: fixed external teacher (NOT the paper's self-distillation) ---
# For a fixed stronger teacher (e.g. Qwen3-32B) instead of self-teaching:
#   1. Launch it on a spare GPU:
#        CUDA_VISIBLE_DEVICES=7 python3 -m sglang.launch_server \
#            --model-path /root/Qwen3-32B --host 0.0.0.0 --port 13141 \
#            --tp 1 --mem-fraction-static 0.85 &
#   2. Reduce actor/rollout GPU counts to leave that GPU free.
#   3. Add to GRPO_ARGS:  --no-sdpo-self-teacher
#   4. Add to RM_ARGS:    --rm-url http://127.0.0.1:13141/generate
