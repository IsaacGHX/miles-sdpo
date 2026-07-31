#!/bin/bash

# RLSD ablation series -- Qwen3-4B on DAPO math, single 8x H200 (141GB) node,
# COLOCATE variant. Sibling of run-qwen3-4B-sdpo-math-colocate.sh: SAME
# rollout/eval/skill machinery and dataset (DAPO math, AIME25+Minerva eval),
# but the RESPONSE teacher signal is RLSD (arXiv:2604.03128, "Self-Distilled
# RLVR") instead of the additive KD-loss (--sdpo-kd-loss) -- the math-domain
# port of run-qwen2.5-7B-sdpo-sci-rl-colocate.sh. Per explicit instruction,
# this ONLY swaps the non-skill (response) mechanism -- self-skill/pitfall-
# injection/skill-KD are configured IDENTICALLY to the KD-loss math script's
# own arms 2/3/4/5, and skill-KD's own divergence computation
# (miles/backends/training_utils/loss_hub/losses.py) is untouched.
#
# RLSD vs SDPO's KD loss (see miles/backends/training_utils/loss_hub/rlsd.py
# for the full derivation):
#   SDPO (--sdpo-kd-loss): an ADDITIVE distribution-matching loss pulling the
#     student toward the teacher's (prompt+correct-peer-prefix+response)
#     distribution -- the paper's OPSD failure mode: the teacher's privileged
#     evaluation enters the gradient DIRECTION, so even a WRONG trace's tokens
#     get pulled toward whatever the (possibly irrelevant) peer's teacher
#     distribution favors -- an irreducible mutual-information leakage term.
#   RLSD (--sdpo-rlsd): a MULTIPLICATIVE reweighting of the GRPO advantage.
#     Direction still comes EXCLUSIVELY from sign(A) (the real task reward);
#     the teacher's per-token evidence ratio P_T(y_t)/P_S(y_t) only modulates
#     MAGNITUDE within a trajectory.
#
#   SDPO_ABLATION_ARM=1    pure vanilla GRPO, no SDPO at all -- IDENTICAL to
#                          run-qwen3-4B-sdpo-math-colocate.sh's own arm 1;
#                          kept here too (not just "compare against that
#                          script's arm 1") so this script is self-contained
#                          and every arm here shares the SAME queue/launcher.
#   SDPO_ABLATION_ARM=1.1  RLSD baseline, NO self-skill at all -- correct-peer
#                          FULL raw trace as the response teacher prefix
#                          (base SDPO's default). RLSD analogue of the
#                          KD-loss script's arm 1.2 ("does RLSD's reweighting
#                          alone help" before any skill machinery). lambda=1.0
#                          constant (no decay) -- see the lambda note below;
#                          math has no separate "paper-default-decay" leg
#                          (unlike the sci-rl script's arm 2 vs 2.2) since
#                          that comparison was already settled there.
#   SDPO_ABLATION_ARM=2    + self-skill, skill-source correct only, NO skill-KD
#   SDPO_ABLATION_ARM=3    + self-skill, skill-source incorrect only, NO skill-KD
#   SDPO_ABLATION_ARM=4    + self-skill, skill-source all, NO skill-KD
#   SDPO_ABLATION_ARM=5.1  + self-skill, skill-source all, WITH skill-KD
#                          (mode=both), --sdpo-skill-max-new-tokens 2048.
#                          Named 5.1 (not 5) to match the sci-rl script's own
#                          numbering (its arm 5.1 raised the skill token cap
#                          from 1024 after observing skill/length_max hit
#                          that ceiling) -- start math directly at the
#                          already-fixed cap rather than re-discovering the
#                          same OOM/truncation issue.
#   SDPO_ABLATION_ARM=5.2  IDENTICAL to arm 5.1 except --sdpo-skill-kd-mode
#                          both-blind (was both): self-success (correct
#                          traces) is replaced by blind-correct, the
#                          symmetric counterpart of pitfall-condense for
#                          correct traces (student regenerates from the
#                          PROBLEM ONLY, teacher = same problem-only prompt +
#                          the trace's own correct solution). See
#                          skill/kl_correct vs skill/kl_pitfall metrics
#                          (losses.py) for which half is contributing.
#
# lambda (--sdpo-rlsd-lambda-init / --sdpo-rlsd-lambda-warmup-steps): every
# arm here uses lambda=1.0 constant / warmup-steps 0 (RLSD's reweighting
# active for the FULL run) -- the sci-rl script's own arm 2 vs 2.2 comparison
# already established that the paper's decay-to-inert schedule (0.5->0 over
# 50 steps) just wastes the back half of a 100-rollout ablation with no
# offsetting benefit, so there is no separate decay leg here.
#
# --sdpo-rlsd requires --sdpo-teacher-backend megatron --sdpo-logprob-mode
# sampled (enforced in miles/utils/arguments.py's validate_args). Mutually
# exclusive with --sdpo-kd-loss / --use-opd.
#
# --use-tis on every SDPO arm: corrects train-vs-rollout-engine log-prob
# drift (vanilla_tis_function's tis=exp(train_log_probs-rollout_log_probs),
# multiplied into pg_loss) -- orthogonal to RLSD's own delta_t/credit_t
# (teacher-vs-student, same training-side snapshot). See rlsd.py's own
# module docstring for why these are two separate axes, not the same thing.
#
# --no-sdpo-pure-distill on every SDPO arm here (UNLIKE the KD-loss script,
# where --sdpo-pure-distill defaults True and is left unset): RLSD's
# advantage reweighting has nothing to reweight if the GRPO advantage is
# always 0 -- the real per-trace correctness reward MUST flow into the GRPO
# advantage for RLSD to do anything.
#
# Grading: --sdpo-grader dapo on every SDPO arm (unchanged from the KD-loss
# math script) -- sdpo_group_reward's _grade_group call is untouched by any
# RLSD work this session, so the already-fixed grader
# (examples/SDPO/sdpo.py's _is_correct dapo-path strict-box +
# grade_answer_verl fallback) is used exactly as-is. Must NOT regress to the
# old broken grader.
#
# Same model/domain/thinking/GPU-sizing/LR reasoning as
# run-qwen3-4B-sdpo-math-colocate.sh (see that script's own docstring) --
# only RM_ARGS/GRPO_ARGS' response-KD flags differ per arm below.
#
# usage:
#   SDPO_ABLATION_ARM=1   bash examples/SDPO/run-qwen3-4B-sdpo-math-rl-colocate.sh
#   SDPO_ABLATION_ARM=1.1 bash examples/SDPO/run-qwen3-4B-sdpo-math-rl-colocate.sh
#   ... etc for 2, 3, 4, 5.1, 5.2

set -exf

export PYTHONBUFFERED=16
SDPO_ABLATION_ARM="${SDPO_ABLATION_ARM:?Set SDPO_ABLATION_ARM to one of: 1 1.1 2 3 4 5.1 5.2}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
echo "SDPO_ABLATION_ARM: ${SDPO_ABLATION_ARM}"

source "/root/miles/scripts/models/qwen3-4B.sh"

SDPO_EXP="${SDPO_EXP:-qwen3-4B-sdpo-math-rl-ablation-arm${SDPO_ABLATION_ARM}_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/miles/sdpo_dumps/${SDPO_EXP}"
echo "SDPO dump dir: ${DUMP_DIR}"

# NO --save / --load: exploratory ablation only, nothing here should be
# resumed or kept around (see module docstring above).
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   --ref-load /root/Qwen3-4B_torch_dist
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)

# DAPO math train set, identical across all arms -- the whole point of the
# ablation is to hold data/rollout/eval fixed and vary only the response-KD
# mechanism (KD-loss vs RLSD) and skill knobs.
#
# enable_thinking:false (UNLIKE run-qwen3-4B-sdpo-math-colocate.sh's own
# ROLLOUT_ARGS, which sets it true): per explicit instruction, this Qwen3-4B
# ablation runs entirely no-think -- both rollout generation AND skill
# generation. Skill generation was already forced no-think regardless
# (_skill_gen_template_kwargs() in sdpo.py hardcodes {"enable_thinking":
# False}, independent of the run's own --apply-chat-template-kwargs), so this
# only changes the ROLLOUT side. Consequently --sdpo-remove-thinking-from-
# demonstration (which strips a peer's <think> block before splicing it into
# the response-SDPO teacher prefix) is dropped from every arm below -- with
# no-think rollouts there is no <think> block to strip, so the flag would be
# a pure no-op scan.
ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
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
        # Plain GRPO, no SDPO machinery at all -- IDENTICAL to
        # run-qwen3-4B-sdpo-math-colocate.sh's own arm 1 (see that script's
        # comment for why plain_grpo_reward + --sdpo-grader dapo, not
        # --rm-type dapo/boxed_dapo).
        RM_ARGS=(
            --custom-rm-path examples.SDPO.sdpo.plain_grpo_reward
            --sdpo-grader dapo
        )
        GRPO_ARGS=(
            --advantage-estimator grpo
            --entropy-coef 0.00
            --observe-training-entropy
            # NO --calculate-per-token-loss here (unlike every SDPO arm below)
            # -- this is the "pure vanilla GRPO" baseline: std-normalized
            # advantages, no dynamic sampling, miles' own default
            # seq-mean-token-mean loss aggregation.
        )
        ;;
    1.1)
        # RLSD baseline, NO self-skill at all: the response teacher prefix is
        # the correct peer's FULL raw trace (base SDPO's Pass-1 pick, `trace`
        # -- --sdpo-self-skill never set, so --sdpo-response-prefix has
        # nothing to swap to). Direct RLSD analogue of the KD-loss script's
        # arm 1.2 -- isolates "does RLSD's multiplicative advantage-
        # reweighting alone help" before layering any self-skill/pitfall
        # machinery on top.
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
        # + self-skill, skill-source correct ONLY, no skill-KD. Response-SDPO
        # teacher prefix switches to the peer's SKILL once self-skill is on
        # (--sdpo-response-prefix skill), matching the KD-loss math script's
        # own arm 2.
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
        # + self-skill, skill-source incorrect ONLY (pitfall warnings from
        # failed traces), no skill-KD. NO --sdpo-response-prefix skill here
        # (skill-source incorrect means no correct peer ever generates a
        # skill -- same fix already applied to the KD-loss math script's
        # own arm 3).
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
        # + self-skill, skill-source ALL (correct traces -> solution roadmap,
        # incorrect traces -> pitfall warnings), no skill-KD. Failed traces
        # keep BOTH the correct-peer skill AND the group pitfall summary
        # (appended, per sdpo.py::_render_prefix -- the sdpo_group_reward fix
        # from the sci-rl ablation), degrading to pitfalls-only only when the
        # group has zero correct traces.
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
        # --sdpo-skill-max-new-tokens 2048 (see module docstring for why this
        # starts at the sci-rl script's already-fixed cap instead of 1024).
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
        # IDENTICAL to arm 5.1 except --sdpo-skill-kd-mode both-blind (was
        # both): self-success (correct traces) replaced by blind-correct,
        # the symmetric counterpart of pitfall-condense for correct traces.
        # See skill/kl_correct vs skill/kl_pitfall metrics (losses.py) for
        # which half is contributing.
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

# Identical across all arms: AIME-2025 + Minerva-Math eval (already staged
# under /root/math_eval on this host -- see build_math_eval.py). Arm 1 (plain
# GRPO, no --group-rm) still uses sdpo_eval_reward: it works standalone and
# keeps the eval metric computation identical across every arm.
#
# NO --skip-eval-before-train: a genuine step-0 (untrained-checkpoint) eval is
# required so every arm's "did training help" comparison is against its OWN
# real baseline.
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

# Same GPU/perf sizing as run-qwen3-4B-sdpo-math-colocate.sh, INCLUDING the
# sci-rl script's arm-5-family OOM fix: any skill-KD arm (5.1, 5.2 --
# --sdpo-skill-kd) appends extra skill-KD samples (self-success/blind-correct
# + pitfall-condense forward+backward), needing a lower per-GPU token cap
# than the skill-KD-free arms (confirmed OOM at 24576 on the sci-rl script's
# own arm 5/5.1/5.2 -- "Tried to allocate 11.75 GiB... GPU 2"). RLSD swaps a
# top-k teacher forward for a cheaper sampled-token one on the RESPONSE span
# only, which does not touch the skill-KD memory pressure that causes this.
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
   --wandb-group "qwen3-4B-sdpo-math-rl-ablation-arm${SDPO_ABLATION_ARM}"
   --wandb-key "${WANDB_API_KEY}"
)

# Same rollout-engine memory margin as run-qwen3-4B-sdpo-math-colocate.sh
# (see that script's own comment on the KV-cache/scratch-spike OOM it fixed).
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
