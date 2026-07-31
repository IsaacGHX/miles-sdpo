#!/bin/bash

# RLSD ablation series -- Qwen2.5-7B-Instruct on SciKnowEval (MCQ), single 8x
# H200 (141GB) node, COLOCATE variant. Sibling of
# run-qwen2.5-7B-sdpo-sci-colocate.sh: SAME rollout/eval/skill machinery and
# SAME $SDPO_ABLATION_ARM values (1.1/2/2.2/3/4/5/5.1/5.2 -- no plain 1/1.2
# here, see below), but the RESPONSE teacher signal is now RLSD (arXiv:2604.03128, "Self-Distilled
# RLVR") instead of the additive KD-loss (--sdpo-kd-loss). Per explicit
# instruction, this ONLY swaps the non-skill (response) mechanism -- self-
# skill / pitfall-injection / skill-KD are configured IDENTICALLY to arms
# 2/3/4/5 in run-qwen2.5-7B-sdpo-sci-colocate.sh, and skill-KD's own
# divergence computation (miles/backends/training_utils/loss_hub/losses.py)
# is untouched (it always runs the top-k KD path on the skill span,
# independent of --sdpo-rlsd -- see actor.py::_compute_sdpo_teacher_log_probs's
# sampled-mode branch, which now also appends skill-KD samples).
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
#     MAGNITUDE within a trajectory. A wrong trace's tokens can never be
#     pulled toward the teacher's favored tokens -- reward keeps the sign, the
#     teacher only reshapes WHERE within that (correctly-signed) trajectory
#     the gradient concentrates.
#
#   SDPO_ABLATION_ARM=1.1  RLSD baseline, NO self-skill at all -- correct-peer
#                          FULL raw trace as the response teacher prefix
#                          (base SDPO's default). RLSD analogue of the
#                          KD-loss script's arm 1.2; isolates "does RLSD's
#                          reweighting alone help" before any skill machinery.
#   SDPO_ABLATION_ARM=2    + self-skill, skill-source correct only, NO skill-KD.
#                          RLSD lambda: paper defaults (0.5 -> 0 decayed over
#                          50 rollouts -- INERT, i.e. plain GRPO, for rollouts
#                          50-100 of this ablation's 100).
#   SDPO_ABLATION_ARM=2.2  IDENTICAL to arm 2 except RLSD lambda is held at a
#                          constant 1.0 (no decay) -- the reweighting stays
#                          ACTIVE for the full run. Isolates whether the
#                          paper's own decay-to-inert schedule matters here.
#   SDPO_ABLATION_ARM=3    + self-skill, skill-source incorrect only, NO skill-KD
#   SDPO_ABLATION_ARM=4    + self-skill, skill-source all, NO skill-KD
#   SDPO_ABLATION_ARM=5    + self-skill, skill-source all, WITH skill-KD (mode=both)
#   SDPO_ABLATION_ARM=5.1  IDENTICAL to arm 5 except --sdpo-skill-max-new-tokens
#                          2048 (was 1024) -- arm 5's own skill/length_max
#                          metric was observed hitting the 1024 cap, so this
#                          widens the budget to see if more room changes the
#                          skill-KD signal.
#   SDPO_ABLATION_ARM=5.2  IDENTICAL to arm 5 except --sdpo-skill-kd-mode
#                          both-blind (was both): self-success (correct
#                          traces, near-zero information gap between student/
#                          teacher -- see the module comment above
#                          _build_skill_self_success_teacher_prompt_str in
#                          examples/SDPO/sdpo.py) is replaced by blind-correct,
#                          the symmetric counterpart of pitfall-condense for
#                          correct traces (student regenerates from the
#                          PROBLEM ONLY, teacher = same problem-only prompt +
#                          this trace's own correct solution). Isolates
#                          whether arm 5's correct-vs-pitfall skill-KD
#                          contribution imbalance is a real effect or an
#                          artifact of self-success barely having privileged
#                          info to distill -- see the new skill/kl_correct vs
#                          skill/kl_pitfall metrics (losses.py) for the split.
#
# Arms 1.1/3/4/5/5.1/5.2 use lambda=1.0/no-decay (not arm 2's paper-default
# decay) -- see the lambda discussion below.
#
# No plain arm 1 (plain GRPO, no SDPO at all -- IDENTICAL to the KD-loss
# script's arm 1, so it is not re-run here; compare against that script's
# arm 1 directly) and no arm 1.2 (that number is the KD-loss script's own
# no-skill baseline; arm 1.1 here is RLSD's analogue of it, keeping the
# "X.1 = RLSD variant, X.2 = KD-loss variant" numbering distinct rather than
# reusing 1.2 for a different mechanism under the same arm number).
#
# lambda (--sdpo-rlsd-lambda-init / --sdpo-rlsd-lambda-warmup-steps): the
# paper's Algorithm 1 formula credit_t = (1-lambda) + lambda*clip(w_t, ...)
# means once lambda decays to 0, credit_t == 1 for every token and RLSD's
# reweighting mechanism is COMPLETELY INERT for the rest of training -- with
# --num-rollout 100 and the paper's own --sdpo-rlsd-lambda-warmup-steps 50,
# that is literally the back HALF of every run here. Arm 2 keeps the paper's
# own defaults (0.5 -> 0 over 50 rollouts) so that run stays reproducible;
# arm 1.1/2.2/3/4/5 instead use lambda=1.0 constant / warmup-steps 0, keeping
# the per-token reweighting active for the FULL run -- matching the paper's
# ALTERNATIVE final-objective formula (Eq. 16, a plain PPO-style min/clip on
# w_t*A with no lambda term at all) rather than Algorithm 1's line 17.
#
# --sdpo-rlsd requires --sdpo-teacher-backend megatron --sdpo-logprob-mode
# sampled (enforced in miles/utils/arguments.py's validate_args): RLSD only
# needs the sampled TOKEN's teacher log-prob (delta_t = log P_T(y_t) -
# log P_S(y_t)), not a top-k distribution, so it reuses the cheaper single-
# forward "sampled" branch of _compute_sdpo_teacher_log_probs (shared with the
# legacy sampled-mode OPD advantage-hook path) instead of the KD-loss's top-k
# branch. Mutually exclusive with --sdpo-kd-loss / --use-opd (all three are
# alternative ways of turning the same teacher-vs-student divergence into a
# training signal; combining them double-counts it).
#
# --use-tis on every arm: RLSD's own delta_t/credit_t (P_T vs P_S, both from
# the SAME training-side no-grad snapshot) is orthogonal to a SEPARATE axis --
# drift between the training-side snapshot and the SGLang rollout engine that
# actually SAMPLED the trace (async/off-policy lag). That axis is exactly what
# --use-tis already corrects (vanilla_tis_function's
# tis=exp(train_log_probs-rollout_log_probs), multiplied into pg_loss). An
# earlier version of rlsd.py duplicated this same correction internally
# (folded into delta_t before its own exp()) -- removed; --use-tis is the
# single place this drift gets corrected now, --tis-clip/--tis-clip-low
# (defaults 2.0/0) control it exactly like it would for any other estimator.
#
# --no-sdpo-pure-distill on every arm here (UNLIKE the KD-loss script, where
# --sdpo-pure-distill defaults True and is left unset): RLSD's advantage
# reweighting has nothing to reweight if the GRPO advantage is always 0 --
# the paper's mechanism is explicitly "environment reward sets direction,
# teacher sets magnitude," so the real per-trace correctness reward MUST flow
# into the GRPO advantage (matching the paper's own ablated-baseline setup,
# and this session's explicit choice for the EPO/RLSD-family experiments).
#
# Grading (--sdpo-grader unset -> default "mcq"): unchanged from the KD-loss
# sci script -- sdpo_group_reward's _grade_group call is untouched by any of
# this session's RLSD work, so the already-fixed grader
# (examples/SDPO/sdpo.py's _is_correct dapo-path strict-box + grade_answer_verl
# fallback) is used exactly as-is. Per explicit instruction, this must NOT
# regress to the old broken grader; nothing in this script or in rlsd.py
# touches grading at all.
#
# Same model/domain/GPU-sizing/LR reasoning as run-qwen2.5-7B-sdpo-sci-
# colocate.sh (see that script's own docstring) -- only RM_ARGS/GRPO_ARGS'
# response-KD flags differ per arm below.
#
# usage:
#   SDPO_ABLATION_ARM=1.1 bash examples/SDPO/run-qwen2.5-7B-sdpo-sci-rl-colocate.sh
#   ... etc for 2, 2.2, 3, 4, 5, 5.1, 5.2

set -exf

export PYTHONBUFFERED=16
SDPO_ABLATION_ARM="${SDPO_ABLATION_ARM:?Set SDPO_ABLATION_ARM to one of: 1.1 2 2.2 3 4 5 5.1 5.2}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
echo "SDPO_ABLATION_ARM: ${SDPO_ABLATION_ARM}"

source "/root/miles/scripts/models/qwen2.5-7B.sh"

SDPO_EXP="${SDPO_EXP:-qwen2.5-7B-sdpo-sci-rl-ablation-arm${SDPO_ABLATION_ARM}_$(date +%Y%m%d_%H%M%S)}"
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

# SciKnowEval MCQ train set, identical to the KD-loss sci script -- the whole
# point of this sibling ablation is to hold data/rollout/eval fixed and vary
# ONLY the response-KD mechanism (KD-loss vs RLSD).
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
    1.1)
        # RLSD baseline, NO self-skill at all: the response teacher prefix is
        # the correct peer's FULL raw trace (base SDPO's Pass-1 pick, `trace`
        # -- --sdpo-self-skill never set, so --sdpo-response-prefix has
        # nothing to swap to). Direct RLSD analogue of the KD-loss script's
        # arm 1.2 (--sdpo-kd-loss, no skill) -- isolates "does RLSD's
        # multiplicative advantage-reweighting alone help" before layering
        # any self-skill/pitfall machinery on top, same role 1.2 played for
        # the additive KD-loss mechanism.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
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
        # + self-skill, skill-source correct ONLY, no skill-KD. IDENTICAL skill
        # config to run-qwen2.5-7B-sdpo-sci-colocate.sh's arm 2 -- only the
        # response-KD block (--sdpo-kd-* -> --sdpo-rlsd-*) differs.
        #
        # lambda-init 0.5 / warmup-steps 50 (the PAPER's own defaults, decaying
        # to a fully-inert credit_t==1 i.e. plain GRPO for the back half of
        # this ablation's 100 rollouts -- see run 2026-07-30 00:32,
        # qwen2.5-7B-sdpo-sci-rl-ablation-arm2_20260730_003203, already tagged
        # "arm2" in wandb): kept as the literal paper config under this arm
        # number so a future re-run of ARM=2 reproduces that SAME run rather
        # than silently diverging. See arm 2.2 for the lambda=1.0/no-decay
        # (RLSD reweighting active for the FULL run) variant, run alongside
        # this one for direct comparison -- same "X vs X.2" naming convention
        # this session already used for 1.1 (first attempt) vs 1.2 (fixed).
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
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
            --sdpo-rlsd-lambda-init 0.5
            --sdpo-rlsd-lambda-warmup-steps 50
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
    2.2)
        # IDENTICAL skill config to arm 2 -- ONLY --sdpo-rlsd-lambda-init/
        # --sdpo-rlsd-lambda-warmup-steps differ: 1.0/0 means credit_t stays
        # w_t (clipped) for the ENTIRE run, never decaying to the inert
        # credit_t==1 plain-GRPO fallback arm 2 settles into after rollout 50.
        # This is "RLSD's reweighting mechanism, on its own, for the full run"
        # -- the direct answer to whether the paper's own decay schedule (vs.
        # keeping it always active) matters on this task/model.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
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
        # (same reasoning as the KD-loss script's arm 3 -- skill-source
        # incorrect means no correct peer ever has a generated skill).
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
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
        # incorrect traces -> pitfall warnings), no skill-KD.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
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
    5)
        # + self-skill, skill-source ALL, WITH skill-KD (mode=both). Per
        # explicit instruction, arm 5 ONLY swaps the non-skill (response)
        # part -- skill-KD's own top-k divergence mechanism, mode, and coef
        # are IDENTICAL to run-qwen2.5-7B-sdpo-sci-colocate.sh's arm 5
        # (actor.py::_compute_sdpo_teacher_log_probs's sampled/RLSD branch now
        # also calls _append_sdpo_skill_samples when --sdpo-skill-kd is set,
        # seeding empty response-span top-k targets so the shared KD-loss path
        # in losses.py only fires on the appended skill tokens).
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
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
            --sdpo-skill-kd
            --sdpo-skill-kd-coef 0.01
            --sdpo-skill-kd-mode both
            --entropy-coef 0.00
            --observe-training-entropy
            --calculate-per-token-loss
        )
        ;;
    5.1)
        # IDENTICAL to arm 5 except --sdpo-skill-max-new-tokens raised from
        # 1024 to 2048 -- arm 5's own skill/length_max metric (skill/ panel,
        # miles/ray/rollout/metrics.py) was observed hitting the 1024 cap,
        # so this widens the budget to see whether skills that get more room
        # (longer pitfall-condense / self-success distillations) change the
        # skill-KD signal.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
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
        # IDENTICAL to arm 5 except --sdpo-skill-kd-mode both-blind (was both):
        # self-success (correct traces) is replaced by blind-correct --
        # student regenerates a knowledge prediction from the PROBLEM ONLY (no
        # solution, no attempt, exactly like pitfall-condense's student),
        # teacher = same problem-only prompt + the trace's own correct
        # solution as privileged info. self-success's student/teacher prompts
        # differed by only ONE hint sentence (SKILL_SELF_SUCCESS_HINT) because
        # the student prompt already states "You are given a CORRECT worked
        # solution" -- almost no information gap for the KD divergence to
        # measure. both-blind gives correct traces the SAME size information
        # gap pitfall-condense already has for failed traces, so the new
        # skill/kl_correct vs skill/kl_pitfall metrics (see
        # miles/backends/training_utils/loss_hub/losses.py) become directly
        # comparable -- answers "was self-success's weak showing (vs
        # pitfall-condense) a real effect, or just an artifact of self-success
        # itself barely having any privileged info to distill from."
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
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
            --sdpo-skill-kd
            --sdpo-skill-kd-coef 0.01
            --sdpo-skill-kd-mode both-blind
            --entropy-coef 0.00
            --observe-training-entropy
            --calculate-per-token-loss
        )
        ;;
    *)
        echo "Unknown SDPO_ABLATION_ARM='${SDPO_ABLATION_ARM}' (expected one of: 1.1 2 2.2 3 4 5 5.1 5.2)" >&2
        exit 1
        ;;
esac

# Identical across all 4 arms: all four SciKnowEval domains (already staged
# under /root/sci on this host -- see build_sci_dataset.py).
#
# NO --skip-eval-before-train: a genuine step-0 (untrained-checkpoint) eval is
# required so every arm's "did training help" comparison is against its OWN
# real baseline (same reasoning as run-qwen2.5-7B-sdpo-sci-colocate.sh's own
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

# Same GPU/perf sizing as run-qwen2.5-7B-sdpo-sci-colocate.sh, INCLUDING its
# arm-5-specific OOM fix: any skill-KD arm (5, 5.1, 5.2 -- all --sdpo-skill-kd)
# appends the same extra skill-KD samples (self-success/blind-correct +
# pitfall-condense forward+backward) regardless of whether the response
# mechanism is KD-loss or RLSD, so the same lower per-GPU token cap applies
# for the same reason (confirmed OOM at 24576 on the KD-loss script's arm 5,
# AND on this script's own arm 5.2 -- "Tried to allocate 11.75 GiB... GPU 2"
# at rollout 32 -- since the exact-string-match "= 5" check didn't cover the
# 5.1/5.2 variants added later). RLSD swaps a top-k teacher forward for a
# cheaper sampled-token one on the RESPONSE span only, which does not touch
# the skill-KD memory pressure that actually causes this OOM.
MAX_TOKENS_PER_GPU=24576
case "${SDPO_ABLATION_ARM}" in
   5|5.1|5.2)
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
   --wandb-group "qwen2.5-7B-sdpo-sci-rl-ablation-arm${SDPO_ABLATION_ARM}"
   --wandb-key "${WANDB_API_KEY}"
)

# Same rollout-engine memory margin as run-qwen2.5-7B-sdpo-sci-colocate.sh
# (see that script's own comment).
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
