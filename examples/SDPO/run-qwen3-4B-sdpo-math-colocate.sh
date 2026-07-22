#!/bin/bash

# SDPO ablation series -- Qwen3-4B on DAPO math, single 8x H200 (141GB) node,
# COLOCATE variant. Sibling of run-olmo3-7B-sdpo-math-colocate.sh (same
# rollout/eval/colocate skeleton), swapped to Qwen3-4B and driven by
# $SDPO_ABLATION_ARM to switch between 6 ablation legs sharing one script
# (same "one script + env-var switch" convention as
# examples/SDPO/enroot-run-sdpo.sh's $SDPO_MODEL and examples/EPO/
# enroot-run-epo.sh's $EPO_MODEL):
#
#   SDPO_ABLATION_ARM=1    pure vanilla GRPO, no SDPO at all: single-sample
#                          reward (plain_grpo_reward, not --rm-type dapo --
#                          see arm 1's own comment for why), no --group-rm,
#                          default std-normalized advantages, no dynamic
#                          sampling, no --calculate-per-token-loss (miles'
#                          own default seq-mean-token-mean aggregation)
#   SDPO_ABLATION_ARM=1.1  SDPO baseline: --group-rm + KD loss (jsd), no skill
#                          anything. PURE distillation for every SDPO arm
#                          below (1.1-5): --sdpo-pure-distill is the DEFAULT
#                          and deliberately left unset (not passed as
#                          --no-sdpo-pure-distill) -- task reward is 0 for
#                          every trace whenever --group-rm's sdpo_group_reward
#                          runs, so the GRPO advantage is exactly 0 and the
#                          entire training signal is the JSD divergence loss
#                          alone: no reward, no advantage, purely KD.
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
# Qwen3-4B is a THINKING model (unlike Qwen2.5-7B, which run-qwen3-8B-sdpo.sh's
# comment notes was chosen specifically to AVOID Qwen3's reasoning-collapse
# risk). Two independent thinking-related knobs, deliberately kept ORTHOGONAL:
#   1. --apply-chat-template-kwargs '{"enable_thinking":true}' -- whether the
#      model is ASKED to think at all (dataset/prompt-rendering time). ON for
#      every arm here (that's the whole point of testing a thinking model).
#   2. --sdpo-remove-thinking-from-demonstration -- whether a peer's <think>
#      block is stripped before splicing it into the RESPONSE-SDPO teacher
#      prefix (examples/SDPO/sdpo.py::_render_prefix/_build_teacher_prompt_str).
#      ON for every SDPO arm (1.1, 2-5) for the same reason run-qwen3-8B-sdpo.sh
#      requires it: an unstripped prefix is huge and teaches the student to
#      echo the peer's reasoning verbatim instead of reasoning independently.
#      This does NOT touch the skill-generation step's OWN input (skill-gen's
#      "WORKED SOLUTION" field is always the trace's raw, un-stripped response,
#      thinking included -- the skill is a SUMMARY of the full reasoning, by
#      design) or the KD/opd divergence loss (which covers the model's entire
#      response span, thinking tokens included -- there is no code path that
#      excludes them; confirmed by reading examples/SDPO/sdpo.py's
#      _compute_kl_for_sample and miles/backends/training_utils/loss_hub/opd.py).
#
# GPU/perf sizing: reasoned from Qwen3-4B (36 layers, hidden=2560, GQA 8 kv
# groups) vs Olmo3-7B (32 layers, hidden=4096, full MHA 32 kv groups) -- ~53%
# the per-token activation cost from hidden-size^2 alone, plus GQA shrinks
# attention activations further. Initial reasoned default (--max-tokens-per-
# gpu 40960 + --recompute-granularity selective) OOM'd during arm 1's actual
# backward pass ("CUDA out of memory... 129.91 GiB is allocated by PyTorch"
# -- colocate leaves less headroom than the pure-training-only comparison
# suggested, since the SGLang engines' KV cache/weights share the same GPUs).
# Reverted to Olmo3-7B's own proven-safe combo (--max-tokens-per-gpu 24576 +
# --recompute-granularity full) rather than re-guessing a second reasoned
# number -- 4B's real memory headroom over 7B is going into --colocate's
# rollout-engine footprint, not into a bigger training microbatch. Watch
# arm 1's re-run for a cleaner OOM margin before the remaining 5 arms.
#
# LR: 1e-6, matching run-olmo3-7B-sdpo-math-colocate.sh (the mixed GRPO+KD
# objective) rather than run-qwen3-8B-sdpo.sh's 1e-5 (pure-KD-only tuning) --
# arm 1 has a REAL GRPO advantage flowing with no KD term at all, so it needs
# the smaller, GRPO-safe LR; keeping one LR across all 6 arms holds this
# variable constant so the ablation isolates the SDPO/skill knobs, not LR.
#
# usage:
#   SDPO_ABLATION_ARM=1   bash examples/SDPO/run-qwen3-4B-sdpo-math-colocate.sh
#   SDPO_ABLATION_ARM=1.1 bash examples/SDPO/run-qwen3-4B-sdpo-math-colocate.sh
#   ... etc for 2, 3, 4, 5

set -exf

export PYTHONBUFFERED=16
SDPO_ABLATION_ARM="${SDPO_ABLATION_ARM:?Set SDPO_ABLATION_ARM to one of: 1 1.1 2 3 4 5}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
echo "SDPO_ABLATION_ARM: ${SDPO_ABLATION_ARM}"

source "/root/miles/scripts/models/qwen3-4B.sh"

SDPO_EXP="${SDPO_EXP:-qwen3-4B-sdpo-ablation-arm${SDPO_ABLATION_ARM}_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/miles/sdpo_dumps/${SDPO_EXP}"
echo "SDPO dump dir: ${DUMP_DIR}"

# NO --save / --load: exploratory ablation only, nothing here should be
# resumed or kept around (see module docstring above -- this also sidesteps
# the disk-fill checkpoint crash a full SDPO_ReAct run hit earlier).
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   --ref-load /root/Qwen3-4B_torch_dist
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)

# DAPO math train set, identical across all 6 arms -- the whole point of the
# ablation is to hold data/rollout/eval fixed and vary only the SDPO/skill
# knobs. enable_thinking:true so Qwen3 actually reasons before answering (its
# chat template defaults to thinking-on already, but set explicitly so this
# is never silently affected by a --tito-model auto-fill).
ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":true}'
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
        # examples.SDPO.sdpo.plain_grpo_reward (NOT --rm-type dapo/boxed_dapo)
        # -- async_rm's "dapo" rm_type calls math_dapo_utils.compute_score
        # with strict_box_verify defaulting to False (Minerva "Answer: X"
        # line-matching), which fails on a bare \boxed{...} response with no
        # such line (confirmed empirically); the "boxed_" prefix doesn't fix
        # this either (it feeds just the extracted boxed string back into the
        # same Minerva-pattern grader, which still finds no "Answer:" line).
        # plain_grpo_reward reuses sdpo.py's own _is_correct under
        # --sdpo-grader dapo, which correctly calls compute_score with
        # strict_box_verify=True -- the SAME grading criterion arms 1.1-5 use
        # via sdpo_group_reward, so the ablation isolates the SDPO/skill
        # knobs, not a grading-rule difference.
        RM_ARGS=(
            --custom-rm-path examples.SDPO.sdpo.plain_grpo_reward
            --sdpo-grader dapo
        )
        GRPO_ARGS=(
            --advantage-estimator grpo
            --entropy-coef 0.00
            --observe-training-entropy
            # NO --calculate-per-token-loss here (unlike every other arm) --
            # this is the "pure原始 GRPO" baseline: std-normalized advantages
            # (default grpo_std_normalization=True, the original DeepSeekMath
            # GRPO formula A_i=(r_i-mean(r))/std(r) -- --disable-grpo-std-
            # normalization is the Dr.GRPO MODIFICATION, not the default), no
            # dynamic-sampling filter, and miles' own default seq-mean-token-
            # mean loss aggregation (NOT the token-mean every SDPO arm below
            # uses) -- confirmed via miles/ray/rollout/train_data_conversion.py
            # and miles/utils/arguments.py's argparse defaults, not assumed.
        )
        ;;
    1.1)
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
            --sdpo-grader dapo
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
            --sdpo-remove-thinking-from-demonstration
            --sdpo-answer-tag answer
            --entropy-coef 0.00
            --observe-training-entropy
            --calculate-per-token-loss
        )
        ;;
    2)
        # + self-skill, skill-source correct ONLY, no skill-KD. Response-SDPO
        # teacher prefix switches to the peer's SKILL (not the full trace) once
        # self-skill is on, matching run-olmo3-7B-sdpo-math-colocate.sh's own
        # choice (--sdpo-response-prefix skill), per explicit instruction to
        # use skill-as-prefix whenever self-skill is active.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
            --sdpo-grader dapo
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
            --sdpo-remove-thinking-from-demonstration
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
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
            --sdpo-grader dapo
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
            --sdpo-remove-thinking-from-demonstration
            --sdpo-answer-tag answer
            --sdpo-self-skill
            --sdpo-skill-source incorrect
            --sdpo-skill-max-new-tokens 1024
            --sdpo-pitfall-summary-backend self
            --sdpo-response-prefix skill
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
            --sdpo-grader dapo
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
            --sdpo-remove-thinking-from-demonstration
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
        # matching run-olmo3-7B-sdpo-math-colocate.sh's exact value per
        # explicit instruction to keep it if present.
        RM_ARGS=(
            --group-rm
            --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
            --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
            --sdpo-grader dapo
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
            --sdpo-remove-thinking-from-demonstration
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
        echo "Unknown SDPO_ABLATION_ARM='${SDPO_ABLATION_ARM}' (expected one of: 1 1.1 2 3 4 5)" >&2
        exit 1
        ;;
esac

# Identical across all 6 arms: AIME-2025 + Minerva-Math eval (already staged
# under /root/math_eval on this host -- see build_math_eval.py). Arm 1 (plain
# GRPO, no --group-rm) still uses sdpo_eval_reward: it works standalone (just
# grades pass@1 via the same DAPO/general math grader) and keeps the eval
# metric computation identical across every arm regardless of training-side
# reward wiring.
#
# NO --skip-eval-before-train: a genuine step-0 (untrained-checkpoint) eval is
# required so every arm's "did training help" comparison is against its OWN
# real baseline, not inferred from a different arm's eval at a nearby step --
# arm 1.1's first eval landed at rollout_id=9 (--eval-interval 10 skips the
# pre-train point), so its early jump could only be compared against arm 1's
# OWN step-9 eval as an approximation, not a true baseline for arm 1.1 itself.
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

# See module docstring for the 4B-vs-7B reasoning behind these two numbers.
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
   --wandb-group "qwen3-4B-sdpo-ablation-arm${SDPO_ABLATION_ARM}"
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
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
