#!/bin/bash
# SDPO_ReAct -- Qwen3-4B on tau2 (tau2-bench: retail/airline/telecom customer-
# service simulation, https://github.com/sierra-research/tau2-bench), single
# 8x H200 (141GB) node, COLOCATE variant. Sibling of run-qwen3-4B-sdpo-react-
# agentic.sh (webshop/alfworld) -- same 7-arm SDPO ablation structure, but a
# DIFFERENT generate path: tau2's own Orchestrator class drives the whole
# agent<->user-simulator<->environment conversation itself, so there is no
# seam for this repo's own <tool_call>-parsing loop (multi_turn.generate,
# used by webshop/alfworld/native) to plug into. This script instead uses
# miles.rollout.generate_hub.agentic_tool_call.generate + --use-session-
# server + --custom-agent-function-path (the SAME mechanism
# examples/experimental/swe-agent-v2/ uses for Harbor-based agents): the
# session server exposes an OpenAI-compatible endpoint (full `tools=`/
# `tool_calls` passthrough confirmed via tests/e2e/sglang/utils/
# session_tool_agent.py) that the tau2 sidecar's LLMAgent talks to via
# litellm's own `api_base` custom-endpoint routing -- see
# tools/tau2/docker/server.py's module docstring for the full wiring.
#
# WHY A SEPARATE FILE (not a domain branch inside the agentic script): the
# generate-path plumbing (CUSTOM_GENERATE_ARGS, TITO/session-server args) is
# categorically different from multi_turn.generate's tool-specs/executor
# wiring -- forcing both into one script's if/elif chain would tangle two
# unrelated rollout architectures together for no benefit. The 7-arm case
# block below is copied VERBATIM, character-for-character, from the native/
# agentic scripts -- RM_ARGS/GRPO_ARGS are pure SDPO/skill/RLSD machinery,
# orthogonal to domain.
#
# Task data: inclusionAI/AReaL-tau2-data (HF dataset) -- the SEA-engine-
# scaled 1982-task RL training set AReaL-SEA-235B-A22B was trained on, NOT
# tau2-bench's own tiny shipped task list (see data/build_tau2_data.py's
# docstring). One task = one episode = one outbound sidecar call (unlike
# webshop/alfworld's per-STEP tool calls); reward is tau2's own
# evaluate_simulation() score, forwarded verbatim onto sample.metadata by
# tools/tau2/agent_function.py.
#
# Env overrides (swap without editing the script):
#   SDPO_REACT_TAU2_MAX_STEPS      (default 40)   tau2 orchestrator turn budget, TRAIN only -- see below
#   SDPO_REACT_TAU2_EVAL_MAX_STEPS (default 100)  tau2 orchestrator turn budget, EVAL only -- see below
#   SDPO_REACT_NUM_ROLLOUT      (default 300)  --num-rollout
#   SDPO_REACT_EVAL_N_SAMPLES   (default 8)    eval samples/prompt (eval yaml)
#   TAU_USER_MODEL_PROVIDER     (default openai)  litellm provider for the user simulator --
#                                                  "openai" routes through the Salesforce
#                                                  Research gateway (SFT_GATEWAY_KEY/
#                                                  OPENAI_API_URL, loaded from ~/gitproj/apis/
#                                                  .env by enroot-run-sdpo-react.sh -- same
#                                                  credentials --sdpo-judge-model already
#                                                  uses in run-qwen3-4B-sdpo-react-native.sh).
#                                                  Override to gemini|deepseek (examples/tau-
#                                                  bench's own convention) + the matching
#                                                  *_API_KEY if you have one and want off the
#                                                  gateway.
#   TAU_USER_MODEL              (default gpt-5.6-luna)
#   GEMINI_API_KEY / DEEPSEEK_API_KEY  -- only used if TAU_USER_MODEL_PROVIDER is switched to one of these
#
# usage: bash examples/SDPO_ReAct/run-qwen3-4B-sdpo-react-tau2.sh
set -exf

export PYTHONBUFFERED=16
export SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-8}"
# tau2 has no separate train/eval turn budget the way webshop/alfworld do
# (--generate-max-turns governs multi_turn.generate's OWN turn loop; tau2's
# turn loop lives entirely inside the sidecar's Orchestrator, driven by
# metadata["tau2_max_steps"] instead -- see agent_function.py). TRAIN budget
# (this env var) and EVAL budget (SDPO_REACT_TAU2_EVAL_MAX_STEPS, forwarded
# below + read by data/eval_tau2.yaml's metadata_overrides) are DELIBERATELY
# split, same pattern as every other SDPO_ReAct domain's SDPO_REACT_TRAIN_
# MAX_TURNS/SDPO_REACT_EVAL_MAX_TURNS.
#
# Train default 40 (NOT the tau2-native 100, NOT 30): tau2-bench's own
# Orchestrator.__init__ defaults to max_steps=100 and its CLI defaults to
# DEFAULT_MAX_STEPS=200 (tau2/config.py) -- every message counts as one step
# (agent turn, user turn, AND each individual tool call each +1, confirmed
# by reading Orchestrator.step()), so a real multi-tool retail/airline task
# (auth + several lookups + a modify + a confirmation, with the user
# simulator's own turns interleaved) routinely needs 40-80+ steps. A too-
# tight budget hits TerminationReason.MAX_STEPS, which tau2's own
# evaluate_simulation() scores as a HARD reward=0 regardless of whether the
# agent was about to get it right -- confirmed live on this exact 9B/tau2
# training run (after the --sglang-tool-call-parser fix below): cap=30 hits
# MAX_STEPS on 33% of episodes, cap=40 only 13% (concentrated in telecom,
# which needs the most tool calls/episode), cap=50 just 4%, cap=100 0% --
# but avg per-episode token cost is ~flat across all four (4275/4452/4412/
# 4429), since most conversations finish well under 40 turns regardless.
# 40 is the chosen train-time tradeoff: keeps most of cap=100's near-zero
# truncation-noise benefit for GRPO/SDPO's group-relative advantage
# (a real completion getting reward=0 purely from running out of turns
# corrupts the whole group's advantage estimate) while capping the long
# tail's contribution to "training is slow" (the original 30 turn cap
# skewed too far toward speed at truncation-noise's expense; 100 is eval-
# only precision paid for at 2.5x the train step count for near-zero
# additional real completions).
SDPO_REACT_TAU2_MAX_STEPS="${SDPO_REACT_TAU2_MAX_STEPS:-40}"
export SDPO_REACT_TAU2_MAX_STEPS
# Eval keeps the full tau2-recommended budget -- see data/eval_tau2.yaml's
# own metadata_overrides comment for why eval and train diverge here.
SDPO_REACT_TAU2_EVAL_MAX_STEPS="${SDPO_REACT_TAU2_EVAL_MAX_STEPS:-100}"
export SDPO_REACT_TAU2_EVAL_MAX_STEPS
# Same ablation-arm convention as run-qwen3-4B-sdpo-react-native.sh (see the
# RM_ARGS/GRPO_ARGS case block below): 1 | 1.1 | 2 | 3 | 4 | 5 | 5.1.
SDPO_REACT_ARM="${SDPO_REACT_ARM:-1.1}"
SDPO_REACT_NUM_ROLLOUT="${SDPO_REACT_NUM_ROLLOUT:-300}"

# Batch sizing MUST stay divisible by the data-parallel size (= TRAIN_GPUS/TP).
SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-8}"
N_SAMPLES_PER_PROMPT=8
SDPO_REACT_TP="${SDPO_REACT_TP:-1}"
DP_SIZE=$((SDPO_REACT_TRAIN_GPUS / SDPO_REACT_TP))
ROLLOUT_BATCH_SIZE="${SDPO_REACT_ROLLOUT_BATCH:-$((DP_SIZE * 4))}"
if [ $((ROLLOUT_BATCH_SIZE % DP_SIZE)) -ne 0 ]; then
    ROLLOUT_BATCH_SIZE=$((DP_SIZE * 4))
    echo "WARN: rollout batch not divisible by dp=${DP_SIZE} (train_gpus=${SDPO_REACT_TRAIN_GPUS}/tp=${SDPO_REACT_TP}); falling back to ${ROLLOUT_BATCH_SIZE}"
fi
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
echo "BATCH: train_gpus=${SDPO_REACT_TRAIN_GPUS} tp=${SDPO_REACT_TP} dp=${DP_SIZE} rollout_batch=${ROLLOUT_BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Only Qwen3-4B (native, TITO tokenizer type "qwen3") is wired here -- unlike
# the native/agentic scripts' model-family case block, tau2's generate path
# additionally needs a TITOTokenizerType entry (see --tito-model below);
# adding another model means confirming ITS TITO support first, not just
# copying a models/*.sh source line.
MODEL_NAME=Qwen3-4B
MODEL_ARG_SH=scripts/models/qwen3-4B.sh
TOOL_PARSER=qwen25
TOOL_GRAMMAR=qwen25
MAX_TOKENS_PER_GPU=12288
echo "MODEL: ${MODEL_NAME} | tool-parser=${TOOL_PARSER} (unused by tau2's own tool schemas, kept for RM_ARGS' --sdpo-tool-grammar below)"
source "$REPO_ROOT/${MODEL_ARG_SH}"

# --- 0. tau2 sidecar (idempotent, ONE container / ONE port for the whole job) ---
bash "$SCRIPT_DIR/tools/tau2/run_tau2_sidecar.sh"

export SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-true}"
SDPO_REACT_PURE_DISTILL="${SDPO_REACT_PURE_DISTILL:-true}"
echo "SDPO_REACT_THINKING: ${SDPO_REACT_THINKING} | SDPO_REACT_PURE_DISTILL: ${SDPO_REACT_PURE_DISTILL}"

# --- 0b. data prep: tau2 (AReaL-tau2-data, retail+airline+telecom combined) --
TAU2_DIR="/root/data/tau2_data"
TRAIN_DATA="$TAU2_DIR/tau2_train.jsonl"
# SDPO_REACT_TAU2_EVAL_CONFIG override: point at data/eval_tau2_
# telecom_only.yaml (retail/airline dropped) once those two domains are
# saturated -- confirmed live: a real ablation run hit airline=100%/
# retail=100% mean reward at eval_0, only telecom sitting lower -- so
# continuing to spend 2/3 of every eval's wall-clock on two domains
# that already read 100% adds no signal. Default stays the full 3-
# domain yaml (safe default: use this only once you have confirmed the
# other two domains are actually saturated for THIS run).
EVAL_CFG="${SDPO_REACT_TAU2_EVAL_CONFIG:-$SCRIPT_DIR/data/eval_tau2.yaml}"
mkdir -p "$TAU2_DIR"
[ -f "$TRAIN_DATA" ] || \
    (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_tau2_data \
        --out-dir "$TAU2_DIR" --n-eval-per-domain "${SDPO_REACT_TAU2_N_EVAL_PER_DOMAIN:-30}")
python - <<PYCK
import json
from collections import Counter
c = Counter(json.loads(l)["metadata"]["tau2_domain"] for l in open("$TRAIN_DATA"))
print(f"tau2 train rows OK: {dict(c)}")
PYCK

[ "$SDPO_REACT_PURE_DISTILL" = "true" ] && _DISTILL=pure || _DISTILL=mixed
[ "$SDPO_REACT_THINKING" = "true" ] && _THINK=think || _THINK=nothink
if [ "$SDPO_REACT_THINKING" = "true" ]; then
    REMOVE_THINKING_ARG=()
else
    REMOVE_THINKING_ARG=(--sdpo-remove-thinking-from-demonstration)
fi
CFG_TAG="${MODEL_NAME}-tau2-${SDPO_REACT_ARM}-${_DISTILL}-${_THINK}"

SDPO_REACT_EXP="${SDPO_REACT_EXP:-qwen3-4B-sdpo-react-tau2-${CFG_TAG}_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/data/sdpo_dumps/${SDPO_REACT_EXP}"
echo "SDPO_ReAct dump dir: ${DUMP_DIR}"

CKPT_DIR="${SDPO_REACT_CKPT_DIR:-/root/data/sdpo_ckpts/qwen3-4B-sdpo-react-tau2-${CFG_TAG}_ckpt}"

EXPLOG="${SDPO_REACT_EXPLOG:-$REPO_ROOT/examples/SDPO_ReAct/explog.jsonl}"
GIT_COMMIT="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(cd "$REPO_ROOT" && [ -n "$(git status --porcelain 2>/dev/null)" ] && echo dirty || echo clean)"
python - <<PYLOG || true
import json, time, os
row = {
    "exp": "${SDPO_REACT_EXP}",
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "arm": "${SDPO_REACT_ARM}",
    "domain": "tau2",
    "model": "${MODEL_NAME}",
    "train_data": "${TRAIN_DATA}",
    "num_rollout": "${SDPO_REACT_NUM_ROLLOUT}",
    "tau2_max_steps": "${SDPO_REACT_TAU2_MAX_STEPS}",
    "wandb_group": "qwen3-4B-sdpo-react-tau2-${SDPO_REACT_ARM}",
    "dump_dir": "${DUMP_DIR}",
    "ckpt_dir": "${CKPT_DIR}",
    "git_commit": "${GIT_COMMIT}",
    "git_state": "${GIT_DIRTY}",
    "note": os.environ.get("SDPO_REACT_NOTE", ""),
}
with open("${EXPLOG}", "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("explog appended ->", "${EXPLOG}")
print(json.dumps(row, ensure_ascii=False, indent=2))
PYLOG

CKPT_ARGS=(
   --hf-checkpoint /root/${MODEL_NAME}
   --ref-load /root/${MODEL_NAME}_torch_dist
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)
if [ "${SDPO_REACT_SAVE_CKPT:-0}" = "1" ]; then
    CKPT_ARGS+=(--save "${CKPT_DIR}" --load "${CKPT_DIR}" --save-interval 10 --override-opt-param-scheduler)
fi

ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_DATA"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --apply-chat-template-kwargs "{\"enable_thinking\":${SDPO_REACT_THINKING}}"
   --rollout-shuffle
   --num-rollout "${SDPO_REACT_NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len 8192
   --rollout-max-context-len 40960
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
   --over-sampling-batch-size "${ROLLOUT_BATCH_SIZE}"
)
if [ "${SDPO_REACT_DYNAMIC_SAMPLE:-true}" != "true" ]; then
    echo "DYNAMIC SAMPLING: OFF (raw batch, no oversample/filter/rebalance)"
elif [ "$SDPO_REACT_ARM" = "1" ] || [ "$SDPO_REACT_PURE_DISTILL" != "true" ]; then
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
else
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_sdpo_group_has_prefix)
    ROLLOUT_ARGS+=(--sdpo-dynamic-filter-min-correct "${SDPO_REACT_MIN_CORRECT:-1}")
fi

# tau2's generate path: agentic_tool_call.generate hands --custom-agent-
# function-path a base_url pointed at the session-server proxy (started
# because --use-session-server is set below) -- tools/tau2/agent_function.py
# forwards that to the tau2 sidecar's /run endpoint, which builds tau2's own
# LLMAgent against it via litellm. --tito-model qwen3 lets miles auto-resolve
# --chat-template-path for this family (see arguments.py's
# should_auto_resolve block); --tito-allowed-append-roles must include BOTH
# "tool" (env responses) and "user" (the user-simulator's own turns --
# genuinely appended mid-conversation here, unlike a plain tool-calling loop
# where only "tool" ever appears) -- the resulting warning about "user" is
# expected, not a misconfiguration (tau2 legitimately interleaves user turns).
CUSTOM_GENERATE_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path examples.SDPO_ReAct.tools.tau2.agent_function.run
   --use-session-server
   --tito-model qwen3
   --tito-allowed-append-roles tool user
)

# Ablation-arm series -- IDENTICAL semantics to run-qwen3-4B-sdpo-react-native
# .sh's own arm block (copied verbatim below; see that script's header
# comment for the full arm-by-arm rationale). RM_ARGS/GRPO_ARGS are pure
# SDPO/skill/RLSD machinery, orthogonal to domain -- tau2 has no LLM-judge
# fallback path (grading is tau2's own evaluate_simulation() score, no text-
# answer ambiguity), same as webshop/alfworld's own lack of that block.
#
#   SDPO_REACT_ARM=1    Plain GRPO, no SDPO machinery at all.
#   SDPO_REACT_ARM=1.1  RLSD baseline, NO self-skill.
#   SDPO_REACT_ARM=2    + self-skill, skill-source correct only, NO skill-KD.
#   SDPO_REACT_ARM=3    + self-skill, skill-source incorrect only, NO skill-KD.
#   SDPO_REACT_ARM=4    + self-skill, skill-source all, NO skill-KD.
#   SDPO_REACT_ARM=5    + self-skill, skill-source all, WITH skill-KD (mode=both).
#   SDPO_REACT_ARM=5.1  IDENTICAL to arm 5 except --sdpo-skill-kd-mode both-blind.
RM_ARGS=(
   --sdpo-answer-tag answer
   "${REMOVE_THINKING_ARG[@]}"
   --sdpo-reframe-multiturn-prefix
   --sdpo-tool-grammar "${TOOL_GRAMMAR}"
   --sdpo-max-prefix-chars 20000
)

case "${SDPO_REACT_ARM}" in
   1)
      RM_ARGS=(
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_plain_grpo_reward
         --sdpo-grader dapo
         --sdpo-answer-tag answer
      )
      GRPO_ARGS=(
         --advantage-estimator grpo
         --entropy-coef 0.00
      )
      ;;
   1.1)
      RM_ARGS+=(
         --group-rm
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
         --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
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
         --sdpo-prefer-tool-use-peer
         --entropy-coef 0.00
         --calculate-per-token-loss
      )
      ;;
   2)
      RM_ARGS+=(
         --group-rm
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
         --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
         --sdpo-grader dapo
         --sdpo-self-skill
         --sdpo-skill-source correct
         --sdpo-skill-max-new-tokens 1024
         --sdpo-response-prefix skill
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
         --sdpo-prefer-tool-use-peer
         --entropy-coef 0.00
         --calculate-per-token-loss
      )
      ;;
   3)
      RM_ARGS+=(
         --group-rm
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
         --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
         --sdpo-grader dapo
         --sdpo-self-skill
         --sdpo-skill-source incorrect
         --sdpo-skill-max-new-tokens 1024
         --sdpo-pitfall-summary-backend self
         --sdpo-env-feedback-max-chars 2000
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
         --sdpo-prefer-tool-use-peer
         --entropy-coef 0.00
         --calculate-per-token-loss
      )
      ;;
   4)
      RM_ARGS+=(
         --group-rm
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
         --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
         --sdpo-grader dapo
         --sdpo-self-skill
         --sdpo-skill-source all
         --sdpo-skill-max-new-tokens 1024
         --sdpo-pitfall-summary-backend self
         --sdpo-response-prefix skill
         --sdpo-env-feedback-max-chars 2000
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
         --sdpo-prefer-tool-use-peer
         --entropy-coef 0.00
         --calculate-per-token-loss
      )
      ;;
   5)
      RM_ARGS+=(
         --group-rm
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
         --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
         --sdpo-grader dapo
         --sdpo-self-skill
         --sdpo-skill-source all
         --sdpo-skill-max-new-tokens 2048
         --sdpo-pitfall-summary-backend self
         --sdpo-response-prefix skill
         --sdpo-env-feedback-max-chars 2000
         --sdpo-skill-kd
         --sdpo-skill-kd-coef 0.01
         --sdpo-skill-kd-mode both
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
         --sdpo-prefer-tool-use-peer
         --entropy-coef 0.00
         --calculate-per-token-loss
      )
      ;;
   5.1)
      RM_ARGS+=(
         --group-rm
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
         --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
         --sdpo-grader dapo
         --sdpo-self-skill
         --sdpo-skill-source all
         --sdpo-skill-max-new-tokens 2048
         --sdpo-pitfall-summary-backend self
         --sdpo-response-prefix skill
         --sdpo-env-feedback-max-chars 2000
         --sdpo-skill-kd
         --sdpo-skill-kd-coef 0.01
         --sdpo-skill-kd-mode both-blind
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
         --sdpo-prefer-tool-use-peer
         --entropy-coef 0.00
         --calculate-per-token-loss
      )
      ;;
   *)
      echo "Unknown SDPO_REACT_ARM='${SDPO_REACT_ARM}' (expected: 1 | 1.1 | 2 | 3 | 4 | 5 | 5.1)" >&2
      exit 1 ;;
esac

EVAL_ARGS=(
   --eval-interval 10
   --eval-config "$EVAL_CFG"
   --n-samples-per-eval-prompt "${SDPO_REACT_EVAL_N_SAMPLES}"
   --log-passrate
)
if [ "${SDPO_REACT_SKIP_EVAL0:-0}" = "1" ]; then
    EVAL_ARGS+=(--skip-eval-before-train)
    echo "EVAL: --skip-eval-before-train (debug: no step-0 eval)"
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${SDPO_REACT_TP:-2}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-12288}"
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
   --wandb-group "qwen3-4B-sdpo-react-tau2-${CFG_TAG}"
   --wandb-key "${WANDB_API_KEY}"
)

# --sglang-tool-call-parser is NOT optional here, unlike the native/agentic
# scripts (which drive multi_turn.generate's OWN text-parsing loop over
# sglang's raw /generate endpoint via --generate-tool-call-parser). tau2
# instead goes through agentic_tool_call.generate -> the session-server proxy
# -> sglang's native /v1/chat/completions endpoint (see tools/tau2/agent_
# function.py's module docstring) -- WITHOUT this flag, sglang never parses
# <tool_call> tags out of the raw text at all, so message.tool_calls comes
# back None for every turn. tau2's own LLMAgent (via litellm) only ever reads
# that structured field, never falls back to text -- confirmed live against a
# real training dump (sdpo-react-ablation-Qwen3.5-9B-tau2-grpo-a-think): 83%
# of a 257-episode sample had ZERO tool/tool_calls messages, with the model
# narrating fake tool results in plain text instead ("I already called
# get_user_details... let me assume the response was..."), episodes running
# to the turn cap with reward=0. This single missing flag is the primary
# cause of eval success rates (33%/56%/4% retail/airline/telecom) landing far
# below the Qwen tech report's 79.1/79.9 -- not a training/algorithm issue.
#
# --sglang-reasoning-parser is DELIBERATELY NOT set here (even with thinking
# on), unlike the native/agentic scripts: it makes sglang split reasoning out
# into a separate message.reasoning_content field, but tau2's own generate()
# (tau2/utils/llm_utils.py) only ever reads response_choice.message.content
# -- there is no reasoning_content field on tau2's AssistantMessage at all
# (confirmed live: not in AssistantMessage.model_fields) -- so tau2 silently
# drops it when relaying the assistant turn back through litellm. On the
# NEXT turn the session-server compares the message it stored (content +
# non-empty reasoning_content) against what tau2 sends back (content only,
# no reasoning_content key) via message_matches() -- TEMPLATE_RELEVANT_KEYS
# includes reasoning_content, so this mismatches, the append-only checkpoint
# detector treats it as a brand-new appended message, and since role=
# 'assistant' is not in --tito-allowed-append-roles (tool/user only) every
# single turn after the first 400s. Confirmed live: a real run hit ~1740
# such 400s in under 10 minutes (litellm's own num_retries=3 backoff on each
# one is also a real contributor to "training is slow"). Leaving reasoning-
# parser unset keeps <think>...</think> inline in content instead, which
# survives the tau2 round-trip unchanged (the TITO jinja templates already
# know how to strip/re-render inline <think> tags).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.75}"
   --sglang-tool-call-parser qwen25
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-8}"
if [ "$SDPO_REACT_TRAIN_GPUS" -lt 8 ]; then
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((SDPO_REACT_TRAIN_GPUS-1)))"
    echo "TRAIN GPUs: ${SDPO_REACT_TRAIN_GPUS} (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
fi

ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
sleep 2

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "${SDPO_REACT_TRAIN_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# --- checkpoint pruner (background): identical to the native/agentic
# scripts' own -- keep only the newest COMPLETED iter dir. ---
if [ "${SDPO_REACT_SAVE_CKPT:-0}" = "1" ]; then
(
    set +f
    while true; do
        sleep 60
        latest_file="${CKPT_DIR}/latest_checkpointed_iteration.txt"
        [ -f "$latest_file" ] || continue
        latest_num="$(cat "$latest_file" 2>/dev/null)"
        [ -n "$latest_num" ] || continue
        case "$latest_num" in ''|*[!0-9]*) continue;; esac
        for d in "${CKPT_DIR}"/iter_*; do
            [ -d "$d" ] || continue
            n="$(basename "$d")"; n="${n#iter_}"
            case "$n" in ''|*[!0-9]*) continue;; esac
            if [ "$((10#$n))" -lt "$((10#$latest_num))" ]; then
                rm -rf "$d"
            fi
        done
    done
) &
PRUNER_PID=$!
trap 'kill "$PRUNER_PID" 2>/dev/null || true' EXIT
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"/root/Megatron-LM/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
        \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
        \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\",
        \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
        \"SDPO_REACT_TAU2_MAX_STEPS\": \"${SDPO_REACT_TAU2_MAX_STEPS}\",
        \"SDPO_REACT_TAU2_EVAL_MAX_STEPS\": \"${SDPO_REACT_TAU2_EVAL_MAX_STEPS}\",
        \"TAU_USER_MODEL_PROVIDER\": \"${TAU_USER_MODEL_PROVIDER:-openai}\",
        \"TAU_USER_MODEL\": \"${TAU_USER_MODEL:-gpt-5.6-luna}\",
        \"OPENAI_API_URL\": \"${OPENAI_API_URL:-}\",
        \"SFT_GATEWAY_KEY\": \"${SFT_GATEWAY_KEY:-}\",
        \"GEMINI_API_KEY\": \"${GEMINI_API_KEY:-}\",
        \"DEEPSEEK_API_KEY\": \"${DEEPSEEK_API_KEY:-}\"
     }
   }" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${SDPO_REACT_TRAIN_GPUS}" \
   --rollout-num-gpus "${SDPO_REACT_TRAIN_GPUS}" \
   --colocate \
   --update-weights-interval 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${CUSTOM_GENERATE_ARGS[@]} \
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
