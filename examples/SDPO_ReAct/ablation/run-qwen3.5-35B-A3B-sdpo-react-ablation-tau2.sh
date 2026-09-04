#!/bin/bash
# SDPO/RLSD/GRPO ablation matrix -- Qwen3.5-35B-A3B MoE on Tau2-bench
# (retail/airline/telecom customer-service simulation, via
# agentic_tool_call.generate + --use-session-server -- tau2's own
# Orchestrator drives the whole agent<->user-simulator<->environment
# conversation, categorically different from the other two ablation
# scripts' native <tool_call> loop), single 8x A100 (80GB) node, COLOCATE
# variant.
#
# MoE wiring (EP=8, R3 rollout-routing-replay, flex token dispatcher) is
# reused VERBATIM from ../run-qwen3.5-27B-35BA3B-sdpo-react-native.sh's own
# MoE NOTES header -- see ../ablation/run-qwen3.5-35B-A3B-sdpo-react-
# ablation-mathcodesearch.sh's header for the full rationale.
#
# Sibling of ../run-qwen3-4B-sdpo-react-tau2.sh: SAME session-server/sidecar
# machinery, but with the two-axis SDPO_ABLATION_ALGO x SDPO_ABLATION_ARM
# switch instead of the single numeric SDPO_REACT_ARM -- see
# ../ablation/run-qwen3.5-4B-sdpo-react-ablation-mathcodesearch.sh's header
# for the full arm/algo -> flag mapping table (identical mapping here; tau2
# has no LLM-judge fallback path, same as webshop/alfworld, so RM_ARGS never
# needs --sdpo-search-judge-fallback).
#
# --tito-model qwen35 (NOT qwen3 -- Qwen3.5 needs its own TITO tokenizer
# type, confirmed present in miles/utils/chat_template_utils/
# tito_tokenizer.py's TITOTokenizerType.QWEN35).
#
# A100-80G retune + path parameterization: see the mathcodesearch MoE
# sibling script's header for the full rationale.
#
# usage:
#   SDPO_ABLATION_ALGO=grpo SDPO_ABLATION_ARM=a \
#     bash examples/SDPO_ReAct/ablation/run-qwen3.5-35B-A3B-sdpo-react-ablation-tau2.sh
set -exf

SDPO_ABLATION_DATA_ROOT="${SDPO_ABLATION_DATA_ROOT:-/root}"
SDPO_ABLATION_MODEL_ROOT="${SDPO_ABLATION_MODEL_ROOT:-/root}"
SDPO_ABLATION_DUMP_ROOT="${SDPO_ABLATION_DUMP_ROOT:-/root/data/sdpo_dumps}"
SDPO_ABLATION_CKPT_ROOT="${SDPO_ABLATION_CKPT_ROOT:-/root/data/sdpo_ckpts}"
SDPO_ABLATION_MEGATRON_PATH="${SDPO_ABLATION_MEGATRON_PATH:-/root/Megatron-LM}"

export PYTHONBUFFERED=16
export SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-8}"
# Default 100, not 30: tau2-bench's own Orchestrator defaults to
# max_steps=100 (CLI defaults to 200, tau2/config.py's DEFAULT_MAX_STEPS) --
# every message (agent turn, user turn, EACH tool call) counts as one step,
# so a real multi-tool task often needs 40-80+ steps. A too-tight budget
# hits TerminationReason.MAX_STEPS, which tau2's own evaluate_simulation()
# scores as a hard reward=0 regardless of task competence (confirmed live:
# 29% of a max_steps=30 ablation run's training episodes hit MAX_STEPS --
# see run-qwen3-4B-sdpo-react-tau2.sh's matching comment for the full data).
# TRAIN budget (this env var) vs EVAL budget (SDPO_REACT_TAU2_EVAL_MAX_STEPS,
# forwarded below + read by data/eval_tau2.yaml's metadata_overrides) are
# DELIBERATELY split, same pattern as every other SDPO_ReAct domain's
# SDPO_REACT_TRAIN_MAX_TURNS/SDPO_REACT_EVAL_MAX_TURNS split. Train default
# 40 (not tau2's own 100, not 30): confirmed live on this exact model/tau2
# combo (after the --sglang-tool-call-parser fix below) that raising the
# cap from 30->40 cuts the MAX_STEPS-truncation rate (a HARD reward=0
# regardless of task competence, which corrupts GRPO/SDPO's group-relative
# advantage) from 33% to 13% (mostly telecom, which needs the most tool
# calls/episode) at essentially flat avg per-episode token cost (~4300-4450
# across cap=30/40/50/100) -- most conversations finish well under 40 turns
# regardless of the cap. 40 keeps most of cap=100's truncation-noise benefit
# while capping the long tail's contribution to overall training speed;
# eval keeps the full 100 since it's infrequent/low-sample and exists to
# report an honest benchmark number.
SDPO_REACT_TAU2_MAX_STEPS="${SDPO_REACT_TAU2_MAX_STEPS:-40}"
export SDPO_REACT_TAU2_MAX_STEPS
SDPO_REACT_TAU2_EVAL_MAX_STEPS="${SDPO_REACT_TAU2_EVAL_MAX_STEPS:-100}"
export SDPO_REACT_TAU2_EVAL_MAX_STEPS
SDPO_ABLATION_ALGO="${SDPO_ABLATION_ALGO:?Set SDPO_ABLATION_ALGO to one of: grpo sdpo rlsd}"
SDPO_ABLATION_ARM="${SDPO_ABLATION_ARM:?Set SDPO_ABLATION_ARM to one of: a b c d e f}"
if [ "$SDPO_ABLATION_ALGO" = "grpo" ]; then
    case "$SDPO_ABLATION_ARM" in
        a|e|f) ;;
        *) echo "GRPO only supports arms a/e/f (got '${SDPO_ABLATION_ARM}')" >&2; exit 1 ;;
    esac
fi
SDPO_REACT_NUM_ROLLOUT="${SDPO_REACT_NUM_ROLLOUT:-51}"

SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-8}"
N_SAMPLES_PER_PROMPT=8
SDPO_REACT_TP="${SDPO_REACT_TP:-2}"
DP_SIZE=$((SDPO_REACT_TRAIN_GPUS / SDPO_REACT_TP))
ROLLOUT_BATCH_SIZE="${SDPO_REACT_ROLLOUT_BATCH:-$((DP_SIZE * 4))}"
if [ $((ROLLOUT_BATCH_SIZE % DP_SIZE)) -ne 0 ]; then
    ROLLOUT_BATCH_SIZE=$((DP_SIZE * 4))
    echo "WARN: rollout batch not divisible by dp=${DP_SIZE}; falling back to ${ROLLOUT_BATCH_SIZE}"
fi
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
echo "BATCH: train_gpus=${SDPO_REACT_TRAIN_GPUS} tp=${SDPO_REACT_TP} dp=${DP_SIZE} rollout_batch=${ROLLOUT_BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REACT_DIR="$REPO_ROOT/examples/SDPO_ReAct"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

MODEL_NAME=Qwen3.5-35B-A3B
MODEL_ARG_SH=scripts/models/qwen3.5-35B-A3B.sh
TOOL_GRAMMAR=qwen3_coder
SDPO_REACT_EP_SIZE="${SDPO_REACT_EP_SIZE:-8}"
MAX_TOKENS_PER_GPU="${SDPO_ABLATION_MAX_TOKENS_PER_GPU:-4096}"
if [ "$SDPO_ABLATION_ARM" = "e" ] || [ "$SDPO_ABLATION_ARM" = "f" ]; then
    MAX_TOKENS_PER_GPU="${SDPO_ABLATION_MAX_TOKENS_PER_GPU:-2048}"
fi
echo "MODEL: ${MODEL_NAME} | ep=${SDPO_REACT_EP_SIZE} | max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"
source "$REPO_ROOT/${MODEL_ARG_SH}"

# --- 0. tau2 sidecar (idempotent, ONE container / ONE port for the whole job) ---
bash "$REACT_DIR/tools/tau2/run_tau2_sidecar.sh"

export SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-true}"

# --- data prep: tau2 (AReaL-tau2-data, retail+airline+telecom combined) ---
TAU2_DIR="${SDPO_ABLATION_DATA_ROOT}/data/tau2_data"
TRAIN_DATA="$TAU2_DIR/tau2_train.jsonl"
# SDPO_REACT_TAU2_EVAL_CONFIG override: point at data/eval_tau2_
# telecom_only.yaml (retail/airline dropped) once those two domains are
# saturated -- confirmed live: a real ablation run hit airline=100%/
# retail=100% mean reward at eval_0, only telecom sitting lower -- so
# continuing to spend 2/3 of every eval's wall-clock on two domains
# that already read 100% adds no signal. Default stays the full 3-
# domain yaml (safe default: use this only once you have confirmed the
# other two domains are actually saturated for THIS run).
EVAL_CFG="${SDPO_REACT_TAU2_EVAL_CONFIG:-$REACT_DIR/data/eval_tau2.yaml}"
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

[ "$SDPO_REACT_THINKING" = "true" ] && _THINK=think || _THINK=nothink
if [ "$SDPO_REACT_THINKING" = "true" ]; then
    REMOVE_THINKING_ARG=()
else
    REMOVE_THINKING_ARG=(--sdpo-remove-thinking-from-demonstration)
fi
CFG_TAG="${MODEL_NAME}-tau2-${SDPO_ABLATION_ALGO}-${SDPO_ABLATION_ARM}-${_THINK}"

SDPO_REACT_EXP="${SDPO_REACT_EXP:-sdpo-react-ablation-${CFG_TAG}_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="${SDPO_ABLATION_DUMP_ROOT}/${SDPO_REACT_EXP}"
echo "SDPO_ReAct dump dir: ${DUMP_DIR}"
CKPT_DIR="${SDPO_REACT_CKPT_DIR:-${SDPO_ABLATION_CKPT_ROOT}/sdpo-react-ablation-${CFG_TAG}_ckpt}"

EXPLOG="${SDPO_REACT_EXPLOG:-$REPO_ROOT/examples/SDPO_ReAct/ablation/explog.jsonl}"
GIT_COMMIT="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(cd "$REPO_ROOT" && [ -n "$(git status --porcelain 2>/dev/null)" ] && echo dirty || echo clean)"
python - <<PYLOG || true
import json, time, os
row = {
    "exp": "${SDPO_REACT_EXP}", "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "algo": "${SDPO_ABLATION_ALGO}", "arm": "${SDPO_ABLATION_ARM}", "domain": "tau2",
    "model": "${MODEL_NAME}", "train_data": "${TRAIN_DATA}", "num_rollout": "${SDPO_REACT_NUM_ROLLOUT}",
    "tau2_max_steps": "${SDPO_REACT_TAU2_MAX_STEPS}",
    "dump_dir": "${DUMP_DIR}", "ckpt_dir": "${CKPT_DIR}", "git_commit": "${GIT_COMMIT}", "git_state": "${GIT_DIRTY}",
    "note": os.environ.get("SDPO_REACT_NOTE", ""),
}
with open("${EXPLOG}", "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("explog appended ->", "${EXPLOG}")
print(json.dumps(row, ensure_ascii=False, indent=2))
PYLOG

CKPT_ARGS=(
   --hf-checkpoint "${SDPO_ABLATION_MODEL_ROOT}/${MODEL_NAME}"
   --ref-load "${SDPO_ABLATION_MODEL_ROOT}/${MODEL_NAME}_torch_dist"
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)
if [ "${SDPO_REACT_SAVE_CKPT:-0}" = "1" ]; then
    CKPT_ARGS+=(--save "${CKPT_DIR}" --load "${CKPT_DIR}" --save-interval "${SDPO_REACT_SAVE_INTERVAL:-10}" --override-opt-param-scheduler)
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
   --rollout-max-context-len 81920
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
   --over-sampling-batch-size "${ROLLOUT_BATCH_SIZE}"
)
if [ "$SDPO_ABLATION_ARM" = "a" ] && [ "$SDPO_ABLATION_ALGO" = "grpo" ]; then
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
else
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_sdpo_group_has_prefix)
    ROLLOUT_ARGS+=(--sdpo-dynamic-filter-min-correct "${SDPO_REACT_MIN_CORRECT:-1}")
fi

CUSTOM_GENERATE_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path examples.SDPO_ReAct.tools.tau2.agent_function.run
   --use-session-server
   --tito-model qwen35
   --tito-allowed-append-roles tool user
)

# ============================================================================ #
# ARM -> skill-related RM_ARGS -- see the mathcodesearch sibling script for
# the full mapping table. tau2 has no LLM-judge fallback (grading is tau2's
# own evaluate_simulation() score), same as webshop/alfworld.
# ============================================================================ #
RM_ARGS=(
   --sdpo-answer-tag answer
   "${REMOVE_THINKING_ARG[@]}"
   --sdpo-reframe-multiturn-prefix
   --sdpo-tool-grammar "${TOOL_GRAMMAR}"
   --sdpo-max-prefix-chars 20000
)

if [ "$SDPO_ABLATION_ARM" = "a" ] && [ "$SDPO_ABLATION_ALGO" = "grpo" ]; then
    RM_ARGS=(
       --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_plain_grpo_reward
       --sdpo-grader dapo
       --sdpo-answer-tag answer
    )
else
    RM_ARGS+=(
       --group-rm
       --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
       --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
       --sdpo-grader dapo
    )
    case "$SDPO_ABLATION_ARM" in
        a) : ;;
        b)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source correct --sdpo-skill-max-new-tokens 2048 --sdpo-response-prefix skill)
            ;;
        c)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source incorrect --sdpo-skill-max-new-tokens 2048 \
                      --sdpo-pitfall-summary-backend self --sdpo-env-feedback-max-chars 2000)
            ;;
        d)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source all --sdpo-skill-max-new-tokens 2048 \
                      --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill --sdpo-env-feedback-max-chars 2000)
            ;;
        e)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source all --sdpo-skill-max-new-tokens 2048 \
                      --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill --sdpo-env-feedback-max-chars 2000 \
                      --sdpo-skill-kd --sdpo-skill-kd-coef 0.01 --sdpo-skill-kd-mode both)
            ;;
        f)
            RM_ARGS+=(--sdpo-self-skill --sdpo-skill-source all --sdpo-skill-max-new-tokens 2048 \
                      --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill --sdpo-env-feedback-max-chars 2000 \
                      --sdpo-skill-kd --sdpo-skill-kd-coef 0.01 --sdpo-skill-kd-mode both-blind)
            ;;
    esac
fi

# ============================================================================ #
# ALGORITHM -> GRPO_ARGS. tau2 has no --sdpo-prefer-tool-use-peer (that's a
# native-tool-calling-domain-only escape-hatch guard, not applicable here).
# ============================================================================ #
case "$SDPO_ABLATION_ALGO" in
    grpo)
        if [ "$SDPO_ABLATION_ARM" = "a" ]; then
            GRPO_ARGS=(
               --advantage-estimator grpo
               --entropy-coef 0.00
            )
        else
            RM_ARGS+=(--no-sdpo-pure-distill)
            GRPO_ARGS=(
               --advantage-estimator grpo
               --sdpo-teacher-backend megatron
               --sdpo-ema-teacher
               --sdpo-ema-teacher-rate 0.05
               --sdpo-logprob-mode sampled
               --sdpo-self-teacher
               --entropy-coef 0.00
               --calculate-per-token-loss
            )
        fi
        ;;
    sdpo)
        RM_ARGS+=(--no-sdpo-pure-distill)
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
           --entropy-coef 0.00
           --calculate-per-token-loss
        )
        ;;
    rlsd)
        RM_ARGS+=(--no-sdpo-pure-distill)
        GRPO_ARGS=(
           --advantage-estimator grpo
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
           --entropy-coef 0.00
           --calculate-per-token-loss
        )
        ;;
    *)
        echo "Unknown SDPO_ABLATION_ALGO='${SDPO_ABLATION_ALGO}' (expected: grpo | sdpo | rlsd)" >&2
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
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${SDPO_REACT_TP}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size "${SDPO_REACT_EP_SIZE}"
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size "${SDPO_REACT_LOGPROBS_CHUNK:-4096}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-warmup-iters 10
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-sdpo
   --wandb-group "sdpo-react-ablation-${CFG_TAG}"
   --wandb-key "${WANDB_API_KEY}"
)

# --sglang-tool-call-parser is NOT optional here: tau2 goes through
# agentic_tool_call.generate -> session-server proxy -> sglang's native
# /v1/chat/completions (see tools/tau2/agent_function.py's module docstring),
# NOT multi_turn.generate's own text-parsing loop over /generate. Without
# this flag sglang never parses <function=..> tags out of the raw text, so
# message.tool_calls comes back None every turn and tau2's LLMAgent (via
# litellm) has no text fallback -- confirmed live against a real training
# dump (9B sibling script): 83% of a 257-episode sample had ZERO tool/
# tool_calls messages, model narrating fake tool results in plain text
# instead, episodes running to the turn cap at reward=0. This single
# missing flag is the primary cause of eval success landing far below the
# Qwen tech report's reference numbers.
#
# --sglang-reasoning-parser is DELIBERATELY NOT set here (even with thinking
# on): it makes sglang split reasoning into a separate message.reasoning_
# content field, but tau2's own generate() (tau2/utils/llm_utils.py) only
# reads response_choice.message.content -- there is no reasoning_content
# field on tau2's AssistantMessage at all (confirmed live: not in
# AssistantMessage.model_fields) -- so tau2 silently drops it when relaying
# the assistant turn back through litellm. On the NEXT turn the session-
# server compares the stored message (content + non-empty reasoning_
# content) against what tau2 sends back (content only) via message_
# matches() -- reasoning_content is one of the TEMPLATE_RELEVANT_KEYS
# compared, so this mismatches, the append-only checkpoint detector treats
# it as a brand-new appended message, and role='assistant' is not in
# --tito-allowed-append-roles (tool/user only) -- every turn after the
# first 400s. Confirmed live: ~1740 such 400s in under 10 minutes on one
# run (litellm's own num_retries backoff on each is also a real contributor
# to "training is slow"). Leaving reasoning-parser unset keeps
# <think>...</think> inline in content instead, which survives the tau2
# round-trip unchanged (the TITO jinja templates already know how to
# strip/re-render inline <think> tags).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static "${SDPO_ABLATION_SGLANG_MEM_FRACTION:-0.5}"
   --sglang-ep-size "${SDPO_REACT_EP_SIZE}"
   --sglang-tool-call-parser qwen3_coder
   # Confirmed live (2026-08-09, 8xH200, 9B sibling script): the default 60s
   # flush_cache timeout (RolloutManager.offload -> SGLangEngine.release_
   # memory_occupation -> flush_cache) can fire while a genuinely still-
   # decoding tau2 session (observed 125K+ tokens) is holding the engine,
   # since /flush_cache only returns 200 once all in-flight generation
   # drains. Real long-tail latency in this domain, not memory pressure --
   # raising --max-tokens-per-gpu/--sglang-mem-fraction-static does not fix
   # it. 300s gives slow episodes room to finish first.
   --sglang-flush-cache-timeout "${SDPO_REACT_FLUSH_CACHE_TIMEOUT:-300}"
)

R3_ARGS=()
if [ "${SDPO_REACT_R3:-true}" = "true" ]; then
    R3_ARGS+=(--use-rollout-routing-replay)
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --distributed-timeout-minutes "${SDPO_REACT_DIST_TIMEOUT_MIN:-60}"
   --moe-token-dispatcher-type flex
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
        \"PYTHONPATH\": \"${SDPO_ABLATION_MEGATRON_PATH}/\",
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
   ${R3_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
