#!/bin/bash
# SDPO/RLSD/GRPO ablation matrix -- Qwen3.5-9B on Math+Code+Search (native
# multi-turn <tool_call> rollout, SDPO_REACT_DOMAIN=multitask), single 8x
# A100 (80GB) node, COLOCATE variant.
#
# Sibling of ../run-qwen3-4B-sdpo-react-native.sh: SAME data-prep/tool/eval
# machinery, but with TWO orthogonal ablation axes instead of one:
#   SDPO_ABLATION_ALGO  grpo | sdpo | rlsd   -- how the teacher-vs-student
#                        divergence (if any) is consumed
#   SDPO_ABLATION_ARM   a | b | c | d | e | f -- which skill-prefix/skill-KD
#                        machinery is layered on top (see table below)
# GRPO only supports arms a/e/f (skill-prefix-only arms b/c/d have no
# GRPO-relevant training signal without a distillation term to differ from
# the baseline). SDPO/RLSD support all 6 arms.
#
# Arm -> flags (verified against examples/SDPO_ReAct/run-qwen3-4B-sdpo-react-
# native.sh's arm 1.1/2/3/4/5/5.1 case block and doc/DESIGN_self_skill.md):
#   a  Baseline               (no self-skill; --sdpo-response-prefix defaults to trace)
#   b  +correct skill prefix  --sdpo-self-skill --sdpo-skill-source correct --sdpo-response-prefix skill
#   c  +pitfall skill prefix  --sdpo-self-skill --sdpo-skill-source incorrect --sdpo-pitfall-summary-backend self
#                              (NO --sdpo-response-prefix skill -- a correct peer never has
#                              a skill under skill-source=incorrect, see DESIGN_self_skill.md)
#   d  +all skill prefix      --sdpo-self-skill --sdpo-skill-source all --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill
#   e  +skill-sd both (self-success + pitfall-condense)  = d + --sdpo-skill-kd --sdpo-skill-kd-mode both
#   f  +skill-sd both-blind (blind-correct + pitfall-condense)  = d + --sdpo-skill-kd --sdpo-skill-kd-mode both-blind
#
# Algorithm -> flags (mutually exclusive teacher-vs-student divergence
# mechanisms; asserted mutually exclusive in miles/utils/arguments.py):
#   grpo  no --sdpo-rlsd, no --sdpo-kd-loss, no --use-opd. Task reward drives
#         the GRPO advantage directly. Arms e/f still need
#         --sdpo-teacher-backend megatron --sdpo-logprob-mode sampled
#         --calculate-per-token-loss (confirmed: self-skill/skill-KD's own
#         teacher-forward dispatch in actor.py::_compute_sdpo_teacher_log_probs
#         only calls _append_sdpo_skill_samples from the "sampled" branch (RLSD's
#         branch) or the "topk"+--sdpo-kd-loss branch (SDPO's) -- the DEFAULT
#         "topk" branch WITHOUT --sdpo-kd-loss falls into the legacy advantage-
#         hook path, which never generates skill samples at all. So GRPO+e/f
#         must explicitly set --sdpo-logprob-mode sampled even though
#         --sdpo-rlsd itself stays OFF -- this is what makes skill-KD's own
#         teacher forward run, orthogonally from response-token RLSD reweighting).
#   sdpo  --sdpo-kd-loss --sdpo-divergence jsd --sdpo-logprob-mode topk
#         --opd-log-prob-top-k 100 --sdpo-is-clip 2.0 --sdpo-kd-coef 1.0
#         --sdpo-kd-max-tokens 8192 (additive distribution KD loss on response
#         tokens -- exact flags from run-qwen2.5-7B-sdpo-sci-colocate.sh's arm 1.2)
#   rlsd  --sdpo-rlsd --sdpo-rlsd-clip-eps 0.2 --sdpo-rlsd-lambda-init 1.0
#         --sdpo-rlsd-lambda-warmup-steps 0 --use-tis --sdpo-logprob-mode
#         sampled (multiplicative advantage reweighting -- exact flags from
#         the native script's own arm 1.1-5.1)
# All three (when not plain GRPO-baseline arm a) share
# --sdpo-teacher-backend megatron --sdpo-ema-teacher --sdpo-ema-teacher-rate
# 0.05 --sdpo-self-teacher --calculate-per-token-loss --entropy-coef 0.00.
# RLSD additionally sets --sdpo-prefer-tool-use-peer (agentic-only, prevents
# tool-call collapse via a bad peer fallback -- native script's own rationale).
#
# A100-80G retune (NOT copied verbatim from the H200-141GB native script,
# which explicitly sizes for 141GB): TP=2 (new derivation, no existing
# precedent at this exact size -- 9B's ~18GB bf16 weights alone are
# manageable at TP=1, but activations+optimizer+KV under colocate push total
# footprint past comfortable 80G headroom; TP=2 + --sequence-parallel buys
# back margin the same way the 27B H200 script's TP=4 does relative to TP=1).
# MAX_TOKENS_PER_GPU default 6144, arms e/f (heaviest -- extra skill forward/
# backward) drop further to 3072, matching the SAME per-arm memory-tuning
# pattern run-qwen2.5-7B-sdpo-sci-colocate.sh already established for its own
# heaviest arm. --sglang-mem-fraction-static default 0.6 (vs H200's 0.75).
# --optimizer-cpu-offload is effectively REQUIRED (not optional) at this
# VRAM budget. THESE ARE STARTING POINTS, not validated against real A100
# hardware (none available in this session, confirmed via nvidia-smi showing
# H200s only) -- retune further on the first real OOM, same iterative
# process every existing script's own memory-tuning comments describe.
#
# Path parameterization (every local save/load path overridable from
# outside, declared here at the top instead of hardcoded deep in the body):
#   SDPO_ABLATION_DATA_ROOT      (default /root)              math/code/search data + eval dirs
#   SDPO_ABLATION_MODEL_ROOT     (default /root)               HF checkpoint + torch_dist dirs
#   SDPO_ABLATION_DUMP_ROOT      (default /root/data/sdpo_dumps)
#   SDPO_ABLATION_CKPT_ROOT      (default /root/data/sdpo_ckpts)
#   SDPO_ABLATION_MEGATRON_PATH  (default /root/Megatron-LM)
#
# Other env overrides:
#   SDPO_ABLATION_ALGO           (required) grpo | sdpo | rlsd
#   SDPO_ABLATION_ARM            (required) a | b | c | d | e | f
#   SDPO_REACT_TRAIN_MAX_TURNS   (default 8)
#   SDPO_REACT_EVAL_MAX_TURNS    (default 20)
#   SDPO_REACT_EVAL_N_SAMPLES    (default 8)
#   SDPO_REACT_NUM_ROLLOUT       (default 51 -- per spec, agentic domains all use 51)
#   SDPO_ABLATION_MAX_TOKENS_PER_GPU  (default 6144, 3072 for arms e/f)
#   SDPO_ABLATION_SGLANG_MEM_FRACTION (default 0.6)
#
# usage:
#   SDPO_ABLATION_ALGO=rlsd SDPO_ABLATION_ARM=d \
#     bash examples/SDPO_ReAct/ablation/run-qwen3.5-9B-sdpo-react-ablation-mathcodesearch.sh
set -exf

# --- path knobs (override from outside; defaults match every existing SDPO_ReAct script) ---
SDPO_ABLATION_DATA_ROOT="${SDPO_ABLATION_DATA_ROOT:-/root}"
SDPO_ABLATION_MODEL_ROOT="${SDPO_ABLATION_MODEL_ROOT:-/root}"
SDPO_ABLATION_DUMP_ROOT="${SDPO_ABLATION_DUMP_ROOT:-/root/data/sdpo_dumps}"
SDPO_ABLATION_CKPT_ROOT="${SDPO_ABLATION_CKPT_ROOT:-/root/data/sdpo_ckpts}"
SDPO_ABLATION_MEGATRON_PATH="${SDPO_ABLATION_MEGATRON_PATH:-/root/Megatron-LM}"

export PYTHONBUFFERED=16
export SDPO_REACT_TRAIN_MAX_TURNS="${SDPO_REACT_TRAIN_MAX_TURNS:-8}"
export SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-20}"
export SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-8}"
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

MODEL_NAME=Qwen3.5-9B
MODEL_ARG_SH=scripts/models/qwen3.5-9B.sh
TOOL_PARSER=qwen3_coder
TOOL_GRAMMAR=qwen3_coder
MODEL_EXTRA_ARGS=(--dist-ckpt-optim-fully-reshardable)
MAX_TOKENS_PER_GPU="${SDPO_ABLATION_MAX_TOKENS_PER_GPU:-6144}"
if [ "$SDPO_ABLATION_ARM" = "e" ] || [ "$SDPO_ABLATION_ARM" = "f" ]; then
    MAX_TOKENS_PER_GPU="${SDPO_ABLATION_MAX_TOKENS_PER_GPU:-3072}"
fi
echo "MODEL: ${MODEL_NAME} | tool-parser=${TOOL_PARSER} | tool-grammar=${TOOL_GRAMMAR} | max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"
source "$REPO_ROOT/${MODEL_ARG_SH}"

# --- 0. sandbox sidecar (idempotent, ONE container / ONE port for the whole job) ---
bash "$REACT_DIR/tools/run_sandbox.sh"

SDPO_REACT_DOMAIN=multitask
echo "SDPO_REACT_DOMAIN: ${SDPO_REACT_DOMAIN} (fixed -- this script is math+code+search only)"

# --- 0a. search sidecar (search/open/find -> the wiki-18 retriever) ---
bash "$REACT_DIR/tools/search/run_search_sidecar.sh"

export SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-true}"
export SDPO_REACT_PROMPT="${SDPO_REACT_PROMPT:-minimal}"
DATA_SUFFIX="$SDPO_REACT_PROMPT"

# --- MULTITASK data prep: shuffled math + code + search (identical to the
# native script's own multitask branch, paths parameterized) ---
MT_DIR="${SDPO_ABLATION_DATA_ROOT}/data/multitask"
TRAIN_DATA="$MT_DIR/train.jsonl"
EVAL_CFG="$REACT_DIR/data/eval_multitask.yaml"
mkdir -p "$MT_DIR" "${SDPO_ABLATION_DATA_ROOT}/dapo-math-17k" "${SDPO_ABLATION_DATA_ROOT}/data/code_data" \
    "${SDPO_ABLATION_DATA_ROOT}/data/code_data_v6eval" "${SDPO_ABLATION_DATA_ROOT}/math_eval/native-${DATA_SUFFIX}"
export SDPO_REACT_EVAL_DIR="${SDPO_ABLATION_DATA_ROOT}/math_eval/native-${DATA_SUFFIX}"
[ -f "${SDPO_ABLATION_DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl" ] || \
    hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir "${SDPO_ABLATION_DATA_ROOT}/dapo-math-17k"
[ -f "${SDPO_ABLATION_DATA_ROOT}/dapo-math-17k/dapo-math-17k-native-${DATA_SUFFIX}.jsonl" ] || \
    (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.native_prompt \
        --in "${SDPO_ABLATION_DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl" \
        --out "${SDPO_ABLATION_DATA_ROOT}/dapo-math-17k/dapo-math-17k-native-${DATA_SUFFIX}.jsonl")
[ -f "${SDPO_ABLATION_DATA_ROOT}/data/code_data/livecodebench_train.jsonl" ] || \
    (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_code_data \
        --out-dir "${SDPO_ABLATION_DATA_ROOT}/data/code_data" --testtype stdin --difficulty medium,hard --n-train 2000 --n-eval 1)
[ -f "${SDPO_ABLATION_DATA_ROOT}/data/code_data_v6eval/livecodebench_eval.jsonl" ] || \
    (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_code_data \
        --out-dir "${SDPO_ABLATION_DATA_ROOT}/data/code_data_v6eval" --testtype stdin --difficulty easy,medium,hard \
        --min-date 2025-02 --n-train 0 --n-eval 80)
[ -f "${SDPO_ABLATION_DATA_ROOT}/math_eval/native-${DATA_SUFFIX}/aime25_native.jsonl" ] || \
    (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_native_eval --out-dir "${SDPO_ABLATION_DATA_ROOT}/math_eval/native-${DATA_SUFFIX}")
SEARCH_TRAIN="${SDPO_ABLATION_DATA_ROOT}/data/search_data/search_train_passk.jsonl"
[ -f "$SEARCH_TRAIN" ] || SEARCH_TRAIN="${SDPO_ABLATION_DATA_ROOT}/data/search_data/search_train.jsonl"
[ -f "$SEARCH_TRAIN" ] || \
    (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_search_data \
        --out-dir "${SDPO_ABLATION_DATA_ROOT}/data/search_data" --n-train-per 1500 --n-val-per 100)
MT_SOURCES=(--source "math:${SDPO_ABLATION_DATA_ROOT}/dapo-math-17k/dapo-math-17k-native-${DATA_SUFFIX}.jsonl"
            --source "code:${SDPO_ABLATION_DATA_ROOT}/data/code_data/livecodebench_train.jsonl"
            --source "search:$SEARCH_TRAIN")
[ -f "$TRAIN_DATA" ] || \
    (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_multitask_data \
        --out "$TRAIN_DATA" "${MT_SOURCES[@]}" --per-domain "${SDPO_REACT_MT_PER_DOMAIN:-400}")
python - <<PYCK
import json
from collections import Counter
c=Counter(json.loads(l)["metadata"]["domain"] for l in open("$TRAIN_DATA"))
print(f"multitask train rows OK: {dict(c)}")
PYCK

# --- 0c. one-time template check ---
python - "$REPO_ROOT" "$SDPO_REACT_DOMAIN" "$MODEL_NAME" "$SDPO_ABLATION_MODEL_ROOT" <<'PYCHECK'
import sys
from transformers import AutoTokenizer
sys.path.insert(0, sys.argv[1])
domain = sys.argv[2]
model_name = sys.argv[3]
model_root = sys.argv[4]
from examples.SDPO_ReAct.tools.tool_specs import tool_specs
from examples.SDPO_ReAct.tools.search.spec import search_specs
tok = AutoTokenizer.from_pretrained(f"{model_root}/{model_name}", trust_remote_code=True)
from examples.SDPO_ReAct.data.build_code_data import CODE_SYSTEM_PROMPT
from examples.SDPO_ReAct.data.build_search_data import SEARCH_SYSTEM_PROMPT
from examples.SDPO_ReAct.native_prompt import build_native_messages
checks = [
    ("code", [{"role": "system", "content": CODE_SYSTEM_PROMPT}, {"role": "user", "content": "add two ints"}], tool_specs),
    ("math", build_native_messages("2+2?"), tool_specs),
    ("search", [{"role": "system", "content": SEARCH_SYSTEM_PROMPT}, {"role": "user", "content": "who directed X?"}], search_specs),
]
for name, msgs, tspecs in checks:
    r = tok.apply_chat_template(msgs, tools=tspecs, tokenize=False, add_generation_prompt=True)
    assert "<tools>" in r, f"{name}: native <tools> block missing"
_SENT = "SDPOSENTINEL"
probe = tok.apply_chat_template([{"role":"user","content":_SENT}], tokenize=False, add_generation_prompt=True)
assert _SENT in probe and probe.split(_SENT,1)[1], "gen_suffix empty -> SDPO splice undefined"
print(f"Native template check OK for {model_name} (domain={domain})")
PYCHECK

[ "$SDPO_REACT_THINKING" = "true" ] && _THINK=think || _THINK=nothink
if [ "$SDPO_REACT_THINKING" = "true" ]; then
    REMOVE_THINKING_ARG=()
else
    REMOVE_THINKING_ARG=(--sdpo-remove-thinking-from-demonstration)
fi
CFG_TAG="${MODEL_NAME}-mathcodesearch-${SDPO_ABLATION_ALGO}-${SDPO_ABLATION_ARM}-${_THINK}-${DATA_SUFFIX}"

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
    "algo": "${SDPO_ABLATION_ALGO}", "arm": "${SDPO_ABLATION_ARM}", "domain": "mathcodesearch",
    "model": "${MODEL_NAME}", "train_data": "${TRAIN_DATA}", "num_rollout": "${SDPO_REACT_NUM_ROLLOUT}",
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
   --tool-key tools
   --apply-chat-template
   --apply-chat-template-kwargs "{\"enable_thinking\":${SDPO_REACT_THINKING}}"
   --num-rollout "${SDPO_REACT_NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len 8192
   --rollout-max-context-len 81920
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
   --over-sampling-batch-size "${ROLLOUT_BATCH_SIZE}"
   # multitask -> preserve balanced per-batch domain mix (no --rollout-shuffle)
)
if [ "$SDPO_ABLATION_ARM" = "a" ] && [ "$SDPO_ABLATION_ALGO" = "grpo" ]; then
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
else
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_sdpo_group_has_prefix)
    ROLLOUT_ARGS+=(--sdpo-dynamic-filter-min-correct "${SDPO_REACT_MIN_CORRECT:-1}")
fi

TOOL_SPECS_PATH="examples.SDPO_ReAct.tools.registry.all_tool_specs"
EXECUTE_TOOL_PATH="examples.SDPO_ReAct.tools.registry.execute_tool"
ROLLOUT_ARGS+=(--tool-specs-resolver-path "$TOOL_SPECS_PATH")
CUSTOM_GENERATE_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate
   --generate-tool-specs-path "$TOOL_SPECS_PATH"
   --generate-execute-tool-function-path "$EXECUTE_TOOL_PATH"
   --generate-tool-call-parser "$TOOL_PARSER"
   --generate-max-turns "${SDPO_REACT_TRAIN_MAX_TURNS}"
)

# ============================================================================ #
# ARM -> skill-related RM_ARGS (shared across all 3 algorithms)
# ============================================================================ #
RM_ARGS=(
   --sdpo-answer-tag answer
   "${REMOVE_THINKING_ARG[@]}"
   --sdpo-reframe-multiturn-prefix
   --sdpo-tool-grammar "${TOOL_GRAMMAR}"
   --sdpo-max-prefix-chars 20000
   --sdpo-search-judge-fallback
   --sdpo-judge-base-url "${OPENAI_API_URL:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
   --sdpo-judge-model "${SDPO_REACT_JUDGE_MODEL:-gpt-5.6-luna}"
   --sdpo-judge-api-key-env SFT_GATEWAY_KEY
)

if [ "$SDPO_ABLATION_ARM" = "a" ] && [ "$SDPO_ABLATION_ALGO" = "grpo" ]; then
    # Plain GRPO, no SDPO machinery at all -- single-sample domain-routed reward.
    RM_ARGS=(
       --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_plain_grpo_reward
       --sdpo-grader dapo
       --sdpo-answer-tag answer
       --sdpo-search-judge-fallback
       --sdpo-judge-base-url "${OPENAI_API_URL:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
       --sdpo-judge-model "${SDPO_REACT_JUDGE_MODEL:-gpt-5.6-luna}"
       --sdpo-judge-api-key-env SFT_GATEWAY_KEY
    )
else
    RM_ARGS+=(
       --group-rm
       --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
       --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
       --sdpo-grader dapo
    )
    case "$SDPO_ABLATION_ARM" in
        a) : ;;  # baseline -- no skill flags, --sdpo-response-prefix defaults to trace
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
# ALGORITHM -> GRPO_ARGS (how the divergence, if any, is consumed)
# ============================================================================ #
case "$SDPO_ABLATION_ALGO" in
    grpo)
        if [ "$SDPO_ABLATION_ARM" = "a" ]; then
            GRPO_ARGS=(
               --advantage-estimator grpo
               --entropy-coef 0.00
            )
        else
            # arms e/f: real GRPO advantage on the response + orthogonal skill-KD
            # loss on the skill tokens. --sdpo-logprob-mode sampled is REQUIRED
            # (not RLSD-specific) -- see the module docstring's actor.py finding.
            RM_ARGS+=(--no-sdpo-pure-distill)
            GRPO_ARGS=(
               --advantage-estimator grpo
               --sdpo-teacher-backend megatron
               --sdpo-ema-teacher
               --sdpo-ema-teacher-rate 0.05
               --sdpo-logprob-mode sampled
               --sdpo-self-teacher
               --sdpo-prefer-tool-use-peer
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
           --sdpo-prefer-tool-use-peer
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
           --sdpo-prefer-tool-use-peer
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
   --eval-tool-key tools
   --n-samples-per-eval-prompt "${SDPO_REACT_EVAL_N_SAMPLES}"
   --log-passrate
)
if [ "${SDPO_REACT_SKIP_EVAL0:-0}" = "1" ]; then
    EVAL_ARGS+=(--skip-eval-before-train)
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
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size "${SDPO_REACT_LOGPROBS_CHUNK:-4096}"
   # TP=2 (see module docstring's A100-80G retune section) needs sequence-
   # parallel to shard the activations/LayerNorm too.
   --sequence-parallel
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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SDPO_ABLATION_SGLANG_MEM_FRACTION:-0.6}"
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
        \"SDPO_REACT_EVAL_MAX_TURNS\": \"${SDPO_REACT_EVAL_MAX_TURNS}\",
        \"SDPO_REACT_EVAL_N_SAMPLES\": \"${SDPO_REACT_EVAL_N_SAMPLES}\",
        \"SDPO_REACT_EVAL_DIR\": \"${SDPO_REACT_EVAL_DIR}\"
     }
   }" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${SDPO_REACT_TRAIN_GPUS}" \
   --rollout-num-gpus "${SDPO_REACT_TRAIN_GPUS}" \
   --colocate \
   --update-weights-interval 1 \
   ${MODEL_ARGS[@]} \
   ${MODEL_EXTRA_ARGS[@]} \
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
