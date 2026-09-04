#!/bin/bash
# Code-only GRPO ablation -- Qwen3.5-9B on LiveCodeBench (native multi-turn
# <tool_call> rollout), single 8x H200 node, COLOCATE variant.
#
# Stripped-down sibling of run-qwen3.5-9B-sdpo-react-ablation-mathcodesearch.sh:
#   - Training data: LiveCodeBench code ONLY (no math, no search)
#   - Eval: LCB v6 ONLY (eval_code.yaml)
#   - NO dynamic sampling on EITHER arm (no --dynamic-sampling-filter-path at all)
#   - Two arms only:
#     a  Plain GRPO baseline (no skill, no dynamic filter)
#     e  GRPO + skill-KD(both): on all-correct or all-incorrect traces (which
#        have zero GRPO advantage), the ONLY training signal is the skill-SD
#        loss on the self-predicted skill tokens. No dynamic filter means these
#        traces are NOT dropped -- they remain in the batch and contribute ONLY
#        the skill-KD gradient. On mixed (has-both-correct-and-incorrect) groups
#        the normal GRPO advantage fires on response tokens AND skill-KD fires
#        on skill tokens.
#
# Same model / parallelism / optimizer config as the mathcodesearch sibling.
# No search sidecar needed (code only uses the sandbox).
#
# usage:
#   SDPO_ABLATION_ARM=a bash examples/SDPO_ReAct/ablation/run-qwen3.5-9B-sdpo-react-ablation-codeonly.sh
#   SDPO_ABLATION_ARM=e bash examples/SDPO_ReAct/ablation/run-qwen3.5-9B-sdpo-react-ablation-codeonly.sh
set -exf

# Ensure container binaries take priority over host ~/.local/bin
export PATH="/usr/local/bin:/usr/bin:/usr/sbin:/bin:/sbin:$PATH"

# --- path knobs ---
SDPO_ABLATION_DATA_ROOT="${SDPO_ABLATION_DATA_ROOT:-/root}"
SDPO_ABLATION_MODEL_ROOT="${SDPO_ABLATION_MODEL_ROOT:-/root}"
SDPO_ABLATION_DUMP_ROOT="${SDPO_ABLATION_DUMP_ROOT:-/root/data/sdpo_dumps}"
SDPO_ABLATION_CKPT_ROOT="${SDPO_ABLATION_CKPT_ROOT:-/root/data/sdpo_ckpts}"
SDPO_ABLATION_MEGATRON_PATH="${SDPO_ABLATION_MEGATRON_PATH:-/root/Megatron-LM}"

export PYTHONBUFFERED=16
export SDPO_REACT_TRAIN_MAX_TURNS="${SDPO_REACT_TRAIN_MAX_TURNS:-8}"
export SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-20}"
export SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-8}"
SDPO_ABLATION_ARM="${SDPO_ABLATION_ARM:?Set SDPO_ABLATION_ARM to one of: a e f}"
case "$SDPO_ABLATION_ARM" in
    a|e|f) ;;
    *) echo "Code-only ablation only supports arms a/e/f (got '${SDPO_ABLATION_ARM}')" >&2; exit 1 ;;
esac
SDPO_ABLATION_ALGO=grpo
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

MODEL_NAME="${CODEONLY_MODEL_NAME:-Qwen3.5-9B}"
MODEL_ARG_SH="${CODEONLY_MODEL_ARG_SH:-scripts/models/qwen3.5-9B.sh}"
TOOL_PARSER=qwen3_coder
TOOL_GRAMMAR=qwen3_coder
MODEL_EXTRA_ARGS=(--dist-ckpt-optim-fully-reshardable --distrib-optim-fully-reshardable-mem-efficient)
MAX_TOKENS_PER_GPU="${SDPO_ABLATION_MAX_TOKENS_PER_GPU:-6144}"
if [ "$SDPO_ABLATION_ARM" = "e" ]; then
    MAX_TOKENS_PER_GPU="${SDPO_ABLATION_MAX_TOKENS_PER_GPU:-3072}"
fi
echo "MODEL: ${MODEL_NAME} | tool-parser=${TOOL_PARSER} | tool-grammar=${TOOL_GRAMMAR} | max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"
source "$REPO_ROOT/${MODEL_ARG_SH}"

# --- 0. sandbox sidecar (code_interpreter) ---
bash "$REACT_DIR/tools/run_sandbox.sh"

export SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-true}"
export SDPO_REACT_PROMPT="${SDPO_REACT_PROMPT:-minimal}"
DATA_SUFFIX="$SDPO_REACT_PROMPT"

# --- CODE-ONLY data prep (LiveCodeBench train set, no math/search) ---
CODE_DATA_DIR="${SDPO_ABLATION_DATA_ROOT}/data/code_data"
TRAIN_DATA="${CODE_DATA_DIR}/livecodebench_train.jsonl"
EVAL_CFG="$REACT_DIR/data/eval_code.yaml"
mkdir -p "$CODE_DATA_DIR" "${SDPO_ABLATION_DATA_ROOT}/data/code_data_v6eval"
[ -f "$TRAIN_DATA" ] || \
    (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_code_data \
        --out-dir "$CODE_DATA_DIR" --testtype stdin --difficulty medium,hard --n-train 2000 --n-eval 1)
[ -f "${SDPO_ABLATION_DATA_ROOT}/data/code_data_v6eval/livecodebench_eval.jsonl" ] || \
    (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_code_data \
        --out-dir "${SDPO_ABLATION_DATA_ROOT}/data/code_data_v6eval" --testtype stdin --difficulty easy,medium,hard \
        --min-date 2025-02 --n-train 0 --n-eval 80)
python - <<PYCK
import json
n = sum(1 for _ in open("$TRAIN_DATA"))
print(f"code-only train rows: {n}")
PYCK

# --- 0c. one-time template check ---
python - "$REPO_ROOT" "$MODEL_NAME" "$SDPO_ABLATION_MODEL_ROOT" <<'PYCHECK'
import sys
from transformers import AutoTokenizer
sys.path.insert(0, sys.argv[1])
model_name = sys.argv[2]
model_root = sys.argv[3]
from examples.SDPO_ReAct.tools.tool_specs import tool_specs
tok = AutoTokenizer.from_pretrained(f"{model_root}/{model_name}", trust_remote_code=True)
from examples.SDPO_ReAct.data.build_code_data import CODE_SYSTEM_PROMPT
checks = [
    ("code", [{"role": "system", "content": CODE_SYSTEM_PROMPT}, {"role": "user", "content": "add two ints"}], tool_specs),
]
for name, msgs, tspecs in checks:
    r = tok.apply_chat_template(msgs, tools=tspecs, tokenize=False, add_generation_prompt=True)
    assert "<tools>" in r, f"{name}: native <tools> block missing"
_SENT = "SDPOSENTINEL"
probe = tok.apply_chat_template([{"role":"user","content":_SENT}], tokenize=False, add_generation_prompt=True)
assert _SENT in probe and probe.split(_SENT,1)[1], "gen_suffix empty -> SDPO splice undefined"
print(f"Native template check OK for {model_name} (code-only)")
PYCHECK

[ "$SDPO_REACT_THINKING" = "true" ] && _THINK=think || _THINK=nothink
if [ "$SDPO_REACT_THINKING" = "true" ]; then
    REMOVE_THINKING_ARG=()
else
    REMOVE_THINKING_ARG=(--sdpo-remove-thinking-from-demonstration)
fi
CFG_TAG="${MODEL_NAME}-codeonly-grpo-${SDPO_ABLATION_ARM}-${_THINK}-${DATA_SUFFIX}"

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
    "algo": "grpo", "arm": "${SDPO_ABLATION_ARM}", "domain": "codeonly",
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
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)
# Use raw mode if torch_dist exists, else bridge mode
if [ -f "${SDPO_ABLATION_MODEL_ROOT}/${MODEL_NAME}_torch_dist/latest_checkpointed_iteration.txt" ]; then
    echo "Using RAW mode (torch_dist found)"
    CKPT_ARGS+=(--ref-load "${SDPO_ABLATION_MODEL_ROOT}/${MODEL_NAME}_torch_dist")
else
    echo "Using BRIDGE mode (no torch_dist)"
    CKPT_ARGS+=(--megatron-to-hf-mode bridge)
fi
if [ "${SDPO_REACT_SAVE_CKPT:-0}" = "1" ]; then
    CKPT_ARGS+=(--save "${CKPT_DIR}" --load "${CKPT_DIR}" --save-interval "${SDPO_REACT_SAVE_INTERVAL:-10}" --override-opt-param-scheduler)
fi

# --- ROLLOUT: NO dynamic sampling filter on EITHER arm ---
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
)

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
# ARM -> RM_ARGS + GRPO_ARGS
# ============================================================================ #
case "$SDPO_ABLATION_ARM" in
    a)
        # Plain GRPO, no SDPO, no skill, no dynamic filter.
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
    e)
        # GRPO + skill-KD(both): skill-SD is the ONLY learning signal on
        # all-correct/all-incorrect groups (GRPO advantage = 0 there).
        # No dynamic filter -> those groups stay in the batch, contributing
        # skill-KD gradient only.
        RM_ARGS=(
           --sdpo-answer-tag answer
           "${REMOVE_THINKING_ARG[@]}"
           --sdpo-reframe-multiturn-prefix
           --sdpo-tool-grammar "${TOOL_GRAMMAR}"
           --sdpo-max-prefix-chars 20000
           --group-rm
           --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
           --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
           --sdpo-grader dapo
           --sdpo-self-skill --sdpo-skill-source all --sdpo-skill-max-new-tokens 2048
           --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill --sdpo-env-feedback-max-chars 2000
           --sdpo-skill-kd --sdpo-skill-kd-coef 0.01 --sdpo-skill-kd-mode both
           --no-sdpo-pure-distill
        )
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
        ;;
    f)
        # GRPO + skill-KD(both-blind): same as arm-e but blind-correct skill.
        RM_ARGS=(
           --sdpo-answer-tag answer
           "${REMOVE_THINKING_ARG[@]}"
           --sdpo-reframe-multiturn-prefix
           --sdpo-tool-grammar "${TOOL_GRAMMAR}"
           --sdpo-max-prefix-chars 20000
           --group-rm
           --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
           --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
           --sdpo-grader dapo
           --sdpo-self-skill --sdpo-skill-source all --sdpo-skill-max-new-tokens 2048
           --sdpo-pitfall-summary-backend self --sdpo-response-prefix skill --sdpo-env-feedback-max-chars 2000
           --sdpo-skill-kd --sdpo-skill-kd-coef 0.01 --sdpo-skill-kd-mode both-blind
           --no-sdpo-pure-distill
        )
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
        ;;
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
   --expert-model-parallel-size "${SDPO_REACT_EP:-1}"
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size "${SDPO_REACT_LOGPROBS_CHUNK:-4096}"
   --sequence-parallel
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-warmup-iters "${SDPO_REACT_LR_WARMUP:-10}"
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

if [ -n "${WANDB_API_KEY}" ]; then
    WANDB_ARGS=(
       --use-wandb
       --wandb-project miles-sdpo
       --wandb-group "sdpo-react-ablation-${CFG_TAG}"
       --wandb-key "${WANDB_API_KEY}"
    )
else
    WANDB_ARGS=()
fi

if [ "${SDPO_REACT_EP:-1}" -gt 1 ]; then
    SGLANG_ARGS=(
       --rollout-num-gpus-per-engine "${SDPO_REACT_TRAIN_GPUS}"
       --sglang-mem-fraction-static "${SDPO_ABLATION_SGLANG_MEM_FRACTION:-0.75}"
       --sglang-ep-size "${SDPO_REACT_EP}"
    )
else
    SGLANG_ARGS=(
       --rollout-num-gpus-per-engine 1
       --sglang-mem-fraction-static "${SDPO_ABLATION_SGLANG_MEM_FRACTION:-0.75}"
    )
fi

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

cd "$REPO_ROOT"
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
        \"PYTHONPATH\": \"${REPO_ROOT}:${SDPO_ABLATION_MEGATRON_PATH}/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
        \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
        \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\",
        \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
        \"SDPO_REACT_EVAL_MAX_TURNS\": \"${SDPO_REACT_EVAL_MAX_TURNS}\",
        \"SDPO_REACT_EVAL_N_SAMPLES\": \"${SDPO_REACT_EVAL_N_SAMPLES}\",
        \"SDPO_REACT_EVAL_DIR\": \"${SDPO_ABLATION_DATA_ROOT}/math_eval/native-${DATA_SUFFIX}\"
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
