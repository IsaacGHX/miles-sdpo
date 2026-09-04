#!/bin/bash
# Standalone LiveCodeBench-v6 FUNCTIONAL (leetcode-style) evaluation -- runs one
# or more HF checkpoints through the SAME multi-turn native tool-calling rollout
# training's step-0 eval uses. Direct sibling of eval-amo-bench.sh (same
# eval-only trick, same arg arrays); only the dataset + grader path differ.
#
# WHY this eval set: every LiveCodeBench TRAIN set we ever built was built with
# --testtype stdin, so the functional/leetcode pool (444 problems overall; 63 in
# v6, 20 of them hard) has never been in any training mix -- while 54-77% of the
# stdin v6 eval problems DID leak into the code training pool. This is currently
# our only genuinely held-out LiveCodeBench slice. Default slice: ALL of v6
# functional (test6.jsonl, contests 2025-01..2025-04) -- 63 problems
# (17 easy + 26 medium + 20 hard) x 15 tests each (2 public + 13 private;
# functional problems ship only ~2 public tests, far too few to call a problem
# solved). LCB_FUNC_DIFFICULTY=hard gives the 20-problem hard-only subset, but 20
# problems is a coarse denominator -- one problem is 5pp of pass@1.
#
# Grading: judge.py's FUNCTIONAL harness (testtype="functional" on the test
# cases) -- exec the candidate, call Solution().<func_name>(*args) with the
# JSON-decoded argument lines, structurally compare the RETURN value. Correct =
# all 15 tests pass. TOOL-MANDATORY by default (--sdpo-code-require-tool): the
# graded candidate is the code the model last RAN through code_interpreter, not
# a text fence -- set LCB_REQUIRE_TOOL=0 to grade the response fence instead.
#
# How it works: runs the standard train.py with --num-rollout 1 (a single no-op
# training step) and WITHOUT --skip-eval-before-train, so the step-0 eval fires
# against the loaded model and then exits. Same pattern every ablation script
# uses for its baseline measurement.
#
# Reports (wandb): pass@1..pass@8 on the eval set, per-sample turns, tool call
# count, tool error rate, zero-tool-call fraction, hit-max-turns fraction.
#
# Required env vars:
#   WANDB_API_KEY     for logging to wandb
#
# Optional env vars:
#   MODEL_PATHS       space-separated HF checkpoint dirs to evaluate SEQUENTIALLY
#                     (default: Qwen3.5-4B + Qwen3.5-9B base under MODEL_STORE)
#   MODEL_PATH        single checkpoint dir; overrides MODEL_PATHS when set
#   MODEL_STORE       (default /root/data/home-static/data/hf_models) base-model dir
#   LCB_FUNC_DATA_DIR (default /root/data/code_data_v6func_all) eval jsonl dir
#   LCB_FUNC_DUMP_DIR (default /root/data/sdpo_dumps) trace dump root
#   LCB_FUNC_FILES    (default test6.jsonl) which LiveCodeBench release files
#   LCB_FUNC_DIFFICULTY (default easy,medium,hard = all 63) comma-separated
#   LCB_REQUIRE_TOOL  (default 1) 0 -> grade the ```python fence, tool optional
#   LCB_FUNC_MAX_RESPONSE_LEN (default 16384) PER-TURN generation budget; 32768
#                     removes the 35% truncation seen on functional/hard
#   SDPO_REACT_EVAL_N_SAMPLES (default 8) samples per prompt for pass@k
#   SDPO_REACT_EVAL_MAX_TURNS (default 20) max tool-calling turns per sample
#   SDPO_REACT_TP     (default 2) tensor parallel size
#   SDPO_REACT_TRAIN_GPUS (default 8) total GPUs
#   MEGATRON_PATH     (default /root/Megatron-LM)
#   SGLANG_MEM_FRACTION (default 0.75)
#   LCB_FUNC_WANDB_PROJECT (default miles-sdpo)
#   LCB_FUNC_WANDB_GROUP   (default lcb-v6-functional-eval)
#
# usage (both base models, the default):
#   bash examples/SDPO_ReAct/ablation/eval-lcb-functional.sh
# usage (one checkpoint, e.g. a trained arm):
#   MODEL_PATH=/root/data/hf_uploads/Qwen3.5-9B-...-arm-e \
#     bash examples/SDPO_ReAct/ablation/eval-lcb-functional.sh
set -exf

# Paths are CONTAINER paths: enroot mounts the host $DATA_DIR (/fsx/data/$USER)
# as /root/data, and /fsx itself is not readable from inside the session.
MODEL_STORE="${MODEL_STORE:-/root/data/home-static/data/hf_models}"
if [ -n "${MODEL_PATH:-}" ]; then
    MODEL_PATHS="$MODEL_PATH"
fi
MODEL_PATHS="${MODEL_PATHS:-${MODEL_STORE}/Qwen3.5-4B ${MODEL_STORE}/Qwen3.5-9B}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

LCB_FUNC_DATA_DIR="${LCB_FUNC_DATA_DIR:-/root/data/code_data_v6func_all}"
export LCB_FUNC_DATA_DIR
LCB_FUNC_FILES="${LCB_FUNC_FILES:-test6.jsonl}"
# ALL of v6 functional (63 = 17 easy + 26 medium + 20 hard), not hard-only: 20
# problems x 8 samples is too small a denominator to separate arms (one problem
# = 5pp of pass@1), and the whole functional pool is equally held out, so
# restricting to hard throws away 2/3 of the signal for no contamination gain.
LCB_FUNC_DIFFICULTY="${LCB_FUNC_DIFFICULTY:-easy,medium,hard}"
# PER-TURN response budget, shared by the eval config (eval_lcb_functional.yaml
# reads this same env var) and the no-op train rollout. 16384 truncates 35% of
# Qwen3.5-9B base samples inside their first thinking block -> zero tool calls
# -> auto-0 under tool-mandatory grading; 32768 still fits the 81920 context.
LCB_FUNC_MAX_RESPONSE_LEN="${LCB_FUNC_MAX_RESPONSE_LEN:-16384}"
export LCB_FUNC_MAX_RESPONSE_LEN

export PYTHONBUFFERED=16
export SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-8}"
export SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-20}"
# Train max turns doesn't matter (no real training), set to 1 so the single
# no-op rollout is cheap.
export SDPO_REACT_TRAIN_MAX_TURNS=1

SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-8}"
N_SAMPLES_PER_PROMPT=8
SDPO_REACT_TP="${SDPO_REACT_TP:-2}"
DP_SIZE=$((SDPO_REACT_TRAIN_GPUS / SDPO_REACT_TP))
ROLLOUT_BATCH_SIZE=$((DP_SIZE * 4))
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
echo "BATCH: train_gpus=${SDPO_REACT_TRAIN_GPUS} tp=${SDPO_REACT_TP} dp=${DP_SIZE} rollout_batch=${ROLLOUT_BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REACT_DIR="$REPO_ROOT/examples/SDPO_ReAct"

TOOL_PARSER=qwen3_coder
TOOL_GRAMMAR=qwen3_coder

# --- 0. sandbox sidecar (code_interpreter AND the judge's harness) ---
bash "$REACT_DIR/tools/run_sandbox.sh"

# --- 0a. data prep: build the v6 functional eval jsonl (all 63 by default) ---
EVAL_JSONL="${LCB_FUNC_DATA_DIR}/livecodebench_eval.jsonl"
mkdir -p "$LCB_FUNC_DATA_DIR"
[ -f "$EVAL_JSONL" ] || \
    (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_code_data \
        --out-dir "$LCB_FUNC_DATA_DIR" \
        --jsonl-files ${LCB_FUNC_FILES} \
        --testtype functional \
        --difficulty "${LCB_FUNC_DIFFICULTY}" \
        --include-private-tests --max-test-chars 20000 --max-tests 15 \
        --n-train 0 --n-eval 200)
echo "LCB functional eval rows: $(wc -l < "$EVAL_JSONL")"

EVAL_CFG="$REACT_DIR/data/eval_lcb_functional.yaml"

# Use thinking by default (Qwen3.5 strong reasoning mode).
export SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-true}"
export SDPO_REACT_PROMPT="${SDPO_REACT_PROMPT:-minimal}"

for MODEL_PATH in ${MODEL_PATHS}; do
MODEL_NAME="$(basename "$MODEL_PATH")"
echo "============================================================"
echo "LCB v6 functional/${LCB_FUNC_DIFFICULTY} eval: ${MODEL_NAME}"
echo "============================================================"

# --- Detect model arch for model args ---
MODEL_ARG_SH=""
if echo "$MODEL_NAME" | grep -qi "qwen3.5-9B"; then
    MODEL_ARG_SH=scripts/models/qwen3.5-9B.sh
elif echo "$MODEL_NAME" | grep -qi "qwen3.5-4B"; then
    MODEL_ARG_SH=scripts/models/qwen3.5-4B.sh
elif echo "$MODEL_NAME" | grep -qi "qwen3.5-2B"; then
    MODEL_ARG_SH=scripts/models/qwen3.5-2B.sh
elif echo "$MODEL_NAME" | grep -qi "qwen3.5-35B\|Qwen3.5-35B-A3B"; then
    MODEL_ARG_SH=scripts/models/qwen3.5-35B-A3B.sh
elif echo "$MODEL_NAME" | grep -qi "qwen3.5-27B\|Qwen3.6-27B"; then
    MODEL_ARG_SH=scripts/models/qwen3.5-27B.sh
elif echo "$MODEL_NAME" | grep -qi "qwen3-4B"; then
    MODEL_ARG_SH=scripts/models/qwen3-4B.sh
fi

if [ -n "$MODEL_ARG_SH" ] && [ -f "$REPO_ROOT/$MODEL_ARG_SH" ]; then
    echo "MODEL: ${MODEL_NAME} (sourcing ${MODEL_ARG_SH})"
    source "$REPO_ROOT/${MODEL_ARG_SH}"
else
    echo "ERROR: Could not find model args script for '${MODEL_NAME}'." >&2
    echo "Set MODEL_ARG_SH env var to point to the correct scripts/models/*.sh" >&2
    exit 1
fi

# --- 0b. one-time template check (same as the ablation scripts') ---
python - "$REPO_ROOT" "$MODEL_PATH" <<'PYCHECK'
import sys
from transformers import AutoTokenizer
sys.path.insert(0, sys.argv[1])
from examples.SDPO_ReAct.data.build_code_data import FUNCTIONAL_SYSTEM_PROMPT
from examples.SDPO_ReAct.tools.registry import all_tool_specs
tok = AutoTokenizer.from_pretrained(sys.argv[2], trust_remote_code=True)
msgs = [{"role": "system", "content": FUNCTIONAL_SYSTEM_PROMPT},
        {"role": "user", "content": "complete class Solution"}]
r = tok.apply_chat_template(msgs, tools=all_tool_specs, tokenize=False, add_generation_prompt=True)
assert "<tools>" in r, "native <tools> block missing"
print("Native template check OK for", sys.argv[2])
PYCHECK

DUMP_DIR="${LCB_FUNC_DUMP_DIR:-/root/data/sdpo_dumps}/lcb-v6-functional-eval-${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
echo "LCB functional eval dump dir: ${DUMP_DIR}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)

# Megatron loads the ACTOR from a torch_dist checkpoint, not from the HF dir:
# with no --load/--ref-load, arguments.py sets args.load = args.ref_load = None
# and setup_model_and_optimizer asserts (miles/backends/megatron_utils/model.py
# :128). Every training/ablation script passes --ref-load <model>_torch_dist;
# do the same, resolving the sibling _torch_dist dir of the HF checkpoint, then
# the per-node asset copy the enroot launcher's prep step writes.
REF_LOAD=""
for cand in "${MODEL_PATH}_torch_dist" "/root/assets/${MODEL_NAME}_torch_dist" "/root/${MODEL_NAME}_torch_dist"; do
    if [ -f "${cand}/latest_checkpointed_iteration.txt" ]; then
        REF_LOAD="$cand"
        break
    fi
done
if [ -z "$REF_LOAD" ]; then
    echo "ERROR: no torch_dist checkpoint for ${MODEL_NAME}. Convert it first:" >&2
    echo "  bash examples/SDPO_ReAct/ablation/convert-all-models-to-torch-dist.sh" >&2
    exit 1
fi
echo "REF_LOAD: ${REF_LOAD}"
CKPT_ARGS+=(--ref-load "${REF_LOAD}")

# We need a dummy train data file (the training loop loads it even for eval-
# only runs). Use the eval data itself -- the single no-op rollout step will
# sample from it but never actually train on it (num-rollout=1 exits after
# step-0 eval).
ROLLOUT_ARGS=(
   --prompt-data "${EVAL_JSONL}"
   --input-key prompt
   --label-key label
   --tool-key tools
   --apply-chat-template
   --apply-chat-template-kwargs "{\"enable_thinking\":${SDPO_REACT_THINKING}}"
   --num-rollout 1
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${LCB_FUNC_MAX_RESPONSE_LEN}"
   --rollout-max-context-len 81920
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
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

# Plain GRPO reward for the no-op training step; it domain-routes code samples
# to the test-case judge, which is exactly what the eval needs too (no
# --eval-custom-rm-path -> eval reuses --custom-rm-path). No LLM judge: code
# correctness here is fully deterministic.
RM_ARGS=(
   --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_plain_grpo_reward
   --sdpo-grader dapo
   --sdpo-answer-tag answer
)
if [ "${LCB_REQUIRE_TOOL:-1}" = "0" ]; then
    RM_ARGS+=(--no-sdpo-code-require-tool)
fi

EVAL_ARGS=(
   # eval-interval 2, NOT 1, with --num-rollout 1: train.py evals once before the
   # loop (rollout_id == start_rollout_id and not --skip-eval-before-train) and
   # then AGAIN at the end of the step if should_run_periodic_action() fires.
   # With interval 1 that second eval runs on weights that just took a GRPO step
   # on --prompt-data -- which for an eval-only run IS the eval set. Both evals
   # log to the same wandb key and the same rollout_data/eval_0.jsonl (the second
   # overwrites the first), so the dumped/tabulated number was the post-leak one:
   # measured 0.3875 -> 0.5188 pass@1 for Qwen3.5-9B on v6 functional/hard, a
   # +13pp phantom gain. interval 2 makes step 1 miss the periodic check, so
   # exactly one eval runs, on the loaded weights.
   --eval-interval 2
   --eval-config "$EVAL_CFG"
   --eval-tool-key tools
   --n-samples-per-eval-prompt "${SDPO_REACT_EVAL_N_SAMPLES}"
   --log-passrate
)
# Do NOT add --skip-eval-before-train: we WANT the step-0 eval to fire.

PERF_ARGS=(
   --tensor-model-parallel-size "${SDPO_REACT_TP}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-6144}"
   --sequence-parallel
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-warmup-iters 0
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${LCB_FUNC_WANDB_PROJECT:-miles-sdpo}"
   --wandb-group "${LCB_FUNC_WANDB_GROUP:-lcb-v6-functional-eval}"
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.75}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

if [ "$SDPO_REACT_TRAIN_GPUS" -lt 8 ]; then
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((SDPO_REACT_TRAIN_GPUS-1)))"
    echo "TRAIN GPUs: ${SDPO_REACT_TRAIN_GPUS} (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
fi

ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
sleep 2

cd "$REPO_ROOT"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "${SDPO_REACT_TRAIN_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# `ray start` returns as soon as the GCS is up, but the dashboard (which serves
# the job-submission API on 8265) is a separate process that can take another
# ~20s to bind -- submitting immediately dies with ConnectionRefused. Wait for
# the API to actually answer.
for i in $(seq 1 60); do
    curl -sf -o /dev/null "http://127.0.0.1:8265/api/version" && break
    if [ "$i" = 60 ]; then
        echo "ERROR: ray dashboard API never came up on 127.0.0.1:8265" >&2
        exit 1
    fi
    sleep 2
done

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"${REPO_ROOT}:${MEGATRON_PATH}/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"NCCL_NVLS_ENABLE\": \"0\",
        \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
        \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\",
        \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
        \"LCB_FUNC_DATA_DIR\": \"${LCB_FUNC_DATA_DIR}\",
        \"LCB_FUNC_MAX_RESPONSE_LEN\": \"${LCB_FUNC_MAX_RESPONSE_LEN}\",
        \"SDPO_REACT_EVAL_N_SAMPLES\": \"${SDPO_REACT_EVAL_N_SAMPLES}\",
        \"SDPO_REACT_EVAL_MAX_TURNS\": \"${SDPO_REACT_EVAL_MAX_TURNS}\",
        \"SDPO_REACT_TRAIN_MAX_TURNS\": \"${SDPO_REACT_TRAIN_MAX_TURNS}\",
        \"SDPO_REACT_PROMPT\": \"${SDPO_REACT_PROMPT}\"
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
pkill -9 -f 'ray::' 2>/dev/null || true
sleep 3
echo "DONE ${MODEL_NAME} -- traces: ${DUMP_DIR}"
done

echo "LCB v6 functional/${LCB_FUNC_DIFFICULTY} eval complete for: ${MODEL_PATHS}"
echo "Check wandb group '${LCB_FUNC_WANDB_GROUP:-lcb-v6-functional-eval}' (eval/livecodebench_v6_functional*)."
