#!/bin/bash
# Standalone OJBench (NOI + ICPC competition programming) evaluation -- runs one
# or more HF checkpoints through the SAME multi-turn native tool-calling rollout
# training's step-0 eval uses. Direct sibling of eval-lcb-functional.sh and
# eval-amo-bench.sh (same eval-only trick, same arg arrays); only the dataset
# differs.
#
# DEFAULT SLICE: medium only (77 problems: 52 NOI + 25 ICPC). Why not all 232:
#   - HARD is dead weight. Across 6 checkpoints, the 86 NOI-hard problems x 8
#     samples yielded 0-3 solves each -- 54% of the eval compute buying no
#     signal, while diluting every average toward zero.
#   - EASY saturates and is dominated by the tool-protocol effect rather than
#     coding (base 25% vs trained arms 38% on the same problems).
#   Medium is where the arms actually separate.
# The full set is still one env var away (OJBENCH_DATA_DIR + a build_ojbench run
# with --difficulty easy,medium,hard).
#
# Grading: judge.py's STDIN harness (whole-program candidates, patched stdin,
# whitespace-normalized stdout compare), all-or-nothing over up to 10 test cases
# per problem. TOOL-MANDATORY by default (--sdpo-code-require-tool), matching
# TRAINING: the graded candidate is the code the model last RAN through
# code_interpreter, not a text fence. Set OJBENCH_REQUIRE_TOOL=0 to grade the
# final ```python fence instead (that is the official OJBench protocol, and it
# scores an untrained base model much higher -- 51.7% vs 25.0% on easy -- because
# a base model's last tool call is usually a test driver, not its submission).
#
# SCAFFOLD = TRAINING SCAFFOLD, deliberately: 16384 per-turn response budget,
# 20 max turns, thinking on, minimal system prompt, 8 samples at temperature 1 --
# the same numbers eval_multitask.yaml/eval_code.yaml use for the step-0 eval
# inside the ablation runs, so these numbers are comparable to the training
# curves rather than to a hand-tuned benchmark config.
#
# How it works: runs the standard train.py with --num-rollout 1 and WITHOUT
# --skip-eval-before-train, so the pre-train (step-0) eval fires against the
# loaded model. See EVAL_ARGS for why --eval-interval is 2 and not 1.
#
# Required env vars:
#   WANDB_API_KEY     for logging to wandb
#
# Optional env vars:
#   MODEL_PATHS       space-separated HF checkpoint dirs to evaluate SEQUENTIALLY
#                     (default: Qwen3.5-4B + Qwen3.5-9B base under MODEL_STORE)
#   MODEL_PATH        single checkpoint dir; overrides MODEL_PATHS when set
#   MODEL_STORE       (default /root/data/home-static/data/hf_models) base-model dir
#   OJBENCH_DATA_DIR  (default /root/data/home-static/data/ojbench_medium)
#   OJBENCH_DIFFICULTY (default medium) comma-separated, only used if the eval
#                     jsonl has to be BUILT (needs network for the HF download)
#   OJBENCH_DUMP_DIR  (default /root/data/sdpo_dumps) trace dump root
#   OJBENCH_MAX_RESPONSE_LEN (default 16384) PER-TURN generation budget
#   OJBENCH_REQUIRE_TOOL (default 1) 0 -> grade the ```python fence
#   SDPO_REACT_EVAL_N_SAMPLES (default 8) samples per prompt for pass@k
#   SDPO_REACT_EVAL_MAX_TURNS (default 20) max tool-calling turns per sample
#   SDPO_REACT_TP     (default 2) tensor parallel size
#   SDPO_REACT_TRAIN_GPUS (default 8) total GPUs
#   MEGATRON_PATH     (default /root/Megatron-LM)
#   SGLANG_MEM_FRACTION (default 0.75)
#   OJBENCH_WANDB_PROJECT (default miles-sdpo)
#   OJBENCH_WANDB_GROUP   (default ojbench-medium-eval)
#   SDPO_EVAL_REF_LOAD    megatron dist-ckpt to load the WEIGHTS from; lets this
#                         script eval a TRAINED checkpoint directly (no HF
#                         round-trip), MODEL_PATH then only gives tokenizer/config
#   SDPO_EVAL_TAG         suffix for dump-dir / log naming, so a base-model
#                         MODEL_PATH + trained SDPO_EVAL_REF_LOAD is not filed
#                         under the base model's name
#
# usage (both base models, the default):
#   bash examples/SDPO_ReAct/ablation/eval-ojbench.sh
# usage (one trained arm):
#   MODEL_PATH=/root/data/home-static/data/hf_models/Qwen3.5-9B-...-arm-e \
#     bash examples/SDPO_ReAct/ablation/eval-ojbench.sh
set -exf

# Paths are CONTAINER paths: enroot mounts the host $DATA_DIR (/fsx/data/$USER)
# as /root/data, and /fsx itself is not readable from inside the session.
MODEL_STORE="${MODEL_STORE:-/root/data/home-static/data/hf_models}"
if [ -n "${MODEL_PATH:-}" ]; then
    MODEL_PATHS="$MODEL_PATH"
fi
MODEL_PATHS="${MODEL_PATHS:-${MODEL_STORE}/Qwen3.5-4B ${MODEL_STORE}/Qwen3.5-9B}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"

OJBENCH_DATA_DIR="${OJBENCH_DATA_DIR:-/root/data/home-static/data/ojbench_medium}"
export OJBENCH_DATA_DIR
OJBENCH_DIFFICULTY="${OJBENCH_DIFFICULTY:-medium}"
# PER-TURN response budget. 16384 is what the training runs' own eval config
# uses (eval_code.yaml / eval_multitask.yaml), so keep it here.
OJBENCH_MAX_RESPONSE_LEN="${OJBENCH_MAX_RESPONSE_LEN:-16384}"
export OJBENCH_MAX_RESPONSE_LEN

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
# NOTE the sidecar's concurrency cap (docker/sandbox_server.py::MAX_CONCURRENCY)
# matters more here than anywhere else: OJBench test cases are CPU-heavy, so an
# oversubscribed sandbox turns correct submissions into wall-clock timeouts
# (measured 5.6% vs 18.5% solved on identical candidates).
bash "$REACT_DIR/tools/run_sandbox.sh"

# --- 0a. data prep ---
EVAL_JSONL="${OJBENCH_DATA_DIR}/ojbench_eval.jsonl"
mkdir -p "$OJBENCH_DATA_DIR"
[ -f "$EVAL_JSONL" ] || \
    (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_ojbench \
        --out-dir "$OJBENCH_DATA_DIR" \
        --difficulty "${OJBENCH_DIFFICULTY}" \
        --max-tests-per-problem 10)
echo "OJBench eval rows: $(wc -l < "$EVAL_JSONL")"

EVAL_CFG="$REACT_DIR/data/eval_ojbench.yaml"

# Use thinking by default (Qwen3.5 strong reasoning mode), same as training.
export SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-true}"
export SDPO_REACT_PROMPT="${SDPO_REACT_PROMPT:-minimal}"

for MODEL_PATH in ${MODEL_PATHS}; do
MODEL_NAME="$(basename "$MODEL_PATH")"
# SDPO_EVAL_TAG only renames things (dump dir, log lines). Needed when
# MODEL_PATH is a BASE HF dir used purely for tokenizer/config while the weights
# come from SDPO_EVAL_REF_LOAD (see the REF_LOAD block below) -- otherwise such
# an eval is filed under the base model's name and is indistinguishable from the
# baseline. Only meaningful for a single-model MODEL_PATH.
MODEL_TAG="${MODEL_NAME}${SDPO_EVAL_TAG:+-${SDPO_EVAL_TAG}}"
echo "============================================================"
echo "OJBench ${OJBENCH_DIFFICULTY} eval: ${MODEL_TAG}"
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

DUMP_DIR="${OJBENCH_DUMP_DIR:-/root/data/sdpo_dumps}/ojbench-${OJBENCH_DIFFICULTY}-eval-${MODEL_TAG}_$(date +%Y%m%d_%H%M%S)"
echo "OJBench eval dump dir: ${DUMP_DIR}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)

# Megatron loads the ACTOR from a torch_dist checkpoint, not from the HF dir:
# with no --load/--ref-load, arguments.py sets args.load = args.ref_load = None
# and setup_model_and_optimizer asserts (miles/backends/megatron_utils/model.py
# :128). Resolve the sibling _torch_dist dir of the HF checkpoint, then the
# per-node asset copy the enroot launcher's prep step writes.
#   SDPO_EVAL_REF_LOAD=<dir> overrides the search with a TRAINED megatron
#   dist-ckpt (a run's --save dir: iter_NNNNNNN/ + latest_checkpointed_iteration
#   .txt). arguments.py copies ref_load into args.load with no_load_optim/
#   no_load_rng/finetune, so the trained WEIGHTS load and the optimizer state in
#   the ckpt is ignored -- i.e. a training checkpoint can be evaluated with no HF
#   round-trip conversion. MODEL_PATH then only supplies tokenizer/config, so
#   point it at the BASE model and set SDPO_EVAL_TAG.
REF_LOAD="${SDPO_EVAL_REF_LOAD:-}"
if [ -z "$REF_LOAD" ]; then
    for cand in "${MODEL_PATH}_torch_dist" "/root/assets/${MODEL_NAME}_torch_dist" "/root/${MODEL_NAME}_torch_dist"; do
        if [ -f "${cand}/latest_checkpointed_iteration.txt" ]; then
            REF_LOAD="$cand"
            break
        fi
    done
fi
if [ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: no torch_dist checkpoint for ${MODEL_NAME}. Convert it first:" >&2
    echo "  sbatch examples/SDPO_ReAct/ablation/sbatch_convert_torch_dist.sh" >&2
    exit 1
fi
echo "REF_LOAD: ${REF_LOAD}"
CKPT_ARGS+=(--ref-load "${REF_LOAD}")

# The training loop needs a --prompt-data even for an eval-only run. Use the
# eval data itself; with --eval-interval 2 (see EVAL_ARGS) the single no-op
# rollout step never feeds an eval that follows it.
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
   --rollout-max-response-len "${OJBENCH_MAX_RESPONSE_LEN}"
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
if [ "${OJBENCH_REQUIRE_TOOL:-1}" = "0" ]; then
    RM_ARGS+=(--no-sdpo-code-require-tool)
fi

EVAL_ARGS=(
   # eval-interval 2, NOT 1, with --num-rollout 1: train.py evals once before the
   # loop (rollout_id == start_rollout_id and not --skip-eval-before-train) and
   # then AGAIN at the end of the step if should_run_periodic_action() fires.
   # With interval 1 that second eval runs on weights that just took a GRPO step
   # on --prompt-data -- which for an eval-only run IS the eval set. Both evals
   # write the same rollout_data/eval_0.jsonl (the second overwrites the first),
   # so what got dumped was the post-leak number. interval 2 makes step 1 miss
   # the periodic check, so exactly one eval runs, on the loaded weights.
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
   --wandb-project "${OJBENCH_WANDB_PROJECT:-miles-sdpo}"
   --wandb-group "${OJBENCH_WANDB_GROUP:-ojbench-medium-eval}"
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
# ~20s to bind -- submitting immediately dies with ConnectionRefused.
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
        \"OJBENCH_DATA_DIR\": \"${OJBENCH_DATA_DIR}\",
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
echo "DONE ${MODEL_TAG} -- traces: ${DUMP_DIR}"
done

echo "OJBench ${OJBENCH_DIFFICULTY} eval complete for: ${MODEL_PATHS}"
echo "Check wandb group '${OJBENCH_WANDB_GROUP:-ojbench-medium-eval}' (eval/ojbench*)."
