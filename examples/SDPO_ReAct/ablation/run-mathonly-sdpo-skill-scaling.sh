#!/bin/bash
# Math-only SDPO ablation (arms a/e/f/g/h) for 4B and 9B.
#
# EVERY arm here runs the SAME algorithm: full/original SDPO (dynamic sampling via
# check_sdpo_group_has_prefix, JSD KD loss, EMA self-teacher). The arms differ ONLY
# in the skill machinery layered on top, so any two rows are directly comparable:
#
#   arm  skill flags                                  response prefix   skill-KD
#   a    none (baseline)                              trace             --
#   e    self-skill all + pitfall-condense            skill             both
#   f    self-skill all + pitfall-condense            skill             both-blind
#   g    self-skill all + pitfall-condense            trace             both
#   h    self-skill all + pitfall-condense            trace             both-blind
#
# Arms a/e/f are the arm definitions from
# run-qwen3.5-4B-sdpo-react-ablation-mathcodesearch.sh (its ARM case block, with
# SDPO_ABLATION_ALGO=sdpo) transplanted onto math-only data. Arms g/h are this
# script's own additions and are identical to e/f EXCEPT that
# --sdpo-response-prefix stays at its default "trace": that single flag is the
# whole e-vs-g and f-vs-h contrast (does the teacher hand over the peer's distilled
# SKILL, or the peer's complete worked TRACE?).
#
# Training data: dapo-math-17k (native prompt format)
# Eval: AIME26 + AMO-Bench
#
# Usage:
#   MATHONLY_MODELS="4B" MATHONLY_ARMS="g" bash examples/SDPO_ReAct/ablation/run-mathonly-sdpo-skill-scaling.sh
#   MATHONLY_MODELS="4B 9B" MATHONLY_ARMS="a e f" bash examples/SDPO_ReAct/ablation/run-mathonly-sdpo-skill-scaling.sh
set -xf

export PATH="/usr/local/bin:/usr/bin:/usr/sbin:/bin:/sbin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REACT_DIR="$REPO_ROOT/examples/SDPO_ReAct"

MATHONLY_MODELS="${MATHONLY_MODELS:-4B 9B}"
MATHONLY_ARMS="${MATHONLY_ARMS:-g h}"

MODEL_STORE="/fsx/data/haoxiang.zhang/home-static/data/hf_models"
# Read from the environment, same as every other run script here -- never hardcode.
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY (do not hardcode it in this script)}"

export PYTHONBUFFERED=16
export SDPO_REACT_TRAIN_MAX_TURNS="${SDPO_REACT_TRAIN_MAX_TURNS:-8}"
export SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-20}"
export SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-8}"
export SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-true}"
export SDPO_REACT_PROMPT="${SDPO_REACT_PROMPT:-minimal}"
SDPO_REACT_NUM_ROLLOUT="${SDPO_REACT_NUM_ROLLOUT:-51}"

SDPO_REACT_TRAIN_GPUS=8
N_SAMPLES_PER_PROMPT=8

# --- Data prep ---
DATA_DIR="/fsx/data/haoxiang.zhang/home-static"
MATH_DATA="${DATA_DIR}/dapo-math-17k/dapo-math-17k-native-minimal.jsonl"
EVAL_DIR="${DATA_DIR}/math_eval/native-minimal"
AMO_DATA_DIR="${DATA_DIR}/data/amo_bench"
AIME26_DATA_DIR="${DATA_DIR}/data/aime26"

# Build data if needed. The raw dapo dump only lives on ephemeral nvme on a fresh
# host (see doc note on /opt/dlami/nvme being wiped on restart) and the enroot
# container does not mount it, so seed the fsx copy before the native-prompt build.
mkdir -p "${DATA_DIR}/dapo-math-17k"
if [ ! -f "$MATH_DATA" ]; then
    if [ ! -f "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl" ]; then
        cp /opt/dlami/nvme/miles-assets/dapo-math-17k/dapo-math-17k.jsonl \
           "${DATA_DIR}/dapo-math-17k/" 2>/dev/null || \
        hf download --repo-type dataset zhuzilin/dapo-math-17k \
            --local-dir "${DATA_DIR}/dapo-math-17k"
    fi
    (cd "$REPO_ROOT" && python3 -m examples.SDPO_ReAct.native_prompt \
        --in "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl" \
        --out "$MATH_DATA")
fi
[ -f "$MATH_DATA" ] || { echo "FATAL: train data '$MATH_DATA' missing" >&2; exit 1; }

[ -f "${EVAL_DIR}/aime25_native.jsonl" ] || (cd "$REPO_ROOT" && python3 -m examples.SDPO_ReAct.data.build_native_eval --out-dir "$EVAL_DIR")

[ -f "${AMO_DATA_DIR}/amo_bench_eval.jsonl" ] || (cd "$REPO_ROOT" && python3 -m examples.SDPO_ReAct.data.build_amo_bench --out-dir "$AMO_DATA_DIR")

[ -f "${AIME26_DATA_DIR}/aime26_eval.jsonl" ] || (cd "$REPO_ROOT" && python3 -m examples.SDPO_ReAct.data.build_aime26 --out-dir "$AIME26_DATA_DIR")

for f in "${AMO_DATA_DIR}/amo_bench_eval.jsonl" "${AIME26_DATA_DIR}/aime26_eval.jsonl"; do
    [ -f "$f" ] || { echo "FATAL: eval data '$f' missing" >&2; exit 1; }
done

# Eval config: use both AIME26 and AMO-Bench
# We'll create a combined eval yaml
EVAL_CFG="$REACT_DIR/data/eval_math_combined.yaml"
if [ ! -f "$EVAL_CFG" ]; then
cat > "$EVAL_CFG" << 'YAML'
eval:
  defaults:
    max_response_len: 16384
    top_p: 1.0
    tool_key: tools
    metadata_key: metadata
  datasets:
    - name: aime26
      path: ${oc.env:AIME26_DATA_DIR,/fsx/data/haoxiang.zhang/home-static/data/aime26}/aime26_eval.jsonl
      rm_type: null
      n_samples_per_eval_prompt: ${oc.decode:${oc.env:SDPO_REACT_EVAL_N_SAMPLES,8}}
      metadata_overrides:
        generate_max_turns: ${oc.decode:${oc.env:SDPO_REACT_EVAL_MAX_TURNS,20}}
    - name: amo_bench
      path: ${oc.env:AMO_BENCH_DATA_DIR,/fsx/data/haoxiang.zhang/home-static/data/amo_bench}/amo_bench_eval.jsonl
      rm_type: null
      n_samples_per_eval_prompt: ${oc.decode:${oc.env:SDPO_REACT_EVAL_N_SAMPLES,8}}
      metadata_overrides:
        generate_max_turns: ${oc.decode:${oc.env:SDPO_REACT_EVAL_MAX_TURNS,20}}
YAML
fi

# --- Sandbox sidecar ---
bash "$REACT_DIR/tools/run_sandbox.sh"

# --- Download models ---
for size in $MATHONLY_MODELS; do
    local_dir="${MODEL_STORE}/Qwen3.5-${size}"
    if [ ! -f "${local_dir}/config.json" ]; then
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-${size}', local_dir='${local_dir}')
"
    fi
done

for size in $MATHONLY_MODELS; do
    for arm in $MATHONLY_ARMS; do
        echo ""
        echo "============================================================"
        echo "Training: Qwen3.5-${size} arm-${arm} (math-only SDPO+skill, eval AIME26+AMO)"
        echo "============================================================"

        # Model config
        case "$size" in
            4B)
                MODEL_ARG_SH=scripts/models/qwen3.5-4B.sh
                SDPO_REACT_TP=2
                MAX_TOKENS_PER_GPU=6144
                ;;
            9B)
                MODEL_ARG_SH=scripts/models/qwen3.5-9B.sh
                SDPO_REACT_TP=2
                MAX_TOKENS_PER_GPU=6144
                ;;
        esac

        DP_SIZE=$((SDPO_REACT_TRAIN_GPUS / SDPO_REACT_TP))
        ROLLOUT_BATCH_SIZE=$((DP_SIZE * 4))
        GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))

        source "$REPO_ROOT/${MODEL_ARG_SH}"

        # Symlinks for eval paths
        mkdir -p /root/data
        ln -sfn "$AIME26_DATA_DIR" /root/data/aime26
        ln -sfn "$AMO_DATA_DIR" /root/data/amo_bench

        [ "$SDPO_REACT_THINKING" = "true" ] && _THINK=think || _THINK=nothink
        REMOVE_THINKING_ARG=()
        [ "$SDPO_REACT_THINKING" != "true" ] && REMOVE_THINKING_ARG=(--sdpo-remove-thinking-from-demonstration)

        CFG_TAG="Qwen3.5-${size}-mathonly-sdpo-${arm}-${_THINK}"
        DUMP_DIR="/fsx/data/haoxiang.zhang/sdpo_dumps/sdpo-react-ablation-${CFG_TAG}_$(date +%Y%m%d_%H%M%S)"
        echo "Dump dir: $DUMP_DIR"

        TOOL_SPECS_PATH="examples.SDPO_ReAct.tools.registry.all_tool_specs"
        EXECUTE_TOOL_PATH="examples.SDPO_ReAct.tools.registry.execute_tool"
        TOOL_PARSER=qwen3_coder
        TOOL_GRAMMAR=qwen3_coder

        # --- RM_ARGS: shared by every arm (the arm-specific skill flags follow) ---
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
           --no-sdpo-pure-distill
           # Match the eval grading behind sdpo_dumps/0_stat/amo_bench_results.md
           # (eval-amo-bench.sh + --sdpo-judge: EVERY eval row is LLM-judged).
           # --sdpo-judge itself is wrong here -- on a pure-math TRAINING group
           # _grade_group would route every sample to the judge and replace
           # --sdpo-grader dapo. This is the eval-only switch; judge backend and
           # model come from their defaults (Bedrock Converse + luna, IAM role).
           --sdpo-eval-judge
        )

        # --- arm -> skill flags (see the header table) ---
        # Identical skill-generation config for e/f/g/h; the only differences are
        # --sdpo-response-prefix (skill for e/f, default trace for g/h) and the
        # skill-KD mode. Arm a gets no skill machinery at all, so its teacher prefix
        # is the correct peer's full trace -- plain original SDPO.
        _SKILL_ALL=(--sdpo-self-skill --sdpo-skill-source all --sdpo-skill-max-new-tokens 2048
                    --sdpo-pitfall-summary-backend self --sdpo-env-feedback-max-chars 2000)
        _SKILL_KD=(--sdpo-skill-kd --sdpo-skill-kd-coef 0.01)
        case "$arm" in
            a) : ;;
            e) RM_ARGS+=("${_SKILL_ALL[@]}" --sdpo-response-prefix skill
                         "${_SKILL_KD[@]}" --sdpo-skill-kd-mode both) ;;
            f) RM_ARGS+=("${_SKILL_ALL[@]}" --sdpo-response-prefix skill
                         "${_SKILL_KD[@]}" --sdpo-skill-kd-mode both-blind) ;;
            g) RM_ARGS+=("${_SKILL_ALL[@]}" "${_SKILL_KD[@]}" --sdpo-skill-kd-mode both) ;;
            h) RM_ARGS+=("${_SKILL_ALL[@]}" "${_SKILL_KD[@]}" --sdpo-skill-kd-mode both-blind) ;;
            # Without this an unknown arm silently trained as "g minus skill-KD",
            # which is a real configuration and so produced believable garbage.
            *) echo "FATAL: unknown arm '$arm' (expected one of: a e f g h)" >&2; exit 1 ;;
        esac

        # --- GRPO_ARGS: full SDPO algorithm (JSD KD loss + dynamic sampling) ---
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

        # --- Ray ---
        ray stop --force 2>/dev/null || true
        pkill -9 -f 'ray::' 2>/dev/null || true
        sleep 2
        cd "$REPO_ROOT"
        ray start --head --node-ip-address 127.0.0.1 --num-gpus "${SDPO_REACT_TRAIN_GPUS}" \
            --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

        ray job submit --address="http://127.0.0.1:8265" \
           --runtime-env-json="{
             \"env_vars\": {
                \"PYTHONPATH\": \"${REPO_ROOT}:/root/Megatron-LM/\",
                \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
                \"NCCL_NVLS_ENABLE\": \"1\",
                \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
                \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\",
                \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
                \"SDPO_REACT_EVAL_MAX_TURNS\": \"${SDPO_REACT_EVAL_MAX_TURNS}\",
                \"SDPO_REACT_EVAL_N_SAMPLES\": \"${SDPO_REACT_EVAL_N_SAMPLES}\",
                \"SDPO_REACT_PROMPT\": \"${SDPO_REACT_PROMPT}\",
                \"AIME26_DATA_DIR\": \"${AIME26_DATA_DIR}\",
                \"AMO_BENCH_DATA_DIR\": \"${AMO_DATA_DIR}\",
                \"MILES_EVAL_METRICS_FILE\": \"${DUMP_DIR}/eval_metrics.json\"
             }
           }" \
           -- python3 train.py \
           --actor-num-nodes 1 \
           --actor-num-gpus-per-node "${SDPO_REACT_TRAIN_GPUS}" \
           --rollout-num-gpus "${SDPO_REACT_TRAIN_GPUS}" \
           --colocate \
           --update-weights-interval 1 \
           ${MODEL_ARGS[@]} \
           --megatron-to-hf-mode bridge \
           --hf-checkpoint "${MODEL_STORE}/Qwen3.5-${size}" \
           --dump-details "${DUMP_DIR}" \
           --no-dump-train-data \
           --no-dump-policy-loss-debug \
           --prompt-data "$MATH_DATA" \
           --input-key prompt \
           --label-key label \
           --tool-key tools \
           --apply-chat-template \
           --apply-chat-template-kwargs "{\"enable_thinking\":${SDPO_REACT_THINKING}}" \
           --num-rollout "${SDPO_REACT_NUM_ROLLOUT}" \
           --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
           --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
           --rollout-max-response-len 8192 \
           --rollout-max-context-len 81920 \
           --rollout-temperature 1 \
           --global-batch-size "${GLOBAL_BATCH_SIZE}" \
           --balance-data \
           --over-sampling-batch-size "${ROLLOUT_BATCH_SIZE}" \
           --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_sdpo_group_has_prefix \
           --sdpo-dynamic-filter-min-correct 1 \
           --tool-specs-resolver-path "$TOOL_SPECS_PATH" \
           --custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate \
           --generate-tool-specs-path "$TOOL_SPECS_PATH" \
           --generate-execute-tool-function-path "$EXECUTE_TOOL_PATH" \
           --generate-tool-call-parser "$TOOL_PARSER" \
           --generate-max-turns "${SDPO_REACT_TRAIN_MAX_TURNS}" \
           ${RM_ARGS[@]} \
           ${GRPO_ARGS[@]} \
           --optimizer adam \
           --lr 1e-6 \
           --lr-decay-style constant \
           --lr-warmup-iters 10 \
           --weight-decay 0.1 \
           --adam-beta1 0.9 \
           --adam-beta2 0.98 \
           --optimizer-cpu-offload \
           --overlap-cpu-optimizer-d2h-h2d \
           --use-precision-aware-optimizer \
           --tensor-model-parallel-size "${SDPO_REACT_TP}" \
           --pipeline-model-parallel-size 1 \
           --context-parallel-size 1 \
           --expert-model-parallel-size 1 \
           --expert-tensor-parallel-size 1 \
           --recompute-granularity full \
           --recompute-method uniform \
           --recompute-num-layers 1 \
           --use-dynamic-batch-size \
           --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
           --log-probs-chunk-size 4096 \
           --sequence-parallel \
           --eval-interval 10 \
           --eval-config "$EVAL_CFG" \
           --eval-tool-key tools \
           --n-samples-per-eval-prompt "${SDPO_REACT_EVAL_N_SAMPLES}" \
           --log-passrate \
           --use-wandb \
           --wandb-project miles-sdpo \
           --wandb-group "sdpo-react-ablation-${CFG_TAG}" \
           --wandb-key "${WANDB_API_KEY}" \
           --rollout-num-gpus-per-engine 1 \
           --sglang-mem-fraction-static 0.85 \
           --attention-dropout 0.0 \
           --hidden-dropout 0.0 \
           --accumulate-allreduce-grads-in-fp32 \
           --attention-softmax-in-fp32 \
           --attention-backend flash

        echo "DONE: Qwen3.5-${size} arm-${arm}"

        ray stop --force 2>/dev/null || true
        pkill -9 -f 'ray::' 2>/dev/null || true
        sleep 5
    done
done

echo ""
echo "========================================"
echo "ALL MATH-ONLY SDPO+SKILL RUNS COMPLETE"
echo "========================================"
