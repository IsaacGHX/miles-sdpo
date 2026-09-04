#!/bin/bash
# Code-only GRPO ablation on LCBv6 -- all model sizes, arm-a and arm-e.
# Runs sequentially from smallest to largest. Each run uses all 8 GPUs.
# No dynamic sampling filter. Bridge mode (direct HF checkpoint loading).
#
# Models: Qwen3.5-2B, 4B, 9B, 35B-A3B
# Arms: a (pure GRPO), e (GRPO + skill-KD both)
#
# Usage:
#   bash examples/SDPO_ReAct/ablation/run-codeonly-lcbv6-all-sizes.sh
#
# Or run a specific model/arm:
#   CODEONLY_MODELS="4B" CODEONLY_ARMS="a" bash examples/SDPO_ReAct/ablation/run-codeonly-lcbv6-all-sizes.sh
set -xf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Which models/arms to run (override with env vars)
CODEONLY_MODELS="${CODEONLY_MODELS:-2B 4B 9B 35B-A3B}"
CODEONLY_ARMS="${CODEONLY_ARMS:-a e}"

# Where to store models (persistent across runs)
MODEL_STORE="/fsx/data/haoxiang.zhang/home-static/data/hf_models"

# Wandb -- read from the environment, same as every other run script here.
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY (do not hardcode it in this script)}"

# --- Download models if needed ---
download_model() {
    local model_name="$1"
    local hf_id="Qwen/Qwen3.5-${model_name}"
    local local_dir="${MODEL_STORE}/Qwen3.5-${model_name}"

    if [ -f "${local_dir}/config.json" ]; then
        echo "Model ${model_name} already downloaded at ${local_dir}"
        return
    fi

    echo "Downloading ${hf_id} -> ${local_dir}"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${hf_id}', local_dir='${local_dir}')
print('Done')
"
}

for size in $CODEONLY_MODELS; do
    download_model "$size"
done

# --- Run training for each model/arm ---
for size in $CODEONLY_MODELS; do
    for arm in $CODEONLY_ARMS; do
        echo ""
        echo "============================================================"
        echo "Training: Qwen3.5-${size} arm-${arm} (code-only GRPO, LCBv6)"
        echo "============================================================"

        export SDPO_ABLATION_ARM="$arm"
        export SDPO_ABLATION_MODEL_ROOT="${MODEL_STORE}"
        export SDPO_ABLATION_DATA_ROOT="/fsx/data/haoxiang.zhang/home-static"
        export SDPO_ABLATION_DUMP_ROOT="/fsx/data/haoxiang.zhang/sdpo_dumps"
        export SDPO_ABLATION_CKPT_ROOT="/fsx/data/haoxiang.zhang/sdpo_ckpts"
        export SDPO_REACT_SAVE_CKPT="${SDPO_REACT_SAVE_CKPT:-1}"
        export SDPO_REACT_SAVE_INTERVAL="${SDPO_REACT_SAVE_INTERVAL:-10}"

        # Select the right codeonly script based on model size
        case "$size" in
            2B)
                export CODEONLY_MODEL_NAME="Qwen3.5-2B"
                export CODEONLY_MODEL_ARG_SH="scripts/models/qwen3.5-2B.sh"
                export SDPO_REACT_TP=2
                export SDPO_ABLATION_MAX_TOKENS_PER_GPU=6144
                export SDPO_ABLATION_SGLANG_MEM_FRACTION=0.6
                ;;
            4B)
                export CODEONLY_MODEL_NAME="Qwen3.5-4B"
                export CODEONLY_MODEL_ARG_SH="scripts/models/qwen3.5-4B.sh"
                export SDPO_REACT_TP=2
                export SDPO_ABLATION_MAX_TOKENS_PER_GPU=6144
                ;;
            9B)
                export CODEONLY_MODEL_NAME="Qwen3.5-9B"
                export CODEONLY_MODEL_ARG_SH="scripts/models/qwen3.5-9B.sh"
                export SDPO_REACT_TP=2
                export SDPO_ABLATION_MAX_TOKENS_PER_GPU=6144
                ;;
            35B-A3B)
                export CODEONLY_MODEL_NAME="Qwen3.5-35B-A3B"
                export CODEONLY_MODEL_ARG_SH="scripts/models/qwen3.5-35B-A3B.sh"
                export SDPO_REACT_TP=2
                export SDPO_REACT_EP=8
                export SDPO_ABLATION_MAX_TOKENS_PER_GPU=4096
                export SDPO_ABLATION_SGLANG_MEM_FRACTION=0.75
                export SDPO_REACT_LOGPROBS_CHUNK=4096
                ;;
        esac

        # Symlink data paths (eval yaml uses /root/data/...)
        mkdir -p /root/data
        ln -sfn "${SDPO_ABLATION_DATA_ROOT}/data/code_data" /root/data/code_data
        ln -sfn "${SDPO_ABLATION_DATA_ROOT}/data/code_data_v6eval" /root/data/code_data_v6eval

        # Use the generic codeonly script (parameterized by env vars)
        bash "$SCRIPT_DIR/run-codeonly-lcbv6-generic.sh"

        echo "DONE: Qwen3.5-${size} arm-${arm}"
    done
done

echo ""
echo "========================================"
echo "ALL TRAINING RUNS COMPLETE"
echo "========================================"
