#!/bin/bash
# Convert HF checkpoints to Megatron torch_dist format for raw-mode training.
# Run inside enroot container with all 8 GPUs.
#
# Usage:
#   enroot start --rw -m /fsx/...:/fsx/... miles-0704 bash examples/SDPO_ReAct/ablation/convert-all-models-to-torch-dist.sh
set -ex

export PATH="/usr/local/bin:/usr/bin:/usr/sbin:/bin:/sbin:$PATH"

REPO_ROOT="/fsx/home/haoxiang.zhang/gitproj/miles-sdpo"
MODEL_STORE="/fsx/data/haoxiang.zhang/home-static/data/hf_models"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:/root/Megatron-LM/"
export CUDA_DEVICE_MAX_CONNECTIONS=1

convert_model() {
    local SIZE="$1"
    local HF_PATH="${MODEL_STORE}/Qwen3.5-${SIZE}"
    local DIST_PATH="${MODEL_STORE}/Qwen3.5-${SIZE}_torch_dist"
    local MODEL_SCRIPT="scripts/models/qwen3.5-${SIZE}.sh"

    if [ -f "${DIST_PATH}/latest_checkpointed_iteration.txt" ]; then
        echo "SKIP: Qwen3.5-${SIZE}_torch_dist already exists"
        return
    fi

    if [ ! -f "${HF_PATH}/config.json" ]; then
        echo "ERROR: HF checkpoint not found at ${HF_PATH}"
        return
    fi

    echo ""
    echo "============================================================"
    echo "Converting: Qwen3.5-${SIZE} -> torch_dist"
    echo "============================================================"

    # Source model args
    source "${REPO_ROOT}/${MODEL_SCRIPT}"

    # Determine TP for conversion (use all 8 GPUs as PP for faster conversion)
    torchrun --nproc_per_node=8 tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint "${HF_PATH}" \
        --save "${DIST_PATH}" \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --expert-tensor-parallel-size 1

    echo "DONE: Qwen3.5-${SIZE}_torch_dist -> ${DIST_PATH}"
}

# Convert all models (skip if already done)
for SIZE in 2B 4B 9B 35B-A3B; do
    convert_model "$SIZE"
done

echo ""
echo "========================================"
echo "ALL CONVERSIONS COMPLETE"
echo "========================================"
ls -d ${MODEL_STORE}/Qwen3.5-*_torch_dist 2>/dev/null
