#!/bin/bash
# ONE sbatch job that converts a list of HF checkpoints to Megatron torch_dist
# (../convert-models-to-torch-dist-generic.sh) on ONE H200 node inside the
# enroot container.
#
# WHY this exists: every eval/training run needs --ref-load <model>_torch_dist
# (raw mode: args.load = args.ref_load, and setup_model_and_optimizer asserts on
# it), but only the BASE checkpoints were ever converted -- the 12 trained arms
# under MODEL_STORE are HF-only, so they cannot be evaluated until converted.
# Conversion needs the 8-GPU torchrun from tools/convert_hf_to_torch_dist.py,
# hence a real slurm allocation rather than a login-node run.
#
# Sizes: 4B -> 7.9G, 9B -> 17G per checkpoint. 12 arms ~= 150G on /fsx/data.
#
# usage (defaults to the 12 arms listed in ./todo):
#   sbatch examples/SDPO_ReAct/ablation/sbatch_convert_torch_dist.sh
#   MODEL_NAMES="Qwen3.5-4B-... Qwen3.5-9B-..." \
#     sbatch examples/SDPO_ReAct/ablation/sbatch_convert_torch_dist.sh
#SBATCH --job-name=hx-convert-torch-dist
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low,ml.p5en.48xlarge-ultra-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/fsx/home/haoxiang.zhang/logs/hx-convert-torch-dist_%j.out
#SBATCH --requeue
set -x

REPO_ROOT="/fsx/home/haoxiang.zhang/gitproj/miles-sdpo"
cd "$REPO_ROOT"

echo "[$(date -u)] job $SLURM_JOB_ID started on $(hostname)"
nvidia-smi -L || true

# The 12 trained arms from ./todo (base models are already converted).
DEFAULT_NAMES="\
Qwen3.5-4B-sdpo-react-rlsd-multitask-arm1 \
Qwen3.5-4B-sdpo-react-rlsd-multitask-arm1.1 \
Qwen3.5-4B-sdpo-react-rlsd-multitask-arm5 \
Qwen3.5-4B-sdpo-react-mathcodesearch-sdpo-arm-a \
Qwen3.5-4B-sdpo-react-mathcodesearch-sdpo-arm-e \
Qwen3.5-4B-sdpo-react-mathcodesearch-grpo-arm-e \
Qwen3.5-9B-sdpo-react-mathcodesearch-rlsd-arm-a \
Qwen3.5-9B-sdpo-react-mathcodesearch-rlsd-arm-e \
Qwen3.5-9B-sdpo-react-mathcodesearch-sdpo-arm-a \
Qwen3.5-9B-sdpo-react-mathcodesearch-sdpo-arm-e \
Qwen3.5-9B-sdpo-react-mathcodesearch-grpo-arm-a \
Qwen3.5-9B-sdpo-react-mathcodesearch-grpo-arm-e"
MODEL_NAMES="${MODEL_NAMES:-$DEFAULT_NAMES}"

IMAGE="${IMAGE:-radixark/miles:dev-cu12-202607040446}"
ENROOT_NVME="${ENROOT_NVME:-/opt/dlami/nvme/miles-enroot}"
SQSH="${SQSH:-$ENROOT_NVME/miles-0704.sqsh}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$ENROOT_NVME/cache}"
CONTAINER="${CONTAINER:-miles-sdpo-react-cu12}"
ASSETS="${ASSETS:-/opt/dlami/nvme/miles-assets}"
CACHES="${CACHES:-$ASSETS/caches}"

mkdir -p "$ENROOT_NVME" "$ENROOT_CACHE_PATH" "$ASSETS" "$ASSETS/hf_cache" \
    "$CACHES/triton" "$CACHES/inductor" "$CACHES/torch_extensions" "$CACHES/nv"

if [ ! -f "$SQSH" ]; then
    enroot import -o "$SQSH" "docker://${IMAGE}"
fi
if ! enroot list 2>/dev/null | grep -qx "$CONTAINER"; then
    enroot create --name "$CONTAINER" "$SQSH"
fi

# NOTE: unlike the eval job this mounts /fsx STRAIGHT THROUGH rather than as
# /root/data, because the converter writes <hf_path>_torch_dist next to the HF
# dir and both the reader (eval scripts, via /root/data/...) and this writer must
# agree on one physical location. /fsx/data is a shared mount, so the converted
# checkpoints are visible to every node afterwards.
enroot start --rw \
    --mount "$REPO_ROOT":/root/miles \
    --mount /fsx/data/haoxiang.zhang:/fsx/data/haoxiang.zhang \
    --mount "$ASSETS/hf_cache":/root/hf_cache \
    --mount "$CACHES":/root/caches \
    --env HF_HOME=/root/hf_cache \
    --env TRITON_CACHE_DIR=/root/caches/triton \
    --env TORCHINDUCTOR_CACHE_DIR=/root/caches/inductor \
    --env TORCH_EXTENSIONS_DIR=/root/caches/torch_extensions \
    --env CUDA_CACHE_PATH=/root/caches/nv \
    --env MODEL_NAMES="$MODEL_NAMES" \
    --env REPO_ROOT=/root/miles \
    --env MODEL_STORE="${MODEL_STORE:-/fsx/data/haoxiang.zhang/home-static/data/hf_models}" \
    "$CONTAINER" \
    bash -c 'cd /root/miles && bash examples/SDPO_ReAct/ablation/convert-models-to-torch-dist-generic.sh'
RC=$?

echo "[$(date -u)] convert rc=$RC"
ls -d /fsx/data/haoxiang.zhang/home-static/data/hf_models/*_torch_dist 2>/dev/null
exit $RC
