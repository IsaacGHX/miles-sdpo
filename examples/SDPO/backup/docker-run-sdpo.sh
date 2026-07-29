#!/bin/bash
# One-click Docker launcher for the SDPO example (Qwen3-8B, single 8x H200 node).
#
# Run this ON THE HOST. It pulls the Miles image, starts a container with GPUs,
# mounts THIS repo over /root/miles (so the container runs your latest SDPO
# code) plus a persistent asset dir, then inside the container:
#   1. downloads Qwen3-8B + builds the SciKnowEval dataset
#   2. converts the student checkpoint to Megatron torch_dist
#   3. runs examples/SDPO/run-qwen3-8B-sdpo.sh
#
# Everything is idempotent: existing models/data/checkpoints are reused.
#
#   IMAGE   (default radixark/miles:latest-cu12)       Miles image to use
#   ASSETS  (default /opt/dlami/nvme/miles-assets)     host dir persisting models+data+ckpts
#   HF_HOME (default $ASSETS/hf_cache)                 host-side HuggingFace cache
#   PREP_ONLY=1  prepare assets but do not start training
#
# NOTE: assets default to the local NVMe (/opt/dlami/nvme, ~27T free) rather
# than $HOME (a near-full shared Lustre mount). Docker's own data-root should
# also live on NVMe — this script checks and warns if it does not.
set -ex

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Default to the CUDA 12.9 image: the ':latest' image is built for CUDA 13, which
# needs NVIDIA driver 580+. This host's driver (570.x) tops out at CUDA 12.8, so
# use ':latest-cu12'. Override with IMAGE=radixark/miles:latest if on driver 580+.
IMAGE="${IMAGE:-radixark/miles:latest-cu12}"
ASSETS="${ASSETS:-/opt/dlami/nvme/miles-assets}"
HF_HOME="${HF_HOME:-$ASSETS/hf_cache}"
# Compile/JIT caches (Triton autotune, torch Inductor, torch_extensions C++/CUDA
# JIT, nvcc PTX). Persist them on NVMe so runs don't re-autotune/re-compile from
# scratch every time — otherwise --rm throws them away with the container.
CACHES="${CACHES:-$ASSETS/caches}"

mkdir -p "$ASSETS" "$HF_HOME" \
    "$CACHES/triton" "$CACHES/inductor" "$CACHES/torch_extensions" "$CACHES/nv"

# --- disk preflight ----------------------------------------------------------
# SDPO needs ~60G+ (Qwen3-8B HF ~16G, torch_dist ~30G, dataset, checkpoints).
avail_gb=$(df -BG --output=avail "$ASSETS" | tail -1 | tr -dc '0-9')
if [ "${avail_gb:-0}" -lt 80 ]; then
    echo "WARNING: only ${avail_gb}G free at $ASSETS (need ~80G+). Set ASSETS= to a bigger disk." >&2
fi
# Docker image layers land in the daemon's data-root; make sure that is not the
# full root partition.
data_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)"
root_avail_gb=$(df -BG --output=avail "$data_root" 2>/dev/null | tail -1 | tr -dc '0-9')
echo "Docker data-root: $data_root (${root_avail_gb:-?}G free)"
if [ "${root_avail_gb:-0}" -lt 40 ]; then
    echo "WARNING: docker data-root has only ${root_avail_gb}G free — 'docker pull' may fail." >&2
    echo "         Point data-root at NVMe via /etc/docker/daemon.json {\"data-root\": \"/opt/dlami/nvme/docker/data-root\"}." >&2
fi

docker pull "$IMAGE"

# The container writes models/data straight into /root (mapped to $ASSETS) so
# they survive across runs and match the /root/... paths in the run script.
docker run --rm \
    --gpus all --ipc=host --shm-size=32g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --network=host \
    -e PREP_ONLY="${PREP_ONLY:-0}" \
    -e HF_HOME=/root/hf_cache \
    -e TRITON_CACHE_DIR=/root/caches/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/root/caches/inductor \
    -e TORCH_EXTENSIONS_DIR=/root/caches/torch_extensions \
    -e CUDA_CACHE_PATH=/root/caches/nv \
    -v "$REPO_ROOT":/root/miles \
    -v "$ASSETS":/root/assets \
    -v "$HF_HOME":/root/hf_cache \
    -v "$CACHES":/root/caches \
    -w /root/miles \
    "$IMAGE" /bin/bash -euxc '
        # Point the run scripts /root/<name> at the persistent assets volume.
        for name in Qwen3-8B Qwen3-8B_torch_dist Qwen3-8B_miles sci; do
            mkdir -p /root/assets/$name
            ln -sfn /root/assets/$name /root/$name
        done

        python -c "import miles; print(\"Miles import OK\")"

        # 1. student model + SDPO science dataset
        [ -n "$(ls -A /root/Qwen3-8B 2>/dev/null)" ] || \
            hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
        [ -f /root/sci/train.jsonl ] || \
            python examples/SDPO/build_sci_dataset.py --out-dir /root/sci --val-ratio 0.1

        # 2. convert student -> Megatron torch_dist (used as --ref-load)
        if [ -z "$(ls -A /root/Qwen3-8B_torch_dist 2>/dev/null)" ]; then
            source scripts/models/qwen3-8B.sh
            PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
                "${MODEL_ARGS[@]}" \
                --hf-checkpoint /root/Qwen3-8B \
                --save /root/Qwen3-8B_torch_dist
        fi

        if [ "$PREP_ONLY" = "1" ]; then
            echo "PREP_ONLY=1 -> assets ready under /root/assets, skipping training."
            exit 0
        fi

        # 3. train
        bash examples/SDPO/run-qwen3-8B-sdpo.sh
    '
