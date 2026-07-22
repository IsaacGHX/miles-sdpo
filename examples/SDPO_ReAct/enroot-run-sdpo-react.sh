#!/bin/bash
# One-click ENROOT launcher for SDPO_ReAct — no sudo, no Docker daemon inside
# the training container. Sibling of examples/SDPO/enroot-run-sdpo.sh: same
# asset/cache/enroot plumbing, pointed at the SDPO_ReAct run script.
#
# Docker note: the code_interpreter sandbox sidecar (tools/docker/) is a REAL
# Docker container, started on the HOST via tools/run_sandbox.sh BEFORE the
# enroot session starts (not from inside enroot -- enroot has no Docker-in-
# Docker story, and doesn't need one: unlike Docker, enroot containers share
# the host's network namespace by default, so 127.0.0.1:8420 inside the
# enroot session already reaches the host-side sandbox container's published
# port). This keeps the "exactly one extra port for the whole job" property:
# the sandbox is a host-level singleton, independent of how many enroot/train
# sessions come and go.
#
#   IMAGE   (default radixark/miles:latest-cu12)   docker image (driver 570 -> cu12)
#   SQSH    (default $ENROOT_NVME/miles-cu12.sqsh)  imported squashfs image
#   CONTAINER (default miles-sdpo-react-cu12)       enroot container name
#   ASSETS  (default /opt/dlami/nvme/miles-assets)  models + data (local nvme, ephemeral)
#   DATA_DIR (default /fsx/data/$USER)              training checkpoints (shared, durable
#                                                    network storage -- NOT /fsx/home, which
#                                                    is much smaller/quota-limited and where a
#                                                    checkpoint write once genuinely failed
#                                                    mid-save from disk pressure)
#   SDPO_REACT_MODEL (default qwen2.5)              qwen2.5 | olmo3 -- picks the HF repo,
#                                                    local asset dir name, megatron model-arg
#                                                    script, and run script (same switch
#                                                    pattern as examples/SDPO/enroot-run-sdpo.sh's
#                                                    SDPO_MODEL)
#   PREP_ONLY=1  prepare assets but do not train
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Auto-load secrets (e.g. WANDB_API_KEY) from examples/SDPO_ReAct/.env if
# present, else fall back to examples/SDPO/.env (shared secrets, same repo).
# The file is gitignored — never commit it.
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; . "$SCRIPT_DIR/.env"; set +a
elif [ -f "$SCRIPT_DIR/../SDPO/.env" ]; then
    set -a; . "$SCRIPT_DIR/../SDPO/.env"; set +a
fi

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# dev-cu12-202607040446 (sglang dev13799) is the validated image both
# examples/SDPO/enroot-run-sdpo.sh and examples/EPO/enroot-run-epo.sh default
# to on this host (driver 570 -> cu12; has the tolist patch + FA3 fix already
# proven working) -- match their default rather than an unvalidated one.
IMAGE="${IMAGE:-radixark/miles:dev-cu12-202607040446}"
ENROOT_NVME="${ENROOT_NVME:-/opt/dlami/nvme/miles-enroot}"
SQSH="${SQSH:-$ENROOT_NVME/miles-0704.sqsh}"

export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$ENROOT_NVME/cache}"
CONTAINER="${CONTAINER:-miles-sdpo-react-cu12}"
ASSETS="${ASSETS:-/opt/dlami/nvme/miles-assets}"
CACHES="${CACHES:-$ASSETS/caches}"
DATA_DIR="${DATA_DIR:-/fsx/data/$USER}"

mkdir -p "$ENROOT_NVME" "$ENROOT_CACHE_PATH" "$ASSETS" "$ASSETS/hf_cache" \
    "$CACHES/triton" "$CACHES/inductor" "$CACHES/torch_extensions" "$CACHES/nv" \
    "$DATA_DIR/sdpo_ckpts"

# --- 0. sandbox sidecar on the HOST (before entering enroot) -----------------
bash "$SCRIPT_DIR/tools/run_sandbox.sh"

# --- 1. import image -> squashfs on NVMe (skip if already imported) ----------
if [ ! -f "$SQSH" ]; then
    enroot import -o "$SQSH" "docker://${IMAGE}"
fi

# --- 2. create container rootfs (unsquashfs, no fuse) ------------------------
if ! enroot list 2>/dev/null | grep -qx "$CONTAINER"; then
    enroot create --name "$CONTAINER" "$SQSH"
fi

# --- 3. run ------------------------------------------------------------------
enroot start --rw \
    --mount "$REPO_ROOT":/root/miles \
    --mount "$ASSETS":/root/assets \
    --mount "$ASSETS/hf_cache":/root/hf_cache \
    --mount "$CACHES":/root/caches \
    --mount "$DATA_DIR":/root/data \
    --env PREP_ONLY="${PREP_ONLY:-0}" \
    --env SDPO_REACT_MODEL="${SDPO_REACT_MODEL:-qwen2.5}" \
    --env HF_HOME=/root/hf_cache \
    --env TRITON_CACHE_DIR=/root/caches/triton \
    --env TORCHINDUCTOR_CACHE_DIR=/root/caches/inductor \
    --env TORCH_EXTENSIONS_DIR=/root/caches/torch_extensions \
    --env CUDA_CACHE_PATH=/root/caches/nv \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    "$CONTAINER" \
    bash -euxc '
        cd /root/miles

        # Pick model by $SDPO_REACT_MODEL (qwen2.5 | olmo3): local dir name, HF
        # repo id, megatron model-arg script, and the SDPO_ReAct run script --
        # same switch pattern as examples/SDPO/enroot-run-sdpo.sh'"'"'s SDPO_MODEL.
        case "${SDPO_REACT_MODEL}" in
            olmo3)
                MODEL_DIR=Olmo-3-7B-Instruct
                HF_REPO=allenai/Olmo-3-7B-Instruct
                MODEL_SH=scripts/models/olmo3-7B.sh
                RUN_SH=examples/SDPO_ReAct/run-olmo3-7B-sdpo-react-dapo-math.sh
                ;;
            *)
                MODEL_DIR=Qwen2.5-7B-Instruct
                HF_REPO=Qwen/Qwen2.5-7B-Instruct
                MODEL_SH=scripts/models/qwen2.5-7B.sh
                RUN_SH=examples/SDPO_ReAct/run-qwen2.5-7B-sdpo-react-dapo-math.sh
                ;;
        esac

        for name in "$MODEL_DIR" "${MODEL_DIR}_torch_dist" "${MODEL_DIR}_miles" dapo-math-17k math_eval; do
            mkdir -p /root/assets/$name
            ln -sfn /root/assets/$name /root/$name
        done

        # Reuse SDPO'"'"'s idempotent sglang tolist patch (shared, algorithm-
        # agnostic infra -- see examples/SDPO/patch-sglang-tolist.sh).
        bash examples/SDPO/patch-sglang-tolist.sh

        python -c "import miles; print(\"Miles import OK\")"

        [ -n "$(ls -A /root/$MODEL_DIR 2>/dev/null)" ] || \
            hf download "$HF_REPO" --local-dir /root/$MODEL_DIR

        if [ -z "$(ls -A /root/${MODEL_DIR}_torch_dist 2>/dev/null)" ]; then
            source "$MODEL_SH"
            PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
                "${MODEL_ARGS[@]}" \
                --hf-checkpoint /root/$MODEL_DIR \
                --save /root/${MODEL_DIR}_torch_dist
        fi

        if [ "$PREP_ONLY" = "1" ]; then
            echo "PREP_ONLY=1 -> assets ready under /root/assets, skipping training."
            exit 0
        fi

        bash "$RUN_SH"
    '
