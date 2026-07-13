#!/bin/bash
# One-click ENROOT launcher for the SDPO example — no sudo, no Docker daemon.
#
# Why enroot: this host runs Docker with the containerd image store, which keeps
# image content under /var/lib/containerd on the full root partition (ignoring
# daemon.json data-root) and needs sudo to fix. enroot instead imports the image
# to a squashfs file on NVMe and runs it entirely as your user — nothing touches
# the root disk. The system enroot.conf already points every path at NVMe.
#
#   IMAGE   (default radixark/miles:latest-cu12)   docker image (driver 570 -> cu12)
#   SQSH    (default $ENROOT_NVME/miles-cu12.sqsh)  imported squashfs image
#   CONTAINER (default miles-cu12)                  enroot container name
#   ASSETS  (default /opt/dlami/nvme/miles-assets)  models + data + checkpoints
#   PREP_ONLY=1  prepare assets but do not train
#
# NOTE: we `enroot create` (unsquashfs -> a real rootfs dir on NVMe) instead of
# starting straight from the .sqsh. On this host the default squashfuse/fuse
# mount HANGS indefinitely; unsquashfs extraction avoids fuse entirely.
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Auto-load secrets (e.g. WANDB_API_KEY) from examples/SDPO/.env if present.
# The file is gitignored — never commit it.
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; . "$SCRIPT_DIR/.env"; set +a
fi

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# dev-cu12-202607040446 (sglang dev13799) is the middle-ground image that works:
#   - has the /begin_weight_update HTTP API miles needs (the 6/17 image lacked it)
#   - its FA3 kernel matches sglang (does NOT pass `only_qv`), so FA3 is usable
#     (fast) — no triton fallback needed
#   - still has the tolist bug, fixed by patch-sglang-tolist.sh (idempotent)
#   - uses full sglang parallel-size arg names, handled by the alias in
#     backends/sglang_utils/arguments.py::validate_args
IMAGE="${IMAGE:-radixark/miles:dev-cu12-202607040446}"
ENROOT_NVME="${ENROOT_NVME:-/opt/dlami/nvme/miles-enroot}"
SQSH="${SQSH:-$ENROOT_NVME/miles-0704.sqsh}"
CONTAINER="${CONTAINER:-miles-0704}"
ASSETS="${ASSETS:-/opt/dlami/nvme/miles-assets}"
CACHES="${CACHES:-$ASSETS/caches}"

mkdir -p "$ENROOT_NVME" "$ASSETS" "$ASSETS/hf_cache" \
    "$CACHES/triton" "$CACHES/inductor" "$CACHES/torch_extensions" "$CACHES/nv"

# --- 1. import image -> squashfs on NVMe (skip if already imported) ----------
# enroot names hub images as 'docker://<image>'. The import lands on NVMe via the
# ENROOT_* paths in /etc/enroot/enroot.conf, never the root disk.
if [ ! -f "$SQSH" ]; then
    enroot import -o "$SQSH" "docker://${IMAGE}"
fi

# --- 2. create container rootfs (unsquashfs, no fuse) ------------------------
# Skip if the named container already exists.
if ! enroot list 2>/dev/null | grep -qx "$CONTAINER"; then
    enroot create --name "$CONTAINER" "$SQSH"
fi

# --- 3. run ------------------------------------------------------------------
# --rw: writable rootfs;  --mount SRC:DST bind-mounts host NVMe dirs.
enroot start --rw \
    --mount "$REPO_ROOT":/root/miles \
    --mount "$ASSETS":/root/assets \
    --mount "$ASSETS/hf_cache":/root/hf_cache \
    --mount "$CACHES":/root/caches \
    --env PREP_ONLY="${PREP_ONLY:-0}" \
    --env SDPO_MODEL="${SDPO_MODEL:-olmo3-sci}" \
    --env HF_HOME=/root/hf_cache \
    --env TRITON_CACHE_DIR=/root/caches/triton \
    --env TORCHINDUCTOR_CACHE_DIR=/root/caches/inductor \
    --env TORCH_EXTENSIONS_DIR=/root/caches/torch_extensions \
    --env CUDA_CACHE_PATH=/root/caches/nv \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    --env SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    "$CONTAINER" \
    bash -euxc '
        cd /root/miles

        # Pick model by $SDPO_MODEL (qwen3 | olmo3 | olmo3-sci): local dir name, HF
        # repo id, megatron model-arg script, and the SDPO run script.
        case "${SDPO_MODEL}" in
            olmo3)
                MODEL_DIR=Olmo-3-7B-Instruct
                HF_REPO=allenai/Olmo-3-7B-Instruct
                MODEL_SH=scripts/models/olmo3-7B.sh
                RUN_SH=examples/SDPO/run-olmo3-7B-sdpo.sh
                DATA_KIND=dapo
                ;;
            olmo3-sci)
                # Olmo3 on the SciKnowEval (MCQ) task instead of DAPO math.
                MODEL_DIR=Olmo-3-7B-Instruct
                HF_REPO=allenai/Olmo-3-7B-Instruct
                MODEL_SH=scripts/models/olmo3-7B.sh
                RUN_SH=examples/SDPO/run-olmo3-7B-sdpo-sci.sh
                DATA_KIND=sci
                ;;
            *)
                MODEL_DIR=Qwen3-8B
                HF_REPO=Qwen/Qwen3-8B
                MODEL_SH=scripts/models/qwen3-8B.sh
                RUN_SH=examples/SDPO/run-qwen3-8B-sdpo.sh
                DATA_KIND=sci
                ;;
        esac

        for name in "$MODEL_DIR" "${MODEL_DIR}_torch_dist" "${MODEL_DIR}_miles" sci dapo-math-17k math_eval; do
            mkdir -p /root/assets/$name
            ln -sfn /root/assets/$name /root/$name
        done

        # Patch the sglang tolist bug in this image (idempotent).
        bash examples/SDPO/patch-sglang-tolist.sh

        python -c "import miles; print(\"Miles import OK\")"

        [ -n "$(ls -A /root/$MODEL_DIR 2>/dev/null)" ] || \
            hf download "$HF_REPO" --local-dir /root/$MODEL_DIR

        if [ "$DATA_KIND" = "dapo" ]; then
            # DAPO math train set + AIME-2025/Minerva-Math eval sets.
            [ -f /root/dapo-math-17k/dapo-math-17k.jsonl ] || \
                hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
            # Guard on minerva_math.jsonl (not aime25) so the eval swap regenerates.
            [ -f /root/math_eval/minerva_math.jsonl ] || \
                python examples/SDPO/build_math_eval.py --out-dir /root/math_eval
        else
            [ -f /root/sci/train.jsonl ] || \
                python examples/SDPO/build_sci_dataset.py --out-dir /root/sci --val-ratio 0.1
        fi

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
