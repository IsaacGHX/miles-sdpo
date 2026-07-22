#!/bin/bash
# One-click ENROOT launcher for the EPO example — no sudo, no Docker daemon.
# Sibling of examples/SDPO/enroot-run-sdpo.sh: identical container/asset
# plumbing, only the model-selection cases point at the EPO run scripts
# (examples/EPO/run-olmo3-7B-epo-*-colocate.sh) instead of SDPO's.
#
# Why enroot: this host runs Docker with the containerd image store, which keeps
# image content under /var/lib/containerd on the full root partition (ignoring
# daemon.json data-root) and needs sudo to fix. enroot instead imports the image
# to a squashfs file on NVMe and runs it entirely as your user — nothing touches
# the root disk. The system enroot.conf already points every path at NVMe.
#
#   IMAGE   (default radixark/miles:latest-cu12)   docker image (driver 570 -> cu12)
#   SQSH    (default $ENROOT_NVME/miles-cu12.sqsh)  imported squashfs image
#   CONTAINER (default miles-epo-cu12)              enroot container name
#   ASSETS  (default /opt/dlami/nvme/miles-assets)  models + data + checkpoints
#   PREP_ONLY=1  prepare assets but do not train
#
# NOTE: we `enroot create` (unsquashfs -> a real rootfs dir on NVMe) instead of
# starting straight from the .sqsh. On this host the default squashfuse/fuse
# mount HANGS indefinitely; unsquashfs extraction avoids fuse entirely.
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Auto-load secrets (e.g. WANDB_API_KEY) from examples/EPO/.env if present, else
# fall back to examples/SDPO/.env (shared secrets, same repo). The file is
# gitignored — never commit it.
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; . "$SCRIPT_DIR/.env"; set +a
elif [ -f "$SCRIPT_DIR/../SDPO/.env" ]; then
    set -a; . "$SCRIPT_DIR/../SDPO/.env"; set +a
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

# The system enroot.conf points ENROOT_CACHE_PATH at /fsx/enroot, but /fsx itself
# is NOT a Lustre mount here — it lives on the tiny (97G, OS-filled) root disk.
# Left alone, `enroot import` downloads image layers onto root and dies with
# "zstd: No space left on device". Force the layer cache onto NVMe like every
# other enroot path. (export so the enroot subprocess inherits it.)
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$ENROOT_NVME/cache}"
CONTAINER="${CONTAINER:-miles-epo-0704}"
ASSETS="${ASSETS:-/opt/dlami/nvme/miles-assets}"
CACHES="${CACHES:-$ASSETS/caches}"

mkdir -p "$ENROOT_NVME" "$ENROOT_CACHE_PATH" "$ASSETS" "$ASSETS/hf_cache" \
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
    --env EPO_MODEL="${EPO_MODEL:-olmo3-math-colocate}" \
    --env MILES_NEMOTRONH_KEEP_MTP="${MILES_NEMOTRONH_KEEP_MTP:-}" \
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

        # Pick model by $EPO_MODEL (olmo3-math-colocate | olmo3-sci-colocate):
        # local dir name, HF repo id, megatron model-arg script, and the EPO run
        # script.
        case "${EPO_MODEL}" in
            olmo3-sci-colocate)
                # Olmo3 on SciKnowEval (MCQ), COLOCATE: all 8 GPUs run both the
                # actor and the SGLang engines (time-shared via offload/onload).
                MODEL_DIR=Olmo-3-7B-Instruct
                HF_REPO=allenai/Olmo-3-7B-Instruct
                MODEL_SH=scripts/models/olmo3-7B.sh
                RUN_SH=examples/EPO/run-olmo3-7B-epo-sci-colocate.sh
                DATA_KIND=sci
                ;;
            *)
                # olmo3-math-colocate (default): DAPO math task (dapo grader,
                # AIME-2025 + Minerva eval), COLOCATE.
                MODEL_DIR=Olmo-3-7B-Instruct
                HF_REPO=allenai/Olmo-3-7B-Instruct
                MODEL_SH=scripts/models/olmo3-7B.sh
                RUN_SH=examples/EPO/run-olmo3-7B-epo-math-colocate.sh
                DATA_KIND=dapo
                ;;
        esac

        for name in "$MODEL_DIR" "${MODEL_DIR}_torch_dist" "${MODEL_DIR}_miles" sci dapo-math-17k math_eval; do
            mkdir -p /root/assets/$name
            ln -sfn /root/assets/$name /root/$name
        done

        # Patch the sglang tolist bug in this image (idempotent). Shared,
        # algorithm-agnostic infra — lives under examples/SDPO/, reused as-is.
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
