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

# The system enroot.conf points ENROOT_CACHE_PATH at /fsx/enroot, but /fsx itself
# is NOT a Lustre mount here — it lives on the tiny (97G, OS-filled) root disk.
# Left alone, `enroot import` downloads image layers onto root and dies with
# "zstd: No space left on device". Force the layer cache onto NVMe like every
# other enroot path. (export so the enroot subprocess inherits it.)
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$ENROOT_NVME/cache}"
CONTAINER="${CONTAINER:-miles-0704}"
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
    --env SDPO_MODEL="${SDPO_MODEL:-olmo3-math-colocate}" \
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

        # Pick model by $SDPO_MODEL
        # (qwen3 | olmo3 | olmo3-sci | olmo3-sci-colocate | olmo3-math-colocate |
        #  nemo-sci | nemo-sci-colocate):
        # local dir name, HF repo id, megatron model-arg script, and the SDPO run
        # script. USE_BRIDGE=1 marks models loaded via AutoBridge straight from the
        # HF checkpoint (no offline _torch_dist conversion needed); default 0.
        USE_BRIDGE=0
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
                # DISAGGREGATED 4+4 (4 train GPUs, 4 SGLang rollout GPUs).
                MODEL_DIR=Olmo-3-7B-Instruct
                HF_REPO=allenai/Olmo-3-7B-Instruct
                MODEL_SH=scripts/models/olmo3-7B.sh
                RUN_SH=examples/SDPO/run-olmo3-7B-sdpo-sci.sh
                DATA_KIND=sci
                ;;
            olmo3-sci-colocate)
                # Same model/task/SDPO config as olmo3-sci, but COLOCATE: all 8 GPUs
                # run both the actor and the SGLang engines (time-shared via
                # offload/onload). The serial generate->train loop leaves half the
                # node idle under 4+4; colocate uses all 8 in both phases -> ~2x
                # rollout throughput. SDPO logic is unchanged (teacher weight swaps
                # are training-side, decoupled from the rollout transport).
                MODEL_DIR=Olmo-3-7B-Instruct
                HF_REPO=allenai/Olmo-3-7B-Instruct
                MODEL_SH=scripts/models/olmo3-7B.sh
                RUN_SH=examples/SDPO/run-olmo3-7B-sdpo-sci-colocate.sh
                DATA_KIND=sci
                ;;
            olmo3-math-colocate)
                # Same colocate / SDPO / skill / pitfall config as olmo3-sci-colocate,
                # but on the DAPO math task (dapo grader, AIME-2025 + Minerva eval).
                MODEL_DIR=Olmo-3-7B-Instruct
                HF_REPO=allenai/Olmo-3-7B-Instruct
                MODEL_SH=scripts/models/olmo3-7B.sh
                RUN_SH=examples/SDPO/run-olmo3-7B-sdpo-math-colocate.sh
                DATA_KIND=dapo
                ;;
            nemo-sci)
                # Nemotron-3-Nano-4B (DENSE nemotron_h = hybrid Mamba+Attention) on
                # the SciKnowEval (MCQ) task. Loads via AutoBridge (the run script
                # passes --megatron-to-hf-mode bridge), so NO _torch_dist conversion.
                # DISAGGREGATED 4+4 (4 train GPUs, 4 SGLang rollout GPUs).
                MODEL_DIR=NVIDIA-Nemotron-3-Nano-4B-BF16
                HF_REPO=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
                MODEL_SH=scripts/models/nemotron-3-nano-4b.sh
                RUN_SH=examples/SDPO/run-nemotron3-4b-sdpo-sci.sh
                DATA_KIND=sci
                USE_BRIDGE=1
                ;;
            nemo-sci-colocate)
                # Same model/task/SDPO config as nemo-sci, but COLOCATE: all 8 GPUs
                # run both the actor and the SGLang engines (time-shared via
                # offload/onload). The serial generate->train loop leaves half the
                # node idle under 4+4; colocate uses all 8 in both phases, so a 4B
                # model gets ~2x rollout throughput. SDPO logic is unchanged (teacher
                # weight swaps are training-side, decoupled from the rollout transport).
                MODEL_DIR=NVIDIA-Nemotron-3-Nano-4B-BF16
                HF_REPO=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
                MODEL_SH=scripts/models/nemotron-3-nano-4b.sh
                RUN_SH=examples/SDPO/run-nemotron3-4b-sdpo-sci-colocate.sh
                DATA_KIND=sci
                USE_BRIDGE=1
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

        # AutoBridge models (USE_BRIDGE=1, e.g. nemotron3-4b) build the Megatron
        # provider from HF config.json at load time, so they skip this offline
        # HF -> torch_dist conversion and ref-load the HF checkpoint directly.
        if [ "$USE_BRIDGE" != "1" ] && [ -z "$(ls -A /root/${MODEL_DIR}_torch_dist 2>/dev/null)" ]; then
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
