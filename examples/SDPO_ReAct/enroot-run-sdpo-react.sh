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
# The search/open/find sidecar (tools/search/docker/, port 8421) and the
# wiki-18 retriever it routes onto (tools/search/run_retrieval.sh, port 8000)
# are STARTED BY THE RUN SCRIPT ITSELF (run-qwen3-4B-sdpo-react-native.sh),
# conditionally on $SDPO_REACT_DOMAIN -- not unconditionally here, since only
# the multitask/search domains need them and the retriever needs its own GPU
# reservation, which is a per-run decision, not a per-enroot-session one.
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
#   SDPO_REACT_RUN_FAMILY (default native)          native (math/code/search, run-*-native.sh)
#                                                    | agentic (webshop/alfworld, run-*-agentic.sh)
#                                                    | tau2 (retail/airline/telecom, run-*-tau2.sh)
#                                                    -- orthogonal to SDPO_REACT_MODEL (which
#                                                    picks weights/tool-grammar within EITHER
#                                                    family); each model-family case branch below
#                                                    sets *_NATIVE_RUN_SH/*_AGENTIC_RUN_SH/
#                                                    *_TAU2_RUN_SH, this var picks which one
#                                                    actually runs (tau2 is only wired for
#                                                    qwen3-native so far, same bootstrap-scope
#                                                    as agentic's own initial wiring).
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
# Salesforce Research LLM gateway credentials (SFT_GATEWAY_KEY, OPENAI_API_URL)
# for --sdpo-judge / --sdpo-search-judge-fallback -- lives in a separate,
# ALSO-gitignored .env outside this repo (shared across projects on this
# host), not duplicated into examples/SDPO_ReAct/.env.
if [ -f "$HOME/gitproj/apis/.env" ]; then
    set -a; . "$HOME/gitproj/apis/.env"; set +a
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
    --env SDPO_REACT_RUN_FAMILY="${SDPO_REACT_RUN_FAMILY:-}" \
    --env SDPO_REACT_DOMAIN="${SDPO_REACT_DOMAIN:-}" \
    --env SDPO_REACT_WEBSHOP_SIDECAR_URL="${SDPO_REACT_WEBSHOP_SIDECAR_URL:-}" \
    --env SDPO_REACT_ALFWORLD_SIDECAR_URL="${SDPO_REACT_ALFWORLD_SIDECAR_URL:-}" \
    --env SDPO_REACT_AGENTIC_PER_DOMAIN="${SDPO_REACT_AGENTIC_PER_DOMAIN:-}" \
    --env SDPO_REACT_WEBSHOP_N_TRAIN="${SDPO_REACT_WEBSHOP_N_TRAIN:-}" \
    --env SDPO_REACT_WEBSHOP_N_EVAL="${SDPO_REACT_WEBSHOP_N_EVAL:-}" \
    --env SDPO_REACT_ALFWORLD_N_TRAIN="${SDPO_REACT_ALFWORLD_N_TRAIN:-}" \
    --env SDPO_REACT_ALFWORLD_N_EVAL_ID="${SDPO_REACT_ALFWORLD_N_EVAL_ID:-}" \
    --env SDPO_REACT_ALFWORLD_N_EVAL_OOD="${SDPO_REACT_ALFWORLD_N_EVAL_OOD:-}" \
    --env SDPO_REACT_TAU2_SIDECAR_URL="${SDPO_REACT_TAU2_SIDECAR_URL:-}" \
    --env SDPO_REACT_TAU2_MAX_STEPS="${SDPO_REACT_TAU2_MAX_STEPS:-}" \
    --env SDPO_REACT_TAU2_N_EVAL_PER_DOMAIN="${SDPO_REACT_TAU2_N_EVAL_PER_DOMAIN:-}" \
    --env TAU_USER_MODEL_PROVIDER="${TAU_USER_MODEL_PROVIDER:-}" \
    --env TAU_USER_MODEL="${TAU_USER_MODEL:-}" \
    --env GEMINI_API_KEY="${GEMINI_API_KEY:-}" \
    --env DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}" \
    --env SDPO_REACT_ARM="${SDPO_REACT_ARM:-}" \
    --env SDPO_REACT_PROMPT="${SDPO_REACT_PROMPT:-}" \
    --env SDPO_REACT_NUM_ROLLOUT="${SDPO_REACT_NUM_ROLLOUT:-}" \
    --env SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-}" \
    --env SDPO_REACT_PURE_DISTILL="${SDPO_REACT_PURE_DISTILL:-}" \
    --env SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-}" \
    --env SDPO_REACT_MT_PER_DOMAIN="${SDPO_REACT_MT_PER_DOMAIN:-}" \
    --env SDPO_REACT_MIN_CORRECT="${SDPO_REACT_MIN_CORRECT:-}" \
    --env SDPO_REACT_DYNAMIC_SAMPLE="${SDPO_REACT_DYNAMIC_SAMPLE:-}" \
    --env SDPO_REACT_ROLLOUT_BATCH="${SDPO_REACT_ROLLOUT_BATCH:-}" \
    --env SDPO_REACT_TP="${SDPO_REACT_TP:-}" \
    --env SDPO_REACT_MAX_TOKENS_PER_GPU="${SDPO_REACT_MAX_TOKENS_PER_GPU:-}" \
    --env SDPO_REACT_EP_SIZE="${SDPO_REACT_EP_SIZE:-}" \
    --env SDPO_REACT_R3="${SDPO_REACT_R3:-}" \
    --env SDPO_REACT_OPT_CPU_OFFLOAD="${SDPO_REACT_OPT_CPU_OFFLOAD:-}" \
    --env SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-}" \
    --env SDPO_REACT_LOGPROBS_CHUNK="${SDPO_REACT_LOGPROBS_CHUNK:-}" \
    --env SDPO_REACT_DIST_TIMEOUT_MIN="${SDPO_REACT_DIST_TIMEOUT_MIN:-}" \
    --env SDPO_REACT_SAVE_INTERVAL="${SDPO_REACT_SAVE_INTERVAL:-}" \
    --env SDPO_REACT_ASYNC_SAVE="${SDPO_REACT_ASYNC_SAVE:-}" \
    --env SDPO_REACT_SKIP_EVAL0="${SDPO_REACT_SKIP_EVAL0:-}" \
    --env SDPO_REACT_SAVE_CKPT="${SDPO_REACT_SAVE_CKPT:-}" \
    --env SDPO_REACT_NOTE="${SDPO_REACT_NOTE:-}" \
    --env SDPO_REACT_TRAIN_MAX_TURNS="${SDPO_REACT_TRAIN_MAX_TURNS:-}" \
    --env SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-}" \
    --env HF_HOME=/root/hf_cache \
    --env TRITON_CACHE_DIR=/root/caches/triton \
    --env TORCHINDUCTOR_CACHE_DIR=/root/caches/inductor \
    --env TORCH_EXTENSIONS_DIR=/root/caches/torch_extensions \
    --env CUDA_CACHE_PATH=/root/caches/nv \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env SFT_GATEWAY_KEY="${SFT_GATEWAY_KEY:-}" \
    --env OPENAI_API_URL="${OPENAI_API_URL:-}" \
    --env SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    "$CONTAINER" \
    bash -euxc '
        cd /root/miles

        # Pick model by $SDPO_REACT_MODEL (qwen2.5 | olmo3): local dir name, HF
        # repo id, megatron model-arg script, and the SDPO_ReAct run script --
        # same switch pattern as examples/SDPO/enroot-run-sdpo.sh'"'"'s SDPO_MODEL.
        # Each branch sets *_NATIVE_RUN_SH/*_AGENTIC_RUN_SH/*_TAU2_RUN_SH (the
        # latter two fall back to the native script wherever no variant
        # exists yet, e.g. olmo3/Qwen2.5 -- SDPO_REACT_RUN_FAMILY=agentic|tau2
        # is only actually validated for qwen3-native so far); the final
        # RUN_SH switch below picks between them via SDPO_REACT_RUN_FAMILY.
        case "${SDPO_REACT_MODEL}" in
            olmo3)
                MODEL_DIR=Olmo-3-7B-Instruct
                HF_REPO=allenai/Olmo-3-7B-Instruct
                MODEL_SH=scripts/models/olmo3-7B.sh
                NATIVE_RUN_SH=examples/SDPO_ReAct/run-olmo3-7B-sdpo-react-dapo-math.sh
                AGENTIC_RUN_SH="$NATIVE_RUN_SH"
                TAU2_RUN_SH="$NATIVE_RUN_SH"
                ;;
            qwen3-native)
                # Qwen3-4B, NATIVE <tool_call> multi-turn rollout + SDPO
                # self-skill with env_feedback prefix -- the proposal-validation
                # run (see run-qwen3-4B-sdpo-react-native.sh header). Uses the
                # model-native tool-calling grammar (qwen25 JSON-in-tags), NOT the
                # plain-text tags the Qwen2.5 leg teaches.
                MODEL_DIR=Qwen3-4B
                HF_REPO=Qwen/Qwen3-4B
                MODEL_SH=scripts/models/qwen3-4B.sh
                NATIVE_RUN_SH=examples/SDPO_ReAct/run-qwen3-4B-sdpo-react-native.sh
                AGENTIC_RUN_SH=examples/SDPO_ReAct/run-qwen3-4B-sdpo-react-agentic.sh
                TAU2_RUN_SH=examples/SDPO_ReAct/run-qwen3-4B-sdpo-react-tau2.sh
                ;;
            qwen3.5-native)
                # Qwen3.5-4B, SAME native multi-turn SDPO-ReAct run script, but
                # the tool-call grammar is qwen3_coder XML (<function=..>
                # <parameter=..>) instead of qwen25 JSON-in-tags. The run script
                # is parameterised by SDPO_REACT_MODEL (model paths + parser +
                # --sdpo-tool-grammar), so no fork. Hybrid linear-attention arch;
                # SGLang rollout engine runs tp=1 (tp>1 hit NCCL; 4B fits 1 GPU).
                MODEL_DIR=Qwen3.5-4B
                HF_REPO=Qwen/Qwen3.5-4B
                MODEL_SH=scripts/models/qwen3.5-4B.sh
                NATIVE_RUN_SH=examples/SDPO_ReAct/run-qwen3-4B-sdpo-react-native.sh
                AGENTIC_RUN_SH=examples/SDPO_ReAct/run-qwen3-4B-sdpo-react-agentic.sh
                # tau2 run script only wires the qwen3 TITO tokenizer type
                # (see that script own header) -- no qwen3.5 variant yet.
                TAU2_RUN_SH="$NATIVE_RUN_SH"
                ;;
            qwen3.5-27B)
                # Qwen3.5-27B DENSE. Same native SDPO-ReAct arms as the 4B run,
                # but the large-model launcher (TP=4, CPU-offloaded optimizer).
                # MODEL_SH here is only for the HF->torch_dist CONVERT below; the
                # run script re-sources it (and sets its own parallelism).
                MODEL_DIR=Qwen3.5-27B
                HF_REPO=Qwen/Qwen3.5-27B
                MODEL_SH=scripts/models/qwen3.5-27B.sh
                NATIVE_RUN_SH=examples/SDPO_ReAct/run-qwen3.5-27B-35BA3B-sdpo-react-native.sh
                AGENTIC_RUN_SH="$NATIVE_RUN_SH"
                TAU2_RUN_SH="$NATIVE_RUN_SH"
                # the large-model run script switches on SDPO_REACT_MODEL, so
                # normalize it to the value that script expects.
                export SDPO_REACT_MODEL=qwen3.5-27B
                ;;
            qwen3.5-35B-A3B)
                # Qwen3.5-35B-A3B MoE (256 experts, top-8, ~3B active). Same arms;
                # the large-model launcher sets EP=8 + R3 rollout-routing-replay
                # (train/inference router alignment). MODEL_SH drives the convert.
                MODEL_DIR=Qwen3.5-35B-A3B
                HF_REPO=Qwen/Qwen3.5-35B-A3B
                MODEL_SH=scripts/models/qwen3.5-35B-A3B.sh
                NATIVE_RUN_SH=examples/SDPO_ReAct/run-qwen3.5-27B-35BA3B-sdpo-react-native.sh
                AGENTIC_RUN_SH="$NATIVE_RUN_SH"
                TAU2_RUN_SH="$NATIVE_RUN_SH"
                export SDPO_REACT_MODEL=qwen3.5-35B-A3B
                ;;
            *)
                MODEL_DIR=Qwen2.5-7B-Instruct
                HF_REPO=Qwen/Qwen2.5-7B-Instruct
                MODEL_SH=scripts/models/qwen2.5-7B.sh
                NATIVE_RUN_SH=examples/SDPO_ReAct/run-qwen2.5-7B-sdpo-react-dapo-math.sh
                AGENTIC_RUN_SH="$NATIVE_RUN_SH"
                TAU2_RUN_SH="$NATIVE_RUN_SH"
                ;;
        esac
        case "${SDPO_REACT_RUN_FAMILY:-native}" in
            agentic) RUN_SH="$AGENTIC_RUN_SH" ;;
            tau2) RUN_SH="$TAU2_RUN_SH" ;;
            *) RUN_SH="$NATIVE_RUN_SH" ;;
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
