#!/bin/bash
# Run BOTH held-out code benchmarks for all 14 checkpoints SEQUENTIALLY on the
# node this script is launched from -- no sbatch, no queue.
#
# Why not sbatch (see sbatch_eval_code_benches.sh, kept for the case where no
# node is held): this node is already allocated to a long-lived holder job
# (h200_4d_nooc.sbatch, 4 days, 8 idle H200s), so queueing 14 more jobs only
# added 14 PENDING entries competing with the holder for the same 5-node
# partition. Running in-place reuses the warm enroot container, the warm
# sandbox sidecar and the page cache, and keeps squeue clean.
#
# PHASE ORDER IS DELIBERATE: LiveCodeBench-v6 functional for EVERY model first,
# then OJBench medium for every model. LCB-v6-functional is the primary
# deliverable, so it must be complete even if the whole run is interrupted
# halfway; OJBench is the secondary one.
#
# Each (model, bench) pair is one `enroot start` of the training container
# running the normal eval script, i.e. the TRAINING scaffold: 16384 per-turn
# budget, 20 turns, thinking, minimal prompt, tool-mandatory grading, 8 samples
# at temperature 1, and exactly ONE step-0 eval (--eval-interval 2, see
# eval-lcb-functional.sh for why not 1).
#
# The 12 trained arms need a Megatron torch_dist checkpoint that a separate
# conversion job writes; rather than a hard dependency, each model WAITS for its
# own <model>_torch_dist/latest_checkpointed_iteration.txt to appear (up to
# WAIT_TORCH_DIST_MINUTES) and is skipped if it never does. So this can be
# started while the conversion is still running.
#
# usage:
#   nohup bash examples/SDPO_ReAct/ablation/run_code_benches_local.sh \
#     > /fsx/home/haoxiang.zhang/logs/local-benches/driver.log 2>&1 &
#
# env:
#   MODELS      space-separated basenames under $MODEL_STORE_HOST (default: the 14)
#   BENCHES     space-separated: lcb ojbench (default "lcb ojbench", in order)
#   WAIT_TORCH_DIST_MINUTES  (default 90)
#   LOG_DIR     (default /fsx/home/haoxiang.zhang/logs/local-benches)
set -u

REPO_ROOT="/fsx/home/haoxiang.zhang/gitproj/miles-sdpo"
cd "$REPO_ROOT"

MODEL_STORE_HOST="/fsx/data/haoxiang.zhang/home-static/data/hf_models"
MODEL_STORE_CTR="/root/data/home-static/data/hf_models"
LOG_DIR="${LOG_DIR:-/fsx/home/haoxiang.zhang/logs/local-benches}"
WAIT_TORCH_DIST_MINUTES="${WAIT_TORCH_DIST_MINUTES:-90}"
BENCHES="${BENCHES:-lcb ojbench}"
mkdir -p "$LOG_DIR"

# Bases first (already converted, so results start landing immediately), then
# the 12 ablation arms from ./todo.
MODELS="${MODELS:-\
Qwen3.5-4B \
Qwen3.5-9B \
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
Qwen3.5-9B-sdpo-react-mathcodesearch-grpo-arm-e}"

# Secrets (WANDB_API_KEY), same gitignored .env chain the enroot launcher uses.
if [ -f examples/SDPO_ReAct/.env ]; then
    set -a; . examples/SDPO_ReAct/.env; set +a
elif [ -f examples/SDPO/.env ]; then
    set -a; . examples/SDPO/.env; set +a
fi

IMAGE="${IMAGE:-radixark/miles:dev-cu12-202607040446}"
ENROOT_NVME="${ENROOT_NVME:-/opt/dlami/nvme/miles-enroot}"
SQSH="${SQSH:-$ENROOT_NVME/miles-0704.sqsh}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$ENROOT_NVME/cache}"
CONTAINER="${CONTAINER:-miles-sdpo-react-cu12}"
ASSETS="${ASSETS:-/opt/dlami/nvme/miles-assets}"
CACHES="${CACHES:-$ASSETS/caches}"
DATA_DIR="${DATA_DIR:-/fsx/data/$USER}"

mkdir -p "$ENROOT_NVME" "$ENROOT_CACHE_PATH" "$ASSETS" "$ASSETS/hf_cache" \
    "$CACHES/triton" "$CACHES/inductor" "$CACHES/torch_extensions" "$CACHES/nv"
[ -f "$SQSH" ] || enroot import -o "$SQSH" "docker://${IMAGE}"
enroot list 2>/dev/null | grep -qx "$CONTAINER" || enroot create --name "$CONTAINER" "$SQSH"

# --- sandbox sidecar, once, on the HOST -------------------------------------
# No docker-in-enroot; the enroot session shares the host network namespace, so
# 127.0.0.1:8420 reaches it. run_sandbox.sh REPLACES a sidecar left over from an
# earlier job whose concurrency cap differs -- which matters a lot here: an
# oversubscribed sandbox silently turns correct submissions into wall-clock
# timeouts (5.6% vs 18.5% solved on identical OJBench candidates).
bash examples/SDPO_ReAct/tools/run_sandbox.sh
SANDBOX_STATS="$(curl -sf http://127.0.0.1:8420/stats || echo unreachable)"
echo "[driver] sandbox stats: ${SANDBOX_STATS}"
# Hard gate, not a warning: the whole point of this run is that the sidecar
# enforces a concurrency cap. A sidecar without one floors every code score
# (~3x on OJBench), and a 14-model table measured that way is worthless -- so
# refuse to start rather than burn 20 GPU-hours on numbers we'd have to redo.
case "$SANDBOX_STATS" in
    *max_concurrency*) : ;;
    *) echo "[driver] FATAL: sandbox sidecar has no max_concurrency (stale server?)." >&2
       echo "[driver] Fix: docker rm -f sdpo-react-sandbox && bash examples/SDPO_ReAct/tools/run_sandbox.sh" >&2
       exit 1 ;;
esac

wait_for_torch_dist() {
    local name="$1"
    local marker="${MODEL_STORE_HOST}/${name}_torch_dist/latest_checkpointed_iteration.txt"
    local deadline=$(( $(date +%s) + WAIT_TORCH_DIST_MINUTES * 60 ))
    [ -f "$marker" ] && return 0
    echo "[driver] waiting for ${name}_torch_dist (up to ${WAIT_TORCH_DIST_MINUTES}m)..."
    while [ "$(date +%s)" -lt "$deadline" ]; do
        sleep 60
        [ -f "$marker" ] && { echo "[driver] ${name}_torch_dist ready"; return 0; }
    done
    return 1
}

run_one() {
    local bench="$1" name="$2"
    local script env_name
    case "$bench" in
        lcb)     script=examples/SDPO_ReAct/ablation/eval-lcb-functional.sh ;;
        ojbench) script=examples/SDPO_ReAct/ablation/eval-ojbench.sh ;;
        *) echo "[driver] unknown bench '$bench'" >&2; return 2 ;;
    esac
    local log="${LOG_DIR}/${bench}_${name}_$(date +%Y%m%d_%H%M%S).log"
    echo "[driver] === ${bench} / ${name} -> ${log}"

    # Anything left over from a previous (possibly killed) eval would make ray
    # start fail or, worse, hold GPU memory the next SGLang engine needs.
    ray stop --force >/dev/null 2>&1 || true
    pkill -9 -f 'ray::' >/dev/null 2>&1 || true
    pkill -9 -f 'sglang::' >/dev/null 2>&1 || true
    sleep 3

    enroot start --rw \
        --mount "$REPO_ROOT":/root/miles \
        --mount "$ASSETS":/root/assets \
        --mount "$ASSETS/hf_cache":/root/hf_cache \
        --mount "$CACHES":/root/caches \
        --mount "$DATA_DIR":/root/data \
        --env HF_HOME=/root/hf_cache \
        --env TRITON_CACHE_DIR=/root/caches/triton \
        --env TORCHINDUCTOR_CACHE_DIR=/root/caches/inductor \
        --env TORCH_EXTENSIONS_DIR=/root/caches/torch_extensions \
        --env CUDA_CACHE_PATH=/root/caches/nv \
        --env SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
        --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
        --env MODEL_PATH="${MODEL_STORE_CTR}/${name}" \
        --env LCB_FUNC_DATA_DIR="${LCB_FUNC_DATA_DIR:-/root/data/code_data_v6func_all}" \
        --env LCB_FUNC_DIFFICULTY="${LCB_FUNC_DIFFICULTY:-easy,medium,hard}" \
        --env LCB_FUNC_MAX_RESPONSE_LEN="${LCB_FUNC_MAX_RESPONSE_LEN:-16384}" \
        --env OJBENCH_DATA_DIR="${OJBENCH_DATA_DIR:-/root/data/home-static/data/ojbench_medium}" \
        --env OJBENCH_DIFFICULTY="${OJBENCH_DIFFICULTY:-medium}" \
        --env OJBENCH_MAX_RESPONSE_LEN="${OJBENCH_MAX_RESPONSE_LEN:-16384}" \
        "$CONTAINER" \
        bash -c "cd /root/miles && bash ${script}" > "$log" 2>&1
    local rc=$?
    echo "[driver] === ${bench} / ${name} rc=${rc} ($(date -u))"
    return $rc
}

echo "[driver] start $(date -u) on $(hostname); benches='${BENCHES}'"
SUMMARY=""
for bench in $BENCHES; do
    for name in $MODELS; do
        if [ ! -d "${MODEL_STORE_HOST}/${name}" ]; then
            echo "[driver] SKIP ${name}: no HF checkpoint"
            SUMMARY="${SUMMARY}\n${bench} ${name} SKIP-no-hf"
            continue
        fi
        if ! wait_for_torch_dist "$name"; then
            echo "[driver] SKIP ${name}: no torch_dist after ${WAIT_TORCH_DIST_MINUTES}m"
            SUMMARY="${SUMMARY}\n${bench} ${name} SKIP-no-torch-dist"
            continue
        fi
        # Never let one bad checkpoint stop the other 13: a failure says nothing
        # about the rest, and a partial table is the thing we are trying to avoid.
        run_one "$bench" "$name"
        SUMMARY="${SUMMARY}\n${bench} ${name} rc=$?"
    done
done

ray stop --force >/dev/null 2>&1 || true
echo "[driver] done $(date -u)"
echo -e "[driver] summary:${SUMMARY}"
curl -sf http://127.0.0.1:8420/stats || true
