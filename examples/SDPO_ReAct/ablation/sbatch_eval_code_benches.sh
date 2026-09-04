#!/bin/bash
# ONE sbatch job that runs BOTH held-out code benchmarks for the given
# checkpoint(s) on ONE H200 node, inside the same enroot container the training
# runs use:
#   1. LiveCodeBench-v6 FUNCTIONAL   (../eval-lcb-functional.sh)   63 problems
#   2. OJBench medium (NOI + ICPC)   (../eval-ojbench.sh)          77 problems
# Both are step-0 evals of the loaded weights, with the TRAINING scaffold
# (16384 per-turn budget, 20 turns, thinking, minimal prompt, tool-mandatory
# grading, 8 samples @ temperature 1).
#
# Both in one job on purpose: each bench is ~30-60 min for one 4B/9B checkpoint,
# the cluster is contended, and a second job for the same model would pay the
# whole queue wait + image/container setup again on a different node.
#
# usage (one model, both benches):
#   MODEL_PATHS=/root/data/home-static/data/hf_models/Qwen3.5-9B \
#     sbatch examples/SDPO_ReAct/ablation/sbatch_eval_code_benches.sh
# usage (skip one bench):
#   RUN_LCB=0 MODEL_PATHS=... sbatch .../sbatch_eval_code_benches.sh
#SBATCH --job-name=hx-eval-code-benches
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low,ml.p5en.48xlarge-ultra-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/fsx/home/haoxiang.zhang/logs/hx-eval-code-benches_%j.out
#SBATCH --requeue
set -x

# NOT derived from ${BASH_SOURCE[0]}: sbatch copies this script to a per-job
# spool path before running it.
REPO_ROOT="/fsx/home/haoxiang.zhang/gitproj/miles-sdpo"
cd "$REPO_ROOT"

echo "[$(date -u)] job $SLURM_JOB_ID started on $(hostname)"
nvidia-smi -L || true

# Secrets (WANDB_API_KEY), same gitignored .env chain the enroot launcher uses.
if [ -f examples/SDPO_ReAct/.env ]; then
    set -a; . examples/SDPO_ReAct/.env; set +a
elif [ -f examples/SDPO/.env ]; then
    set -a; . examples/SDPO/.env; set +a
fi

# --- 0. code_interpreter / judge sandbox sidecar, on the HOST ---------------
# Must be started outside enroot (no docker-in-enroot); the enroot session
# shares the host network namespace, so 127.0.0.1:8420 reaches it. run_sandbox.sh
# also REPLACES a sidecar left over from an older job on this node whose
# concurrency cap differs -- which matters here, because an oversubscribed
# sandbox silently turns correct submissions into wall-clock timeouts.
bash examples/SDPO_ReAct/tools/run_sandbox.sh
curl -sf http://127.0.0.1:8420/stats || true

# --- 1..2. enroot image -> squashfs -> container rootfs (per node) ----------
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

if [ ! -f "$SQSH" ]; then
    enroot import -o "$SQSH" "docker://${IMAGE}"
fi
if ! enroot list 2>/dev/null | grep -qx "$CONTAINER"; then
    enroot create --name "$CONTAINER" "$SQSH"
fi

# --- 3. run the evals -------------------------------------------------------
# Paths handed in below are CONTAINER paths (/fsx is not readable from inside
# the session; $DATA_DIR is mounted as /root/data).
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
    --env RUN_LCB="${RUN_LCB:-1}" \
    --env RUN_OJBENCH="${RUN_OJBENCH:-1}" \
    --env MODEL_PATHS="${MODEL_PATHS:-}" \
    --env MODEL_PATH="${MODEL_PATH:-}" \
    --env LCB_FUNC_DATA_DIR="${LCB_FUNC_DATA_DIR:-/root/data/code_data_v6func_all}" \
    --env LCB_FUNC_FILES="${LCB_FUNC_FILES:-}" \
    --env LCB_FUNC_DIFFICULTY="${LCB_FUNC_DIFFICULTY:-easy,medium,hard}" \
    --env LCB_FUNC_MAX_RESPONSE_LEN="${LCB_FUNC_MAX_RESPONSE_LEN:-16384}" \
    --env LCB_REQUIRE_TOOL="${LCB_REQUIRE_TOOL:-}" \
    --env LCB_FUNC_WANDB_GROUP="${LCB_FUNC_WANDB_GROUP:-lcb-v6-functional-eval}" \
    --env OJBENCH_DATA_DIR="${OJBENCH_DATA_DIR:-/root/data/home-static/data/ojbench_medium}" \
    --env OJBENCH_DIFFICULTY="${OJBENCH_DIFFICULTY:-medium}" \
    --env OJBENCH_MAX_RESPONSE_LEN="${OJBENCH_MAX_RESPONSE_LEN:-16384}" \
    --env OJBENCH_REQUIRE_TOOL="${OJBENCH_REQUIRE_TOOL:-}" \
    --env OJBENCH_WANDB_GROUP="${OJBENCH_WANDB_GROUP:-ojbench-medium-eval}" \
    --env SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-}" \
    --env SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-}" \
    --env SDPO_REACT_TP="${SDPO_REACT_TP:-}" \
    --env SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-}" \
    "$CONTAINER" \
    bash -c '
        cd /root/miles
        # Drop the empty pass-throughs: the eval scripts use ${VAR:-default},
        # which an exported-but-empty VAR would defeat.
        for v in MODEL_PATHS MODEL_PATH LCB_FUNC_FILES LCB_REQUIRE_TOOL \
                 OJBENCH_REQUIRE_TOOL SDPO_REACT_EVAL_N_SAMPLES \
                 SDPO_REACT_EVAL_MAX_TURNS SDPO_REACT_TP SGLANG_MEM_FRACTION; do
            [ -z "${!v}" ] && unset "$v"
        done
        RC=0
        # Run BOTH benches even if the first one fails: a missing torch_dist or a
        # bad checkpoint for one bench says nothing about the other, and a half
        # -populated results table is worse than two independent verdicts.
        if [ "${RUN_LCB}" = "1" ]; then
            bash examples/SDPO_ReAct/ablation/eval-lcb-functional.sh || RC=$?
            echo "[bench] lcb-functional rc=$RC"
        fi
        if [ "${RUN_OJBENCH}" = "1" ]; then
            bash examples/SDPO_ReAct/ablation/eval-ojbench.sh || RC=$?
            echo "[bench] ojbench rc=$RC"
        fi
        exit $RC
    '
RC=$?

echo "[$(date -u)] evals rc=$RC"
curl -sf http://127.0.0.1:8420/stats || true
ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
pkill -9 -f 'sglang::' 2>/dev/null || true
exit $RC
