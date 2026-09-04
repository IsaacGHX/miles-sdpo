#!/bin/bash
# ONE sbatch job that runs the LiveCodeBench-v6 FUNCTIONAL (leetcode-style)
# eval (../eval-lcb-functional.sh) for BOTH base models sequentially on ONE
# H200 node, inside the same enroot container the training runs use.
#
# WHY sbatch and not just `enroot start` on the login/interactive node: the
# ablation queue jobs (sbatch_run_*_*.sh) hold all 8 GPUs of whatever node they
# land on for days, and they start the next combo within seconds of the
# previous one finishing -- so a hand-launched eval on that node either finds no
# free GPUs or races the queue for them (observed: an eval submitted into a
# ~20s gap between combos lost the GPUs to the next combo). Asking slurm for a
# node of our own is the only conflict-free way to run this.
#
# Unlike the sibling queue scripts this does NOT go through
# ../enroot-run-sdpo-react.sh: that launcher is a per-model TRAINING launcher
# (it picks weights/run-script from SDPO_REACT_MODEL and ends in a run-*.sh),
# while the eval script is model-agnostic and loops over MODEL_PATHS itself.
# The enroot plumbing below is the same as that launcher's (same image, same
# squashfs path, same mounts), just pointed at the eval script.
#
# usage:
#   sbatch examples/SDPO_ReAct/ablation/sbatch_eval_lcb_functional.sh
#   MODEL_PATHS="/root/data/home-static/data/hf_models/Qwen3.5-9B" \
#     sbatch examples/SDPO_ReAct/ablation/sbatch_eval_lcb_functional.sh
#SBATCH --job-name=hx-eval-lcb-functional
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low,ml.p5en.48xlarge-ultra-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --output=/fsx/home/haoxiang.zhang/logs/hx-eval-lcb-functional_%j.out
#SBATCH --requeue
set -x

# NOT derived from ${BASH_SOURCE[0]}: sbatch copies this script to a per-job
# spool path before running it (see sbatch_run_4B_alfworld_webshop.sh).
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
# shares the host network namespace, so 127.0.0.1:8420 reaches it. On a node
# that has never run this, run_sandbox.sh builds the image first (~minutes).
bash examples/SDPO_ReAct/tools/run_sandbox.sh

# --- 1..2. enroot image -> squashfs -> container rootfs (per node) ----------
# All three live on the node-local NVMe, so a node that has never run this job
# family pays a one-time image import here.
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

# --- 3. run the eval ------------------------------------------------------
# Paths handed in below are CONTAINER paths (/fsx is not readable from inside
# the session; $DATA_DIR is mounted as /root/data). The eval jsonl and both
# base checkpoints live under $DATA_DIR, i.e. on shared storage -- nothing
# node-local is needed, so this runs on whatever node slurm gives us.
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
    --env MODEL_PATHS="${MODEL_PATHS:-}" \
    --env MODEL_PATH="${MODEL_PATH:-}" \
    --env LCB_FUNC_DATA_DIR="${LCB_FUNC_DATA_DIR:-}" \
    --env LCB_FUNC_FILES="${LCB_FUNC_FILES:-}" \
    --env LCB_FUNC_DIFFICULTY="${LCB_FUNC_DIFFICULTY:-}" \
    --env LCB_REQUIRE_TOOL="${LCB_REQUIRE_TOOL:-}" \
    --env LCB_FUNC_MAX_RESPONSE_LEN="${LCB_FUNC_MAX_RESPONSE_LEN:-}" \
    --env LCB_FUNC_WANDB_GROUP="${LCB_FUNC_WANDB_GROUP:-lcb-v6-functional-eval}" \
    --env SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-}" \
    --env SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-}" \
    --env SDPO_REACT_TP="${SDPO_REACT_TP:-}" \
    --env SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-}" \
    "$CONTAINER" \
    bash -c '
        cd /root/miles
        # Drop the empty pass-throughs: the eval script uses ${VAR:-default},
        # which an exported-but-empty VAR would defeat.
        for v in MODEL_PATHS MODEL_PATH LCB_FUNC_DATA_DIR LCB_FUNC_FILES \
                 LCB_FUNC_DIFFICULTY LCB_REQUIRE_TOOL LCB_FUNC_MAX_RESPONSE_LEN \
                 SDPO_REACT_EVAL_N_SAMPLES \
                 SDPO_REACT_EVAL_MAX_TURNS SDPO_REACT_TP SGLANG_MEM_FRACTION; do
            [ -z "${!v}" ] && unset "$v"
        done
        bash examples/SDPO_ReAct/ablation/eval-lcb-functional.sh
    '
RC=$?

echo "[$(date -u)] eval rc=$RC"
ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
pkill -9 -f 'sglang::' 2>/dev/null || true
exit $RC
