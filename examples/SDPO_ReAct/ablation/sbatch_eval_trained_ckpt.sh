#!/bin/bash
# Final held-out report for ONE TRAINED checkpoint, on one H200 node, inside the
# same enroot container the training runs use:
#   1. MATH  (../eval-amo-bench.sh + data/eval_math_final.yaml)
#            AIME24 (30) + AIME25 (30) + AMO-Bench (50)  -- one sglang boot
#   2. CODE  (../eval-ojbench.sh)   OJBench medium (77: 52 NOI + 25 ICPC)
# Both are step-0 evals of the loaded weights with the TRAINING scaffold (16384
# per-turn budget, 20 turns, thinking on, minimal prompt, 8 samples @ temp 1).
#
# WHY THIS EXISTS next to sbatch_eval_code_benches.sh: that one evaluates HF
# checkpoint dirs (MODEL_PATH + a sibling <MODEL_PATH>_torch_dist), which means a
# trained run first has to be converted megatron->HF->torch_dist. This one loads
# a training run's `--save` dist-ckpt DIRECTLY via SDPO_EVAL_REF_LOAD -- see the
# REF_LOAD block in eval-ojbench.sh: arguments.py copies ref_load into args.load
# with no_load_optim/no_load_rng/finetune, so the trained weights load and the
# optimizer state in the checkpoint is ignored. MODEL_PATH here is only the BASE
# model, supplying tokenizer/config; SDPO_EVAL_TAG is what names the results.
#
# Required env vars:
#   CKPT           container path of the trained dist-ckpt dir (the run's --save
#                  dir, containing iter_NNNNNNN/ + latest_checkpointed_iteration
#                  .txt). Paths are CONTAINER paths: /root/data == /fsx/data/$USER.
#   TAG            short name for this checkpoint (dump dirs, wandb groups)
#
# Optional env vars:
#   MODEL_PATH     (default /root/data/home-static/data/hf_models/Qwen3.5-9B)
#                  BASE HF dir for tokenizer/config; its basename also selects
#                  scripts/models/*.sh, so it must match the trained arch.
#   RUN_MATH / RUN_OJBENCH   (default 1/1) skip a half
#   SDPO_REACT_EVAL_N_SAMPLES (default 8), SDPO_REACT_EVAL_MAX_TURNS (default 20)
#   SDPO_REACT_TP  (default 2)
#   OJBENCH_DATA_DIR (default /root/data/home-static/data/ojbench_medium)
#   AMO_BENCH_DATA_DIR (default /root/data/home-static/data/amo_bench)
#   SDPO_REACT_EVAL_DIR (default /root/data/math_eval/native-minimal) where
#                  aime24_native.jsonl / aime25_native.jsonl live. Build with
#                  SDPO_REACT_PROMPT=minimal python -m
#                  examples.SDPO_ReAct.data.build_native_eval --out-dir <dir>
#                  -- the system prompt is baked in, so it must match the arm's.
#
# usage:
#   CKPT=/root/data/sdpo_ckpts/<run>_step29_ckpt TAG=9B-grpo-e-step29 \
#     sbatch examples/SDPO_ReAct/ablation/sbatch_eval_trained_ckpt.sh
#   # or directly on an already-allocated node:
#   CKPT=... TAG=... bash examples/SDPO_ReAct/ablation/sbatch_eval_trained_ckpt.sh
#SBATCH --job-name=hx-eval-trained-ckpt
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low,ml.p5en.48xlarge-ultra-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/fsx/home/haoxiang.zhang/logs/hx-eval-trained-ckpt_%j.out
#SBATCH --requeue
set -x

: "${CKPT:?Set CKPT to the trained dist-ckpt dir (CONTAINER path)}"
: "${TAG:?Set TAG to a short name for this checkpoint}"

# NOT derived from ${BASH_SOURCE[0]}: sbatch copies this script to a per-job
# spool path before running it.
REPO_ROOT="/fsx/home/haoxiang.zhang/gitproj/miles-sdpo"
cd "$REPO_ROOT"

echo "[$(date -u)] eval-trained-ckpt ${TAG} started on $(hostname)"
nvidia-smi -L || true

# Secrets (WANDB_API_KEY, SFT_GATEWAY_KEY for the AMO judge), same gitignored
# .env chain the enroot launcher uses. `set +x` so the keys are not echoed into
# the job log.
set +x
if [ -f examples/SDPO_ReAct/.env ]; then
    set -a; . examples/SDPO_ReAct/.env; set +a
elif [ -f examples/SDPO/.env ]; then
    set -a; . examples/SDPO/.env; set +a
fi
set -x

# --- keep the OC idle-guard disabled for the whole eval ----------------------
# Same guard that SIGTERMs training jobs to grab their GPUs (see
# sbatch_run_family_queue.sh); an eval is just as killable. Rewrite BOTH names
# in a loop rather than setting a long expiry once -- the file has been observed
# to vanish before its own timestamp.
mkdir -p "$HOME/.oc-filler"
(
    while true; do
        fut=$(( $(date +%s) + 6*3600 ))
        echo "$fut" > "$HOME/.oc-filler/disabled_until"
        echo "$fut" > "$HOME/.oc-filler/disabled_until.$(hostname -s)"
        sleep 60
    done
) >/dev/null 2>&1 &
OC_GUARD_PID=$!
trap 'kill "$OC_GUARD_PID" 2>/dev/null || true' EXIT

# --- code sandbox (port 8420), on the HOST ----------------------------------
# Must be started outside enroot (no docker-in-enroot). BOTH halves need it:
# OJBench grades through the sandbox, and the math eval's code_interpreter tool
# calls go there too -- a dead sandbox silently turns tool calls into errors.
bash examples/SDPO_ReAct/tools/run_sandbox.sh
curl -sf http://127.0.0.1:8420/health >/dev/null 2>&1 \
    || { echo "FATAL: code sandbox not healthy on 8420" >&2; exit 1; }

# --- enroot image -> squashfs -> container rootfs (per node) -----------------
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
if ! enroot list 2>/dev/null | grep -qx "$CONTAINER"; then
    enroot create --name "$CONTAINER" "$SQSH"
fi

# --root: the training launcher's sessions own /root/* inside this container
# (the prep step writes the model dirs there as uid 0), so a non-remapped
# session cannot even read /root/<model>_torch_dist.
enroot start --root --rw \
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
    --env OPENAI_API_URL="${OPENAI_API_URL:-}" \
    --env SFT_GATEWAY_KEY="${SFT_GATEWAY_KEY:-}" \
    --env RUN_MATH="${RUN_MATH:-1}" \
    --env RUN_OJBENCH="${RUN_OJBENCH:-1}" \
    --env SDPO_EVAL_REF_LOAD="$CKPT" \
    --env SDPO_EVAL_TAG="$TAG" \
    --env MODEL_PATH="${MODEL_PATH:-/root/data/home-static/data/hf_models/Qwen3.5-9B}" \
    --env AMO_BENCH_DATA_DIR="${AMO_BENCH_DATA_DIR:-/root/data/home-static/data/amo_bench}" \
    --env SDPO_REACT_EVAL_DIR="${SDPO_REACT_EVAL_DIR:-/root/data/math_eval/native-minimal}" \
    --env OJBENCH_DATA_DIR="${OJBENCH_DATA_DIR:-/root/data/home-static/data/ojbench_medium}" \
    --env OJBENCH_DIFFICULTY="${OJBENCH_DIFFICULTY:-medium}" \
    --env OJBENCH_REQUIRE_TOOL="${OJBENCH_REQUIRE_TOOL:-}" \
    --env SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-}" \
    --env SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-}" \
    --env SDPO_REACT_TP="${SDPO_REACT_TP:-}" \
    --env SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-}" \
    --env MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-}" \
    "$CONTAINER" \
    bash -c '
        cd /root/miles
        # Drop the empty pass-throughs: the eval scripts use ${VAR:-default},
        # which an exported-but-empty VAR would defeat.
        for v in OJBENCH_REQUIRE_TOOL SDPO_REACT_EVAL_N_SAMPLES \
                 SDPO_REACT_EVAL_MAX_TURNS SDPO_REACT_TP SGLANG_MEM_FRACTION \
                 MAX_TOKENS_PER_GPU OPENAI_API_URL SFT_GATEWAY_KEY; do
            [ -z "${!v}" ] && unset "$v"
        done
        if [ ! -f "${SDPO_EVAL_REF_LOAD}/latest_checkpointed_iteration.txt" ]; then
            echo "FATAL: ${SDPO_EVAL_REF_LOAD} is not a megatron dist-ckpt dir" >&2
            exit 1
        fi
        RC=0
        # Run BOTH halves even if the first fails: math and code failures are
        # independent, and a half-populated results table is worse than two
        # independent verdicts.
        if [ "${RUN_MATH}" = "1" ]; then
            SDPO_EVAL_CONFIG=eval_math_final.yaml \
            AMO_BENCH_WANDB_GROUP="math-final-eval-${SDPO_EVAL_TAG}" \
                bash examples/SDPO_ReAct/ablation/eval-amo-bench.sh || RC=$?
            echo "[bench] math (aime24+aime25+amo) rc=$RC"
        fi
        if [ "${RUN_OJBENCH}" = "1" ]; then
            OJBENCH_WANDB_GROUP="ojbench-medium-eval-${SDPO_EVAL_TAG}" \
                bash examples/SDPO_ReAct/ablation/eval-ojbench.sh || RC=$?
            echo "[bench] ojbench rc=$RC"
        fi
        exit $RC
    '
RC=$?

echo "[$(date -u)] evals rc=$RC for ${TAG}"
curl -sf http://127.0.0.1:8420/stats || true
ray stop --force 2>/dev/null || true
# Bracketed patterns: an unbracketed 'ray::' / 'sglang::' also matches THIS
# script's own command line under `pkill -f`, which kills the job itself.
pkill -9 -f 'ra[y]::' 2>/dev/null || true
pkill -9 -f 'sglan[g]::' 2>/dev/null || true
exit $RC
