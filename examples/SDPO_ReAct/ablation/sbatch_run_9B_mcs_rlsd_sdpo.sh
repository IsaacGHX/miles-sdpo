#!/bin/bash
# ONE sbatch job that runs the remaining Qwen3.5-9B mathcodesearch ablation
# combos (rlsd a/e, sdpo a/e, plus a grpo-a RERUN to get its checkpoint) on
# its OWN allocated node -- separate from grpo-f, which is running directly
# on an already-allocated host (job 1640) and should not be blocked behind
# these. Sibling of sbatch_run_9B_mathcodesearch.sh / sbatch_run_4B_alfworld_
# webshop.sh -- same one-sbatch-job-loops-internally rationale (see those
# scripts' headers for the cluster fairness-enforcer rationale).
#
# All 5 combos save+prune to a single latest-iter checkpoint each
# (SDPO_REACT_SAVE_CKPT=1), per examples/SDPO_ReAct/ablation/run-qwen3.5-9B-
# sdpo-react-ablation-mathcodesearch.sh's own pruner loop.
#
# usage:
#   sbatch examples/SDPO_ReAct/ablation/sbatch_run_9B_mcs_rlsd_sdpo.sh
#   # or run a single combo (own node -- lets rlsd/sdpo a/e run in PARALLEL
#   # across separate low-pri jobs instead of queued one-at-a-time on one
#   # node; each submission is still its own single sbatch job, so it never
#   # trips the cluster fairness enforcer -- see sbatch_run_4B_alfworld_
#   # webshop.sh's header for that rationale):
#   COMBOS_OVERRIDE="rlsd a" sbatch examples/SDPO_ReAct/ablation/sbatch_run_9B_mcs_rlsd_sdpo.sh
#SBATCH --job-name=hx-sdpo-9B-mcs-rlsd-sdpo-queue
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low,ml.p5en.48xlarge-ultra-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=16
#SBATCH --time=4-00:00:00
#SBATCH --output=/fsx/home/haoxiang.zhang/logs/hx-sdpo-9B-mcs-rlsd-sdpo-queue_%j.out
#SBATCH --requeue
set -x

REPO_ROOT="/fsx/home/haoxiang.zhang/gitproj/miles-sdpo"
cd "$REPO_ROOT"

echo "[$(date -u)] job $SLURM_JOB_ID started on $(hostname)"
nvidia-smi -L || true

LOG_DIR="/fsx/home/haoxiang.zhang/logs"

# --- bootstrap the search stack on THIS (fresh) node -------------------------
# This job lands on a brand-new node with no BM25 retrieval server / search
# sidecar running -- unlike a host that's already been manually warmed up.
# Without this, every combo below would crash at the search-sidecar health
# check before any GPU work starts, OR WORSE, silently run with a dead
# sidecar the whole time (see this session's grpo-e postmortem in
# sdpo_dumps/0_stat/olmo3_vs_qwen25_sci_ablation.md section 9: 100% of its
# search tool calls failed for the entire run with no crash, just silently
# wrong eval numbers). Recreate both pieces from the cached base sqsh, then
# verify with a live /retrieve probe BEFORE launching any combo.
DATA_DIR="/fsx/data/${USER}"
ENROOT_NVME="/opt/dlami/nvme/miles-enroot"
SQSH="$ENROOT_NVME/miles-0704.sqsh"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$ENROOT_NVME/cache}"
mkdir -p "$ENROOT_NVME" "$ENROOT_CACHE_PATH"

# A genuinely fresh node (never ran a miles job before) has no base sqsh
# imported at all -- confirmed live on ip-10-1-40-128: "enroot create"
# failed with "No such file or directory" for a sqsh path this script
# assumed was already cached. Mirrors enroot-run-sdpo-react.sh's own
# import-if-missing check.
if [ ! -f "$SQSH" ]; then
    enroot import -o "$SQSH" "docker://radixark/miles:latest-cu12"
fi

launch_bm25() {
    nohup enroot start --rw --mount "$REPO_ROOT":/root/miles --mount "$DATA_DIR":/root/data \
        --env RETRIEVAL_BACKEND=bm25 --env RETRIEVAL_TOPK=3 --env BM25_ALLOW_DEP_SURGERY=1 \
        miles-bm25 bash -euxc '
            cd /root/miles
            bash examples/SDPO_ReAct/tools/search/run_retrieval.sh
            tail -f /root/data/wiki18_bm25/retrieval_server.log
        ' > "$LOG_DIR/hx-sdpo-9B-mcs-rlsd-sdpo-bm25_${SLURM_JOB_ID}.log" 2>&1 &
    for _ in $(seq 1 60); do
        curl -sf -X POST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' \
            -d '{"queries":["health probe"],"topk":1}' >/dev/null 2>&1 && return 0
        sleep 5
    done
    return 1
}

# --name reuse can be stale/broken on a node whose ephemeral NVMe was reset
# (enroot list still shows the name, but /opt/dlami/nvme/tmp/enroot/data/...
# is gone -- confirmed live: "enroot start" failed with "No such file or
# directory" for a container "enroot list" claimed existed). Remove first so
# recreate is unconditional and never silently starts against a missing
# rootfs.
enroot remove -f miles-bm25 2>/dev/null || true
enroot create --name miles-bm25 "$SQSH"
enroot list 2>/dev/null | grep -qx miles-bm25 \
    || { echo "FATAL: enroot create miles-bm25 did not produce a listed container" >&2; exit 1; }

launch_bm25 || {
    echo "BM25 retrieval server did not become healthy on first attempt -- recreating container and retrying once" >&2
    enroot remove -f miles-bm25 2>/dev/null || true
    enroot create --name miles-bm25 "$SQSH"
    launch_bm25 || { echo "FATAL: BM25 retrieval server never became healthy after retry" >&2; exit 1; }
}

RETRIEVAL_PORT=8000 bash examples/SDPO_ReAct/tools/search/run_search_sidecar.sh

curl -sf -X POST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' \
    -d '{"queries":["health probe"],"topk":1}' >/dev/null 2>&1 \
    || { echo "FATAL: search stack failed final health check" >&2; exit 1; }
echo "search stack (BM25 + sidecar) healthy on $(hostname)"

export IMAGE="${IMAGE:-radixark/miles:latest-cu12}"
export SDPO_REACT_MODEL=qwen3.5-9B
export SDPO_REACT_RUN_FAMILY=native
export SDPO_REACT_SAVE_CKPT=1
unset SDPO_ABLATION_ARM
unset SDPO_ABLATION_ALGO

# COMBOS_OVERRIDE (env var, pipe-separated "algo arm" entries) runs a SUBSET
# instead of the full 5 -- used to split the matrix across multiple nodes/
# jobs (each still its own single sbatch job, so this stays safe against the
# cluster fairness enforcer noted in sbatch_run_4B_alfworld_webshop.sh's
# header -- don't submit more than 2-3 of these concurrently).
if [ -n "${COMBOS_OVERRIDE:-}" ]; then
    IFS='|' read -ra COMBOS <<< "$COMBOS_OVERRIDE"
else
    COMBOS=(
        "grpo a"
        "rlsd a"
        "rlsd e"
        "sdpo a"
        "sdpo e"
    )
fi

STATUS_FILE="$LOG_DIR/sbatch_run_9B_mcs_rlsd_sdpo_status_${SLURM_JOB_ID}.txt"
echo "queue started $(date -u) on job $SLURM_JOB_ID" > "$STATUS_FILE"

DONE_MARKER="$LOG_DIR/sbatch_run_9B_mcs_rlsd_sdpo_done_${SLURM_JOB_ID}.txt"
touch "$DONE_MARKER"

for combo in "${COMBOS[@]}"; do
    if grep -qxF "$combo" "$DONE_MARKER"; then
        echo "=== SKIP (already done) algo/arm: $combo ===" | tee -a "$STATUS_FILE"
        continue
    fi

    algo=$(echo "$combo" | cut -d' ' -f1)
    arm=$(echo "$combo" | cut -d' ' -f2)

    # Re-verify search health before EVERY combo, not just once at job start --
    # a mid-batch sidecar death (this session's grpo-e incident) produces no
    # crash, just silently wrong numbers, so catching it only at startup is
    # not enough.
    if ! curl -sf -X POST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' \
            -d '{"queries":["health probe"],"topk":1}' >/dev/null 2>&1; then
        echo "FATAL: search stack unhealthy before starting algo=$algo arm=$arm -- aborting rather than run silently broken" \
            | tee -a "$STATUS_FILE" >&2
        exit 1
    fi

    echo "=== STARTING algo=$algo arm=$arm at $(date -u) ===" | tee -a "$STATUS_FILE"

    export SDPO_ABLATION_ALGO="$algo"
    export SDPO_ABLATION_ARM="$arm"

    LOG="$LOG_DIR/hx-sdpo-9B-mcs-${algo}-${arm}_${SLURM_JOB_ID}.log"
    bash examples/SDPO_ReAct/enroot-run-sdpo-react.sh > "$LOG" 2>&1
    RC=$?

    echo "=== FINISHED algo=$algo arm=$arm rc=$RC at $(date -u) ===" | tee -a "$STATUS_FILE"
    echo "$combo" >> "$DONE_MARKER"

    ray stop --force 2>/dev/null || true
    pkill -9 -f 'ray::' 2>/dev/null || true
    pkill -9 -f 'sglang::' 2>/dev/null || true
    sleep 5
done

echo "queue finished $(date -u)" | tee -a "$STATUS_FILE"
