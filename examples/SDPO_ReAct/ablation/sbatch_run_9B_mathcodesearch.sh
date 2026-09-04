#!/bin/bash
# ONE sbatch job that runs the Qwen3.5-9B math+code+search ablation matrix
# (see ../run-qwen3.5-9B-sdpo-react-ablation-mathcodesearch.sh) SEQUENTIALLY,
# on ONE H200 node, via the enroot launcher (../enroot-run-sdpo-react.sh).
# Sibling of sbatch_run_4B_alfworld_webshop.sh -- same rationale (one sbatch
# job looping internally to avoid the cluster fairness enforcer, low-pri
# account/partitions, --requeue + done-marker to resume after preemption).
# See that script's header for the full rationale writeup.
#
# Default combos: grpo a/e/f only (per explicit instruction -- the
# mathcodesearch script's own memory tuning is an A100-80G retune never
# validated on real hardware; start with the arms GRPO actually supports
# and confirm the run is healthy before widening to sdpo/rlsd x a-f).
#
# usage:
#   sbatch examples/SDPO_ReAct/ablation/sbatch_run_9B_mathcodesearch.sh
#   COMBOS_OVERRIDE="sdpo a|rlsd a" \
#     sbatch examples/SDPO_ReAct/ablation/sbatch_run_9B_mathcodesearch.sh
#SBATCH --job-name=hx-sdpo-9B-mcs-ablation-queue
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low,ml.p5en.48xlarge-ultra-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=16
#SBATCH --time=4-00:00:00
#SBATCH --output=/fsx/home/haoxiang.zhang/logs/hx-sdpo-9B-mcs-ablation-queue_%j.out
#SBATCH --requeue
set -x

# NOT derived from ${BASH_SOURCE[0]}/dirname -- sbatch copies this script to a
# per-job spool path before running it (see sbatch_run_4B_alfworld_webshop.sh
# for the confirmed failure mode if this trick is used instead).
REPO_ROOT="/fsx/home/haoxiang.zhang/gitproj/miles-sdpo"
cd "$REPO_ROOT"

echo "[$(date -u)] job $SLURM_JOB_ID started on $(hostname)"
nvidia-smi -L || true

export IMAGE="${IMAGE:-radixark/miles:latest-cu12}"
export SDPO_REACT_MODEL=qwen3.5-9B
export SDPO_REACT_RUN_FAMILY=native
unset SDPO_REACT_NUM_ROLLOUT
unset SDPO_REACT_SKIP_EVAL0
unset SDPO_ABLATION_ARM
unset SDPO_ABLATION_ALGO

if [ -n "${COMBOS_OVERRIDE:-}" ]; then
    IFS='|' read -ra COMBOS <<< "$COMBOS_OVERRIDE"
else
    COMBOS=(
        "grpo a" "grpo e" "grpo f"
    )
fi

LOG_DIR="/fsx/home/haoxiang.zhang/logs"
STATUS_FILE="$LOG_DIR/sbatch_run_9B_mathcodesearch_status_${SLURM_JOB_ID}.txt"
echo "queue started $(date -u) on job $SLURM_JOB_ID" > "$STATUS_FILE"

# Skip combos already marked done in the status file from an earlier
# --requeue of this SAME job id (see sbatch_run_4B_alfworld_webshop.sh).
DONE_MARKER="$LOG_DIR/sbatch_run_9B_mathcodesearch_done_${SLURM_JOB_ID}.txt"
touch "$DONE_MARKER"

for combo in "${COMBOS[@]}"; do
    if grep -qxF "$combo" "$DONE_MARKER"; then
        echo "=== SKIP (already done) algo/arm: $combo ===" | tee -a "$STATUS_FILE"
        continue
    fi

    algo=$(echo "$combo" | cut -d' ' -f1)
    arm=$(echo "$combo" | cut -d' ' -f2)
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
