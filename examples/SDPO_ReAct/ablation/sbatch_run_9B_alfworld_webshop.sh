#!/bin/bash
# ONE sbatch job that runs the Qwen3.5-9B ALFWorld+WebShop ablation matrix
# (see ../run-qwen3.5-9B-sdpo-react-ablation-alfworld-webshop.sh) SEQUENTIALLY,
# on ONE H200 node, via the enroot launcher (../enroot-run-sdpo-react.sh).
# Sibling of sbatch_run_4B_alfworld_webshop.sh / sbatch_run_9B_mcs_rlsd_sdpo.sh
# -- same one-sbatch-job-loops-internally rationale (see those scripts'
# headers for the cluster fairness-enforcer rationale), same low-pri
# account/partitions, --requeue + done-marker to resume after preemption.
#
# Default combos: grpo/sdpo/rlsd x a/e (6 combos) -- the a/e pair lets us
# compare "baseline" vs "self-skill all + skill-KD(both)" per mechanism on
# this domain, mirroring the 9B mathcodesearch matrix's own a/e comparison.
#
# usage:
#   sbatch examples/SDPO_ReAct/ablation/sbatch_run_9B_alfworld_webshop.sh
#   COMBOS_OVERRIDE="grpo a|grpo e" \
#     sbatch examples/SDPO_ReAct/ablation/sbatch_run_9B_alfworld_webshop.sh
#SBATCH --job-name=hx-sdpo-9B-aw-ablation-queue
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low,ml.p5en.48xlarge-ultra-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=16
#SBATCH --time=4-00:00:00
#SBATCH --output=/fsx/home/haoxiang.zhang/logs/hx-sdpo-9B-aw-ablation-queue_%j.out
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
export SDPO_REACT_RUN_FAMILY=agentic
export SDPO_REACT_SAVE_CKPT=1
unset SDPO_REACT_NUM_ROLLOUT
unset SDPO_REACT_SKIP_EVAL0
unset SDPO_ABLATION_ARM
unset SDPO_ABLATION_ALGO

if [ -n "${COMBOS_OVERRIDE:-}" ]; then
    IFS='|' read -ra COMBOS <<< "$COMBOS_OVERRIDE"
else
    COMBOS=(
        "grpo a" "grpo e"
        "sdpo a" "sdpo e"
        "rlsd a" "rlsd e"
    )
fi

LOG_DIR="/fsx/home/haoxiang.zhang/logs"
STATUS_FILE="$LOG_DIR/sbatch_run_9B_alfworld_webshop_status_${SLURM_JOB_ID}.txt"
echo "queue started $(date -u) on job $SLURM_JOB_ID" > "$STATUS_FILE"

# Skip combos already marked done in the status file from an earlier
# --requeue of this SAME job id (see sbatch_run_4B_alfworld_webshop.sh).
DONE_MARKER="$LOG_DIR/sbatch_run_9B_alfworld_webshop_done_${SLURM_JOB_ID}.txt"
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

    LOG="$LOG_DIR/hx-sdpo-9B-aw-${algo}-${arm}_${SLURM_JOB_ID}.log"
    bash examples/SDPO_ReAct/enroot-run-sdpo-react.sh > "$LOG" 2>&1
    RC=$?

    echo "=== FINISHED algo=$algo arm=$arm rc=$RC at $(date -u) ===" | tee -a "$STATUS_FILE"
    echo "$combo" >> "$DONE_MARKER"

    # cleanup between runs -- same sequence as the sibling queue scripts
    # (kill ray/sglang so the next combo starts from a clean GPU, not
    # fighting leftover processes from the previous one).
    ray stop --force 2>/dev/null || true
    pkill -9 -f 'ray::' 2>/dev/null || true
    pkill -9 -f 'sglang::' 2>/dev/null || true
    sleep 5
done

echo "queue finished $(date -u)" | tee -a "$STATUS_FILE"
