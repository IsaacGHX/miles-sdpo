#!/bin/bash
# ONE sbatch job that runs the FULL Qwen3.5-4B alfworld+webshop ablation
# matrix (see ../run-qwen3.5-4B-sdpo-react-ablation-alfworld-webshop.sh)
# SEQUENTIALLY, on ONE H200 node, via the enroot launcher
# (../enroot-run-sdpo-react.sh). This is the low-pri equivalent of
# ../../.claude/.../run_9_arms.sh's queueing loop -- same idea (one combo at
# a time, clean up ray/GPU between runs), but wrapped in a single #SBATCH
# script instead of a bash loop babysat interactively.
#
# WHY ONE JOB LOOPING INTERNALLY (not N separate sbatch jobs, whether
# submitted all at once or --dependency-chained): a cluster-wide fairness
# enforcer (uid 3132 / becky.peng's monitor, not this repo's code)
# auto-scancels this user's jobs once too many are concurrently in the
# QUEUE (pending OR running) -- confirmed live it cancelled jobs at 15
# submitted at once, then again at a --dependency chain capped at 4
# concurrent, then again down to 3, landing at 2 survivors. A single sbatch
# job never trips this: SLURM sees exactly ONE job from this script,
# regardless of how many arm combos it runs internally before exiting.
#
# ACCOUNT: low-pri (not interactive-ai) -- per explicit instruction, this is
# the account that actually succeeds. Uses partitions
# ml.p5en.48xlarge-low,ml.p5en.48xlarge-ultra-low (see
# $HOME/occupy_lowpri_h200.sbatch.DISABLED for the reference: -low is Tier
# 10, -ultra-low Tier 5 -- lower tier is preempted first, so -low is
# preferred, -ultra-low is the fallback). These are PreemptMode=REQUEUE
# partitions -- --requeue is set so a preempted run resumes on this SAME
# job id once a node frees, rather than needing a human to resubmit; the
# training script's own checkpoint/resume handling (not this script) is
# what makes that safe to restart.
#
# usage:
#   sbatch examples/SDPO_ReAct/ablation/sbatch_run_4B_alfworld_webshop.sh
#   # or split the matrix across two nodes/jobs (each still one sbatch job,
#   # so neither trips the fairness enforcer on its own):
#   COMBOS_OVERRIDE="grpo a|grpo e|grpo f|rlsd a|rlsd e|rlsd f|sdpo a" \
#     sbatch examples/SDPO_ReAct/ablation/sbatch_run_4B_alfworld_webshop.sh
#   COMBOS_OVERRIDE="sdpo e|sdpo f|rlsd b|rlsd c|rlsd d|sdpo b|sdpo c|sdpo d" \
#     sbatch examples/SDPO_ReAct/ablation/sbatch_run_4B_alfworld_webshop.sh
#SBATCH --job-name=hx-sdpo-4B-ablation-queue
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low,ml.p5en.48xlarge-ultra-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=16
#SBATCH --time=4-00:00:00
#SBATCH --output=/fsx/home/haoxiang.zhang/logs/hx-sdpo-4B-ablation-queue_%j.out
#SBATCH --requeue
set -x

# NOT derived from ${BASH_SOURCE[0]}/dirname: sbatch copies this script to a
# per-job spool path (/var/spool/slurmd/job0XXXX/slurm_script) before running
# it, so that trick (which works for a script run directly with `bash`)
# silently resolves to /var and every enroot-run-sdpo-react.sh invocation
# below fails with "No such file or directory" (confirmed live: all 15
# combos "finished" in under a second, rc=127, because cd landed in /var).
REPO_ROOT="/fsx/home/haoxiang.zhang/gitproj/miles-sdpo"
cd "$REPO_ROOT"

echo "[$(date -u)] job $SLURM_JOB_ID started on $(hostname)"
nvidia-smi -L || true

export IMAGE="${IMAGE:-radixark/miles:latest-cu12}"
export SDPO_REACT_MODEL=qwen3.5-4B-ablation
export SDPO_REACT_RUN_FAMILY=agentic
export SDPO_ABLATION_SGLANG_MEM_FRACTION="${SDPO_ABLATION_SGLANG_MEM_FRACTION:-0.8}"
unset SDPO_REACT_NUM_ROLLOUT
unset SDPO_REACT_SKIP_EVAL0
unset SDPO_ABLATION_ARM
unset SDPO_ABLATION_ALGO

# Priority order per the user's explicit ask: grpo/rlsd/sdpo x a/e/f FIRST
# (the 9 arms every algo supports), then the rest of the arm matrix
# (rlsd/sdpo x b/c/d -- GRPO only supports a/e/f, enforced by the ablation
# script's own arm-validation check).
#
# COMBOS_OVERRIDE (env var, pipe-separated "algo arm" entries) runs a SUBSET
# instead of the full 15 -- used to split the matrix across two nodes/jobs
# (see usage comment above): each job still only ever has ONE sbatch entry
# in squeue, so splitting this way doesn't trip the fairness enforcer either.
if [ -n "${COMBOS_OVERRIDE:-}" ]; then
    IFS='|' read -ra COMBOS <<< "$COMBOS_OVERRIDE"
else
    COMBOS=(
        "grpo a" "grpo e" "grpo f"
        "rlsd a" "rlsd e" "rlsd f"
        "sdpo a" "sdpo e" "sdpo f"
        "rlsd b" "rlsd c" "rlsd d"
        "sdpo b" "sdpo c" "sdpo d"
    )
fi

LOG_DIR="/fsx/home/haoxiang.zhang/logs"
STATUS_FILE="$LOG_DIR/sbatch_run_4B_alfworld_webshop_status_${SLURM_JOB_ID}.txt"
echo "queue started $(date -u) on job $SLURM_JOB_ID" > "$STATUS_FILE"

# Skip combos already marked done in the status file from an earlier
# --requeue of this SAME job id (preemption resumed this job -- without
# this check it would restart the whole matrix from combo 1 instead of
# picking up where it left off). A fresh job id always starts a fresh
# status file, so this is a no-op on first run.
DONE_MARKER="$LOG_DIR/sbatch_run_4B_alfworld_webshop_done_${SLURM_JOB_ID}.txt"
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

    LOG="$LOG_DIR/hx-sdpo-${algo}-${arm}_${SLURM_JOB_ID}.log"
    bash examples/SDPO_ReAct/enroot-run-sdpo-react.sh > "$LOG" 2>&1
    RC=$?

    echo "=== FINISHED algo=$algo arm=$arm rc=$RC at $(date -u) ===" | tee -a "$STATUS_FILE"
    echo "$combo" >> "$DONE_MARKER"

    # cleanup between runs -- same sequence as the interactive queue script
    # this mirrors (kill ray/sglang so the next combo starts from a clean
    # GPU, not fighting leftover processes from the previous one).
    ray stop --force 2>/dev/null || true
    pkill -9 -f 'ray::' 2>/dev/null || true
    pkill -9 -f 'sglang::' 2>/dev/null || true
    sleep 5
done

echo "queue finished $(date -u)" | tee -a "$STATUS_FILE"
