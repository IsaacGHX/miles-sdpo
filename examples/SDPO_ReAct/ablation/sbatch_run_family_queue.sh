#!/bin/bash
# ONE sbatch job that runs model x domain combos SEQUENTIALLY on ONE H200 node,
# via the enroot launcher (../enroot-run-sdpo-react.sh), for a SINGLE algo/arm.
#
# Unlike its per-model siblings (sbatch_run_9B_mathcodesearch.sh,
# sbatch_run_4B_alfworld_webshop.sh, ...) a combo here is "MODEL_KEY FAMILY",
# not "algo arm": the algo/arm is fixed for the whole queue (SDPO_ABLATION_ALGO
# / SDPO_ABLATION_ARM, default grpo/z) and the axis that varies is which
# model/domain script runs. That lets a single job cover e.g. both
# mathcodesearch runs (sharing ONE search-stack bootstrap) or both
# alfworld+webshop runs.
#
# PREFER THIS over the per-model siblings for any `native` (mathcodesearch) run:
# they have no sidecar bootstrap at all, which is how the 9B grpo-e run came to
# train for hours against a search server that returned 500 for 100% of ~1900
# calls without ever crashing. This script starts BM25 + the search sidecar, and
# re-probes both before EVERY combo.
#
# WHY ONE JOB LOOPING INTERNALLY: a cluster-wide fairness enforcer (uid 3132's
# monitor, not this repo's code) auto-scancels this user's jobs once too many are
# concurrently in the queue -- it settled at 2 survivors when 15, then 4, then 3
# were submitted. See sbatch_run_4B_alfworld_webshop.sh's header for the full
# writeup. Don't have more than ~2 of these in flight at once.
#
# MODEL_KEY / FAMILY are enroot-run-sdpo-react.sh's own vocabulary:
#   MODEL_KEY  qwen3.5-4B-ablation | qwen3.5-9B   (SDPO_REACT_MODEL)
#   FAMILY     native (math+code+search) | agentic (alfworld+webshop)
#              (SDPO_REACT_RUN_FAMILY -> that script's NATIVE_RUN_SH/AGENTIC_RUN_SH)
#
# usage:
#   # both alfworld+webshop runs of arm z on one node (no search stack needed):
#   COMBOS_OVERRIDE="qwen3.5-4B-ablation agentic|qwen3.5-9B agentic" \
#     sbatch examples/SDPO_ReAct/ablation/sbatch_run_family_queue.sh
#   # 9B mathcodesearch, grpo arm e (bootstraps BM25 + search sidecar):
#   SDPO_ABLATION_ALGO=grpo SDPO_ABLATION_ARM=e \
#     COMBOS_OVERRIDE="qwen3.5-9B native" \
#     sbatch examples/SDPO_ReAct/ablation/sbatch_run_family_queue.sh
#   # or run directly on an ALREADY-allocated node (SLURM_JOB_ID unset -> a
#   # timestamp tag is used for log/marker names instead):
#   COMBOS_OVERRIDE="..." bash examples/SDPO_ReAct/ablation/sbatch_run_family_queue.sh
#SBATCH --job-name=hx-sdpo-family-queue
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low,ml.p5en.48xlarge-ultra-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=16
#SBATCH --time=4-00:00:00
#SBATCH --output=/fsx/home/haoxiang.zhang/logs/hx-sdpo-family-queue_%j.out
#SBATCH --requeue
set -x

# NOT derived from ${BASH_SOURCE[0]}: sbatch copies this script to a per-job
# spool path (/var/spool/slurmd/jobNNNN/slurm_script) before running it, so
# dirname-based path resolution silently lands in /var and every launch below
# fails rc=127 (confirmed live -- see sbatch_run_4B_alfworld_webshop.sh).
REPO_ROOT="/fsx/home/haoxiang.zhang/gitproj/miles-sdpo"
cd "$REPO_ROOT"

# Runnable both as an sbatch job and directly on an already-allocated node.
TAG="${SLURM_JOB_ID:-local$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="/fsx/home/haoxiang.zhang/logs"
mkdir -p "$LOG_DIR"

echo "[$(date -u)] family queue $TAG started on $(hostname)"
nvidia-smi -L || true

# --- keep the OC idle-guard disabled for the WHOLE queue ---------------------
# The host's "oc filler" idle-guard SIGTERMs (exit -15) a running training job
# to grab its GPUs. `oc-off` alone is not reliable (historic path mismatch: the
# monitor read the bare file, oc-off wrote the host-suffixed one), and the
# disable file has been observed to VANISH well before its own timestamp -- so
# rewrite BOTH names in a loop rather than setting a long expiry once. A -15
# that kills every sglang engine simultaneously is this, not an sglang bug.
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

# WANDB_API_KEY et al. -- an unset key makes train.py die with
# "argument --wandb-key: expected one argument" ~1s in (confirmed live on the
# code-only launch), which reads like a code bug rather than a missing secret.
if [ -f examples/SDPO/.env ]; then
    set +x  # don't echo the key into the job log
    set -a; . examples/SDPO/.env; set +a
    set -x
fi

if [ -n "${COMBOS_OVERRIDE:-}" ]; then
    IFS='|' read -ra COMBOS <<< "$COMBOS_OVERRIDE"
else
    COMBOS=(
        "qwen3.5-4B-ablation native"
        "qwen3.5-4B-ablation agentic"
        "qwen3.5-9B native"
        "qwen3.5-9B agentic"
    )
fi

NEEDS_SEARCH=0
for combo in "${COMBOS[@]}"; do
    [ "$(echo "$combo" | cut -d' ' -f2)" = "native" ] && NEEDS_SEARCH=1
done

# --- host-level code sandbox (port 8420) -------------------------------------
# The sandbox is a HOST docker singleton; there is no `docker` binary inside the
# enroot training container, where run_sandbox.sh degrades to a bare health check
# that only WARNS. So a run launched without this started first trains for hours
# against a dead code_interpreter and silently scores 0 on every code task
# (confirmed live twice -- see sdpo_dumps/0_stat/olmo3_vs_qwen25_sci_ablation.md
# section 9). Needed by BOTH families: the agentic scripts keep code_interpreter
# in their toolset.
bash examples/SDPO_ReAct/tools/run_sandbox.sh
curl -sf http://127.0.0.1:8420/health >/dev/null 2>&1 \
    || { echo "FATAL: code sandbox not healthy on 8420 after run_sandbox.sh" >&2; exit 1; }

# --- search stack (BM25 retrieval server + search sidecar) -------------------
# native/mathcodesearch only. Verbatim the bootstrap+probe sequence from
# sbatch_run_9B_mcs_rlsd_sdpo.sh (which exists because a run whose sidecar was
# never started returned a 500 for 100% of ~1900 search calls without crashing).
DATA_DIR="/fsx/data/${USER}"
ENROOT_NVME="/opt/dlami/nvme/miles-enroot"
SQSH="$ENROOT_NVME/miles-0704.sqsh"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$ENROOT_NVME/cache}"

probe_search() {
    curl -sf -X POST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' \
        -d '{"queries":["health probe"],"topk":1}' >/dev/null 2>&1
}

launch_bm25() {
    nohup enroot start --rw --mount "$REPO_ROOT":/root/miles --mount "$DATA_DIR":/root/data \
        --env RETRIEVAL_BACKEND=bm25 --env RETRIEVAL_TOPK=3 --env BM25_ALLOW_DEP_SURGERY=1 \
        miles-bm25 bash -euxc '
            cd /root/miles
            bash examples/SDPO_ReAct/tools/search/run_retrieval.sh
            tail -f /root/data/wiki18_bm25/retrieval_server.log
        ' > "$LOG_DIR/hx-sdpo-family-bm25_${TAG}.log" 2>&1 &
    # 30 min, NOT the 5 min this used to allow. On a container freshly created
    # from the pinned sqsh, run_retrieval.sh first does its dep surgery
    # (`pip install --no-cache-dir pyserini==0.44.0` + a JDK) before it ever
    # binds :8000, and that alone runs well past 5 min. The old 300s budget
    # expired mid-install, the retry below then `enroot remove -f`'d the
    # container out from under that pip, and the second attempt started the
    # same install from scratch on the same budget -- i.e. the retry could
    # never succeed where the first attempt failed, and the queue died FATAL
    # having done nothing but thrash (observed live).
    for _ in $(seq 1 360); do
        probe_search && return 0
        sleep 5
    done
    return 1
}

if [ "$NEEDS_SEARCH" = "1" ] && ! probe_search; then
    mkdir -p "$ENROOT_NVME" "$ENROOT_CACHE_PATH"
    # /opt/dlami/nvme is ephemeral and wiped on host restart, so a genuinely
    # fresh node has neither the base sqsh nor the container rootfs, even when
    # `enroot list` still shows the name.
    # SAME pinned tag as the launcher's own IMAGE default, NOT :latest-cu12.
    # $SQSH is shared: the sqsh created here is what the TRAINING container also
    # gets built from, so a mutable tag here silently decides the training
    # stack's vintage. Two nodes already diverged this way (Jul-3 vs Aug-4
    # sglang source, needing different sgl_kernel builds).
    [ -f "$SQSH" ] || enroot import -o "$SQSH" "docker://radixark/miles:dev-cu12-202607040446"
    enroot remove -f miles-bm25 2>/dev/null || true
    enroot create --name miles-bm25 "$SQSH"
    launch_bm25 || {
        echo "BM25 unhealthy on first attempt -- recreating container and retrying once" >&2
        enroot remove -f miles-bm25 2>/dev/null || true
        enroot create --name miles-bm25 "$SQSH"
        launch_bm25 || { echo "FATAL: BM25 retrieval server never became healthy" >&2; exit 1; }
    }
fi
if [ "$NEEDS_SEARCH" = "1" ]; then
    RETRIEVAL_PORT=8000 bash examples/SDPO_ReAct/tools/search/run_search_sidecar.sh
    probe_search || { echo "FATAL: search stack failed final health check" >&2; exit 1; }
    echo "search stack (BM25 + sidecar) healthy on $(hostname)"
fi

# NO `export IMAGE=...latest-cu12` here on purpose. That tag is mutable, and the
# sqsh it would produce is what BOTH the BM25 and the TRAINING container get
# created from -- so pinning it wrong silently decides the whole training stack's
# vintage. The launcher already pins a digest-stable tag; let it win.
export SDPO_ABLATION_ALGO="${SDPO_ABLATION_ALGO:-grpo}"
export SDPO_ABLATION_ARM="${SDPO_ABLATION_ARM:-z}"
# Checkpointing is decided PER-COMBO by family, not once for the queue: only the
# native/mathcodesearch arms need weights on disk (they feed the AMO-Bench and
# LCB-functional evals afterwards); the agentic ones are read off wandb curves
# only, and a 9B agentic run writes 110G per saved iter for nothing. Set
# SDPO_REACT_SAVE_CKPT explicitly to force one setting on the whole queue.
SAVE_CKPT_OVERRIDE="${SDPO_REACT_SAVE_CKPT:-}"
# These used to be force-unset here so a leftover smoke-test value
# (SDPO_REACT_NUM_ROLLOUT=2, SKIP_EVAL0=1) in the submitting shell could not
# silently shorten a real queue. Keep that as the DEFAULT -- unset stays unset,
# and the run scripts' own `${VAR:-51}` / `${VAR:-0}` defaults apply -- but let an
# EXPLICIT value through: a rerun that deliberately stops just past the observed
# eval peak has to set num_rollout, and there is no other channel to it (the
# enroot launcher only forwards whitelisted --env vars).
export SDPO_REACT_NUM_ROLLOUT="${SDPO_REACT_NUM_ROLLOUT:-}"
export SDPO_REACT_SKIP_EVAL0="${SDPO_REACT_SKIP_EVAL0:-}"

# algo/arm is part of every filename: two queues can share a TAG (same job id
# after a requeue, or two direct launches in the same second) but differ in arm,
# and a shared DONE_MARKER would make one silently skip the other's combos.
RUNID="${TAG}_${SDPO_ABLATION_ALGO}-${SDPO_ABLATION_ARM}"
STATUS_FILE="$LOG_DIR/sbatch_run_family_queue_status_${RUNID}.txt"
echo "queue started $(date -u): algo=$SDPO_ABLATION_ALGO arm=$SDPO_ABLATION_ARM on $TAG / $(hostname)" > "$STATUS_FILE"
# Resume marker: skip combos already finished by an earlier --requeue of this
# SAME job id (preemption on the low-pri partitions is expected).
DONE_MARKER="$LOG_DIR/sbatch_run_family_queue_done_${RUNID}.txt"
touch "$DONE_MARKER"

for combo in "${COMBOS[@]}"; do
    if grep -qxF "$combo" "$DONE_MARKER"; then
        echo "=== SKIP (already done): $combo ===" | tee -a "$STATUS_FILE"
        continue
    fi

    model=$(echo "$combo" | cut -d' ' -f1)
    family=$(echo "$combo" | cut -d' ' -f2)

    # Re-probe every sidecar before EVERY combo, not just once at job start: a
    # mid-queue sidecar death produces no crash, only silently wrong numbers.
    curl -sf http://127.0.0.1:8420/health >/dev/null 2>&1 \
        || bash examples/SDPO_ReAct/tools/run_sandbox.sh
    if ! curl -sf http://127.0.0.1:8420/health >/dev/null 2>&1; then
        echo "FATAL: code sandbox unhealthy before $combo -- aborting rather than run silently broken" \
            | tee -a "$STATUS_FILE" >&2
        exit 1
    fi
    if [ "$family" = "native" ] && ! probe_search; then
        echo "FATAL: search stack unhealthy before $combo -- aborting rather than run silently broken" \
            | tee -a "$STATUS_FILE" >&2
        exit 1
    fi

    if [ -n "$SAVE_CKPT_OVERRIDE" ]; then
        export SDPO_REACT_SAVE_CKPT="$SAVE_CKPT_OVERRIDE"
    elif [ "$family" = "native" ]; then
        export SDPO_REACT_SAVE_CKPT=1
    else
        export SDPO_REACT_SAVE_CKPT=0
    fi

    echo "=== STARTING $combo (algo=$SDPO_ABLATION_ALGO arm=$SDPO_ABLATION_ARM save_ckpt=$SDPO_REACT_SAVE_CKPT) at $(date -u) ===" \
        | tee -a "$STATUS_FILE"
    export SDPO_REACT_MODEL="$model"
    export SDPO_REACT_RUN_FAMILY="$family"

    LOG="$LOG_DIR/hx-sdpo-${model}-${family}_${RUNID}.log"
    bash examples/SDPO_ReAct/enroot-run-sdpo-react.sh > "$LOG" 2>&1
    RC=$?

    echo "=== FINISHED $combo rc=$RC at $(date -u) (log $LOG) ===" | tee -a "$STATUS_FILE"
    echo "$combo" >> "$DONE_MARKER"

    ray stop --force 2>/dev/null || true
    pkill -9 -f 'ray::' 2>/dev/null || true
    pkill -9 -f 'sglang::' 2>/dev/null || true
    sleep 5
done

echo "queue finished $(date -u)" | tee -a "$STATUS_FILE"
