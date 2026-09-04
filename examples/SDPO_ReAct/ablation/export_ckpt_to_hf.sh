#!/bin/bash
# Export ONE megatron torch_dist checkpoint to an HF model dir, and optionally
# push it to the Hub.
#
# Runs the conversion INSIDE the enroot container: the host python's
# transformers is too old for these configs (`load_hf_config` on a Qwen3.5 dir
# dies with "module 'torch.utils._pytree' has no attribute 'register_constant'").
# CPU-only, no GPUs needed -- tools/convert_torch_dist_to_hf.py loads with
# no_dist=True and its planner skips the optimizer keys, so a 110G saved iter
# yields a ~17G HF dir.
#
# Required env vars:
#   ITER_DIR   the iter_NNNNNNN dir itself (NOT its parent --save dir), as a
#              CONTAINER path: /root/data == /fsx/data/$USER
#   OUT_NAME   HF dir basename, created under $HF_MODELS (and used as the repo
#              name under $HF_ORG when PUSH=1)
#
# Optional env vars:
#   BASE_HF    (default /root/data/home-static/data/hf_models/Qwen3.5-9B)
#              --origin-hf-dir: supplies tokenizer/config/chat_template, and its
#              config class name selects the converter (needs "qwen3_5" in it for
#              Qwen3.5, see megatron_to_hf/__init__.py::_convert_to_hf_core).
#   HF_MODELS  (default /root/data/home-static/data/hf_models) output parent
#   VOCAB_SIZE (default 248320, = Qwen3.5's text_config.vocab_size) passed as
#              --vocab-size so embedding/output_layer get sliced back down if
#              megatron padded them. Harmless no-op when there is no padding.
#              Getting this WRONG silently ships an lm_head that disagrees with
#              config.json, so keep it in sync with the base model's config.
#   PUSH       (default 0) 1 -> also `hf upload` to $HF_ORG/$OUT_NAME
#   HF_ORG     (default ipfipfipf)
#   PRIVATE    (default 0) 1 -> create the repo private
#   CONVERT    (default 1) 0 -> skip the conversion and only verify (+push) an
#              OUT_DIR that already exists. The intended two-step flow: convert
#              with PUSH=0, eyeball the verifier, then CONVERT=0 PUSH=1 to
#              publish, so nothing reaches the Hub before it is checked.
#
# NOTE the conversion drops the VISION TOWER: megatron only ever trained the
# language model, so the output has ~427 tensors vs the base model's ~775 while
# config.json still declares Qwen3_5ForConditionalGeneration. Every previously
# published ablation checkpoint in this org has the same shape, and the language
# model loads/serves fine -- but these repos are not usable for image input.
#
# usage:
#   ITER_DIR=/root/data/sdpo_ckpts/<run>_step29_ckpt/iter_0000029 \
#   OUT_NAME=Qwen3.5-9B-sdpo-react-mathcodesearch-grpo-arm-e-step29 \
#   PUSH=1 bash examples/SDPO_ReAct/ablation/export_ckpt_to_hf.sh
set -eu

: "${ITER_DIR:?Set ITER_DIR to the iter_NNNNNNN dir (container path)}"
: "${OUT_NAME:?Set OUT_NAME to the HF dir basename / repo name}"

REPO_ROOT="/fsx/home/haoxiang.zhang/gitproj/miles-sdpo"
cd "$REPO_ROOT"

# HF_TOKEN lives in the gitignored .env chain. `set +x` is not enough on its own
# here -- this script never enables -x, precisely so the token cannot land in a log.
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
DATA_DIR="${DATA_DIR:-/fsx/data/$USER}"

[ -f "$SQSH" ] || enroot import -o "$SQSH" "docker://${IMAGE}"
if ! enroot list 2>/dev/null | grep -qx "$CONTAINER"; then
    enroot create --name "$CONTAINER" "$SQSH"
fi

# --root for the same reason the eval launcher needs it: /root/* inside this
# shared container is owned by uid 0 from the training launcher's prep step.
enroot start --root --rw \
    --mount "$REPO_ROOT":/root/miles \
    --mount "$DATA_DIR":/root/data \
    --env HF_HOME=/root/data/hf_cache_export \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env ITER_DIR="$ITER_DIR" \
    --env OUT_NAME="$OUT_NAME" \
    --env BASE_HF="${BASE_HF:-/root/data/home-static/data/hf_models/Qwen3.5-9B}" \
    --env HF_MODELS="${HF_MODELS:-/root/data/home-static/data/hf_models}" \
    --env VOCAB_SIZE="${VOCAB_SIZE:-248320}" \
    --env PUSH="${PUSH:-0}" \
    --env HF_ORG="${HF_ORG:-ipfipfipf}" \
    --env PRIVATE="${PRIVATE:-0}" \
    --env CONVERT="${CONVERT:-1}" \
    "$CONTAINER" \
    bash -euc '
        cd /root/miles
        export PYTHONPATH="/root/miles:/root/Megatron-LM/"
        OUT_DIR="${HF_MODELS}/${OUT_NAME}"

        if [ "${CONVERT}" = "1" ]; then
            for f in .metadata common.pt; do
                [ -f "${ITER_DIR}/${f}" ] || { echo "FATAL: ${ITER_DIR} has no ${f} -- is it really an iter_NNNNNNN dir?" >&2; exit 1; }
            done

            echo "=== convert: ${ITER_DIR} -> ${OUT_DIR} (vocab ${VOCAB_SIZE}) ==="
            python3 tools/convert_torch_dist_to_hf.py \
                --input-dir "${ITER_DIR}" \
                --output-dir "${OUT_DIR}" \
                --origin-hf-dir "${BASE_HF}" \
                --vocab-size "${VOCAB_SIZE}"
        else
            echo "=== CONVERT=0: reusing existing ${OUT_DIR} ==="
            [ -d "${OUT_DIR}" ] || { echo "FATAL: CONVERT=0 but ${OUT_DIR} does not exist" >&2; exit 1; }
        fi

        # A separate file, NOT an inline `python3 -c`: the verifier needs single
        # quotes, and this whole block is already inside `bash -euc "..."` with
        # single-quote delimiters, so an inline script silently ends the string.
        echo "=== verify ==="
        python3 examples/SDPO_ReAct/ablation/verify_hf_export.py "${OUT_DIR}"

        if [ "${PUSH}" = "1" ]; then
            [ -n "${HF_TOKEN}" ] || { echo "FATAL: PUSH=1 but HF_TOKEN is empty" >&2; exit 1; }
            REPO="${HF_ORG}/${OUT_NAME}"
            echo "=== push -> ${REPO} ==="
            PRIV_FLAG=""
            [ "${PRIVATE}" = "1" ] && PRIV_FLAG="--private"
            # `hf repos create`, not `hf repo create` -- the latter warns it is deprecated.
            hf repos create "${REPO}" --repo-type model ${PRIV_FLAG} --exist-ok
            hf upload "${REPO}" "${OUT_DIR}" . --repo-type model \
                --commit-message "Add ${OUT_NAME} (converted from ${ITER_DIR##*/})"
            echo "pushed: https://huggingface.co/${REPO}"
        fi
        echo "DONE ${OUT_NAME} -> ${OUT_DIR}"
    '
