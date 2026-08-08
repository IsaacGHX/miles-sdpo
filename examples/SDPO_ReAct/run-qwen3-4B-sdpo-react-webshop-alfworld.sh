#!/bin/bash
# Dedicated combined-domain entrypoint: WebShop + ALFWorld trained AND
# evaluated together, SAME 7-arm SDPO ablation structure as
# run-qwen3-4B-sdpo-react-native.sh (math/code/search) -- pins
# run-qwen3-4B-sdpo-react-agentic.sh's SDPO_REACT_DOMAIN to "agentic" (that
# script already fully implements this: build_multitask_data.py combines
# webshop_train.jsonl + alfworld_train.jsonl for training,
# data/eval_agentic.yaml evaluates webshop + alfworld_id + alfworld_ood as
# three separate eval datasets every --eval-interval steps) and gives this
# dataset combo's --dump-details dumps their OWN "webshop_alfworld/"
# subfolder under sdpo_dumps/, distinct from single-domain webshop-only /
# alfworld-only runs of the SAME agentic script.
#
# A THIN WRAPPER, not a duplicate: unlike native vs agentic (different
# sidecars/tool specs/data builders per domain, hence two full ~700-line
# copies there -- see that script's own header comment for the tradeoff),
# this file and run-qwen3-4B-sdpo-react-agentic.sh share EVERY line of
# actual rollout/reward/arm logic; the only difference is which value
# SDPO_REACT_DOMAIN takes plus where dumps land. Sourcing (not copying) is
# the right call for a same-script parameter preset -- copying here would
# just be 700 lines that silently drift out of sync with the real agentic
# script's arm block on the next edit.
#
# Env overrides (forwarded straight through to run-qwen3-4B-sdpo-react-
# agentic.sh -- see that script's own header for the full list):
#   SDPO_REACT_ARM              (default 1.1)   1 | 1.1 | 2 | 3 | 4 | 5 | 5.1
#   SDPO_REACT_WEBSHOP_N_TRAIN  (default 400)
#   SDPO_REACT_WEBSHOP_N_EVAL   (default 100)
#   SDPO_REACT_ALFWORLD_N_TRAIN (default 400)   out of 3553 available train games
#   SDPO_REACT_ALFWORLD_N_EVAL_ID/_OOD (default 100 each)
#   SDPO_REACT_AGENTIC_PER_DOMAIN (default 400) rows/domain in the combined train.jsonl
#
# usage: bash examples/SDPO_ReAct/run-qwen3-4B-sdpo-react-webshop-alfworld.sh
set -exf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SDPO_REACT_DOMAIN=agentic

SDPO_REACT_ARM="${SDPO_REACT_ARM:-1.1}"
export SDPO_REACT_ARM

# Own dump subtree: /root/data/sdpo_dumps/webshop_alfworld/<exp-name>/ -- so
# every run of THIS script groups together under one folder, distinguishable
# at a glance from single-domain agentic-script runs (which dump flat under
# sdpo_dumps/<exp-name>/) and from the native (math/code/search) dumps.
export SDPO_REACT_EXP="${SDPO_REACT_EXP:-webshop_alfworld/qwen3-4B-sdpo-react-webshop-alfworld-${SDPO_REACT_ARM}_$(date +%Y%m%d_%H%M%S)}"

exec bash "$SCRIPT_DIR/run-qwen3-4B-sdpo-react-agentic.sh"
