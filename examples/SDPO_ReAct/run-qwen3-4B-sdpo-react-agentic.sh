#!/bin/bash
# SDPO_ReAct (NATIVE tool-calling) -- Qwen3-4B on WebShop/ALFWorld, multi-turn
# rollout, single 8x H200 (141GB) node, COLOCATE variant. Sibling of
# run-qwen3-4B-sdpo-react-native.sh (math/code/search domains) -- this script
# is the AGENTIC counterpart: same 7-arm SDPO ablation structure, same tool-
# calling infra, different domains (webshop_step/alfworld_step instead of
# code_interpreter/search/open/find).
#
# WHY A SEPARATE FILE (not a domain branch inside the native script): copied
# in full and edited only at the domain-specific blocks (sidecar-start,
# data-prep, tool-specs/executor select, CFG_TAG/wandb naming). The 7-arm
# case block below is copied VERBATIM, character-for-character, from the
# native script -- RM_ARGS/GRPO_ARGS are pure SDPO/skill/RLSD machinery,
# orthogonal to domain, so duplicating it here (rather than sourcing a shared
# fragment) keeps this script's correctness fully independent of the native
# script's (no shared-file coupling risk); see the port's design plan for the
# full tradeoff writeup. Selected via enroot-run-sdpo-react.sh's
# SDPO_REACT_RUN_FAMILY=agentic (default native).
#
# Both webshop_step and alfworld_step are single-action tools (one string
# param) wrapping a STATEFUL per-trajectory sidecar (tools/webshop/,
# tools/alfworld/) -- same architecture as the native script's search tool
# (session-keyed pool of live env instances), NOT SDAR's own raw <action>
# tag-parsing loop. Task assignment (WHICH webshop task / ALFWorld game file
# a row plays) is pinned per-row by the data builders
# (data/build_webshop_data.py, data/build_alfworld_data.py) via
# metadata['webshop_task_id']/['alfworld_game_file'], read out-of-band by
# each tool's client.py through a dedicated contextvar
# (miles.rollout.generate_hub.multi_turn.current_trajectory_metadata) that
# never enters the rendered prompt -- so the SAME task replays across every
# arm/checkpoint's train/eval pass, making success rate directly comparable.
#
# Env overrides (swap without editing the script):
#   SDPO_REACT_TRAIN_MAX_TURNS  (default: domain-dependent, see below) training turn budget
#   SDPO_REACT_EVAL_MAX_TURNS   (default: domain-dependent, see below) eval turn budget (eval yaml)
#   SDPO_REACT_EVAL_N_SAMPLES   (default 8)   eval samples/prompt (eval yaml)
#   SDPO_REACT_NUM_ROLLOUT      (default 300) --num-rollout
#   SDPO_REACT_DOMAIN           (default webshop) webshop | alfworld | agentic
#
# usage: bash examples/SDPO_ReAct/run-qwen3-4B-sdpo-react-agentic.sh
set -exf

export PYTHONBUFFERED=16
export SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-8}"
# Same ablation-arm convention as run-qwen3-4B-sdpo-react-native.sh (see the
# RM_ARGS/GRPO_ARGS case block below): 1 | 1.1 | 2 | 3 | 4 | 5 | 5.1.
SDPO_REACT_ARM="${SDPO_REACT_ARM:-1.1}"
SDPO_REACT_NUM_ROLLOUT="${SDPO_REACT_NUM_ROLLOUT:-300}"

# Batch sizing MUST stay divisible by the data-parallel size (= TRAIN_GPUS/TP).
SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-8}"
N_SAMPLES_PER_PROMPT=8
# Megatron tensor-model-parallel size. Same rationale as the native script:
# TP=2 shards Qwen3.5's vocab-parallel cross-entropy logits buffer AND the
# full-attention + MLP params, buying back headroom against long-tail
# sequences (webshop/alfworld episodes can run many turns before a terminal
# step). dp = TRAIN_GPUS / TP (pp=cp=1).
SDPO_REACT_TP="${SDPO_REACT_TP:-1}"
DP_SIZE=$((SDPO_REACT_TRAIN_GPUS / SDPO_REACT_TP))
ROLLOUT_BATCH_SIZE="${SDPO_REACT_ROLLOUT_BATCH:-$((DP_SIZE * 4))}"
if [ $((ROLLOUT_BATCH_SIZE % DP_SIZE)) -ne 0 ]; then
    ROLLOUT_BATCH_SIZE=$((DP_SIZE * 4))
    echo "WARN: rollout batch not divisible by dp=${DP_SIZE} (train_gpus=${SDPO_REACT_TRAIN_GPUS}/tp=${SDPO_REACT_TP}); falling back to ${ROLLOUT_BATCH_SIZE}"
fi
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
echo "BATCH: train_gpus=${SDPO_REACT_TRAIN_GPUS} tp=${SDPO_REACT_TP} dp=${DP_SIZE} rollout_batch=${ROLLOUT_BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Model family select -- identical to the native script (this axis is
# orthogonal to domain; SDPO_REACT_RUN_FAMILY picks WHICH script runs,
# SDPO_REACT_MODEL picks WHICH model weights within it).
case "${SDPO_REACT_MODEL:-qwen3-native}" in
    qwen3.5-native)
        MODEL_NAME=Qwen3.5-4B
        MODEL_ARG_SH=scripts/models/qwen3.5-4B.sh
        TOOL_PARSER=qwen3_coder
        TOOL_GRAMMAR=qwen3_coder
        MODEL_EXTRA_ARGS=(--dist-ckpt-optim-fully-reshardable)
        MAX_TOKENS_PER_GPU="${SDPO_REACT_MAX_TOKENS_PER_GPU:-2048}"
        ;;
    *)
        MODEL_NAME=Qwen3-4B
        MODEL_ARG_SH=scripts/models/qwen3-4B.sh
        TOOL_PARSER=qwen25
        TOOL_GRAMMAR=qwen25
        MODEL_EXTRA_ARGS=()
        MAX_TOKENS_PER_GPU=12288
        ;;
esac
echo "MODEL: ${MODEL_NAME} | tool-parser=${TOOL_PARSER} | tool-grammar=${TOOL_GRAMMAR}"
source "$REPO_ROOT/${MODEL_ARG_SH}"

# NOTE: unlike the native script, no unconditional code-sandbox start here --
# no arm/domain in this script's scope ever declares code_interpreter in its
# tools list, so starting that sidecar would just be an unused container.

# Task domain (webshop | alfworld | agentic). agentic = both combined via
# build_multitask_data.py, mirroring the native script's "multitask".
SDPO_REACT_DOMAIN="${SDPO_REACT_DOMAIN:-webshop}"
echo "SDPO_REACT_DOMAIN: ${SDPO_REACT_DOMAIN}"

# Turn-budget defaults are DOMAIN-DEPENDENT, set only now that
# SDPO_REACT_DOMAIN is known -- confirmed via a 10-episode gpt-5.6-luna
# quick-look (examples/agentic dashboard, dump_roots/{webshop,alfworld}_luna_
# quicklook) that the two domains need very different budgets:
#   webshop:  steps=[5,5,10,4,3,6,5,7,7,5]   avg=5.7  max=10  (search -> click
#             -> buy is a short, mostly-linear flow; longest observed run only
#             paged through search results a few extra times)
#   alfworld: steps=[5,7,9,7,7,10,18,7,12,26] avg=10.7 max=26 (exploring rooms/
#             opening containers to find the target object has a much longer
#             tail -- the ONLY 2 losses in that run were the model giving up
#             early, not hitting a turn cap, so budget wasn't even the
#             bottleneck for failures, but a too-tight cap still truncates
#             genuine in-progress wins)
# "agentic" (both combined) keeps webshop's tighter numbers as a floor -- a
# combined run mixes both domains' rows through the SAME --generate-max-turns,
# so if you know your run is alfworld-heavy, override
# SDPO_REACT_TRAIN_MAX_TURNS/SDPO_REACT_EVAL_MAX_TURNS up manually.
case "$SDPO_REACT_DOMAIN" in
    alfworld)
        DEFAULT_TRAIN_MAX_TURNS=15
        DEFAULT_EVAL_MAX_TURNS=30
        ;;
    *)
        DEFAULT_TRAIN_MAX_TURNS=8
        DEFAULT_EVAL_MAX_TURNS=20
        ;;
esac
export SDPO_REACT_TRAIN_MAX_TURNS="${SDPO_REACT_TRAIN_MAX_TURNS:-$DEFAULT_TRAIN_MAX_TURNS}"
export SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-$DEFAULT_EVAL_MAX_TURNS}"
echo "TURN BUDGET: train=${SDPO_REACT_TRAIN_MAX_TURNS} eval=${SDPO_REACT_EVAL_MAX_TURNS} (domain=${SDPO_REACT_DOMAIN})"

# --- 0. domain sidecars (idempotent, ONE container / ONE port each for the
# whole job) -- started conditionally, mirroring the native script's search-
# sidecar conditional. ---
if [ "$SDPO_REACT_DOMAIN" = "agentic" ] || [ "$SDPO_REACT_DOMAIN" = "webshop" ]; then
    bash "$SCRIPT_DIR/tools/webshop/run_webshop_sidecar.sh"
fi
if [ "$SDPO_REACT_DOMAIN" = "agentic" ] || [ "$SDPO_REACT_DOMAIN" = "alfworld" ]; then
    bash "$SCRIPT_DIR/tools/alfworld/run_alfworld_sidecar.sh"
fi

export SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-true}"
SDPO_REACT_PURE_DISTILL="${SDPO_REACT_PURE_DISTILL:-true}"
echo "SDPO_REACT_THINKING: ${SDPO_REACT_THINKING} | SDPO_REACT_PURE_DISTILL: ${SDPO_REACT_PURE_DISTILL}"

if [ "$SDPO_REACT_DOMAIN" = "agentic" ]; then
    # --- AGENTIC: webshop + alfworld combined -----------------------------------
    MT_DIR="/root/data/agentic"
    TRAIN_DATA="$MT_DIR/train.jsonl"
    EVAL_CFG="$SCRIPT_DIR/data/eval_agentic.yaml"
    mkdir -p "$MT_DIR" /root/data/webshop_data /root/data/alfworld_data
    [ -f /root/data/webshop_data/webshop_train.jsonl ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_webshop_data \
            --out-dir /root/data/webshop_data --n-train "${SDPO_REACT_WEBSHOP_N_TRAIN:-400}" --n-eval "${SDPO_REACT_WEBSHOP_N_EVAL:-100}")
    [ -f /root/data/alfworld_data/alfworld_train.jsonl ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_alfworld_data \
            --out-dir /root/data/alfworld_data --n-train "${SDPO_REACT_ALFWORLD_N_TRAIN:-400}" \
            --n-eval-id "${SDPO_REACT_ALFWORLD_N_EVAL_ID:-100}" --n-eval-ood "${SDPO_REACT_ALFWORLD_N_EVAL_OOD:-100}")
    MT_SOURCES=(--source "webshop:/root/data/webshop_data/webshop_train.jsonl"
                --source "alfworld:/root/data/alfworld_data/alfworld_train.jsonl")
    [ -f "$TRAIN_DATA" ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_multitask_data \
            --out "$TRAIN_DATA" "${MT_SOURCES[@]}" --per-domain "${SDPO_REACT_AGENTIC_PER_DOMAIN:-400}")
    python - <<PYCK
import json
from collections import Counter
c=Counter(json.loads(l)["metadata"]["domain"] for l in open("$TRAIN_DATA"))
print(f"agentic train rows OK: {dict(c)}")
PYCK
elif [ "$SDPO_REACT_DOMAIN" = "alfworld" ]; then
    # --- ALFWORLD domain only ----------------------------------------------------
    TRAIN_DATA="/root/data/alfworld_data/alfworld_train.jsonl"
    EVAL_CFG="$SCRIPT_DIR/data/eval_alfworld.yaml"
    mkdir -p /root/data/alfworld_data
    [ -f "$TRAIN_DATA" ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_alfworld_data \
            --out-dir /root/data/alfworld_data --n-train "${SDPO_REACT_ALFWORLD_N_TRAIN:-400}" \
            --n-eval-id "${SDPO_REACT_ALFWORLD_N_EVAL_ID:-100}" --n-eval-ood "${SDPO_REACT_ALFWORLD_N_EVAL_OOD:-100}")
    python - <<PYCK
import json
r = json.loads(open("$TRAIN_DATA").readline())
assert r["metadata"]["domain"] == "alfworld" and r["metadata"]["alfworld_game_file"], "alfworld rows missing domain/game_file"
print("alfworld train rows OK")
PYCK
else
    # --- WEBSHOP domain (default) ------------------------------------------------
    TRAIN_DATA="/root/data/webshop_data/webshop_train.jsonl"
    EVAL_CFG="$SCRIPT_DIR/data/eval_webshop.yaml"
    mkdir -p /root/data/webshop_data
    [ -f "$TRAIN_DATA" ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_webshop_data \
            --out-dir /root/data/webshop_data --n-train "${SDPO_REACT_WEBSHOP_N_TRAIN:-400}" --n-eval "${SDPO_REACT_WEBSHOP_N_EVAL:-100}")
    python - <<PYCK
import json
r = json.loads(open("$TRAIN_DATA").readline())
assert r["metadata"]["domain"] == "webshop" and r["metadata"]["webshop_task_id"] is not None, "webshop rows missing domain/task_id"
print("webshop train rows OK")
PYCK
fi

# --- 0c. one-time template check (fails loudly if the native render is wrong) --
python - "$REPO_ROOT" "$SDPO_REACT_DOMAIN" "$MODEL_NAME" <<'PYCHECK'
import sys
from transformers import AutoTokenizer
sys.path.insert(0, sys.argv[1])
domain = sys.argv[2]
model_name = sys.argv[3]
from examples.SDPO_ReAct.tools.webshop.spec import webshop_specs
from examples.SDPO_ReAct.tools.alfworld.spec import alfworld_specs
tok = AutoTokenizer.from_pretrained(f"/root/{model_name}", trust_remote_code=True)
checks = []
if domain in ("webshop", "agentic"):
    checks.append(("webshop", [{"role": "user", "content": "find me a mouse"}], webshop_specs, "webshop_step"))
if domain in ("alfworld", "agentic"):
    checks.append(("alfworld", [{"role": "user", "content": "go to the kitchen"}], alfworld_specs, "alfworld_step"))
for name, msgs, tspecs, tool_name in checks:
    r = tok.apply_chat_template(msgs, tools=tspecs, tokenize=False, add_generation_prompt=True)
    assert "<tools>" in r, f"{name}: native <tools> block missing"
    assert tool_name in r, f"{name}: {tool_name} not in rendered <tools> block"
_SENT = "SDPOSENTINEL"
probe = tok.apply_chat_template([{"role":"user","content":_SENT}], tokenize=False, add_generation_prompt=True)
assert _SENT in probe and probe.split(_SENT,1)[1], "gen_suffix empty -> SDPO splice undefined"
print(f"Native template check OK for {model_name} (domain={domain}, checked={[n for n,_,_,_ in checks]})")
PYCHECK

[ "$SDPO_REACT_PURE_DISTILL" = "true" ] && _DISTILL=pure || _DISTILL=mixed
[ "$SDPO_REACT_THINKING" = "true" ] && _THINK=think || _THINK=nothink
if [ "$SDPO_REACT_THINKING" = "true" ]; then
    REMOVE_THINKING_ARG=()
else
    REMOVE_THINKING_ARG=(--sdpo-remove-thinking-from-demonstration)
fi
# Distinct "sdpo-react-agentic-..." naming (vs the native script's
# "sdpo-react-native-...") so runs never collide across ckpt/wandb dirs.
CFG_TAG="${MODEL_NAME}-agentic-${SDPO_REACT_DOMAIN}-${SDPO_REACT_ARM}-${_DISTILL}-${_THINK}"

SDPO_REACT_EXP="${SDPO_REACT_EXP:-qwen3-4B-sdpo-react-agentic-${CFG_TAG}_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/data/sdpo_dumps/${SDPO_REACT_EXP}"
echo "SDPO_ReAct dump dir: ${DUMP_DIR}"

CKPT_DIR="${SDPO_REACT_CKPT_DIR:-/root/data/sdpo_ckpts/qwen3-4B-sdpo-react-agentic-${CFG_TAG}_ckpt}"

EXPLOG="${SDPO_REACT_EXPLOG:-$REPO_ROOT/examples/SDPO_ReAct/explog.jsonl}"
GIT_COMMIT="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(cd "$REPO_ROOT" && [ -n "$(git status --porcelain 2>/dev/null)" ] && echo dirty || echo clean)"
python - <<PYLOG || true
import json, time, os
row = {
    "exp": "${SDPO_REACT_EXP}",
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "arm": "${SDPO_REACT_ARM}",
    "domain": "${SDPO_REACT_DOMAIN}",
    "model": "${MODEL_NAME}",
    "train_data": "${TRAIN_DATA}",
    "num_rollout": "${SDPO_REACT_NUM_ROLLOUT}",
    "train_max_turns": "${SDPO_REACT_TRAIN_MAX_TURNS}",
    "eval_max_turns": "${SDPO_REACT_EVAL_MAX_TURNS}",
    "wandb_group": "qwen3-4B-sdpo-react-agentic-${SDPO_REACT_ARM}",
    "dump_dir": "${DUMP_DIR}",
    "ckpt_dir": "${CKPT_DIR}",
    "git_commit": "${GIT_COMMIT}",
    "git_state": "${GIT_DIRTY}",
    "note": os.environ.get("SDPO_REACT_NOTE", ""),
}
with open("${EXPLOG}", "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("explog appended ->", "${EXPLOG}")
print(json.dumps(row, ensure_ascii=False, indent=2))
PYLOG

CKPT_ARGS=(
   --hf-checkpoint /root/${MODEL_NAME}
   --ref-load /root/${MODEL_NAME}_torch_dist
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)
if [ "${SDPO_REACT_SAVE_CKPT:-0}" = "1" ]; then
    CKPT_ARGS+=(--save "${CKPT_DIR}" --load "${CKPT_DIR}" --save-interval 10 --override-opt-param-scheduler)
fi

if [ "$SDPO_REACT_DOMAIN" = "agentic" ]; then
    SHUFFLE_ARGS=()
    echo "ROLLOUT: --rollout-shuffle OFF (agentic -> preserve balanced per-batch domain mix)"
else
    SHUFFLE_ARGS=(--rollout-shuffle)
fi
ROLLOUT_ARGS=(
   --prompt-data "$TRAIN_DATA"
   --input-key prompt
   --label-key label
   --tool-key tools
   --apply-chat-template
   --apply-chat-template-kwargs "{\"enable_thinking\":${SDPO_REACT_THINKING}}"
   "${SHUFFLE_ARGS[@]}"
   --num-rollout "${SDPO_REACT_NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len 8192
   --rollout-max-context-len 40960
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
   --over-sampling-batch-size "${ROLLOUT_BATCH_SIZE}"
)
if [ "${SDPO_REACT_DYNAMIC_SAMPLE:-true}" != "true" ]; then
    echo "DYNAMIC SAMPLING: OFF (raw batch, no oversample/filter/rebalance)"
elif [ "$SDPO_REACT_ARM" = "1" ] || [ "$SDPO_REACT_PURE_DISTILL" != "true" ]; then
    # arm 1 (plain GRPO, sdpo_react_plain_grpo_reward, no --group-rm) never
    # stamps sample.metadata["sdpo_correct"] -- see the native script's
    # identical comment on this same branch.
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
else
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_sdpo_group_has_prefix)
    ROLLOUT_ARGS+=(--sdpo-dynamic-filter-min-correct "${SDPO_REACT_MIN_CORRECT:-1}")
fi

# Tool specs/executor: agentic -> registry.agentic_tool_specs (webshop_step +
# alfworld_step ONLY -- deliberately NOT all_tool_specs, so a mixed rollout
# can't accidentally parse a stray code_interpreter/search call no row in
# this domain ever declares). Single-domain -> that domain's own single-tool
# spec/client (no registry needed at all).
if [ "$SDPO_REACT_DOMAIN" = "agentic" ]; then
    TOOL_SPECS_PATH="examples.SDPO_ReAct.tools.registry.agentic_tool_specs"
    EXECUTE_TOOL_PATH="examples.SDPO_ReAct.tools.registry.execute_tool"
elif [ "$SDPO_REACT_DOMAIN" = "alfworld" ]; then
    TOOL_SPECS_PATH="examples.SDPO_ReAct.tools.alfworld.spec.alfworld_specs"
    EXECUTE_TOOL_PATH="examples.SDPO_ReAct.tools.alfworld.client.execute_tool"
else
    TOOL_SPECS_PATH="examples.SDPO_ReAct.tools.webshop.spec.webshop_specs"
    EXECUTE_TOOL_PATH="examples.SDPO_ReAct.tools.webshop.client.execute_tool"
fi
ROLLOUT_ARGS+=(--tool-specs-resolver-path "$TOOL_SPECS_PATH")
CUSTOM_GENERATE_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate
   --generate-tool-specs-path "$TOOL_SPECS_PATH"
   --generate-execute-tool-function-path "$EXECUTE_TOOL_PATH"
   --generate-tool-call-parser "$TOOL_PARSER"
   --generate-max-turns "${SDPO_REACT_TRAIN_MAX_TURNS}"
)

# Ablation-arm series -- IDENTICAL semantics to run-qwen3-4B-sdpo-react-native
# .sh's own arm block (copied verbatim below; see that script's header
# comment for the full arm-by-arm rationale, RLSD design note, and why
# --no-sdpo-pure-distill is set on every SDPO arm). RM_ARGS/GRPO_ARGS are
# pure SDPO/skill/RLSD machinery, orthogonal to domain -- webshop/alfworld
# episodes just have no LLM-judge fallback path (grading is a hard boolean
# from the sidecar, no text-answer ambiguity), so the native script's
# search-judge-fallback block has NO counterpart here.
#
#   SDPO_REACT_ARM=1    Plain GRPO, no SDPO machinery at all.
#   SDPO_REACT_ARM=1.1  RLSD baseline, NO self-skill.
#   SDPO_REACT_ARM=2    + self-skill, skill-source correct only, NO skill-KD.
#   SDPO_REACT_ARM=3    + self-skill, skill-source incorrect only, NO skill-KD.
#   SDPO_REACT_ARM=4    + self-skill, skill-source all, NO skill-KD.
#   SDPO_REACT_ARM=5    + self-skill, skill-source all, WITH skill-KD (mode=both).
#   SDPO_REACT_ARM=5.1  IDENTICAL to arm 5 except --sdpo-skill-kd-mode both-blind.
RM_ARGS=(
   --sdpo-answer-tag answer
   "${REMOVE_THINKING_ARG[@]}"
   --sdpo-reframe-multiturn-prefix
   --sdpo-tool-grammar "${TOOL_GRAMMAR}"
   --sdpo-max-prefix-chars 20000
)

case "${SDPO_REACT_ARM}" in
   1)
      RM_ARGS=(
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_plain_grpo_reward
         --sdpo-grader dapo
         --sdpo-answer-tag answer
      )
      GRPO_ARGS=(
         --advantage-estimator grpo
         --entropy-coef 0.00
      )
      ;;
   1.1)
      RM_ARGS+=(
         --group-rm
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
         --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
         --sdpo-grader dapo
      )
      GRPO_ARGS=(
         --advantage-estimator grpo
         --no-sdpo-pure-distill
         --sdpo-teacher-backend megatron
         --sdpo-ema-teacher
         --sdpo-ema-teacher-rate 0.05
         --sdpo-logprob-mode sampled
         --sdpo-rlsd
         --sdpo-rlsd-clip-eps 0.2
         --sdpo-rlsd-lambda-init 1.0
         --sdpo-rlsd-lambda-warmup-steps 0
         --use-tis
         --sdpo-self-teacher
         --sdpo-prefer-tool-use-peer
         --entropy-coef 0.00
         --calculate-per-token-loss
      )
      ;;
   2)
      RM_ARGS+=(
         --group-rm
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
         --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
         --sdpo-grader dapo
         --sdpo-self-skill
         --sdpo-skill-source correct
         --sdpo-skill-max-new-tokens 1024
         --sdpo-response-prefix skill
      )
      GRPO_ARGS=(
         --advantage-estimator grpo
         --no-sdpo-pure-distill
         --sdpo-teacher-backend megatron
         --sdpo-ema-teacher
         --sdpo-ema-teacher-rate 0.05
         --sdpo-logprob-mode sampled
         --sdpo-rlsd
         --sdpo-rlsd-clip-eps 0.2
         --sdpo-rlsd-lambda-init 1.0
         --sdpo-rlsd-lambda-warmup-steps 0
         --use-tis
         --sdpo-self-teacher
         --sdpo-prefer-tool-use-peer
         --entropy-coef 0.00
         --calculate-per-token-loss
      )
      ;;
   3)
      RM_ARGS+=(
         --group-rm
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
         --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
         --sdpo-grader dapo
         --sdpo-self-skill
         --sdpo-skill-source incorrect
         --sdpo-skill-max-new-tokens 1024
         --sdpo-pitfall-summary-backend self
         --sdpo-env-feedback-max-chars 2000
      )
      GRPO_ARGS=(
         --advantage-estimator grpo
         --no-sdpo-pure-distill
         --sdpo-teacher-backend megatron
         --sdpo-ema-teacher
         --sdpo-ema-teacher-rate 0.05
         --sdpo-logprob-mode sampled
         --sdpo-rlsd
         --sdpo-rlsd-clip-eps 0.2
         --sdpo-rlsd-lambda-init 1.0
         --sdpo-rlsd-lambda-warmup-steps 0
         --use-tis
         --sdpo-self-teacher
         --sdpo-prefer-tool-use-peer
         --entropy-coef 0.00
         --calculate-per-token-loss
      )
      ;;
   4)
      RM_ARGS+=(
         --group-rm
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
         --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
         --sdpo-grader dapo
         --sdpo-self-skill
         --sdpo-skill-source all
         --sdpo-skill-max-new-tokens 1024
         --sdpo-pitfall-summary-backend self
         --sdpo-response-prefix skill
         --sdpo-env-feedback-max-chars 2000
      )
      GRPO_ARGS=(
         --advantage-estimator grpo
         --no-sdpo-pure-distill
         --sdpo-teacher-backend megatron
         --sdpo-ema-teacher
         --sdpo-ema-teacher-rate 0.05
         --sdpo-logprob-mode sampled
         --sdpo-rlsd
         --sdpo-rlsd-clip-eps 0.2
         --sdpo-rlsd-lambda-init 1.0
         --sdpo-rlsd-lambda-warmup-steps 0
         --use-tis
         --sdpo-self-teacher
         --sdpo-prefer-tool-use-peer
         --entropy-coef 0.00
         --calculate-per-token-loss
      )
      ;;
   5)
      RM_ARGS+=(
         --group-rm
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
         --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
         --sdpo-grader dapo
         --sdpo-self-skill
         --sdpo-skill-source all
         --sdpo-skill-max-new-tokens 2048
         --sdpo-pitfall-summary-backend self
         --sdpo-response-prefix skill
         --sdpo-env-feedback-max-chars 2000
         --sdpo-skill-kd
         --sdpo-skill-kd-coef 0.01
         --sdpo-skill-kd-mode both
      )
      GRPO_ARGS=(
         --advantage-estimator grpo
         --no-sdpo-pure-distill
         --sdpo-teacher-backend megatron
         --sdpo-ema-teacher
         --sdpo-ema-teacher-rate 0.05
         --sdpo-logprob-mode sampled
         --sdpo-rlsd
         --sdpo-rlsd-clip-eps 0.2
         --sdpo-rlsd-lambda-init 1.0
         --sdpo-rlsd-lambda-warmup-steps 0
         --use-tis
         --sdpo-self-teacher
         --sdpo-prefer-tool-use-peer
         --entropy-coef 0.00
         --calculate-per-token-loss
      )
      ;;
   5.1)
      RM_ARGS+=(
         --group-rm
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
         --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
         --sdpo-grader dapo
         --sdpo-self-skill
         --sdpo-skill-source all
         --sdpo-skill-max-new-tokens 2048
         --sdpo-pitfall-summary-backend self
         --sdpo-response-prefix skill
         --sdpo-env-feedback-max-chars 2000
         --sdpo-skill-kd
         --sdpo-skill-kd-coef 0.01
         --sdpo-skill-kd-mode both-blind
      )
      GRPO_ARGS=(
         --advantage-estimator grpo
         --no-sdpo-pure-distill
         --sdpo-teacher-backend megatron
         --sdpo-ema-teacher
         --sdpo-ema-teacher-rate 0.05
         --sdpo-logprob-mode sampled
         --sdpo-rlsd
         --sdpo-rlsd-clip-eps 0.2
         --sdpo-rlsd-lambda-init 1.0
         --sdpo-rlsd-lambda-warmup-steps 0
         --use-tis
         --sdpo-self-teacher
         --sdpo-prefer-tool-use-peer
         --entropy-coef 0.00
         --calculate-per-token-loss
      )
      ;;
   *)
      echo "Unknown SDPO_REACT_ARM='${SDPO_REACT_ARM}' (expected: 1 | 1.1 | 2 | 3 | 4 | 5 | 5.1)" >&2
      exit 1 ;;
esac

EVAL_ARGS=(
   --eval-interval 10
   --eval-config "$EVAL_CFG"
   --eval-tool-key tools
   --n-samples-per-eval-prompt "${SDPO_REACT_EVAL_N_SAMPLES}"
   --log-passrate
)
if [ "${SDPO_REACT_SKIP_EVAL0:-0}" = "1" ]; then
    EVAL_ARGS+=(--skip-eval-before-train)
    echo "EVAL: --skip-eval-before-train (debug: no step-0 eval)"
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${SDPO_REACT_TP:-1}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-12288}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-warmup-iters 10
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-sdpo
   --wandb-group "qwen3-4B-sdpo-react-agentic-${CFG_TAG}"
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.75}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-8}"
if [ "$SDPO_REACT_TRAIN_GPUS" -lt 8 ]; then
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((SDPO_REACT_TRAIN_GPUS-1)))"
    echo "TRAIN GPUs: ${SDPO_REACT_TRAIN_GPUS} (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
fi

ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
sleep 2

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "${SDPO_REACT_TRAIN_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# --- checkpoint pruner (background): identical to the native script's own --
# keep only the newest COMPLETED iter dir. See that script's comment for the
# race-fix rationale (only delete STRICTLY OLDER iter dirs than `latest`).
if [ "${SDPO_REACT_SAVE_CKPT:-0}" = "1" ]; then
(
    set +f
    while true; do
        sleep 60
        latest_file="${CKPT_DIR}/latest_checkpointed_iteration.txt"
        [ -f "$latest_file" ] || continue
        latest_num="$(cat "$latest_file" 2>/dev/null)"
        [ -n "$latest_num" ] || continue
        case "$latest_num" in ''|*[!0-9]*) continue;; esac
        for d in "${CKPT_DIR}"/iter_*; do
            [ -d "$d" ] || continue
            n="$(basename "$d")"; n="${n#iter_}"
            case "$n" in ''|*[!0-9]*) continue;; esac
            if [ "$((10#$n))" -lt "$((10#$latest_num))" ]; then
                rm -rf "$d"
            fi
        done
    done
) &
PRUNER_PID=$!
trap 'kill "$PRUNER_PID" 2>/dev/null || true' EXIT
fi

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"/root/Megatron-LM/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
        \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
        \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\",
        \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
        \"SDPO_REACT_EVAL_MAX_TURNS\": \"${SDPO_REACT_EVAL_MAX_TURNS}\",
        \"SDPO_REACT_EVAL_N_SAMPLES\": \"${SDPO_REACT_EVAL_N_SAMPLES}\"
     }
   }" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${SDPO_REACT_TRAIN_GPUS}" \
   --rollout-num-gpus "${SDPO_REACT_TRAIN_GPUS}" \
   --colocate \
   --update-weights-interval 1 \
   ${MODEL_ARGS[@]} \
   ${MODEL_EXTRA_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${CUSTOM_GENERATE_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
