#!/bin/bash
# SDPO_ReAct (NATIVE tool-calling) -- LARGE Qwen3.5 models: dense Qwen3.5-27B OR
# MoE Qwen3.5-35B-A3B (35B total / ~3B active, 256 experts, top-8). Same native
# <function=..> multi-turn tool-calling rollout + SDPO self-skill ablation ARMS
# (1 | 1.1 | 2 | 3 | 4 | 5 | 5.1) as run-qwen3-4B-sdpo-react-native.sh -- this is
# a MODEL swap of that script, NOT a new experiment design. All the arm/RM/GRPO
# logic, tools, data prep, eval config, and dynamic-sampling machinery are
# IDENTICAL; only the model block (parallelism + MoE-specific flags) differs.
#
# Why a separate script (not just a new case in the 4B one): the 27B/35B need
# real model-parallelism (TP for the dense 27B, expert-parallel + R3 for the MoE
# 35B-A3B) and much lower per-GPU token budgets, so the batch/parallelism math
# and the MoE router-alignment flags are large enough that folding them into the
# 4B launcher would obscure both. The arm case block below is copied verbatim
# from run-qwen3-4B-sdpo-react-native.sh so the ablation stays comparable.
#
# ============================ MoE NOTES (READ) ==============================
# Qwen3.5-35B-A3B is a fine-grained MoE. Two things matter for RL correctness on
# top of the usual dense training:
#
# 1. EXPERT PARALLELISM + LOAD. We shard the 256 experts across the 8 GPUs with
#    --expert-model-parallel-size 8 (--expert-tensor-parallel-size 1), matching
#    scripts/run-qwen3.5-35B-A3B-mtp.sh. The model .sh already sets the arch-level
#    MoE flags (--num-experts 256, --moe-router-topk 8, --moe-grouped-gemm,
#    --moe-token-drop-policy probs, --moe-router-dtype fp32, --moe-aux-loss-coeff 0,
#    --moe-shared-expert-gate, ...). NOTE aux-loss-coeff is 0 by design: in RL we
#    do NOT add an auxiliary load-balance loss (it would fight the policy
#    gradient); balance is handled by the drop policy + the fp32 router, exactly
#    as the reference MoE launchers do. We also switch the token dispatcher to
#    `flex` (--moe-token-dispatcher-type flex in MISC_ARGS) as the 35B-A3B-mtp
#    reference does -- it's the dispatcher that supports the variable-length
#    packed sequences this multi-turn rollout produces.
#
# 2. TRAIN/INFERENCE ROUTER ALIGNMENT (R3). This is the MoE-specific analogue of
#    the log-prob mismatch TIS already corrects for dense models. The rollout
#    (SGLang) and the training forward (Megatron) are two different MoE kernels
#    whose routers can pick DIFFERENT top-k experts for the same token -> the
#    training gradient is computed through a different sub-network than the one
#    that generated the trajectory, i.e. an off-policy routing mismatch on TOP of
#    the usual logprob mismatch. miles implements Rollout Routing Replay (R3,
#    arXiv:2510.11370): SGLang returns the per-token routed-expert indices
#    (enable_return_routed_experts), they ride on sample.rollout_routed_experts
#    through the SAME compute_request_payload / update_sample_from_response path
#    this native multi_turn.generate uses (verified: generate_endpoint_utils.py
#    sets return_routed_experts from --use-rollout-routing-replay), and the
#    training POLICY forward replays exactly those expert choices instead of
#    re-routing. Enabled here for the MoE model via --use-rollout-routing-replay
#    (train) + --sglang-ep-size 8 (rollout). It is a NO-OP / not applicable for
#    the dense 27B (no experts), so we only add it for the MoE model.
#
#    SDPO INTERACTION (the prefix-differs question): R3 replays routing ONLY on
#    the policy log-prob forward over the ACTUAL rollout tokens (stage
#    "replay_forward"). The SDPO teacher / ref / EMA-teacher forwards -- the ones
#    that see a DIFFERENT prefix (peer trace / skill / pitfall) -- are explicitly
#    run under replay stage "fallthrough" (actor.py:_set_replay_stage), i.e. they
#    re-route normally and do NOT consume the rollout's replay list. So R3 fixes
#    the on-policy routing mismatch for the term that MUST be on-policy (the PG /
#    RLSD reweighting term over the sampled trajectory) without corrupting the
#    teacher-prefix forwards whose token sequence isn't the rollout's anyway.
#    That is exactly what we want: it stays useful under SDPO/RLSD.
# ============================================================================
#
# Env overrides (same names as the 4B script where they overlap):
#   SDPO_REACT_MODEL   qwen3.5-27B | qwen3.5-35B-A3B   (default qwen3.5-35B-A3B)
#   SDPO_REACT_ARM     1 | 1.1 | 2 | 3 | 4 | 5 | 5.1   (default 1.1)
#   SDPO_REACT_DOMAIN  math | code | search | multitask (default multitask)
#   SDPO_REACT_R3      true|false  force R3 on/off (default: on for MoE, off dense)
#   SDPO_REACT_EP_SIZE expert-parallel size for the MoE model (default 8)
#   ... plus every SDPO_REACT_* the 4B script honors (TRAIN_MAX_TURNS, PROMPT,
#       THINKING, PURE_DISTILL, NUM_ROLLOUT, MT_PER_DOMAIN, SAVE_CKPT, ...).
#
# usage (via the enroot wrapper, which downloads+converts the model):
#   SDPO_REACT_MODEL=qwen3.5-35B-A3B SDPO_REACT_ARM=1.1 \
#     bash examples/SDPO_ReAct/enroot-run-sdpo-react.sh
set -exf

export PYTHONBUFFERED=16
export SDPO_REACT_TRAIN_MAX_TURNS="${SDPO_REACT_TRAIN_MAX_TURNS:-8}"
export SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-20}"
export SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-8}"
SDPO_REACT_ARM="${SDPO_REACT_ARM:-1.1}"
SDPO_REACT_NUM_ROLLOUT="${SDPO_REACT_NUM_ROLLOUT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-8}"
N_SAMPLES_PER_PROMPT=8

# --------------------------------------------------------------------------- #
# Model family select. BOTH models are Qwen3.5 (same tokenizer/vocab=248320,
# same qwen3_coder XML tool grammar, same qwen3_5 megatron spec); they differ in
# arch (dense vs MoE) -> parallelism, per-GPU token budget, and MoE-only flags.
#   qwen3.5-27B     -> DENSE 27B: TP=4 (weights + vocab-parallel CE need sharding)
#   qwen3.5-35B-A3B -> MoE 35B/A3B: EP=8, R3 router-replay, flex dispatcher, MTP off
# --------------------------------------------------------------------------- #
SDPO_REACT_MODEL="${SDPO_REACT_MODEL:-qwen3.5-35B-A3B}"
TOOL_PARSER=qwen3_coder      # Qwen3.5 tool-call grammar is XML <function=..>
TOOL_GRAMMAR=qwen3_coder
IS_MOE=0

case "${SDPO_REACT_MODEL}" in
    qwen3.5-27B)
        MODEL_NAME=Qwen3.5-27B
        MODEL_ARG_SH=scripts/models/qwen3.5-27B.sh
        # Dense 27B (~54GB bf16 weights): shard across the node with TP. TP=4
        # matches scripts/run-qwen3.5-27B.sh (the standalone dense launcher);
        # dp = TRAIN_GPUS/TP = 2. Qwen3.5's attention-output-gate + vocab=248320
        # cross-entropy make TP essential (the same vocab-parallel CE buffer that
        # forced the 4B down to 2048 tokens/gpu is now sharded 4-way).
        SDPO_REACT_TP="${SDPO_REACT_TP:-4}"
        SDPO_REACT_EP_SIZE=1
        # --dist-ckpt-optim-fully-reshardable: Qwen3.5's hybrid attention gives
        # uneven distributed-optimizer buckets (same reason as the 4B script's
        # note); keep it for safe checkpoint save at any dp.
        MODEL_EXTRA_ARGS=(--dist-ckpt-optim-fully-reshardable)
        MAX_TOKENS_PER_GPU="${SDPO_REACT_MAX_TOKENS_PER_GPU:-8192}"
        # Rollout SGLang: TP>1 produces garbage for Qwen3.5 on the pinned sglang
        # 0.5.9 (github.com/sgl-project/sglang/issues/21039); run the engine at
        # tp=1 per GPU (a 27B dense fits one H200 for inference) + low mem-frac so
        # the colocated train step gets VRAM back.
        ROLLOUT_GPUS_PER_ENGINE=1
        SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.5}"
        R3_DEFAULT=false
        ;;
    qwen3.5-35B-A3B)
        MODEL_NAME=Qwen3.5-35B-A3B
        MODEL_ARG_SH=scripts/models/qwen3.5-35B-A3B.sh
        IS_MOE=1
        # CHECKPOINT-SAVE OOM fix (the RIGHT one -- do NOT shrink mem-fraction):
        # the rollout-1/10 save OOM'd on a 15.4 GB alloc even after the KV cache
        # was offloaded (colocate defaults --offload-rollout-level "kv_cache
        # weight", so KV is already freed before save -- verified). The culprit is
        # --dist-ckpt-optim-fully-reshardable: it GATHERS the full distributed
        # optimizer state into a transient 15.4 GB GPU buffer at save
        # ("Storing distributed optimizer sharded state of type fully_reshardable"
        # -> TorchMemorySaver OOM). That flag was only ever needed for the dp=7
        # "empty bucket" assert (retriever held GPU7); at TP=2 dp=8/2=4 (even) the
        # default dp_reshardable save has no empty bucket AND does the per-shard
        # save with NO full gather -- exactly what the reference TP2/EP8 launcher
        # (run_qwen3_5_35b_a3b_mtp_cp2_ep8.py) uses. So the MoE does NOT set
        # fully_reshardable (see MODEL_EXTRA_ARGS below). This keeps mem-fraction
        # high (fast rollout) AND lets the save fit.
        # MoE: keep TP=1 and shard the 256 experts with EXPERT parallelism instead
        # (EP=8 across the node), matching scripts/run-qwen3.5-35B-A3B-mtp.sh. With
        # ep=8/tp=1/pp=1/cp=1 the data-parallel size = TRAIN_GPUS (the batch guard
        # below divides by dp accordingly). Only ~3B params are active per token,
        # so per-GPU compute is light; memory is dominated by holding the expert
        # weights sharded 8-way.
        SDPO_REACT_TP="${SDPO_REACT_TP:-1}"
        SDPO_REACT_EP_SIZE="${SDPO_REACT_EP_SIZE:-8}"
        # CHECKPOINT save/resume -- back to dp_reshardable (the default, no flag),
        # NOT fully_reshardable. History of this decision:
        #   - dp_reshardable saves fine (per-shard, no full-gather) but its RESUME
        #     used to fail: loading the optimizer param_state hit "Cannot merge two
        #     lists with different lengths (93 and 196)" -- Megatron's own merge()
        #     (dist_checkpointing/dict_utils.py) already has a truncation guard for
        #     this padding mismatch, but ONLY when the on-disk list is SHORTER than
        #     the current run's expected list; our case was the OPPOSITE (on-disk
        #     196 > current 93, from loading under a reshaped TP=2 layout), which
        #     the upstream guard doesn't cover -> raises instead of truncating.
        #   - fully_reshardable resumes fine (resharding-agnostic) but its
        #     synchronous save full-GATHERS the optimizer into a 15.4 GB GPU buffer
        #     per rank -> GPU OOM. Pairing it with --async-save doesn't help either:
        #     getting async_save to actually take (see async_save.yaml) just moves
        #     the same full-gather from GPU to a HOST-memory copy PER RANK (~215 GB
        #     each x8 ranks, on top of --optimizer-cpu-offload's already-resident
        #     copy) -> HOST OOM (ray killed a worker at 1996/2000 GB). fully_reshardable's
        #     per-rank-near-complete-gather design just doesn't fit this model's size
        #     on this host, GPU or CPU, sync or async.
        # FIX (the right one): widen Megatron's own truncation guard to BOTH
        # directions via monkey_patch_dist_checkpointing_merge()
        # (miles/utils/reloadable_process_group.py, called from
        # MegatronTrainRayActor.init before load_checkpoint) instead of avoiding
        # dp_reshardable altogether. dp_reshardable's per-rank memory footprint is
        # ~1/8 of fully_reshardable's, so no OOM on either side, AND resume now works.
        MODEL_EXTRA_ARGS=()
        MAX_TOKENS_PER_GPU="${SDPO_REACT_MAX_TOKENS_PER_GPU:-8192}"
        # Rollout: one 8-GPU EP engine for the MoE (matches the reference: the MoE
        # inference wants all experts co-resident, so the engine spans the node).
        ROLLOUT_GPUS_PER_ENGINE=8
        # mem-fraction stays HIGH (0.7) for fast rollout -- we do NOT shrink it to
        # survive the save (that both slows rollout and doesn't fix the real cause,
        # which was fully_reshardable's full-gather, now removed). KV cache is
        # already offloaded before the save by colocate's default
        # --offload-rollout-level "kv_cache weight", so the save has headroom.
        SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.7}"
        # R3 (rollout routing replay) ON by default for the MoE -- see the MoE
        # NOTES header. This is the train/inference router alignment.
        R3_DEFAULT=true
        ;;
    *)
        echo "Unknown SDPO_REACT_MODEL='${SDPO_REACT_MODEL}' (expected: qwen3.5-27B | qwen3.5-35B-A3B)" >&2
        exit 1 ;;
esac

# R3 final decision: honor an explicit SDPO_REACT_R3 override, else the per-model
# default (on for MoE, off for dense). Guard: never enable R3 on a dense model
# (it has no experts to replay -> no-op at best).
SDPO_REACT_R3="${SDPO_REACT_R3:-$R3_DEFAULT}"
if [ "$IS_MOE" != "1" ]; then SDPO_REACT_R3=false; fi
echo "MODEL: ${MODEL_NAME} | moe=${IS_MOE} tp=${SDPO_REACT_TP} ep=${SDPO_REACT_EP_SIZE} R3=${SDPO_REACT_R3} | tool-grammar=${TOOL_GRAMMAR}"
source "$REPO_ROOT/${MODEL_ARG_SH}"

# Batch sizing: divisible by data-parallel size (dp = TRAIN_GPUS/TP; ep does NOT
# reduce dp -- expert parallel is orthogonal to the data-parallel batch split).
DP_SIZE=$((SDPO_REACT_TRAIN_GPUS / SDPO_REACT_TP))
ROLLOUT_BATCH_SIZE="${SDPO_REACT_ROLLOUT_BATCH:-$((DP_SIZE * 4))}"
if [ $((ROLLOUT_BATCH_SIZE % DP_SIZE)) -ne 0 ]; then
    ROLLOUT_BATCH_SIZE=$((DP_SIZE * 4))
    echo "WARN: rollout batch not divisible by dp=${DP_SIZE}; falling back to ${ROLLOUT_BATCH_SIZE}"
fi
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
echo "BATCH: train_gpus=${SDPO_REACT_TRAIN_GPUS} tp=${SDPO_REACT_TP} dp=${DP_SIZE} rollout_batch=${ROLLOUT_BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE}"

# --- 0. sandbox sidecar (idempotent) ---
bash "$SCRIPT_DIR/tools/run_sandbox.sh"

SDPO_REACT_DOMAIN="${SDPO_REACT_DOMAIN:-multitask}"
echo "SDPO_REACT_DOMAIN: ${SDPO_REACT_DOMAIN}"

# --- 0a. search sidecar (only for domains that use search) ---
if [ "$SDPO_REACT_DOMAIN" = "multitask" ] || [ "$SDPO_REACT_DOMAIN" = "search" ]; then
    bash "$SCRIPT_DIR/tools/search/run_search_sidecar.sh"
fi

export SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-true}"
SDPO_REACT_PURE_DISTILL="${SDPO_REACT_PURE_DISTILL:-true}"
echo "SDPO_REACT_THINKING: ${SDPO_REACT_THINKING} | SDPO_REACT_PURE_DISTILL: ${SDPO_REACT_PURE_DISTILL}"

export SDPO_REACT_PROMPT="${SDPO_REACT_PROMPT:-minimal}"
DATA_SUFFIX="$SDPO_REACT_PROMPT"

# ============================ DATA PREP (unchanged from the 4B script) ========
if [ "$SDPO_REACT_DOMAIN" = "multitask" ]; then
    MT_DIR="/root/data/multitask"
    TRAIN_DATA="$MT_DIR/train.jsonl"
    EVAL_CFG="$SCRIPT_DIR/data/eval_multitask.yaml"
    mkdir -p "$MT_DIR" /root/dapo-math-17k /root/data/code_data /root/data/code_data_v6eval "/root/math_eval/native-${DATA_SUFFIX}"
    export SDPO_REACT_EVAL_DIR="/root/math_eval/native-${DATA_SUFFIX}"
    [ -f /root/dapo-math-17k/dapo-math-17k.jsonl ] || \
        hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
    [ -f "/root/dapo-math-17k/dapo-math-17k-native-${DATA_SUFFIX}.jsonl" ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.native_prompt \
            --in /root/dapo-math-17k/dapo-math-17k.jsonl \
            --out "/root/dapo-math-17k/dapo-math-17k-native-${DATA_SUFFIX}.jsonl")
    [ -f /root/data/code_data/livecodebench_train.jsonl ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_code_data \
            --out-dir /root/data/code_data --testtype stdin --difficulty medium,hard --n-train 2000 --n-eval 1)
    [ -f /root/data/code_data_v6eval/livecodebench_eval.jsonl ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_code_data \
            --out-dir /root/data/code_data_v6eval --testtype stdin --difficulty easy,medium,hard \
            --min-date 2025-02 --n-train 0 --n-eval 80)
    [ -f "/root/math_eval/native-${DATA_SUFFIX}/aime25_native.jsonl" ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_native_eval --out-dir "/root/math_eval/native-${DATA_SUFFIX}")
    SEARCH_TRAIN=/root/data/search_data/search_train_passk.jsonl
    [ -f "$SEARCH_TRAIN" ] || SEARCH_TRAIN=/root/data/search_data/search_train.jsonl
    [ -f "$SEARCH_TRAIN" ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_search_data \
            --out-dir /root/data/search_data --n-train-per 1500 --n-val-per 100)
    MT_SOURCES=(--source "math:/root/dapo-math-17k/dapo-math-17k-native-${DATA_SUFFIX}.jsonl"
                --source "code:/root/data/code_data/livecodebench_train.jsonl"
                --source "search:$SEARCH_TRAIN")
    [ -f "$TRAIN_DATA" ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_multitask_data \
            --out "$TRAIN_DATA" "${MT_SOURCES[@]}" --per-domain "${SDPO_REACT_MT_PER_DOMAIN:-400}")
    python - <<PYCK
import json
from collections import Counter
c=Counter(json.loads(l)["metadata"]["domain"] for l in open("$TRAIN_DATA"))
print(f"multitask train rows OK: {dict(c)}")
PYCK
elif [ "$SDPO_REACT_DOMAIN" = "code" ]; then
    CODE_DIR="/root/data/code_data"
    CODE_EVAL_DIR="/root/data/code_data_v6eval"
    TRAIN_DATA="$CODE_DIR/livecodebench_train.jsonl"
    EVAL_CFG="$SCRIPT_DIR/data/eval_code.yaml"
    mkdir -p "$CODE_DIR" "$CODE_EVAL_DIR"
    [ -f "$TRAIN_DATA" ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_code_data \
            --out-dir "$CODE_DIR" --testtype stdin --difficulty medium,hard --n-train 2000 --n-eval 1)
    [ -f "$CODE_EVAL_DIR/livecodebench_eval.jsonl" ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_code_data \
            --out-dir "$CODE_EVAL_DIR" --testtype stdin --difficulty easy,medium,hard \
            --min-date 2025-02 --n-train 0 --n-eval 80)
    python - <<PYCK
import json
r = json.loads(open("$TRAIN_DATA").readline())
assert r["metadata"]["domain"] == "code" and r["metadata"]["test_cases"], "code rows missing domain/test_cases"
print(f"code train rows OK; example has {len(r['metadata']['test_cases'])} test cases")
PYCK
else
    TRAIN_DATA="/root/dapo-math-17k/dapo-math-17k-native-${DATA_SUFFIX}.jsonl"
    EVAL_DIR="/root/math_eval/native-${DATA_SUFFIX}"
    EVAL_CFG="$SCRIPT_DIR/data/eval_native_math.yaml"
    export SDPO_REACT_EVAL_DIR="$EVAL_DIR"
    mkdir -p /root/dapo-math-17k "$EVAL_DIR"
    [ -f /root/dapo-math-17k/dapo-math-17k.jsonl ] || \
        hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
    [ -f "$TRAIN_DATA" ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.native_prompt \
            --in /root/dapo-math-17k/dapo-math-17k.jsonl \
            --out "$TRAIN_DATA")
    [ -f "$EVAL_DIR/aime25_native.jsonl" ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_native_eval --out-dir "$EVAL_DIR")
fi

# --- 0c. one-time template check (same as the 4B script) --------------------
python - "$REPO_ROOT" "$SDPO_REACT_DOMAIN" "$MODEL_NAME" <<'PYCHECK'
import sys
from transformers import AutoTokenizer
sys.path.insert(0, sys.argv[1])
domain = sys.argv[2]
model_name = sys.argv[3]
from examples.SDPO_ReAct.tools.tool_specs import tool_specs
from examples.SDPO_ReAct.tools.search.spec import search_specs
tok = AutoTokenizer.from_pretrained(f"/root/{model_name}", trust_remote_code=True)
from examples.SDPO_ReAct.data.build_code_data import CODE_SYSTEM_PROMPT
from examples.SDPO_ReAct.data.build_search_data import SEARCH_SYSTEM_PROMPT
from examples.SDPO_ReAct.native_prompt import build_native_messages
checks = []
if domain in ("code", "multitask"):
    checks.append(("code", [{"role": "system", "content": CODE_SYSTEM_PROMPT}, {"role": "user", "content": "add two ints"}], tool_specs))
if domain in ("math", "multitask") or domain not in ("code", "search"):
    checks.append(("math", build_native_messages("2+2?"), tool_specs))
if domain in ("search", "multitask"):
    checks.append(("search", [{"role": "system", "content": SEARCH_SYSTEM_PROMPT}, {"role": "user", "content": "who directed X?"}], search_specs))
for name, msgs, tspecs in checks:
    r = tok.apply_chat_template(msgs, tools=tspecs, tokenize=False, add_generation_prompt=True)
    assert "<tools>" in r, f"{name}: native <tools> block missing"
    if name == "code":
        assert "code_interpreter" in CODE_SYSTEM_PROMPT and ("grad" in CODE_SYSTEM_PROMPT.lower() or "run" in CODE_SYSTEM_PROMPT.lower()), "code prompt missing tool-submit contract"
    elif name == "search":
        assert "\"search\"" in r and "<answer>" in SEARCH_SYSTEM_PROMPT, "search prompt missing search tool/<answer>"
    else:
        assert "<answer>" in r, "math prompt missing <answer> contract"
_SENT = "SDPOSENTINEL"
probe = tok.apply_chat_template([{"role":"user","content":_SENT}], tokenize=False, add_generation_prompt=True)
assert _SENT in probe and probe.split(_SENT,1)[1], "gen_suffix empty -> SDPO splice undefined"
print(f"Native template check OK for {model_name} (domain={domain}, checked={[n for n,_,_ in checks]})")
PYCHECK

# Full-config tag: MODEL_NAME + arm + distill + think + prompt (same axes as 4B).
[ "$SDPO_REACT_PURE_DISTILL" = "true" ] && _DISTILL=pure || _DISTILL=mixed
[ "$SDPO_REACT_THINKING" = "true" ] && _THINK=think || _THINK=nothink
if [ "$SDPO_REACT_THINKING" = "true" ]; then
    REMOVE_THINKING_ARG=()
else
    REMOVE_THINKING_ARG=(--sdpo-remove-thinking-from-demonstration)
fi
CFG_TAG="${MODEL_NAME}-${SDPO_REACT_DOMAIN}-${SDPO_REACT_ARM}-${_DISTILL}-${_THINK}-${DATA_SUFFIX}"

SDPO_REACT_EXP="${SDPO_REACT_EXP:-sdpo-react-native-${CFG_TAG}_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/data/sdpo_dumps/${SDPO_REACT_EXP}"
echo "SDPO_ReAct dump dir: ${DUMP_DIR}"
CKPT_DIR="${SDPO_REACT_CKPT_DIR:-/root/data/sdpo_ckpts/sdpo-react-native-${CFG_TAG}_ckpt}"

# --- experiment log (explog) ---
EXPLOG="${SDPO_REACT_EXPLOG:-$REPO_ROOT/examples/SDPO_ReAct/explog.jsonl}"
GIT_COMMIT="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(cd "$REPO_ROOT" && [ -n "$(git status --porcelain 2>/dev/null)" ] && echo dirty || echo clean)"
python - <<PYLOG || true
import json, time, os
row = {
    "exp": "${SDPO_REACT_EXP}",
    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "arm": "${SDPO_REACT_ARM}",
    "prompt_variant": "${SDPO_REACT_PROMPT}",
    "model": "${MODEL_NAME}",
    "moe": "${IS_MOE}",
    "tp": "${SDPO_REACT_TP}",
    "ep": "${SDPO_REACT_EP_SIZE}",
    "r3": "${SDPO_REACT_R3}",
    "train_data": "${TRAIN_DATA}",
    "eval_dir": "${EVAL_DIR}",
    "num_rollout": "${SDPO_REACT_NUM_ROLLOUT}",
    "train_max_turns": "${SDPO_REACT_TRAIN_MAX_TURNS}",
    "eval_max_turns": "${SDPO_REACT_EVAL_MAX_TURNS}",
    "toolset": os.environ.get("SDPO_REACT_TOOLSET", "math"),
    "wandb_group": "sdpo-react-native-${MODEL_NAME}-${SDPO_REACT_ARM}",
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
    # save-interval overridable (default 10) so a debug run can force an EARLY
    # save -- e.g. SDPO_REACT_SAVE_INTERVAL=1 saves at rollout 1 to validate the
    # dist-ckpt save path (NCCL-watchdog / dist-timeout fix) without waiting for
    # rollout 10.
    CKPT_ARGS+=(--save "${CKPT_DIR}" --load "${CKPT_DIR}" --save-interval "${SDPO_REACT_SAVE_INTERVAL:-10}" --override-opt-param-scheduler)
    # Checkpoint-save reliability fix, ON by default (SDPO_REACT_ASYNC_SAVE=0 to
    # disable). async_save.yaml sets ckpt_fully_parallel_save: false via
    # --custom-config-path -- see that file's docstring: Megatron's fully-parallel
    # save strategy's async PyTorchStreamWriter hit a real corruption assert on
    # /fsx ("unexpected pos 704 vs 598") on a plain dp_reshardable save; disabling
    # it routes each rank's shard through the simpler synchronous writer, avoiding
    # the concurrent-writer race that trips it on network storage.
    if [ "${SDPO_REACT_ASYNC_SAVE:-1}" = "1" ]; then
        CKPT_ARGS+=(--custom-config-path "$SCRIPT_DIR/async_save.yaml")
    fi
fi

# NATIVE tool-calling rollout (identical to the 4B script). Multitask keeps the
# balanced round-robin interleave (no --rollout-shuffle).
if [ "$SDPO_REACT_DOMAIN" = "multitask" ]; then
    SHUFFLE_ARGS=()
    echo "ROLLOUT: --rollout-shuffle OFF (multitask -> preserve balanced per-batch domain mix)"
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
# Dynamic sampling (same regime routing as the 4B script).
if [ "${SDPO_REACT_DYNAMIC_SAMPLE:-true}" != "true" ]; then
    echo "DYNAMIC SAMPLING: OFF (raw batch)"
elif [ "$SDPO_REACT_ARM" = "1" ] || [ "$SDPO_REACT_PURE_DISTILL" != "true" ]; then
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
else
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_sdpo_group_has_prefix)
    ROLLOUT_ARGS+=(--sdpo-dynamic-filter-min-correct "${SDPO_REACT_MIN_CORRECT:-1}")
fi

# Tool specs/executor (same as 4B): registry all-tools for multitask, code-only otherwise.
if [ "$SDPO_REACT_DOMAIN" = "multitask" ]; then
    TOOL_SPECS_PATH="examples.SDPO_ReAct.tools.registry.all_tool_specs"
    EXECUTE_TOOL_PATH="examples.SDPO_ReAct.tools.registry.execute_tool"
else
    TOOL_SPECS_PATH="examples.SDPO_ReAct.tools.tool_specs.tool_specs"
    EXECUTE_TOOL_PATH="examples.SDPO_ReAct.tools.tool_client.execute_tool"
fi
ROLLOUT_ARGS+=(--tool-specs-resolver-path "$TOOL_SPECS_PATH")
CUSTOM_GENERATE_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate
   --generate-tool-specs-path "$TOOL_SPECS_PATH"
   --generate-execute-tool-function-path "$EXECUTE_TOOL_PATH"
   --generate-tool-call-parser "$TOOL_PARSER"
   --generate-max-turns "${SDPO_REACT_TRAIN_MAX_TURNS}"
)

# ============================================================================ #
# ABLATION ARMS -- copied verbatim from run-qwen3-4B-sdpo-react-native.sh so the
# 27B/35B ablation is directly comparable to the 4B one. Only the model block
# (parallelism + MoE flags) above differs. See that script's header for the full
# per-arm rationale (RLSD teacher, prefer-tool-use peer, tool-grammar reframing).
# ============================================================================ #
RM_ARGS=(
   --sdpo-answer-tag answer
   "${REMOVE_THINKING_ARG[@]}"
   --sdpo-reframe-multiturn-prefix
   --sdpo-tool-grammar "${TOOL_GRAMMAR}"
   --sdpo-max-prefix-chars 20000
)
if [ "$SDPO_REACT_DOMAIN" = "multitask" ] || [ "$SDPO_REACT_DOMAIN" = "search" ]; then
    RM_ARGS+=(
       --sdpo-search-judge-fallback
       --sdpo-judge-base-url "${OPENAI_API_URL:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
       --sdpo-judge-model "${SDPO_REACT_JUDGE_MODEL:-gpt-5.6-luna}"
       --sdpo-judge-api-key-env SFT_GATEWAY_KEY
    )
fi

case "${SDPO_REACT_ARM}" in
   1)
      RM_ARGS=(
         --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_plain_grpo_reward
         --sdpo-grader dapo
         --sdpo-answer-tag answer
      )
      if [ "$SDPO_REACT_DOMAIN" = "multitask" ] || [ "$SDPO_REACT_DOMAIN" = "search" ]; then
          RM_ARGS+=(
             --sdpo-search-judge-fallback
             --sdpo-judge-base-url "${OPENAI_API_URL:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
             --sdpo-judge-model "${SDPO_REACT_JUDGE_MODEL:-gpt-5.6-luna}"
             --sdpo-judge-api-key-env SFT_GATEWAY_KEY
          )
      fi
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

# PERF: TP for dense, EP for MoE. cp=pp=1. The MoE model also switches the token
# dispatcher to `flex` (in MISC_ARGS) -- the variable-length dispatcher the 35B
# reference uses. --expert-tensor-parallel-size 1 always (experts aren't further
# TP-sharded here).
PERF_ARGS=(
   --tensor-model-parallel-size "${SDPO_REACT_TP}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size "${SDPO_REACT_EP_SIZE}"
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   # Chunk the log-probs logits computation to avoid OOM. At TP=1 the
   # vocab-parallel cross-entropy / logits buffer [tokens, vocab=248320] is NOT
   # sharded, and the FIRST training-step log_probs forward is exactly where the
   # 35B-A3B run died (attempt-1 log cut off cleanly right after data_preprocess,
   # no traceback = silent CUDA OOM in the logits buffer). The reference MoE
   # launcher (scripts/run_qwen3_5_35b_a3b_mtp_cp2_ep8.py) sets this same flag
   # "to avoid OOM when computing log probs". Default 4096; override via
   # SDPO_REACT_LOGPROBS_CHUNK.
   --log-probs-chunk-size "${SDPO_REACT_LOGPROBS_CHUNK:-4096}"
)
# TP>1 (dense 27B) needs sequence-parallel to shard the activations/LayerNorm too.
if [ "${SDPO_REACT_TP}" -gt 1 ]; then
    PERF_ARGS+=(--sequence-parallel)
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-warmup-iters 10
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
# Large models: offload the optimizer state to CPU (frees VRAM for the bigger
# param/activation footprint), as every reference 27B/35B launcher does.
if [ "${SDPO_REACT_OPT_CPU_OFFLOAD:-1}" = "1" ]; then
    OPTIMIZER_ARGS+=(
       --optimizer-cpu-offload
       --overlap-cpu-optimizer-d2h-h2d
       --use-precision-aware-optimizer
    )
fi

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-sdpo
   --wandb-group "sdpo-react-native-${CFG_TAG}"
   --wandb-key "${WANDB_API_KEY}"
)

# SGLANG: dense = tp=1 per engine (sglang TP>1 garbage on Qwen3.5 0.5.9); MoE =
# one 8-GPU EP engine + --sglang-ep-size 8 (all experts co-resident) + R3's
# --use-rollout-routing-replay (train side) which flips SGLang's
# enable_return_routed_experts so the rollout emits per-token expert indices.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
)
R3_ARGS=()
if [ "$IS_MOE" = "1" ]; then
    SGLANG_ARGS+=(--sglang-ep-size "${SDPO_REACT_EP_SIZE}")
    if [ "$SDPO_REACT_R3" = "true" ]; then
        # R3: train-side flag turns on routing replay; it ALSO drives the rollout
        # payload's return_routed_experts (generate_endpoint_utils reads the same
        # arg), so no separate sglang flag is needed. See MoE NOTES header.
        R3_ARGS+=(--use-rollout-routing-replay)
        echo "R3: rollout routing replay ON (train/inference router alignment)"
    fi
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   # Widen the NCCL/process-group watchdog window. Default is 10 min
   # (--distributed-timeout-minutes, arguments.py). The 35B-A3B distributed
   # checkpoint save (dist-ckpt gathering the fully-reshardable optimizer state)
   # is slow enough that ranks sit in the save collective past 10 min -> NCCL
   # HeartbeatMonitor fires and kills the actor with c10::Error /
   # ActorUnavailableError at the rollout-10 save (observed: every attempt died
   # ~mid-save at iteration 9->10, no ckpt written). Raise to 60 min so a heavy
   # save can't trip the watchdog. Overridable via SDPO_REACT_DIST_TIMEOUT_MIN.
   --distributed-timeout-minutes "${SDPO_REACT_DIST_TIMEOUT_MIN:-60}"
)
# MoE: use the flex token dispatcher (variable-length packed sequences), matching
# scripts/run-qwen3.5-35B-A3B-mtp.sh. The arch-level MoE flags (num-experts,
# router-topk, grouped-gemm, drop-policy, router-dtype, aux-loss-coeff 0,
# shared-expert-gate) all come from scripts/models/qwen3.5-35B-A3B.sh.
if [ "$IS_MOE" = "1" ]; then
    MISC_ARGS+=(--moe-token-dispatcher-type flex)
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# GPU budget for TRAINING (search retriever may own GPU7 for multitask/search).
if [ "$SDPO_REACT_TRAIN_GPUS" -lt 8 ]; then
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((SDPO_REACT_TRAIN_GPUS-1)))"
    echo "TRAIN GPUs: ${SDPO_REACT_TRAIN_GPUS} (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES); retriever owns the rest"
fi

ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
sleep 2

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "${SDPO_REACT_TRAIN_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# --- checkpoint pruner (background): keep only iters < latest, never latest ---
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
        \"SDPO_REACT_EVAL_N_SAMPLES\": \"${SDPO_REACT_EVAL_N_SAMPLES}\",
        \"SDPO_REACT_EVAL_DIR\": \"${SDPO_REACT_EVAL_DIR}\"
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
   ${R3_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
