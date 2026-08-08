#!/bin/bash
# SDPO_ReAct (NATIVE tool-calling) -- Qwen3-4B on DAPO math, multi-turn rollout
# with ONE tool (code_interpreter, isolated Docker sandbox), single 8x H200
# (141GB) node, COLOCATE variant. This is the proposal-validation run: it MERGES
# the two halves that previously lived in separate examples --
#   - examples/SDPO_ReAct (multi-turn tool rollout, but Qwen2.5 + no skill), and
#   - examples/SDPO/run-qwen3-4B-sdpo-math-colocate.sh (full self-skill machinery,
#     but single-turn + no tools)
# -- into ONE Qwen3-4B run: native <tool_call> multi-turn rollout + SDPO
# self-skill whose PREFIX is the model's own env-grounded skill.
#
# Why NATIVE tool-calling (not the plain-text <code>/<answer> tags the Qwen2.5
# launcher uses): Qwen3 was trained on the <tool_call> grammar and its chat
# template DOES inject a <tools> schema block from apply_chat_template(tools=...)
# (verified on Qwen3-4B; the opposite of Qwen2.5, whose template silently drops
# it once the prompt is a string -> the ~98% zero-tool-call that forced the tag
# path). So we use miles.rollout.generate_hub.multi_turn.generate with the
# native qwen25 FunctionCallParser (qwen3 shares qwen25's <tool_call> XML).
#
# Integration with SDPO's skill-prefix splice, WITHOUT forking core code:
# native_prompt.py writes each row as a message list PLUS a `tools` field;
# --apply-chat-template --tool-key tools renders apply_chat_template(tools=...)
# ONCE at load time, baking the <tools> block into the STRING prompt. That
# string satisfies BOTH multi_turn.generate (a string prompt is not re-injected)
# AND sdpo.py's teacher-prefix splice (which needs isinstance(prompt, str)).
#
# The env_feedback skill source (the proposal's core mechanism) is now LIVE:
# multi_turn.generate records the trajectory's {tool_call, observation} pairs on
# sample.metadata["tool_trace"], which sdpo.py::_render_env_feedback grounds a
# failed trace's self-generated pitfall skill in. (This path was dead code
# before: nothing populated tool_trace and the keys mismatched.)
#
# Turn budget: SDPO_REACT_TRAIN_MAX_TURNS (default 8) during TRAINING,
# SDPO_REACT_EVAL_MAX_TURNS (default 20) during EVAL -- eval gets a longer
# budget per the "train short, infer long" spec (see eval_native_math.yaml).
#
# Env overrides (swap without editing the script):
#   SDPO_REACT_TRAIN_MAX_TURNS  (default 8)   training tool-call turn budget
#   SDPO_REACT_EVAL_MAX_TURNS   (default 20)  eval turn budget (eval yaml)
#   SDPO_REACT_EVAL_N_SAMPLES   (default 8)   eval samples/prompt (eval yaml)
#   SDPO_REACT_SKILL_SOURCE     (default env_feedback)  --sdpo-skill-source
#   SDPO_REACT_NUM_ROLLOUT      (default 300) --num-rollout
#
# usage: bash examples/SDPO_ReAct/run-qwen3-4B-sdpo-react-native.sh
set -exf

export PYTHONBUFFERED=16
export SDPO_REACT_TRAIN_MAX_TURNS="${SDPO_REACT_TRAIN_MAX_TURNS:-8}"
export SDPO_REACT_EVAL_MAX_TURNS="${SDPO_REACT_EVAL_MAX_TURNS:-20}"
export SDPO_REACT_EVAL_N_SAMPLES="${SDPO_REACT_EVAL_N_SAMPLES:-8}"
# Ablation-arm switch (see the RM_ARGS/GRPO_ARGS case block below for what
# each does): 1 | 1.1 | 2 | 3 | 4 | 5 | 5.1 -- same convention/numbering as
# examples/SDPO/run-qwen2.5-7B-sdpo-sci-rl-colocate.sh, ported onto the
# multi-turn native tool-calling rollout (this launcher's own machinery:
# search-judge-fallback, tool-grammar-aware trace reframing, prefer-tool-use
# peer selection -- all shared across every arm below, unchanged from before
# this ablation series existed).
SDPO_REACT_ARM="${SDPO_REACT_ARM:-1.1}"
SDPO_REACT_NUM_ROLLOUT="${SDPO_REACT_NUM_ROLLOUT:-300}"

# Batch sizing MUST stay divisible by the data-parallel size (= TRAIN_GPUS, since
# TP=PP=CP=1). Now that search is served by the CPU BM25 sidecar (no GPU), all 8
# GPUs are free for training again, so we go back to the clean 8-GPU config:
# rollout-batch-size = TRAIN_GPUS*4 = 32 prompts -> global-batch = 32*8 = 256, no
# 7-GPU gymnastics (the 112-override / 224-fallback existed only because 256 % 7
# != 0 when GPU7 was reserved for the e5 retriever). global_batch = RBS*N_SAMPLES
# is always divisible by TRAIN_GPUS by construction.
SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-8}"
N_SAMPLES_PER_PROMPT=8
# Megatron tensor-model-parallel size. Default 1 (Qwen3 path, unchanged). For
# Qwen3.5-4B, TP=2 shards the vocab-parallel cross-entropy logits buffer
# ([longest_seq,1,vocab=248320] -- the exact `buf9` that forced max-tokens-per-gpu
# down to 2048) AND the full-attention + MLP params, so it directly buys back the
# fla-backward headroom. dp = TRAIN_GPUS / TP (pp=cp=1), so the batch guard below
# divides by dp, NOT TRAIN_GPUS (only equal at TP=1).
SDPO_REACT_TP="${SDPO_REACT_TP:-1}"
DP_SIZE=$((SDPO_REACT_TRAIN_GPUS / SDPO_REACT_TP))
# rollout-batch-size = prompts per step. Default dp*4 (dp=8 -> 32 prompts, GBS
# 256; dp=4 at TP=2 -> 16 prompts, GBS 128). Overridable via
# SDPO_REACT_ROLLOUT_BATCH; falls back to dp*4 if the override isn't divisible by
# the data-parallel size.
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

# Model family select (parameterised so ONE run script drives Qwen3-4B AND
# Qwen3.5-4B). SDPO_REACT_MODEL comes from the enroot wrapper.
#   qwen3-native   -> Qwen3-4B,   parser qwen25 (JSON-in-<tool_call>), grammar qwen25
#   qwen3.5-native -> Qwen3.5-4B, parser qwen3_coder (XML <function=..>), grammar qwen3_coder
case "${SDPO_REACT_MODEL:-qwen3-native}" in
    qwen3.5-native)
        MODEL_NAME=Qwen3.5-4B
        MODEL_ARG_SH=scripts/models/qwen3.5-4B.sh
        TOOL_PARSER=qwen3_coder
        TOOL_GRAMMAR=qwen3_coder
        # NOTE: the rollout SGLang engine hardcodes trust_remote_code=True
        # (sglang_engine.py) and the HF->torch_dist converter already loads the
        # qwen3_5 tokenizer/config via register_hf_config_aliases(), so no
        # separate --trust-remote-code train flag is needed (and it isn't a
        # registered train.py arg -- passing it would crash).
        # --dist-ckpt-optim-fully-reshardable: REQUIRED for Qwen3.5. Its hybrid
        # linear+full attention gives uneven distributed-optimizer param buckets;
        # at dp=7 the default dp_reshardable save hits "AssertionError: empty
        # bucket encountered" (some rank owns 0 params in a bucket) in
        # distrib_optimizer.sharded_param_state_dp_reshardable at checkpoint save.
        # fully_reshardable gathers the full optimizer state instead -> no per-
        # bucket assert. (Qwen3's plain attention doesn't need it.)
        MODEL_EXTRA_ARGS=(--dist-ckpt-optim-fully-reshardable)
        # Qwen3.5's hybrid linear-attention: the fla gated_delta_rule BACKWARD
        # Triton kernel allocates workspace scaling with NT=ceil(tokens/64) chunks
        # per microbatch (chunk_size=64 is HARDCODED in fla, no tunable knob), so
        # the main lever is the per-GPU token budget. 12288 OOM'd on the backward;
        # 4096 fits. 6144 OOM'd EARLIER but only under mem-fraction 0.60 (during
        # the rollout->train transition sglang KV may not have fully released,
        # squeezing the fla-backward peak). Now that mem-fraction is back to 0.75
        # AND we've confirmed the KV pool is time-shared (TorchMemorySaver sleeps
        # the engine + releases KV before the train step, so training gets the
        # full GPU). 8192 OOM'd on the fla backward (Tried 12 GiB, 12.78 free) even
        # at mem-fraction 0.75; 4096 is the verified-safe value (v8 trained fine).
        # With thinking ON (longer sequences), keep 4096 -- bigger would OOM.
        # (4096 itself OOM'd once on compute_entropy_from_logits --
        # --observe-training-entropy allocates a tokens*vocab_size fp32 buffer,
        # huge at Qwen3.5's vocab=248320 -- fixed by dropping that diagnostic-only
        # flag below, not by shrinking this further.)
        # (expandable_segments is INCOMPATIBLE with the TorchMemorySaver -- no.)
        #
        # 4096 OOM'd again under the multitask (math+code+search) ablation series
        # (confirmed: rollout 0-8 fine, rollout 9's batch skewed longer --
        # response_len/mean 3704->4861, math domain 5948->7349 -- and the
        # vocab-parallel cross-entropy buffer (buf9, shape [s10,1,248320]) blew
        # the remaining margin: "Tried to allocate 14.53 GiB... 10.06 GiB free").
        # multitask's wider/heavier-tailed sequence-length distribution (vs the
        # single-domain runs 4096 was tuned against) means occasional batches
        # exceed the old safe margin. Drop to 3072 for headroom against that tail
        # -- allow SDPO_REACT_MAX_TOKENS_PER_GPU to override for a future re-tune.
        #
        # 3072 STILL OOM'd twice more in this same multitask ablation series
        # (both times the identical buf9 = empty_strided_cuda((s10,1,248320),...)
        # vocab-parallel cross-entropy buffer during log_probs/backward -- e.g.
        # "Tried to allocate 12.57 GiB... 8.35 GiB free"). The failure isn't
        # sensitive to the exact token budget in a linear way (this vocab=248320
        # buffer scales with the LONGEST sequence in the micro-batch, not the
        # average), so drop further to 2048 for more headroom against that tail.
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

# --- 0. sandbox sidecar (idempotent, ONE container / ONE port for the whole job) ---
bash "$SCRIPT_DIR/tools/run_sandbox.sh"

# Task domain (math | code). math = DAPO-math train + AIME eval (label grading).
# code = LiveCodeBench train+eval (test-case grading; correctness routed by
# sdpo.py's domain dispatch on metadata["domain"]=="code"). User plan: validate
# code SINGLE-domain first, then merge. Each domain sets its own train/eval data
# + eval-config below; everything else (SDPO core, arms, tools) is shared.
SDPO_REACT_DOMAIN="${SDPO_REACT_DOMAIN:-math}"
echo "SDPO_REACT_DOMAIN: ${SDPO_REACT_DOMAIN}"

# --- 0a. search sidecar (search/open/find -> the wiki-18 retriever). Only for
# domains that actually use it -- the retriever server itself must already be
# up (tools/search/run_retrieval.sh, needs its own GPU) before this. ---
if [ "$SDPO_REACT_DOMAIN" = "multitask" ] || [ "$SDPO_REACT_DOMAIN" = "search" ]; then
    bash "$SCRIPT_DIR/tools/search/run_search_sidecar.sh"
fi

# These two also feed the CFG_TAG (ckpt/exp/wandb scoping) below, so default
# them HERE (before the tag is built). Detailed usage is where they're consumed.
export SDPO_REACT_THINKING="${SDPO_REACT_THINKING:-true}"
SDPO_REACT_PURE_DISTILL="${SDPO_REACT_PURE_DISTILL:-true}"
echo "SDPO_REACT_THINKING: ${SDPO_REACT_THINKING} | SDPO_REACT_PURE_DISTILL: ${SDPO_REACT_PURE_DISTILL}"

# System-prompt variant (default | force_tool). force_tool makes tool use
# MANDATORY -- the "tool is necessary" hypothesis for the tool-use-collapse fix.
# native_prompt.py / build_native_eval.py read SDPO_REACT_PROMPT at data-prep
# time and bake the chosen prompt into each row, so train and eval MUST use the
# same variant -> per-variant data files (suffix keeps them from colliding).
# "minimal" (Q2 redesign): shared question+format-only prompt with ALL tools
# exposed (see native_prompt.py / registry.MINIMAL_SYSTEM_PROMPT). Also feeds
# DATA_SUFFIX + CFG_TAG, so switching to minimal gives fresh data filenames and a
# distinct ckpt/wandb scope -- no collision with the old verbose "default" run.
export SDPO_REACT_PROMPT="${SDPO_REACT_PROMPT:-minimal}"
DATA_SUFFIX="$SDPO_REACT_PROMPT"

if [ "$SDPO_REACT_DOMAIN" = "multitask" ]; then
    # --- MULTITASK: shuffled math + code (+ search when staged) ------------------
    # Proposal step 3 (env diversity ↑ -> ceiling ↑). Each row carries its own
    # system prompt + metadata['domain']; grading auto-routes per-domain in
    # sdpo.py (math->answer, code->test cases). Build the per-domain native sets
    # first (reuse the same builders), then interleave+shuffle. Eval on the code
    # v6 held-out set (hardest, tool-necessary signal) + AIME math via a combined
    # eval config. SDPO_REACT_MT_SOURCES lets you add search later.
    MT_DIR="/root/data/multitask"
    TRAIN_DATA="$MT_DIR/train.jsonl"
    EVAL_CFG="$SCRIPT_DIR/data/eval_multitask.yaml"
    mkdir -p "$MT_DIR" /root/dapo-math-17k /root/data/code_data /root/data/code_data_v6eval "/root/math_eval/native-${DATA_SUFFIX}"
    export SDPO_REACT_EVAL_DIR="/root/math_eval/native-${DATA_SUFFIX}"
    # per-domain sources (math native + code livecodebench), built if missing.
    # native_prompt.py below reads the RAW dapo-math-17k.jsonl -- download it
    # first (same guard as the math-only branch further down; multitask needs
    # it too since it shares the same source file).
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
    # SEARCH domain: HotpotQA + 2WikiMultiHop (FlashRAG), difficulty-FILTERED by
    # base-model pass@k (see memory search-needs-passk-difficulty-filter): TRAIN =
    # 3000 rows in the 15-75% learnable band (search_train_passk.jsonl), not the
    # raw pool (84% of which is too-easy/too-hard and made search REGRESS). Built
    # offline by data/passk_filter_search.py; fall back to a raw build only if missing.
    # search/open/find need the wiki-18 retriever server up (retrieval_server.py
    # on its own GPU) -- launch it separately BEFORE this run (see
    # tools/search/run_retrieval.sh); the search sidecar itself (which routes
    # search/open/find onto that retriever) is started automatically above.
    SEARCH_TRAIN=/root/data/search_data/search_train_passk.jsonl
    [ -f "$SEARCH_TRAIN" ] || SEARCH_TRAIN=/root/data/search_data/search_train.jsonl
    [ -f "$SEARCH_TRAIN" ] || \
        (cd "$REPO_ROOT" && python -m examples.SDPO_ReAct.data.build_search_data \
            --out-dir /root/data/search_data --n-train-per 1500 --n-val-per 100)
    # combine + shuffle all THREE domains (SDPO_REACT_MT_PER_DOMAIN caps each; default 400)
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
    # --- CODE domain: LiveCodeBench (own system prompt + test cases) -------------
    # TRAIN on the larger medium+hard set (all dates) for RL volume; EVAL on the
    # held-out LCB v6 window (contest_date >= 2025-02, post-Qwen3-4B-cutoff) so
    # eval measures GENUINE generalization, not memorization. Two builds into
    # separate dirs; eval_code.yaml points at the v6 eval file.
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
    # sanity: rows carry domain=code + test_cases
    python - <<PYCK
import json
r = json.loads(open("$TRAIN_DATA").readline())
assert r["metadata"]["domain"] == "code" and r["metadata"]["test_cases"], "code rows missing domain/test_cases"
print(f"code train rows OK; example has {len(r['metadata']['test_cases'])} test cases")
PYCK
else
    # --- MATH domain (default): DAPO math train + AIME24/25 native eval ---------
    TRAIN_DATA="/root/dapo-math-17k/dapo-math-17k-native-${DATA_SUFFIX}.jsonl"
    EVAL_DIR="/root/math_eval/native-${DATA_SUFFIX}"
    EVAL_CFG="$SCRIPT_DIR/data/eval_native_math.yaml"
    export SDPO_REACT_EVAL_DIR="$EVAL_DIR"  # read by eval_native_math.yaml
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

# --- 0c. one-time template check (fails loudly if the native render is wrong) --
# The prompt (math <answer> contract, or code ```python fence contract) must
# render with the model's <tools> block, and _gen_prompt_suffix must derive a
# real assistant marker for the SDPO splice.
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
# Whitespace-safe sentinel: Qwen3.5's template strips leading/trailing spaces
# from user content, so a space-padded probe wouldn't be found verbatim (this
# check crashed on Qwen3.5 before). Use a bare token and confirm a non-empty
# gen_suffix follows it (the assistant generation marker the SDPO splice needs).
_SENT = "SDPOSENTINEL"
probe = tok.apply_chat_template([{"role":"user","content":_SENT}], tokenize=False, add_generation_prompt=True)
assert _SENT in probe and probe.split(_SENT,1)[1], "gen_suffix empty -> SDPO splice undefined"
print(f"Native template check OK for {model_name} (domain={domain}, checked={[n for n,_,_ in checks]})")
PYCHECK

# Full-config tag: distinct experiments (domain / arm / pure-vs-mixed / thinking /
# prompt-design) MUST get distinct ckpt dirs, or a new run silently RESUMES a
# checkpoint trained under a DIFFERENT config -- e.g. a pure-SDPO run continuing a
# GRPO-trained ckpt, which invalidates the comparison (observed: the first pure
# run resumed iter_39 from an earlier mixed-GRPO code run because both used the
# bare code-trace name; AGAIN observed 2026-07-25: the new minimal-prompt/all-tool
# /filtered-search run resumed iter_29 from the old verbose-prompt run because the
# tag omitted the prompt design). So the tag encodes ALL config axes that change
# what the weights learn -- INCLUDING DATA_SUFFIX (=SDPO_REACT_PROMPT), since the
# prompt/tool/data redesign is a different training distribution entirely.
[ "$SDPO_REACT_PURE_DISTILL" = "true" ] && _DISTILL=pure || _DISTILL=mixed
[ "$SDPO_REACT_THINKING" = "true" ] && _THINK=think || _THINK=nothink
# Thinking ON: KEEP the peer's reasoning in the teacher prefix (distil the
# reason->verify->answer pattern, which is what keeps tool use alive). Thinking
# OFF: strip it (there's no useful reasoning to demonstrate). See the tool-use
# collapse under no-think: peers answered from memory + token-called the tool,
# so the student learned tool use is optional and it decayed (3.8->0.6 calls).
if [ "$SDPO_REACT_THINKING" = "true" ]; then
    REMOVE_THINKING_ARG=()
else
    REMOVE_THINKING_ARG=(--sdpo-remove-thinking-from-demonstration)
fi
# MODEL_NAME in the tag too: Qwen3-4B vs Qwen3.5-4B are different weights +
# different tool grammar -- must never share a ckpt/wandb dir.
CFG_TAG="${MODEL_NAME}-${SDPO_REACT_DOMAIN}-${SDPO_REACT_ARM}-${_DISTILL}-${_THINK}-${DATA_SUFFIX}"

SDPO_REACT_EXP="${SDPO_REACT_EXP:-qwen3-4B-sdpo-react-native-${CFG_TAG}_$(date +%Y%m%d_%H%M%S)}"
# Dumps under /root/data (-> /fsx/data, DURABLE data volume), NOT /root/miles
# (-> /fsx/home, quota-limited HOME volume -- never store ckpt/dumps there).
DUMP_DIR="/root/data/sdpo_dumps/${SDPO_REACT_EXP}"
echo "SDPO_ReAct dump dir: ${DUMP_DIR}"

# Checkpoint dir STABLE across restarts (independent of the timestamped exp
# name) so a debug restart resumes instead of retraining from 0, but scoped by
# the FULL config tag so distinct experiments never share/contaminate a ckpt.
# Under /root/data (durable shared storage), NOT /root/miles (quota-limited).
CKPT_DIR="${SDPO_REACT_CKPT_DIR:-/root/data/sdpo_ckpts/qwen3-4B-sdpo-react-native-${CFG_TAG}_ckpt}"

# --- experiment log (explog): one append-only JSONL row per launch, so every
# run's arm/prompt/hyperparams/paths/git are in ONE place for later analysis
# (which arm+prompt produced which wandb curve). Lives under $REPO_ROOT so it
# persists with the checkout. Human-readable + jq-friendly.
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
    "train_data": "${TRAIN_DATA}",
    "eval_dir": "${EVAL_DIR}",
    "num_rollout": "${SDPO_REACT_NUM_ROLLOUT}",
    "train_max_turns": "${SDPO_REACT_TRAIN_MAX_TURNS}",
    "eval_max_turns": "${SDPO_REACT_EVAL_MAX_TURNS}",
    "toolset": os.environ.get("SDPO_REACT_TOOLSET", "math"),
    "wandb_group": "qwen3-4B-sdpo-react-native-${SDPO_REACT_ARM}",
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

# SDPO_REACT_SAVE_CKPT: default off. These are throwaway experiment launches
# (relaunched repeatedly while iterating on config), and a stale ckpt from an
# earlier session silently auto-resuming (--load pointing at a still-present
# dir from a prior run of the SAME config tag) has already caused a real
# incident. Set =1 to opt back into save+resume for a run meant to be durable.
CKPT_ARGS=(
   --hf-checkpoint /root/${MODEL_NAME}
   --ref-load /root/${MODEL_NAME}_torch_dist
   --dump-details "${DUMP_DIR}"
   --no-dump-train-data
   --no-dump-policy-loss-debug
)
if [ "${SDPO_REACT_SAVE_CKPT:-0}" = "1" ]; then
    # --override-opt-param-scheduler: the LR scheduler's total-iteration count is
    # derived from --num-rollout and baked into the checkpoint; if a resume run
    # changes --num-rollout from what the checkpoint was saved under, Megatron's
    # own load_state_dict asserts old-vs-new iteration counts match and crashes
    # (observed live: "class input value 13056 and checkpoint value 25856").
    # This flag tells it to keep the CURRENT run's schedule instead of the
    # checkpoint's -- exactly what we want when deliberately re-tuning
    # --num-rollout on resume; it does not affect optimizer/RNG state loading.
    CKPT_ARGS+=(--save "${CKPT_DIR}" --load "${CKPT_DIR}" --save-interval 10 --override-opt-param-scheduler)
fi

# NATIVE tool-calling rollout: prompt stays a message list + `tools` field;
# --apply-chat-template --tool-key tools bakes the <tools> block into the string
# prompt at load time (see native_prompt.py).
#
# SDPO_REACT_THINKING (defaulted up top): enable_thinking in the chat template.
# Set false (no-thinking) to remove Qwen3's one-shot reasoning crutch -- with
# thinking on, the strong 4B solves problems inside <think> and never needs the
# tool; no-thinking forces it to actually RUN code to make progress, which
# (together with tool-mandatory grading) is what makes the tool necessary.
# Multitask: keep the balanced round-robin interleave (38/37/37 per batch) by
# NOT reshuffling each epoch -- data_source's --rollout-shuffle would re-randomize
# and unbalance every batch. Single-domain: shuffle as usual (order irrelevant).
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
   # Qwen3-4B's true max context (config.json max_position_embeddings). REQUIRED
   # for multi-turn: compute_request_payload (generate_endpoint_utils.py) clamps
   # each turn's max_new_tokens to (rollout_max_context_len - len(input_ids))
   # ONLY when this is set. Without it, a long multi-turn trajectory requests
   # input + max_new_tokens > 40960 and sglang rejects it with a 400 that
   # crashes eval entirely (observed: eval input 25220 + completion 16384 =
   # 41604 > 40960). Setting it makes the engine return TRUNCATED gracefully
   # instead. eval_max_context_len defaults to this too (arguments.py:3090).
   --rollout-max-context-len 40960
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
   # MUST be >= rollout-batch-size (arguments.py asserts). Set to the batch size
   # so each data-fetch pulls exactly one balanced window; with the round-robin
   # interleave that keeps the per-batch domain mix uniform.
   --over-sampling-batch-size "${ROLLOUT_BATCH_SIZE}"
)
# Dynamic sampling. Two regimes:
#  - NOT pure-distill: a real GRPO advantage flows, so drop zero-reward-variance
#    groups (check_reward_nonzero_std), the usual DAPO filter.
#  - PURE distill (default here): task reward is always 0, so reward-variance is
#    meaningless. Instead drop groups with too few CORRECT traces
#    (check_sdpo_group_has_prefix on metadata['sdpo_correct']): under pure SDPO a
#    group with 0 correct traces gives NO teacher prefix -> every trace's KD loss
#    is 0 -> the whole group is a dead batch slot (observed: ~40-50% of math/
#    search groups had no correct peer -> no prefix). Threshold
#    SDPO_REACT_MIN_CORRECT (default 1: keep groups with >=1 correct, so a prefix
#    exists for the incorrect traces; set 2 for full coverage incl. the correct
#    trace itself). Needs over-sampling headroom to refill dropped groups.
# SDPO_REACT_DYNAMIC_SAMPLE (default true) gates the whole dynamic-sampling
# machinery. Set false to train on the raw rollout batch with NO oversample / no
# group filtering / no domain rebalance -- simpler and cheaper. Under pure distill
# a group with 0 correct traces then just contributes 0 KD loss (a wasted slot,
# not a crash); acceptable once the batch is big enough (256 on 8 GPUs) that the
# with-prefix groups dominate the gradient.
if [ "${SDPO_REACT_DYNAMIC_SAMPLE:-true}" != "true" ]; then
    echo "DYNAMIC SAMPLING: OFF (raw batch, no oversample/filter/rebalance)"
elif [ "$SDPO_REACT_ARM" = "1" ] || [ "$SDPO_REACT_PURE_DISTILL" != "true" ]; then
    # arm 1 (plain GRPO, sdpo_react_plain_grpo_reward, no --group-rm) never
    # stamps sample.metadata["sdpo_correct"] -- that's exclusively written by
    # sdpo_group_reward (see examples/SDPO/sdpo.py), which arm 1 does not use.
    # check_sdpo_group_has_prefix hard-requires that key and raises (crashing
    # the whole job, not a graceful skip) when it's absent. Route arm 1 to the
    # standard DAPO reward-variance filter instead, exactly like the
    # non-pure-distill branch below -- arm 1's reward is real (not zeroed), so
    # "drop zero-reward-variance groups" is the correct filter for it anyway.
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
else
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_sdpo_group_has_prefix)
    ROLLOUT_ARGS+=(--sdpo-dynamic-filter-min-correct "${SDPO_REACT_MIN_CORRECT:-1}")
    # Multitask: --rollout-domain-balanced-n (keep the ACCEPTED batch balanced
    # across domains as the filter drops no-prefix groups) does NOT exist as a
    # registered train.py arg in this checkout -- it belongs to a separate,
    # not-yet-landed multi-task domain-balancing patch (see
    # miles/rollout/inference_rollout/inference_rollout_train.py). Passing it
    # crashes argparse ("unrecognized arguments"). Omitted here; the batch may
    # skew toward whichever domain's groups pass the filter more often until
    # that patch lands.
fi

# Multi-turn NATIVE tool-calling generate function (not the plain-text-tag
# generate_with_tools.generate). Parser is qwen25 (qwen3 shares its <tool_call>
# XML format). tool specs + executor are the SAME modules the training-side skill
# machinery / sandbox already use.
# Tool specs/executor: for multitask (math+code+search) use the composable
# registry with ALL tools (code_interpreter + web_search) so the rollout parses
# tool calls from every domain's rows + dispatches each to its backend. Single-
# domain math/code keep the code-only tool_specs (web_search backend needs the
# retriever sidecar up; don't require it for a math/code-only run).
if [ "$SDPO_REACT_DOMAIN" = "multitask" ]; then
    TOOL_SPECS_PATH="examples.SDPO_ReAct.tools.registry.all_tool_specs"
    EXECUTE_TOOL_PATH="examples.SDPO_ReAct.tools.registry.execute_tool"
else
    TOOL_SPECS_PATH="examples.SDPO_ReAct.tools.tool_specs.tool_specs"
    EXECUTE_TOOL_PATH="examples.SDPO_ReAct.tools.tool_client.execute_tool"
fi
# --tool-specs-resolver-path re-derives the <tools> block miles/utils/data.py
# bakes into the prompt string from THIS live TOOL_SPECS_PATH every time the
# training/eval Dataset is constructed (once per rollout-actor/eval-cache-key
# startup) instead of trusting whatever static "tools" snapshot a jsonl was
# built with -- prevents the exact bug hit earlier this session (data built
# under an older registry.py kept declaring a since-renamed tool ("web_search")
# forever, because nothing re-checked it against the CURRENT registry). Same
# module path as --generate-tool-specs-path, so the model is always TOLD
# exactly the tools the rollout parser is prepared to PARSE.
ROLLOUT_ARGS+=(--tool-specs-resolver-path "$TOOL_SPECS_PATH")
CUSTOM_GENERATE_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate
   --generate-tool-specs-path "$TOOL_SPECS_PATH"
   --generate-execute-tool-function-path "$EXECUTE_TOOL_PATH"
   --generate-tool-call-parser "$TOOL_PARSER"
   --generate-max-turns "${SDPO_REACT_TRAIN_MAX_TURNS}"
)

# Ablation-arm series -- ports examples/SDPO/run-qwen2.5-7B-sdpo-sci-rl-colocate.sh's
# RLSD arm structure onto this launcher's multi-turn native tool-calling
# rollout (search-judge-fallback, tool-grammar-aware trace reframing,
# prefer-tool-use peer selection -- all defined ABOVE this block and shared
# by every arm, unchanged). Response-teacher mechanism is RLSD (--sdpo-rlsd,
# arXiv:2604.03128) instead of the additive KD-loss the old trace/skill/nokd
# arms used: direction comes EXCLUSIVELY from the real task reward's
# sign(advantage); the teacher's per-token evidence ratio only reweights
# MAGNITUDE within an already-correctly-signed trajectory. This was adopted
# specifically because the additive JSD-KD arms showed tool-call collapse
# (--sdpo-prefer-tool-use-peer's own "no tool-using correct peer -> falls
# back to random peer" escape hatch let zero-tool-call "correct" traces
# become KD teachers once enough of them appeared -- confirmed live: that
# escape hatch fired on 0% of groups at rollout 0-10, rising to 54.5% by
# rollout 35, i.e. a self-reinforcing collapse). RLSD's reward-gated
# direction cannot pull a trace toward an irrelevant peer's tool-free
# style, so this ablation isolates whether that failure mode is specific to
# the additive-KD mechanism.
#
#   SDPO_REACT_ARM=1    Plain GRPO, no SDPO machinery at all: single-sample
#                       domain-routed reward (sdpo_react_plain_grpo_reward),
#                       no --group-rm, default std-normalized advantages.
#   SDPO_REACT_ARM=1.1  RLSD baseline, NO self-skill -- response teacher
#                       prefix = correct peer's FULL raw trace (tool calls
#                       included). Isolates "does RLSD's reweighting alone
#                       help" before any skill machinery.
#   SDPO_REACT_ARM=2    + self-skill, skill-source correct only, NO skill-KD.
#                       RLSD lambda=1.0/no decay (reweighting active for the
#                       FULL run, not the paper's own decay-to-inert default).
#   SDPO_REACT_ARM=3    + self-skill, skill-source incorrect only, NO skill-KD.
#   SDPO_REACT_ARM=4    + self-skill, skill-source all, NO skill-KD.
#   SDPO_REACT_ARM=5    + self-skill, skill-source all, WITH skill-KD
#                       (mode=both), --sdpo-skill-max-new-tokens 2048.
#   SDPO_REACT_ARM=5.1  IDENTICAL to arm 5 except --sdpo-skill-kd-mode
#                       both-blind (self-success -> blind-correct): gives
#                       correct traces the SAME information gap pitfall-
#                       condense already has for failed traces, so
#                       skill/kl_correct vs skill/kl_pitfall become directly
#                       comparable.
#
# --no-sdpo-pure-distill on every SDPO arm (UNLIKE the old JSD-KD arms, where
# --sdpo-pure-distill defaulted True): RLSD's advantage reweighting has
# nothing to reweight if the GRPO advantage is always 0 -- the real per-trace
# correctness reward MUST flow into the GRPO advantage (RLSD's own "reward
# sets direction, teacher sets magnitude" design).
#
# NO --observe-training-entropy (unlike the sci-rl-colocate script's every
# arm): Qwen3.5's vocab (248320) makes that diagnostic's tokens*vocab_size
# fp32 buffer OOM at this launcher's batch sizing (confirmed earlier this
# session) -- --entropy-coef stays 0.00 (inert either way) but the buffer
# itself is never allocated.
RM_ARGS=(
   --sdpo-answer-tag answer
   "${REMOVE_THINKING_ARG[@]}"
   # Reframe the multi-turn native trace's ChatML <|im_*|> turn boundaries into
   # clean NLP markers before splicing into the teacher prefix (keeps <think>/
   # <tool_call>/<tool_response> verbatim). No-op for single-turn.
   --sdpo-reframe-multiturn-prefix
   # Tool-call grammar the teacher prefix/skill renders in -- MUST match
   # --generate-tool-call-parser (qwen25 JSON for Qwen3-4B, qwen3_coder XML for
   # Qwen3.5-4B) so the distilled text byte-matches the student's own emissions.
   --sdpo-tool-grammar "${TOOL_GRAMMAR}"
   # Cap a peer trace's reframed prose to its last 20K chars before splicing
   # into the teacher prefix. Without this, one pathologically long peer trace
   # (a code-domain debugging loop -- observed teacher prompts up to 65K chars
   # / response_length 18K tokens) becomes every other sample's teacher prefix
   # and OOMs the vocab-parallel forward alone (can't be split across a
   # dynamic-batch-size microbatch). See qwen35-oom-long-code-traces memory.
   --sdpo-max-prefix-chars 20000
)
# Search/QA double-check: EM grades first (cheap, exact); only an EM MISS gets
# a second opinion from an LLM judge before being finalized "incorrect" -- EM's
# normalized string equality has no tolerance for a correct answer phrased
# differently than the one golden string it happens to compare against, which
# is common in multi-hop QA (aliases, abbreviations, differing specificity).
# Gateway: Salesforce Research's OpenAI-compatible endpoint (SFT_GATEWAY_KEY /
# OPENAI_API_URL loaded from ~/gitproj/apis/.env by enroot-run-sdpo-react.sh),
# model gpt-5.6-luna. Only wired for domains that actually train/eval search
# (--sdpo-judge itself stays OFF -- that flag replaces math/mcq grading
# entirely, which this run does not want).
if [ "$SDPO_REACT_DOMAIN" = "multitask" ] || [ "$SDPO_REACT_DOMAIN" = "search" ]; then
    RM_ARGS+=(
       --sdpo-search-judge-fallback
       --sdpo-judge-base-url "${OPENAI_API_URL:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
       --sdpo-judge-model "${SDPO_REACT_JUDGE_MODEL:-gpt-5.6-luna}"
       --sdpo-judge-api-key-env SFT_GATEWAY_KEY
    )
fi

# --- ablation-arm-specific RM_ARGS / GRPO_ARGS -------------------------------
case "${SDPO_REACT_ARM}" in
   1)
      # Plain GRPO, no SDPO machinery at all: single-sample reward, no
      # --group-rm, no --sdpo-*/--use-* flags whatsoever. sdpo_react_plain_grpo_reward
      # domain-routes per sample (math/dapo, code test-cases, search EM) so the
      # grading criterion matches every other arm's sdpo_react_group_reward.
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
         # NO --calculate-per-token-loss (unlike every SDPO arm below) -- the
         # "pure vanilla GRPO" baseline: std-normalized advantages, miles' own
         # default seq-mean-token-mean loss aggregation.
      )
      ;;
   1.1)
      # RLSD baseline, NO self-skill: response teacher prefix = correct peer's
      # FULL raw trace (--sdpo-response-prefix defaults to "trace").
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
      # + self-skill, skill-source correct ONLY, no skill-KD. RLSD lambda=1.0/
      # no-decay (reweighting active for the FULL run -- merged from the old
      # sci script's "2.2" variant; this run never uses the paper's own
      # decay-to-inert-by-rollout-50 default).
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
      # + self-skill, skill-source incorrect ONLY (pitfall warnings from
      # failed traces), no skill-KD. NO --sdpo-response-prefix skill (a
      # correct peer never has a generated skill under skill-source=incorrect
      # -- see doc/DESIGN_self_skill.md): the response teacher prefix stays
      # the peer's full trace, only the failed-trace pitfall injection differs.
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
      # + self-skill, skill-source ALL (correct -> solution roadmap, incorrect
      # -> pitfall warnings), no skill-KD.
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
      # + self-skill, skill-source ALL, WITH skill-KD (mode=both: correct
      # traces get self-success solution-skill KD, failed traces get
      # pitfall-condense KD). --sdpo-skill-max-new-tokens 2048 (merged from
      # the old sci script's "5.1" variant -- the 1024 default was observed
      # hitting its cap).
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
      # IDENTICAL to arm 5 except --sdpo-skill-kd-mode both-blind (was both):
      # self-success (correct traces) replaced by blind-correct -- student
      # regenerates from the PROBLEM ONLY, teacher = problem-only prompt +
      # this trace's own correct solution as privileged info. Gives correct
      # traces the SAME information gap pitfall-condense has for failed
      # traces, so skill/kl_correct vs skill/kl_pitfall become comparable.
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

# --eval-interval 10 + genuine step-0 eval (no --skip-eval-before-train) so
# every "did training help" comparison is against the untrained baseline.
# --n-samples-per-eval-prompt MUST be set globally too (metrics.py's
# compute_pass_rate reads the GLOBAL arg, not the per-dataset one) or no
# val-core pass@1 panel appears.
EVAL_ARGS=(
   --eval-interval 10
   --eval-config "$EVAL_CFG"
   --eval-tool-key tools
   --n-samples-per-eval-prompt "${SDPO_REACT_EVAL_N_SAMPLES}"
   --log-passrate
)
# Debug speed-up: SDPO_REACT_SKIP_EVAL0=1 skips the step-0 (before-train) eval so
# a debug iteration reaches the first TRAIN step ~5 min sooner (no baseline eval).
# Leave OFF for real runs -- the untrained-baseline eval is the comparison anchor.
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
   # Training micro-batch token budget. Bigger rollout batch (112 -> global 896)
   # doesn't change per-microbatch memory by itself, BUT the SDPO teacher-forward
   # + JSD top-k KD roughly doubles activation memory per token vs plain PG, and
   # at global 896 the seqlen-balanced packing pushed a microbatch to ~22 GiB and
   # OOM'd GPU0 (colocate: sglang KV already holds ~0.75 of VRAM). Halved to
   # 12288 so each train microbatch fits alongside the rollout engine; costs a bit
   # of train throughput, keeps batch 112 + the balanced domain mix intact.
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-12288}"
)

# GRPO_ARGS is set per-arm in the case block above (each arm needs a
# different advantage-estimator/RLSD/pure-distill combination) -- no
# top-level default here anymore.

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
   --wandb-group "qwen3-4B-sdpo-react-native-${CFG_TAG}"
   --wandb-key "${WANDB_API_KEY}"
)

# --sglang-mem-fraction-static 0.75: matches run-qwen3-4B-sdpo-math-colocate.sh's
# tuned value for Qwen3-4B under colocate (0.85 OOM'd its rollout engine on the
# 256-request x 8192-token batch). Multi-turn tool rollout grows context per
# turn, so keep the more conservative fraction.
#
# NO --sglang-router-policy here: multi_turn.generate's tool args
# (--generate-tool-specs-path etc.) only register under
# MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 (miles/utils/arguments.py:2533 gates
# add_user_provided_function_arguments on it), and that mode forbids
# --sglang-router-policy (miles/backends/sglang_utils/arguments.py:167). The
# router defaults to a working policy without it. The experimental path's old
# blocker -- no --eval-custom-rm-path support under --group-rm for eval -- is
# fixed (inference_rollout_eval.py:38-39,123-124 now handle it), so eval works.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # Qwen3.5's fla linear-attention backward Triton kernel needs training-side
   # workspace; give training headroom by shrinking the colocated sglang KV pool
   # (SGLANG_MEM_FRACTION=0.60 for Qwen3.5, default 0.75 for Qwen3). NOTE: do NOT
   # use PYTORCH_CUDA_ALLOC_CONF=expandable_segments -- it's incompatible with
   # SGLang colocate's TorchMemorySaver (crashes the engine at init).
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

# GPU budget for TRAINING. Default 8 (whole node). For the 3-domain run the
# wiki-18 retriever (retrieval_server.py) holds GPU 7, so set
# SDPO_REACT_TRAIN_GPUS=7 and mask GPU7 off so training only sees GPUs 0-6.
SDPO_REACT_TRAIN_GPUS="${SDPO_REACT_TRAIN_GPUS:-8}"
if [ "$SDPO_REACT_TRAIN_GPUS" -lt 8 ]; then
    # expose exactly GPUs 0..(N-1) to ray/training; the retriever owns the rest.
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((SDPO_REACT_TRAIN_GPUS-1)))"
    echo "TRAIN GPUs: ${SDPO_REACT_TRAIN_GPUS} (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES); retriever owns the rest"
fi

ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
sleep 2

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus "${SDPO_REACT_TRAIN_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# --- checkpoint pruner (background): keep only the newest COMPLETED iter dir ---
# Megatron's --save-interval has no retention limit; every save keeps its own
# ~50GB+ dir forever, and --load only reads the newest. Left unpruned a long run
# fills the shared volume. Poll every 60s and delete iter dirs.
#
# CRITICAL race fix: latest_checkpointed_iteration.txt is updated only AFTER a
# save COMPLETES, so while iter_N is being written `latest` still points at the
# PREVIOUS iter. The old pruner ("delete everything != latest") therefore
# deleted the in-progress iter_N mid-write -> the save crashed with
# `FileNotFoundError: .../iter_0000019/.metadata.tmp` and killed the whole job
# (observed: trace arm died at step ~18 on the iter_19 save). Fix: only delete
# iter dirs STRICTLY OLDER (smaller iter number) than `latest` -- never touch
# `latest` itself nor any dir NEWER than it (a newer dir is an in-flight save).
#
# No-op entirely when SDPO_REACT_SAVE_CKPT!=1 -- CKPT_DIR is never written to.
if [ "${SDPO_REACT_SAVE_CKPT:-0}" = "1" ]; then
(
    set +f
    while true; do
        sleep 60
        latest_file="${CKPT_DIR}/latest_checkpointed_iteration.txt"
        [ -f "$latest_file" ] || continue
        latest_num="$(cat "$latest_file" 2>/dev/null)"
        [ -n "$latest_num" ] || continue
        case "$latest_num" in ''|*[!0-9]*) continue;; esac  # numeric guard
        for d in "${CKPT_DIR}"/iter_*; do
            [ -d "$d" ] || continue
            n="$(basename "$d")"; n="${n#iter_}"
            case "$n" in ''|*[!0-9]*) continue;; esac
            # strip leading zeros for numeric compare (base-10)
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
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
