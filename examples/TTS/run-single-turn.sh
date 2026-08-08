#!/bin/bash
# One-click launcher for the TTS (test-time scaffolding) single-turn loop.
#
# Unlike the SDPO / SDPO_ReAct launchers, this does NOT start Ray / Megatron /
# training -- TTS is a STANDALONE test-time loop that optimizes a PROMPT (the
# skill-generation prompt), not model weights. So this script only needs to:
#   1. (optionally) launch a local SGLang server exposing the model we scaffold
#      over its OpenAI-compatible endpoint (the "solver"/"skill_writer" role);
#   2. load API keys (for a remote "optimizer" role, e.g. gpt-5.6-luna) from a
#      gitignored .env, same convention as the other examples;
#   3. run the loop (examples/TTS/run_tts.py).
#
# Everything about WHICH engine plays each role is in the YAML --config, so you
# switch a role between local and remote WITHOUT touching this script:
#   CONFIG=configs/single_turn_math.yaml       local solver + remote gpt-5.6-luna optimizer
#   CONFIG=configs/single_turn_all_local.yaml  everything on the local server
#   CONFIG=configs/single_turn_sci.yaml        science MCQ + Anthropic optimizer
#
# Env knobs (all optional; defaults shown):
#   MODEL_PATH   local HF checkpoint to serve      (/root/models/Qwen2.5-7B-Instruct)
#   SGLANG_PORT  local server port                 (30000)
#   SGLANG_TP    tensor parallel for the server    (1)
#   START_SERVER launch the local server here      (1; set 0 if already running
#                                                    or if the config is fully remote)
#   CONFIG       which YAML config to run           (configs/single_turn_math.yaml)
#   TRAIN_DATA   train jsonl(s)                     (/root/math_eval/aime25.jsonl)
#   EVAL_DATA    held-out jsonl(s)                  (/root/math_eval/minerva_math.jsonl)
#   ROUNDS       optimization rounds                (from config)
#   OUT_DIR      logs + best prompt                 (/root/tts_out)
set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load secrets (OPENAI_API_KEY / ANTHROPIC_API_KEY) from a gitignored .env, same
# order the SDPO_ReAct launcher uses. NEVER commit these files.
for env_file in "$SCRIPT_DIR/.env" "$SCRIPT_DIR/../SDPO/.env" "$HOME/gitproj/apis/.env"; do
    if [ -f "$env_file" ]; then set -a; . "$env_file"; set +a; fi
done

MODEL_PATH="${MODEL_PATH:-/root/models/Qwen2.5-7B-Instruct}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_TP="${SGLANG_TP:-1}"
START_SERVER="${START_SERVER:-1}"
CONFIG="${CONFIG:-$SCRIPT_DIR/configs/single_turn_math.yaml}"
TRAIN_DATA="${TRAIN_DATA:-/root/math_eval/aime25.jsonl}"
EVAL_DATA="${EVAL_DATA:-/root/math_eval/minerva_math.jsonl}"
OUT_DIR="${OUT_DIR:-/root/tts_out}"

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [ "$START_SERVER" = "1" ]; then
    echo "[TTS] launching local SGLang server: $MODEL_PATH on :$SGLANG_PORT (tp=$SGLANG_TP)"
    python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host 0.0.0.0 --port "$SGLANG_PORT" \
        --tp "$SGLANG_TP" \
        --mem-fraction-static 0.85 &
    SERVER_PID=$!
    # wait for the OpenAI-compatible endpoint to come up
    echo "[TTS] waiting for server health on :$SGLANG_PORT ..."
    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${SGLANG_PORT}/health" >/dev/null 2>&1 \
           || curl -sf "http://127.0.0.1:${SGLANG_PORT}/v1/models" >/dev/null 2>&1; then
            echo "[TTS] server ready."; break
        fi
        sleep 5
    done
fi

ROUNDS_ARG=""
[ -n "${ROUNDS:-}" ] && ROUNDS_ARG="--rounds $ROUNDS"

cd "$REPO_ROOT"
echo "[TTS] running loop: config=$CONFIG train=$TRAIN_DATA eval=$EVAL_DATA"
PYTHONPATH="$REPO_ROOT" python3 -m examples.TTS.run_tts \
    --config "$CONFIG" \
    --train-data $TRAIN_DATA \
    --eval-data $EVAL_DATA \
    --out-dir "$OUT_DIR" \
    $ROUNDS_ARG

echo "[TTS] done. Best skill-gen prompt: $OUT_DIR/best_skill_gen_prompt.txt"
echo "[TTS] full round log:            $OUT_DIR/tts_log.jsonl"
