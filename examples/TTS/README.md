# TTS — Test-Time Scaffolding (scaffold-search over a skill-generation prompt)

A standalone harness that **optimizes a prompt, not weights**. It runs a
two-layer scaffold: a layer-1 *solving scaffold* answers each query with a
self-generated "skill" in context, and a layer-2 *meta-optimizer* rewrites the
skill-generation prompt from the round's traces. Repeat for several rounds.

This is the test-time analogue of [`examples/SDPO`](../SDPO): SDPO trains model
weights with a **fixed** skill-generation prompt; TTS keeps the weights fixed
and makes that **skill-generation prompt the thing being optimized**. It reuses
SDPO's skill format (`[Knowledge/Rule]/[Details/Examples]`) as the seed and
SDPO's grader verbatim, so results are directly comparable.

## The loop

```
                    ┌──────────────────────── round r ────────────────────────┐
  skill-gen prompt ─┤                                                          │
  (the free var)    │  LAYER 1 (per query):                                    │
        │           │    skill  = skill_writer(skill_gen_prompt, query[, prior])│
        │           │    answer = solver(solver_system, skill ⊕ query)          │
        │           │    correct? = grade(answer, label)     ← SDPO's grader    │
        │           │                                                          │
        │           │  LAYER 2 (once):                                         │
        └───────────┤    new_prompt = optimizer(cur_prompt, [(query,skill,ok)])│
      (next round)  │  eval new_prompt on held-out set; keep the best          │
                    └──────────────────────────────────────────────────────────┘
```

- **Layer 1 — solving scaffold** (`scaffold.py`): for each query, the
  `skill_writer` engine turns the query (and, on later rounds, the solver's own
  prior attempts) into a compact skill; the `solver` engine then answers the
  **original** query with that skill spliced into its user turn.
- **Layer 2 — meta-optimizer** (`optimizer.py`): the `optimizer` engine reads the
  current skill-gen prompt plus a sample of `(query, generated-skill, correct?)`
  triples and proposes an improved skill-gen prompt for the next round.
- **Grading** (`grading.py`): reuses `examples/SDPO/reward.py::_is_correct`
  (DAPO boxed/numeric for math, single-letter match for science MCQ).

## Engines — local **or** remote, per role

Every role picks its engine independently from the YAML config, so you can mix a
local model with a remote API however you like. Three backends, all over raw
HTTP (no `openai`/`anthropic` SDK required):

| backend      | endpoint                     | use for                                            |
| ------------ | ---------------------------- | -------------------------------------------------- |
| `openai`     | `{base_url}/chat/completions`| a **local SGLang** server *and* hosted OpenAI-protocol models (e.g. `gpt-5.6-luna`) — same backend, only `base_url`/`model`/`api_key_env` differ |
| `anthropic`  | `{base_url}/messages`        | Claude models                                      |
| `sglang`     | `{base_url}/generate`        | SGLang's native token-in/out endpoint (needs a local tokenizer; returns logprobs) |

A local SGLang server launched with `python -m sglang.launch_server` exposes the
OpenAI-compatible endpoint, so the **same** `openai` backend drives a local
Qwen2.5-7B and a remote `gpt-5.6-luna` — that is exactly the two-engine setup in
the shipped `configs/single_turn_math.yaml` (local solver + remote optimizer).
Roles with identical settings share one engine instance.

### Example config (`configs/single_turn_math.yaml`)

```yaml
engines:
  solver:       {backend: openai, model: qwen2.5-7b-instruct, base_url: "http://127.0.0.1:30000/v1"}
  skill_writer: {backend: openai, model: qwen2.5-7b-instruct, base_url: "http://127.0.0.1:30000/v1"}
  optimizer:    {backend: openai, model: gpt-5.6-luna, base_url: "https://api.openai.com/v1", api_key_env: OPENAI_API_KEY}
```

Shipped configs: `single_turn_math.yaml` (local solver + remote OpenAI
optimizer), `single_turn_all_local.yaml` (everything on one local server),
`single_turn_sci.yaml` (science MCQ + Anthropic optimizer).

## Why standalone (no `train.py`)

Remote APIs return **text**, not `input_ids` or per-token logprobs, so they
cannot supply miles' token-in/token-out training signal (loss mask, rollout
logprobs). TTS never needs it — it optimizes a prompt string — so a remote model
is a first-class engine here for **any** role, unlike the training rollout path
where a remote backend could only serve eval.

## Data

Reuses SDPO's datasets — JSONL rows of `{"prompt": [...], "label": ..., "metadata": {"domain": ...}}`:

```bash
python examples/SDPO/build_math_eval.py   --out-dir /root/math_eval   # aime25.jsonl, minerva_math.jsonl
python examples/SDPO/build_sci_dataset.py --out-dir /root/sci         # train.jsonl, val_<domain>.jsonl
```

`data.py` extracts the bare question + label + domain (TTS supplies its own
solver system prompt / answer contract). Any JSONL in the same shape works.

## Run

```bash
# 1. all-local (no API keys):
CONFIG=examples/TTS/configs/single_turn_all_local.yaml \
TRAIN_DATA=/root/math_eval/aime25.jsonl EVAL_DATA=/root/math_eval/minerva_math.jsonl \
bash examples/TTS/run-single-turn.sh

# 2. local solver + remote gpt-5.6-luna optimizer (put OPENAI_API_KEY in examples/TTS/.env):
CONFIG=examples/TTS/configs/single_turn_math.yaml bash examples/TTS/run-single-turn.sh

# 3. if the server is already up (or config is fully remote), skip launching it:
START_SERVER=0 CONFIG=... bash examples/TTS/run-single-turn.sh
```

Or drive the loop directly:

```bash
PYTHONPATH=. python -m examples.TTS.run_tts \
    --config examples/TTS/configs/single_turn_math.yaml \
    --train-data /root/math_eval/aime25.jsonl \
    --eval-data  /root/math_eval/minerva_math.jsonl \
    --rounds 4 --out-dir /root/tts_out
```

## Outputs (`--out-dir`)

- `best_skill_gen_prompt.txt` — the highest held-out-accuracy skill-gen prompt found.
- `tts_log.jsonl` — one record per phase (seed eval, each round's optimize +
  eval), including the optimizer's diagnosis, old/new prompts, and per-query
  `(problem, skill, correct)` samples for inspection.

## Files

| file            | role                                                                 |
| --------------- | -------------------------------------------------------------------- |
| `engines.py`    | `Engine` backends (openai/anthropic/sglang) + config-driven factory  |
| `prompts.py`    | seed skill-gen prompt (layer 1) + meta-optimizer prompt (layer 2)    |
| `scaffold.py`   | layer-1 solve: generate skill → answer with skill                    |
| `optimizer.py`  | layer-2: rewrite the skill-gen prompt from traces                    |
| `grading.py`    | thin reuse of `examples/SDPO/reward.py::_is_correct`                 |
| `data.py`       | load SDPO-format JSONL → `{problem, label, domain}`                  |
| `run_tts.py`    | the multi-round orchestration loop                                   |
| `run-single-turn.sh` | launcher: (optional) local SGLang server + the loop             |
| `configs/*.yaml`| per-role engine settings (local/remote)                              |

## Multi-turn code variant (next)

A multi-turn variant on **code** tasks with a **Qwen3.5-4B** scaffold —
mirroring [`examples/SDPO_ReAct`](../SDPO_ReAct)'s tool-calling ReAct loop and
reusing its `tools/registry.py` — will be added after review of this single-turn
version. It swaps the single solver call for a multi-turn tool loop (the skill
still conditions the first turn; the optimizer still rewrites the skill-gen
prompt from whole-trajectory traces).
```
