# SDPO-ReAct — multi-turn tool-calling rollout for SDPO (base version)

Extends [`examples/SDPO`](../SDPO) (prefix-conditioned self-distillation) from
single-turn GRPO rollout to a **ReAct-style multi-turn tool-calling rollout**:
the model can call a `code_interpreter` tool (Python code, run in an isolated
Docker sandbox) across up to `--generate-max-turns` turns before giving its
final answer. SDPO's own reward/prefix machinery
(`examples.SDPO.sdpo.sdpo_group_reward`) is reused **unchanged** — it already
operates purely on `Sample.response`/`Sample.tokens`/`Sample.metadata`, which
holds for a multi-turn trajectory exactly like a single-turn one, as long as
the whole trajectory stays one `Sample` (`--generate-multi-samples` off).

Base version scope: **one tool (Python code execution)**, DAPO math training
data, AIME-2024 eval, Qwen2.5-7B-Instruct. `cli_exec` / `web_search` are
declared as schema-only stubs in `tools/tool_specs.py` for a later pass.

## Why this design

- **Rollout loop**: reused as-is from `miles.rollout.generate_hub.multi_turn.generate`
  (SGLang's native `FunctionCallParser`, `--generate-tool-call-parser qwen25`).
  No hand-rolled `<search>/<information>/<answer>`-style regex tag loop (unlike
  `examples/search-r1`) — tool specs, tool calls, and tool responses all
  round-trip through `tokenizer.apply_chat_template`, so they are always
  byte-exact with whatever chat template the installed tokenizer defines. See
  `docs/user-guide/agentic-chat-template.md` for the append-only-prefix
  invariant this relies on.
- **One-shot example**: `react_prompt.py` builds the worked example (think →
  tool call → observation → final `\boxed{}` answer) as **real chat messages**
  with a genuine `tool_calls` field — never a string-literal tag — so it always
  renders through the model's own template.
- **Docker sandbox**: exactly **one** long-lived sidecar container, exposing
  exactly **one** fixed port for the entire training job (`tools/run_sandbox.sh`
  + `tools/docker/`), reached over `miles.utils.http_utils.post` — the same
  pattern already used by `examples/experimental/swe-agent-v2`'s Harbor sidecar.
  No port is opened per GPU/engine/rollout worker, matching the host's hard
  port-count limit.
- **SDPO integration**: `sdpo_react.py` is a thin wrapper (same pattern as
  `examples/EPO/epo.py`) around `examples.SDPO.sdpo.sdpo_group_reward` /
  `sdpo_eval_reward` — no fork, no duplicated logic. It only adds tool-call
  bookkeeping (`tool_call_count`, `tool_trace`) on `sample.metadata` before
  delegating.
- **Env-feedback dense prefix**: fills in `examples/SDPO/sdpo.py`'s previously
  unimplemented `--sdpo-skill-source env_feedback` branch — a failed trace's
  own tool-execution trace (code + result/error) grounds its self-generated
  pitfall skill, the direct analogue of lasgroup/SDPO's "reprompt with
  environment feedback" idea. Reuses the existing prefix splice point
  (`_build_teacher_prompt_str`); no new training-side machinery.

## Files

```text
examples/SDPO_ReAct/
├── react_prompt.py                            # system prompt + one-shot example (real tool_calls messages)
├── sdpo_react.py                               # thin group-RM wrapper around examples.SDPO.sdpo + trace dump
├── build_aime24_eval.py                        # writes {prompt,label} aime24.jsonl
├── eval_aime24.yaml                            # --eval-config: 8 samples/prompt, 20-turn eval budget (env-overridable)
├── run-qwen2.5-7B-sdpo-react-dapo-math.sh      # launcher: DAPO train (5 turns) + AIME24 eval (20 turns)
├── enroot-run-sdpo-react.sh                    # one-click no-sudo launcher (sibling of examples/SDPO's)
└── tools/
    ├── tool_specs.py                           # code_interpreter spec (+ cli_exec/web_search stubs)
    ├── tool_client.py                          # execute_tool() -> HTTP call to the sandbox sidecar
    ├── run_sandbox.sh                          # idempotent: build+run the ONE sandbox container/port
    └── docker/
        ├── Dockerfile                          # python3-slim + sympy/numpy/scipy + FastAPI sidecar
        ├── requirements.txt
        └── sandbox_server.py                   # POST /execute {code} -> {stdout, error, timed_out}
```

## Wiring

```bash
--custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate
--generate-tool-specs-path examples.SDPO_ReAct.tools.tool_specs.tool_specs
--generate-execute-tool-function-path examples.SDPO_ReAct.tools.tool_client.execute_tool
--generate-tool-call-parser qwen25
--generate-max-turns 5                          # training; eval overrides to 20 (see eval_aime24.yaml)

--group-rm
--custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
--eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
--sdpo-grader dapo --sdpo-teacher-backend megatron

--sdpo-self-skill --sdpo-skill-source env_feedback --sdpo-env-feedback-max-chars 2000
```

Eval running MORE turns than training (5 train / 20 eval, per spec) works via
a new per-sample metadata override: `eval_aime24.yaml`'s `metadata_overrides:
{generate_max_turns: 20}` is injected into each eval sample's metadata by
`EvalDatasetConfig.inject_metadata`, and `multi_turn.generate` now reads that
override before falling back to the global `--generate-max-turns` (see the
`max_turns = sample.metadata.get("generate_max_turns", args.generate_max_turns)`
line added there) — the same override pattern `Sample.generate_function_path`
already uses for per-eval-dataset custom generate functions.

## Quickstart

```bash
bash examples/SDPO_ReAct/run-qwen2.5-7B-sdpo-react-dapo-math.sh
# or, no-sudo / no-Docker-daemon host (enroot instead of docker for the
# TRAINING container; the sandbox sidecar itself is still real Docker,
# started on the host before entering enroot -- see the script's header):
bash examples/SDPO_ReAct/enroot-run-sdpo-react.sh
```

This will (idempotently): start the sandbox sidecar, download+prep DAPO math
and AIME-2024, prepend the ReAct system/one-shot prompt to every training row,
then launch training via `ray job submit`.

### Extending to a new task / bigger tool set

Everything that changes per task is an env var, not a script edit:

| Env var | Default | Purpose |
|---|---|---|
| `SDPO_REACT_TOOL_SPECS_PATH` | `examples.SDPO_ReAct.tools.tool_specs.tool_specs` | point at your own tool set |
| `SDPO_REACT_EXECUTE_TOOL_PATH` | `examples.SDPO_ReAct.tools.tool_client.execute_tool` | point at your own executor |
| `SDPO_REACT_TRAIN_MAX_TURNS` | `5` | training turn budget |
| `SDPO_REACT_EVAL_MAX_TURNS` | `20` | eval turn budget (read by `eval_aime24.yaml`) |
| `SDPO_REACT_EVAL_N_SAMPLES` | `8` | eval samples/prompt (read by `eval_aime24.yaml`) |

Same "override via env, not by editing the example" flexibility as
`examples/EPO/enroot-run-epo.sh`'s `EPO_MODEL` switch. To add a CLI-exec or
web-search tool: write a new `tool_specs`/`execute_tool` module following
`tools/tool_specs.py`'s / `tools/tool_client.py`'s shape (a plain list of
OpenAI function specs + an async `execute_tool(name, params) -> str`), then
point `SDPO_REACT_TOOL_SPECS_PATH`/`SDPO_REACT_EXECUTE_TOOL_PATH` at it.

## Monitoring

Two complementary things land under `--dump-details <dir>` (set by the
launcher to `sdpo_dumps/<exp>/`), plus a wandb panel:

- **`<dir>/agentic_traces/*.jsonl`** (written by `sdpo_react.py::
  _dump_agentic_traces`, one file per rollout group, rollout-side): a full
  reconstructed **message-dict trace** per sample — `[{"role": "user", ...},
  {"role": "assistant", "content": ..., "tool_calls": [...]}, {"role": "tool",
  ...}, ...]` — plus `label`, `tool_call_count`, `tool_error_count`,
  `sdpo_correct`, `status`. This is the human-readable "what did the model
  reason, what did it call, what came back, was it right" view.
- **`<dir>/sdpo_prompts/*.jsonl` / `<dir>/skill/*.jsonl`** (written by
  `MegatronTrainRayActor._dump_sdpo_prompts`, training-side, unchanged from
  `examples/SDPO`): the decoded student/teacher full sequences and, when
  `--sdpo-self-skill` is on (it is, here, via `--sdpo-skill-source
  env_feedback`), the self-generated skill/pitfall text — complementary to
  the message-dict dump above, not a replacement for it.
- **wandb `agentic/*` panel** (`miles.ray.rollout.metrics.py::
  _compute_agentic_tool_metrics`, generic — not SDPO_ReAct-specific, fires for
  ANY run using `multi_turn.generate`):
  - `agentic/round_number_{mean,max,min}` — turns used per trajectory
    (`round_number` is now populated by `multi_turn.generate` itself for any
    caller, feeding the pre-existing `--log-multi-turn` panel too).
  - `agentic/hit_max_turns_frac` — fraction that used the FULL turn budget
    (ran out of turns before answering).
  - `agentic/tool_call_count_{mean,max}`, `agentic/zero_tool_call_frac`.
  - `agentic/tool_error_rate` — errors / total tool calls (SDPO_ReAct tags
    `tool_error_count` from the sandbox's `error:`/`[timeout]` observation
    prefixes; other tools can populate the same key to get this metric).

## Verification (do these IN ORDER before a real training run)

1. **Template / tool-call round-trip** — confirm `tool_specs.py`'s spec renders
   correctly via `tokenizer.apply_chat_template(tools=...)` for
   Qwen2.5-7B-Instruct, and a canned `<tool_call>{...}</tool_call>` completion
   parses via SGLang's `qwen25` `FunctionCallParser` and round-trips through
   `tool_call_utils._tokenize_postfix_messages`'s append-only assertion. See
   `tests/fast/examples/test_sdpo_react.py`.
2. **Sandbox smoke test**:
   ```bash
   bash examples/SDPO_ReAct/tools/run_sandbox.sh
   curl -s localhost:8420/execute -d '{"code":"print(2+2)"}' -H 'Content-Type: application/json'
   # -> {"stdout":"4\n","error":null,"timed_out":false}
   docker port sdpo-react-sandbox   # confirm exactly one port is published
   ```
3. **Single-rollout dry run** — run `multi_turn.generate` directly against a
   handful of DAPO math prompts (1 GPU, no training step); confirm the model
   emits a tool call, the sandbox result is visibly used in the next turn, a
   no-tool-call response still terminates cleanly with a final boxed answer,
   and a forced-timeout case doesn't hang the rollout.
4. **Reward/prefix wiring** — run `sdpo_react_group_reward` over a small
   synthetic multi-turn group and confirm `sample.metadata["sdpo_teacher_prompt_tokens"]`
   is populated for traces with a correct peer (CPU-only unit test, see below).
5. **End-to-end short run** — small `--num-rollout` first (watch wandb for
   `rewards`, `sdpo_correct`, tool-call-count), then a full run to compare
   AIME24 pass@1 against the no-tool `examples/SDPO` baseline.

## Tests

```bash
pytest tests/fast/examples/test_sdpo_react.py -v
```

## Limitations (base version)

- Only `code_interpreter` has a real backend; `cli_exec`/`web_search` are
  schema stubs that raise `NotImplementedError` if ever invoked.
- The sandbox sidecar is single-container/single-host — no multi-node scaling
  of tool execution capacity in this pass.
- `env_feedback` skill source only fires for traces that actually called the
  tool; rollouts with zero tool calls behave exactly like plain `--sdpo-self-skill
  --sdpo-skill-source incorrect` (env_feedback text is empty, a no-op).
