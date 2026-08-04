# SDPO-ReAct — multi-turn tool-calling rollout for SDPO

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
data, AIME-2024 eval, Qwen2.5-7B-Instruct (`tools/tool_specs.py` /
`tools/tool_client.py`, now thin back-compat shims -- see `tools/code/`).

Since the base version, the tool set has grown into a composable registry
(`tools/registry.py`) with THREE real backends -- `code_interpreter`/`cli_exec`
(one Docker sandbox, `tools/code/` + `tools/cli/`) and `search`/`open`/`find`
(a second sidecar wrapping gpt-oss's `simple_browser` tool over the existing
wiki-18 retriever, `tools/search/`) -- switchable via `$SDPO_REACT_TOOLSET`,
used by the multitask/native launcher
(`run-qwen3-4B-sdpo-react-native.sh`). See "Files" and "Extending" below.

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
- **One sidecar per capability, ONE fixed port each**: `code_interpreter`/
  `cli_exec` share the code sandbox (`tools/run_sandbox.sh` + `tools/docker/`);
  `search`/`open`/`find` get their OWN sidecar (`tools/search/
  run_search_sidecar.sh` + `tools/search/docker/`) -- kept separate because
  it wraps gpt-oss's `simple_browser` tool, which requires Python >=3.12, a
  different runtime from both the training image and the code sandbox. Every
  sidecar is reached over `miles.utils.http_utils.post` — the same pattern
  already used by `examples/experimental/swe-agent-v2`'s Harbor sidecar. No
  port is opened per GPU/engine/rollout worker, matching the host's hard
  port-count limit.
- **search/open/find = i-DeepSearch's own code, unmodified**:
  `tools/search/docker/browser.py` is a VERBATIM copy of i-DeepSearch's
  `tools/browser.py` (https://github.com/i-DeepSearch/observation-masking) --
  `BrowserTool`, `LocalServiceBrowserBackend`, `BrowserPool`, the
  `【id†url】` citation rendering, the page-stack/cursor model, all of it,
  unchanged. The only repo-specific addition is `retrieval_adapter.py`, a
  thin `/search`+`/get_content` HTTP shim in front of the wiki-18 retriever
  this repo already has staged, so `LocalServiceBrowserBackend`'s existing
  HTTP contract (built for i-DeepSearch's own BrowseComp-Plus search service)
  works against a different corpus with zero code changes.
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
├── native_prompt.py                            # native <tool_call> variant (multitask/Qwen3 launcher)
├── sdpo_react.py                               # thin group-RM wrapper around examples.SDPO.sdpo + trace dump
├── data/
│   ├── build_aime24_eval.py / build_native_eval.py # writes {prompt,label} eval jsonl (legacy / native)
│   ├── build_code_data.py / build_search_data.py   # LiveCodeBench / HotpotQA+2Wiki row builders (multitask)
│   ├── build_multitask_data.py                     # interleave+shuffle per-domain sources into one train.jsonl
│   ├── passk_filter_search.py                      # base-model pass@k learnability filter (search domain)
│   └── eval_aime24.yaml / eval_native_math.yaml /
│       eval_code.yaml / eval_multitask.yaml         # --eval-config per launcher/domain
├── docs/
│   └── prepare_doc.md                              # design/discussion notes (not a standalone README)
├── debug/                                          # manual one-off tools, never called from the run scripts
│   ├── rerender_search_rows.py                     # rewrite already-filtered search rows to the current prompt/tool design
│   ├── test_message_template.py                    # standalone chat-template render/round-trip check
│   ├── _debug_rollout.py                           # standalone multi-turn rollout probe against a live sglang server
│   └── analyze_explog.py                           # join explog.jsonl rows with their wandb eval curves into a table
├── run-qwen2.5-7B-sdpo-react-dapo-math.sh      # BASE launcher: DAPO train (5 turns) + AIME24 eval (20 turns),
│                                                 legacy plain-text tags, code_interpreter ONLY
├── run-qwen3-4B-sdpo-react-native.sh           # NATIVE/multitask launcher: native <tool_call>, math|code|
│                                                 search|multitask domains, full self-skill wiring
├── enroot-run-sdpo-react.sh                    # one-click no-sudo launcher (sibling of examples/SDPO's)
└── tools/
    ├── registry.py                              # tool registry (name -> spec/handler/set), $SDPO_REACT_TOOLSET
    ├── tool_specs.py / tool_client.py           # back-compat shims -> tools/code/ (single-tool base version)
    ├── reframe_trace.py                         # domain-agnostic debug tool (independent of react_prompt)
    ├── test_tools_docker.py                     # cross-sidecar smoke test (no miles/GPU)
    ├── run_sandbox.sh                           # idempotent: build+run the code sandbox container/port
    ├── docker/                                   # code sandbox image (code_interpreter + cli_exec backend)
    │   ├── Dockerfile                            # python3-slim + sympy/numpy/scipy + FastAPI sidecar
    │   ├── requirements.txt
    │   └── sandbox_server.py                     # POST /execute {code} -> {stdout, error, timed_out}
    ├── code/                                     # code_interpreter: spec + HTTP client + LiveCodeBench judge
    │   ├── spec.py / client.py / judge.py
    ├── cli/                                      # cli_exec: spec + client (wraps a shell cmd, reuses tools/code's sandbox)
    │   ├── spec.py / client.py
    └── search/                                   # search/open/find: spec + client + its OWN sidecar
        ├── spec.py / client.py
        ├── retrieval_server.py                    # torch-GPU wiki-18 dense retriever (drop-in for search-r1's)
        ├── run_retrieval.sh                       # start the retriever ABOVE (needs its own GPU)
        ├── run_search_sidecar.sh                  # idempotent: build+run the search/open/find sidecar/port
        ├── bench_retrieval.py                     # concurrency/latency benchmark for the retriever
        └── docker/                                # gpt-oss simple_browser sidecar (needs Python >=3.12,
            ├── Dockerfile                          # separate runtime from both the training image and the
            ├── requirements.txt                    # code sandbox)
            ├── browser.py                          # i-DeepSearch's BrowserTool/BrowserPool, copied VERBATIM
            ├── retrieval_adapter.py                 # /search+/get_content shim -> ../retrieval_server.py
            └── server.py                            # POST /session/{id}/call {tool, args} -> {observation}
```

## Wiring

```bash
--custom-generate-function-path miles.rollout.generate_hub.multi_turn.generate
--generate-tool-specs-path examples.SDPO_ReAct.tools.tool_specs.tool_specs
--generate-execute-tool-function-path examples.SDPO_ReAct.tools.tool_client.execute_tool
--generate-tool-call-parser qwen25
--generate-max-turns 5                          # training; eval overrides to 20 (see data/eval_aime24.yaml)

--group-rm
--custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
--eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
--sdpo-grader dapo --sdpo-teacher-backend megatron

--sdpo-self-skill --sdpo-skill-source env_feedback --sdpo-env-feedback-max-chars 2000
```

Eval running MORE turns than training (5 train / 20 eval, per spec) works via
a new per-sample metadata override: `data/eval_aime24.yaml`'s `metadata_overrides:
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

For the BASE launcher (single tool, `tools/tool_specs.py`/`tools/tool_client.py`),
turn-budget env vars are the only override point:

| Env var | Default | Purpose |
|---|---|---|
| `SDPO_REACT_TRAIN_MAX_TURNS` | `5` | training turn budget |
| `SDPO_REACT_EVAL_MAX_TURNS` | `20` | eval turn budget (read by `data/eval_aime24.yaml`) |
| `SDPO_REACT_EVAL_N_SAMPLES` | `8` | eval samples/prompt (read by `data/eval_aime24.yaml`) |

For the NATIVE/multitask launcher (`run-qwen3-4B-sdpo-react-native.sh`), the
active tool SET is an env var, not a script edit:

| Env var | Default | Purpose |
|---|---|---|
| `SDPO_REACT_TOOLSET` | `math` (code_interpreter only) | `code` \| `search` \| `cli` \| `all` -- see `tools/registry.py::_REGISTRY` |
| `SDPO_REACT_DOMAIN` | `math` | `math` \| `code` \| `search` \| `multitask` -- picks train/eval data AND starts the matching sidecar(s) |

Same "override via env, not by editing the example" flexibility as
`examples/EPO/enroot-run-epo.sh`'s `EPO_MODEL` switch. To add a genuinely NEW
tool: write a new subpackage under `tools/<name>/` following `tools/cli/`'s
shape (a `spec.py` OpenAI function spec + a `client.py` with an async
`execute_tool(name, params) -> str` for single-tool use), add one entry to
`tools/registry.py::_REGISTRY` (spec + handler + which sets it belongs to) --
no edits to the rollout loop, SDPO reward, or any OTHER tool's files, per the
registry's own design goal.

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

## Limitations

Base launcher (`run-qwen2.5-7B-sdpo-react-dapo-math.sh`) only:
- Only `code_interpreter` has a real backend via `tools/tool_specs.py`'s single-
  tool spec list; that launcher never registers `cli_exec`/`search`.

Both launchers:
- Every sidecar (code sandbox, search sidecar, wiki-18 retriever) is single-
  container/single-host — no multi-node scaling of tool execution capacity in
  this pass.
- `env_feedback` skill source only fires for traces that actually called a
  tool; rollouts with zero tool calls behave exactly like plain `--sdpo-self-skill
  --sdpo-skill-source incorrect` (env_feedback text is empty, a no-op).
- `search`/`open`/`find` state (which page is "currently open") lives ONLY in
  the search sidecar's process memory, keyed by a per-trajectory session id
  (`miles.rollout.generate_hub.multi_turn.current_trajectory_session_id`) --
  a sidecar restart mid-training silently resets every in-flight trajectory's
  browsing state (a fresh `search` starts a new page stack; no crash, just as
  if the trajectory had never opened anything).
