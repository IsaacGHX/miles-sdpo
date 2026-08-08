"""Composable multi-task tool registry for SDPO_ReAct.

Design goal (per the proposal's extensibility requirement): adding a new tool /
environment must be a NON-INVASIVE, one-place change -- no edits to the rollout
loop, the SDPO reward, or existing tools. This module is that one place; the
tools THEMSELVES live one-per-capability under tools/code/, tools/cli/,
tools/search/ (spec + client, and for search a docker/ sidecar -- see each
subpackage's own docstring), so a new tool is a new subpackage plus one entry
in ``_REGISTRY`` below, never an edit to an existing tool's files.

A "tool" here is a triple:
  - an OpenAI function spec (what the model is TOLD it can call),
  - an async backend `handler(params: dict) -> str` (what actually runs),
  - membership in one or more named tool SETS (math / code / search / all).

The active tool set is chosen by the `SDPO_REACT_TOOLSET` env var (default
"math", i.e. just code_interpreter -- identical to the base version). The
launcher points `--generate-tool-specs-path` at `active_tool_specs` and
`--generate-execute-tool-function-path` at `execute_tool`, both of which read
that env var, so switching environments is a launcher env change, never a code
edit. `tool_specs` (the base name other modules import) stays an alias for the
math set so nothing that imported it breaks.

Nothing here is model-specific; it works for the native (multi_turn.generate)
and legacy (generate_with_tools.generate) paths alike, since both ultimately
call the same `execute_tool(name, params) -> str` contract.
"""

import json
import os

from examples.SDPO_ReAct.tools.alfworld.client import call_alfworld_tool
from examples.SDPO_ReAct.tools.alfworld.spec import ALFWORLD_STEP_SPEC
from examples.SDPO_ReAct.tools.cli.client import run_command
from examples.SDPO_ReAct.tools.cli.spec import CLI_EXEC_SPEC
from examples.SDPO_ReAct.tools.code.client import run_code
from examples.SDPO_ReAct.tools.code.spec import CODE_INTERPRETER_SPEC
from examples.SDPO_ReAct.tools.search.client import call_search_tool
from examples.SDPO_ReAct.tools.search.spec import FIND_SPEC, OPEN_SPEC, SEARCH_SPEC
from examples.SDPO_ReAct.tools.webshop.client import call_webshop_tool
from examples.SDPO_ReAct.tools.webshop.spec import WEBSHOP_STEP_SPEC

# --------------------------------------------------------------------------- #
# Backends. Each is `async handler(params: dict) -> str`. Thin adapters over
# each tool subpackage's own client -- see tools/code/client.py,
# tools/cli/client.py, tools/search/client.py for the actual HTTP calls.
# --------------------------------------------------------------------------- #


async def _handle_code_interpreter(params: dict) -> str:
    return await run_code(params.get("code", ""), stdin=params.get("stdin"))


async def _handle_cli_exec(params: dict) -> str:
    return await run_command(params.get("command", ""))


async def _handle_search(params: dict) -> str:
    return await call_search_tool("search", params)


async def _handle_open(params: dict) -> str:
    return await call_search_tool("open", params)


async def _handle_find(params: dict) -> str:
    return await call_search_tool("find", params)


async def _handle_webshop_step(params: dict) -> str:
    return await call_webshop_tool(params)


async def _handle_alfworld_step(params: dict) -> str:
    return await call_alfworld_tool(params)


# --------------------------------------------------------------------------- #
# Registry: name -> (spec, handler, set-membership). Add a tool HERE only --
# the spec/handler themselves live in that tool's own subpackage.
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "code_interpreter": {"spec": CODE_INTERPRETER_SPEC, "handler": _handle_code_interpreter, "sets": {"math", "code", "all"}},
    "cli_exec": {"spec": CLI_EXEC_SPEC, "handler": _handle_cli_exec, "sets": {"code", "cli", "all"}},
    "search": {"spec": SEARCH_SPEC, "handler": _handle_search, "sets": {"search", "deepsearch", "all"}},
    "open": {"spec": OPEN_SPEC, "handler": _handle_open, "sets": {"search", "deepsearch", "all"}},
    "find": {"spec": FIND_SPEC, "handler": _handle_find, "sets": {"search", "deepsearch", "all"}},
    "webshop_step": {"spec": WEBSHOP_STEP_SPEC, "handler": _handle_webshop_step, "sets": {"webshop", "agentic", "all"}},
    "alfworld_step": {"spec": ALFWORLD_STEP_SPEC, "handler": _handle_alfworld_step, "sets": {"alfworld", "agentic", "all"}},
}


def _active_set_name() -> str:
    return os.environ.get("SDPO_REACT_TOOLSET", "math").strip().lower()


def active_tool_specs() -> list[dict]:
    """The specs for the tool set named by $SDPO_REACT_TOOLSET (default "math").
    Pointed at by --generate-tool-specs-path (called with no args by
    load_function, so this is a zero-arg callable returning the list)."""
    s = _active_set_name()
    return [t["spec"] for t in _REGISTRY.values() if s in t["sets"]]


# Back-compat alias: the base modules import `tool_specs` as a plain list.
# Evaluate the default ("math") set at import time so existing imports keep a
# list, while new launchers can call active_tool_specs() for env-driven sets.
tool_specs = [t["spec"] for t in _REGISTRY.values() if "math" in t["sets"]]

# Plain module-level lists for --generate-tool-specs-path (which load_function
# dereferences to the object directly -- a LIST, not a function, so the rollout
# parser gets the specs without calling anything). all_tool_specs = every tool
# (code_interpreter + cli_exec + search/open/find + webshop_step/alfworld_step)
# for a multi-domain run where the rollout must parse tool calls from ALL
# domains' rows.
all_tool_specs = [t["spec"] for t in _REGISTRY.values() if "all" in t["sets"]]

# agentic_tool_specs = just webshop_step + alfworld_step, for the agentic run
# script's "agentic" (webshop+alfworld combined) domain -- deliberately NOT
# all_tool_specs, so a mixed webshop/alfworld rollout can't accidentally
# parse a stray code_interpreter/search call no row in that domain ever
# declares (harmless if it happened, but confusing/wasteful).
agentic_tool_specs = [t["spec"] for t in _REGISTRY.values() if "agentic" in t["sets"]]


# --------------------------------------------------------------------------- #
# Shared MINIMAL system prompt (one source of truth for all domains).
#
# Design (per the proposal's "let the model choose + no over-prompting" intent):
# every row exposes ALL tools (all_tool_specs) and the prompt says NOTHING about
# which tool to use, how many hops to take, or any task-specific workflow -- the
# tool schemas are injected separately by the native <tools> block, and the
# CAPABILITY (multi-hop search, code verification, tool selection) must come from
# TRAINING, not from a hand-written recipe in the prompt. The only non-question
# content is the final-answer FORMAT contract, which grading depends on.
# --------------------------------------------------------------------------- #
MINIMAL_SYSTEM_PROMPT = (
    "You solve the user's problem step by step. You have tools available "
    "(declared below); call any that help, as many times as you need (multiple "
    "tool calls in one turn run concurrently). Do NOT answer from memory alone: "
    "before you commit to a final answer, you MUST use a tool to VERIFY it -- "
    "run code to check a computation, or search to confirm a fact. Only after a "
    "tool has confirmed your reasoning, give your final answer inside <answer> "
    "and </answer> tags, e.g. <answer>42</answer>. Put ONLY the final answer "
    "inside the tags."
)


async def execute_tool(name: str, params) -> str:
    """The --generate-execute-tool-function-path target. Normalizes `params`
    (an untrained policy can emit non-dict/double-encoded arguments) then
    dispatches to the registered backend. Unknown/inactive tools return an
    error observation rather than raising, so a bad call never crashes the
    rollout task."""
    if isinstance(params, str):
        try:
            decoded = json.loads(params)
            params = decoded if isinstance(decoded, dict) else {"code": params}
        except json.JSONDecodeError:
            params = {"code": params}
    elif not isinstance(params, dict):
        params = {}

    entry = _REGISTRY.get(name)
    if entry is None:
        return f"error:\nunknown tool '{name}'"
    return await entry["handler"](params)
