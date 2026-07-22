"""Tool executor for SDPO_ReAct: the ``--generate-execute-tool-function-path``
target consumed by ``miles.rollout.generate_hub.multi_turn.generate`` (via
``miles.rollout.generate_utils.tool_call_utils.execute_tool_calls``).

Talks to the single, long-lived sandbox sidecar (tools/docker/sandbox_server.py,
started once by tools/run_sandbox.sh) over ONE fixed port, reusing the same
retrying async HTTP client (`miles.utils.http_utils.post`) the rollout code
already uses for `/generate` -- no new port is opened per rollout worker, per
GPU, or per tool call; every concurrent trace across every SGLang engine shares
this one endpoint.
"""

import json
import os

from miles.utils.http_utils import post

SANDBOX_URL = os.environ.get("SDPO_REACT_SANDBOX_URL", "http://127.0.0.1:8420")
DEFAULT_CODE_TIMEOUT_SECONDS = 10.0
MAX_RESULT_CHARS = 4000


def _format_result(payload: dict) -> str:
    if payload.get("timed_out"):
        return f"[timeout] {payload.get('error', 'execution timed out')}"
    stdout = (payload.get("stdout") or "").strip()
    error = payload.get("error")
    if error:
        return f"stdout:\n{stdout}\nerror:\n{error}" if stdout else f"error:\n{error}"
    return stdout if stdout else "(no output)"


async def _execute_code(code: str) -> str:
    try:
        # `action` is the real httpx.AsyncClient method name to call
        # (`getattr(client, action)`), NOT a free-form label -- "post" here,
        # not the tool name.
        payload = await post(
            f"{SANDBOX_URL}/execute",
            {"code": code, "timeout": DEFAULT_CODE_TIMEOUT_SECONDS},
            max_retries=3,
            action="post",
        )
    except Exception as e:
        # A dead/unreachable sandbox should surface as a tool-observation error
        # (so the rollout keeps going and dynamic filters like check_no_aborted
        # can catch it at the group level) rather than crashing the rollout task.
        return f"error:\ncode_interpreter sandbox unreachable: {e}"

    result = _format_result(payload)
    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS] + "\n...[truncated]..."
    return result


async def execute_tool(name: str, params) -> str:
    # An untrained/weak model can emit tool-call arguments that are valid JSON
    # but not a dict -- e.g. a JSON-encoded string (double-encoded arguments)
    # or a bare code string instead of {"code": ...}. tool_call_utils.py's
    # execute_tool_calls does `json.loads(call.arguments)` with no shape
    # check, so `params` here is whatever that produced. Normalize defensively
    # rather than letting `.get()` crash the whole rollout task (observed
    # failure: "AttributeError: 'str' object has no attribute 'get'").
    if isinstance(params, str):
        try:
            decoded = json.loads(params)
            params = decoded if isinstance(decoded, dict) else {"code": params}
        except json.JSONDecodeError:
            params = {"code": params}
    elif not isinstance(params, dict):
        params = {}

    if name == "code_interpreter":
        code = params.get("code", "")
        if not isinstance(code, str) or not code.strip():
            return "error:\nno code provided"
        return await _execute_code(code)

    if name in ("cli_exec", "web_search"):
        raise NotImplementedError(
            f"tool '{name}' is registered in tool_specs.py but has no backend yet "
            "(see the module docstring there) -- do not include it in `tool_specs` "
            "until this is implemented."
        )

    return f"error:\nunknown tool '{name}'"
