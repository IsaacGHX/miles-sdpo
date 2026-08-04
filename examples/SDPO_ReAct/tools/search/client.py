"""Training-side client for the search/open/find tool set -- forwards
``execute_tool(name, params)`` calls to the search sidecar
(tools/search/docker/server.py, started by tools/search/run_search_sidecar.sh)
over ONE fixed port, reusing the same retrying async HTTP client
(``miles.utils.http_utils.post``) the rollout code already uses for
``/generate`` -- same pattern as tools/code/client.py.

Session id: search/open/find are STATEFUL within one trajectory (open/find
both refer to "the page most recently shown"), so every call in the same
trajectory must reach the sidecar with the SAME session id. That id comes
from ``miles.rollout.generate_hub.multi_turn``'s per-trajectory ContextVar
(see that module's ``_SESSION_ID`` and this file's ``_session_id`` below) --
NOT from ``Sample.session_id``, which is only populated under
``--sglang-router-policy consistent_hashing`` and is unrelated to this tool's
notion of a session.
"""

import os

from miles.rollout.generate_hub.multi_turn import current_trajectory_session_id
from miles.utils.http_utils import post

SEARCH_SIDECAR_URL = os.environ.get("SDPO_REACT_SEARCH_SIDECAR_URL", "http://127.0.0.1:8421")
MAX_RESULT_CHARS = 4000

_TOOL_NAMES = {"search", "open", "find"}


def _clip(text: str) -> str:
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "\n...[truncated]..."


async def call_search_tool(name: str, args: dict) -> str:
    session_id = current_trajectory_session_id()
    try:
        payload = await post(
            f"{SEARCH_SIDECAR_URL}/session/{session_id}/call",
            {"tool": name, "args": args},
            max_retries=3,
            action="post",
        )
    except Exception as e:
        return f"error:\nsearch sidecar unreachable: {e}"
    return _clip(payload.get("observation", "(no output)"))


async def execute_tool(name: str, params) -> str:
    """Single-tool ``--generate-execute-tool-function-path`` target for a
    search-ONLY run. Multi-tool runs should point at
    ``examples.SDPO_ReAct.tools.registry.execute_tool`` instead."""
    if name not in _TOOL_NAMES:
        return f"error:\nunknown tool '{name}' (this executor only handles search|open|find)"
    args = params if isinstance(params, dict) else {}
    return await call_search_tool(name, args)
