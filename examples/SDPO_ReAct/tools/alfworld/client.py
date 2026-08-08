"""Training-side client for the ``alfworld_step`` tool -- forwards
``execute_tool(name, params)`` calls to the ALFWorld sidecar
(tools/alfworld/docker/server.py, started by
tools/alfworld/run_alfworld_sidecar.sh) over ONE fixed port, reusing the same
retrying async HTTP client (``miles.utils.http_utils.post``) every other
SDPO_ReAct tool client uses -- same pattern as tools/search/client.py.

Two things ALFWorld needs beyond a plain stateless tool (mirroring search's
own stateful-session precedent, plus one new wrinkle search doesn't have):

1. Session id: an ALFWorld episode is STATEFUL across the whole trajectory
   (the game engine holds room/inventory state between steps), so every call
   in one trajectory must reach the sidecar with the SAME session id -- from
   ``current_trajectory_session_id()``, exactly like search.

2. Task assignment: UNLIKE search (which just answers whatever question is
   IN the prompt), which specific ALFWorld game file to play is not implied
   by the model's own text -- it must be assigned OUT OF BAND so the SAME
   game gets replayed on every train/eval pass for a given row (comparable
   success rate across arms/checkpoints). The data builder
   (data/build_alfworld_data.py) stamps this onto each row's
   metadata["alfworld_game_file"] (plus metadata["alfworld_split"] -- one of
   "train"/"eval_in_distribution"/"eval_out_of_distribution", since
   AlfredTWEnv reads a DIFFERENT config path per split and the sidecar must
   load the matching one to find that game_file at all); this client reads
   both back via ``current_trajectory_metadata()`` and forwards them to the
   sidecar's /session/{id}/step endpoint, which only consumes them on the
   FIRST call for that session (session already has a live env after that --
   see docker/server.py's init_session).

The reverse direction uses the SAME metadata channel: when a step response
comes back ``done=True``, this client stamps ``episode_won``/``task_type``
directly onto ``current_trajectory_metadata()`` -- the identical dict object
backing ``sample.metadata`` (see multi_turn.py's docstring), so
examples/SDPO/reward.py's ``_grade_one_alfworld`` can read it after
``generate()`` returns with zero extra plumbing.
"""

import os

from miles.rollout.generate_hub.multi_turn import current_trajectory_metadata, current_trajectory_session_id
from miles.utils.http_utils import post

ALFWORLD_SIDECAR_URL = os.environ.get("SDPO_REACT_ALFWORLD_SIDECAR_URL", "http://127.0.0.1:8423")
MAX_RESULT_CHARS = 4000

_TOOL_NAMES = {"alfworld_step"}


def _clip(text: str) -> str:
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "\n...[truncated]..."


async def call_alfworld_tool(args: dict) -> str:
    session_id = current_trajectory_session_id()
    metadata = current_trajectory_metadata()
    game_file = metadata.get("alfworld_game_file")
    split = metadata.get("alfworld_split", "train")
    try:
        payload = await post(
            f"{ALFWORLD_SIDECAR_URL}/session/{session_id}/step",
            {"action": args.get("action", ""), "game_file": game_file, "split": split},
            max_retries=3,
            action="post",
        )
    except Exception as e:
        return f"error:\nalfworld sidecar unreachable: {e}"
    if payload.get("done"):
        metadata["episode_won"] = bool(payload.get("won", False))
    return _clip(payload.get("observation", "(no output)"))


async def execute_tool(name: str, params) -> str:
    """Single-tool ``--generate-execute-tool-function-path`` target for an
    alfworld-ONLY run. Multi-domain runs should point at
    ``examples.SDPO_ReAct.tools.registry.execute_tool`` instead."""
    if name not in _TOOL_NAMES:
        return f"error:\nunknown tool '{name}' (this executor only handles alfworld_step)"
    args = params if isinstance(params, dict) else {}
    return await call_alfworld_tool(args)
