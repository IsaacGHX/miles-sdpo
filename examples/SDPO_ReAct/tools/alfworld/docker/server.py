"""Sidecar server exposing ``alfworld_step`` for SDPO_ReAct, built directly on
the ``alfworld`` pip package's TextWorld env (``AlfredTWEnv``) -- no
tag-parsing/text-action-loop reimplemented here, just this repo's session-pool
HTTP wrapper around the package's own ``step``/``init_env``.

Runs INSIDE its own container (Python + a pinned gymnasium/stable-baselines3/
alfworld stack that would otherwise conflict with the training image's own
torch/transformers pins -- see Dockerfile's docstring), one long-lived sidecar
for the whole training job, exposing exactly one HTTP port -- same pattern as
../../search/docker/server.py and ../../docker/sandbox_server.py.

State model: AlfworldPool.sessions keeps one live TextWorld env PER SESSION
id for the container's lifetime -- a session corresponds to one model
trajectory (miles/rollout/generate_hub/multi_turn.py assigns it a fresh id
per rollout; see ../client.py for the caller side). No persistence/cleanup
thread by design, same rationale as search's BrowserPool.

Game-file pinning: ``AlfredTWEnv`` has no official "reset to exactly this
game file" API -- ``self.game_files`` is a plain mutable list TextWorld's own
``init_env`` shuffles into a fixed-seed round-robin cycle. To pin a specific
file (so the SAME game replays across arms/checkpoints for a given training
row -- see ../client.py's docstring), ``init_session`` filters
``env.game_files`` down to exactly ``[game_file]`` BEFORE calling
``init_env(batch_size=1)``, so that cycle has only one entry to draw from.
This is the only available mechanism (confirmed by reading the installed
package's source); there is no seed/index parameter that achieves the same
thing more cleanly.

Contract (see ../client.py for the caller side):
    POST /session/{session_id}/step {"action": str, "game_file": str|None}
      -> {"observation": str, "done": bool, "won": bool}
    GET  /health -> {"status": "ok"}
    DELETE /session/{session_id} -> {"status": "ok"}
"""

import logging
import os

import yaml
from alfworld.agents.environment import get_environment
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("alfworld_sidecar")

app = FastAPI()

_CONFIG_PATH = os.environ.get("ALFWORLD_CONFIG", "/app/config_tw.yaml")
with open(_CONFIG_PATH) as f:
    _CONFIG = yaml.safe_load(f)


class AlfworldPool:
    """One live TextWorld env per session id. See module docstring for the
    game-file-pinning mechanism (the only non-obvious part of this class)."""

    def __init__(self, config: dict):
        self._config = config
        self.sessions: dict[str, object] = {}

    def init_session(self, session_id: str, game_file: str | None = None, split: str = "train") -> None:
        # split must be one of "train" | "eval_in_distribution" |
        # "eval_out_of_distribution" (AlfredTWEnv.collect_game_files reads a
        # DIFFERENT dataset.*_data_path config key per split -- confirmed by
        # reading the installed package's source). The data builder stamps
        # which split a game_file came from onto metadata["alfworld_split"];
        # the sidecar must load that SAME split's directory listing or the
        # requested game_file won't be found in game_files to filter down to.
        env_type = self._config["env"]["type"]
        base_env = get_environment(env_type)(self._config, train_eval=split)
        if game_file:
            # See module docstring: game_files is a plain mutable list;
            # filtering it to one entry before init_env is the only way to
            # pin ALFWorld's shuffle-cycle to a specific file.
            base_env.game_files = [game_file]
        env = base_env.init_env(batch_size=1)
        env.reset()  # REQUIRED before the first step() -- initializes TextWorld's
        # internal per-env state (self.last); the model's first tool call is
        # executed as an action AFTER this reset, same as data/build_alfworld_data.py
        # captures the reset's own initial observation for the row's prompt.
        self.sessions[session_id] = env

    def step(self, session_id: str, action: str) -> tuple[str, bool, bool]:
        env = self.sessions[session_id]
        obs, _scores, dones, infos = env.step([action])
        observation = obs[0]
        done = bool(dones[0])
        won = bool(infos.get("won", [False])[0]) if done else False
        admissible = infos.get("admissible_commands", [[]])[0]
        if admissible:
            observation = f"{observation}\nAdmissible actions: {', '.join(admissible)}"
        return observation, done, won

    def cleanup(self, session_id: str) -> None:
        env = self.sessions.pop(session_id, None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


_POOL = AlfworldPool(_CONFIG)
_INITIALIZED_SESSIONS: set[str] = set()


class StepRequest(BaseModel):
    action: str
    game_file: str | None = None
    split: str = "train"


class StepResponse(BaseModel):
    observation: str
    done: bool
    won: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/session/{session_id}/step", response_model=StepResponse)
def step(session_id: str, req: StepRequest) -> StepResponse:
    if session_id not in _INITIALIZED_SESSIONS:
        _POOL.init_session(session_id, game_file=req.game_file, split=req.split)
        _INITIALIZED_SESSIONS.add(session_id)

    try:
        observation, done, won = _POOL.step(session_id, req.action)
        return StepResponse(observation=observation, done=done, won=won)
    except Exception as e:  # a tool-usage error must surface as an observation, not a 500
        logger.warning("session=%s action=%r failed: %r", session_id, req.action, e)
        return StepResponse(observation=f"error:\n{e}", done=False, won=False)


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    """Explicit cleanup hook (not required for correctness -- see the module
    docstring -- but frees a live TextWorld env proactively instead of
    waiting for process exit)."""
    _POOL.cleanup(session_id)
    _INITIALIZED_SESSIONS.discard(session_id)
    return {"status": "ok"}
