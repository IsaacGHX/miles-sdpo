"""Sidecar server exposing ``webshop_step`` for SDPO_ReAct, built directly on
WebShop's own ``WebAgentTextEnv-v0`` gym env (github.com/princeton-nlp/WebShop,
vendored verbatim at build time -- see Dockerfile) -- no tag-parsing/text-
action-loop reimplemented here, just this repo's session-pool HTTP wrapper.

Runs INSIDE its own container (Python<=3.10 + WebShop's own old gym/pyserini/
torch pins, which would otherwise conflict with the training image -- see
Dockerfile's docstring), one long-lived sidecar for the whole training job,
exposing exactly one HTTP port -- same pattern as ../../alfworld/docker/
server.py and ../../search/docker/server.py.

State model: WebshopPool.sessions keeps one live gym env PER SESSION id for
the container's lifetime -- a session corresponds to one model trajectory
(miles/rollout/generate_hub/multi_turn.py assigns it a fresh id per rollout;
see ../client.py for the caller side). No persistence/cleanup thread by
design, same rationale as every other sidecar's session pool.

Task pinning: WebShop's own ``WebAgentTextEnv.reset(session=task_id)`` DOES
natively index into a fixed, seeded-shuffle goal list by session_int -- but
that's only HALF the story. ``get_human_goals`` (web_agent_site/engine/
goal.py) also picks each goal's price-upper-bound via an UNSEEDED
``random.sample(price_range, 2)`` call, made once per gym.make() at
SimServer-construction time, BEFORE WebAgentTextEnv.__init__ ever calls its
own ``random.seed(233)`` (which only covers the goal-list SHUFFLE, not this).
Confirmed live: two independent sessions pinned to the SAME task_id produced
DIFFERENT price thresholds in their instruction text ("...lower than 60.00
dollars" vs "...lower than 30.00 dollars") until this was seeded. Fix:
``random.seed(0)`` right before EVERY ``gym.make()`` call below, so the
process-global unseeded call inside get_goals() becomes deterministic too --
this is the only way to get the SAME task_id to mean the exact same goal text
on every replay, which reproducible train/eval assignment (see
../client.py's docstring) genuinely requires.

Contract (see ../client.py for the caller side):
    POST /session/{session_id}/step {"action": str, "task_id": int|str|None}
      -> {"observation": str, "done": bool, "won": bool, "task_score": float}
    GET  /health -> {"status": "ok"}
    DELETE /session/{session_id} -> {"status": "ok"}
"""

import logging
import random
from typing import Optional, Union

import gym
from fastapi import FastAPI
from pydantic import BaseModel

# Vendored WebShop's own gym env registration (see Dockerfile: cloned into
# /app/webshop, PYTHONPATH set there) -- importing this module registers
# WebAgentTextEnv-v0 with gym.
import web_agent_site.envs  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("webshop_sidecar")

app = FastAPI()


class WebshopPool:
    """One live WebAgentTextEnv per session id."""

    def __init__(self):
        self.sessions: dict[str, object] = {}

    def init_session(self, session_id: str, task_id=None) -> None:
        # See module docstring: seeds the process-global `random` module
        # BEFORE gym.make() so get_goals()'s unseeded price-threshold sample
        # is deterministic too, not just the goal-list shuffle order.
        random.seed(0)
        env = gym.make("WebAgentTextEnv-v0", observation_mode="text")
        if task_id is not None:
            env.reset(session=task_id)
        else:
            env.reset()
        self.sessions[session_id] = env

    def step(self, session_id: str, action: str) -> tuple[str, bool, bool, float]:
        env = self.sessions[session_id]
        # WebAgentTextEnv.step()'s 4th return value (`info`) is ALWAYS None --
        # confirmed by reading web_agent_site/envs/web_agent_text_env.py:
        # `step()` hardcodes `info = None` and never assigns it. The real
        # cumulative reward (0.0 while shopping, the get_reward() score in
        # [0, 1] once a purchase is made) is the 2nd return value. Reading it
        # from `info.get("task_score")` (the original bug here) always fell
        # through the `isinstance(info, dict)` guard to 0.0, silently
        # discarding every episode's true reward regardless of outcome --
        # confirmed live: a purchase that actually scored 1.0 reported 0.0.
        obs, reward, done, _info = env.step(action)
        task_score = float(reward)
        won = bool(done and task_score >= 1.0)
        return obs, bool(done), won, task_score

    def cleanup(self, session_id: str) -> None:
        env = self.sessions.pop(session_id, None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


_POOL = WebshopPool()
_INITIALIZED_SESSIONS: set[str] = set()


class StepRequest(BaseModel):
    action: str
    # pydantic 1.8.2 (pinned by WebShop's own spacy<3.4 requirement -- see
    # requirements.txt) doesn't support PEP 604 `X | Y` union syntax; use
    # typing.Optional/Union instead. IMPORTANT: pydantic 1.x's Union coercion
    # tries member types IN ORDER and keeps the first that doesn't error --
    # `Union[int, str]` with a JSON string body value like "9" should coerce
    # to int 9, but empirically (confirmed live) it was landing as the STRING
    # "9". WebAgentTextEnv.reset()'s own `isinstance(session, int)` check then
    # silently fails, falling through to a RANDOM goal every time (which,
    # combined with our seeded `random.seed(0)`, deterministically picked the
    # SAME "random" goal regardless of the requested task_id -- the exact bug
    # this comment is here to prevent regressing). Use plain `int` (not a
    # Union) and cast at the call site instead of trusting pydantic to coerce.
    task_id: Optional[int] = None


class StepResponse(BaseModel):
    observation: str
    done: bool
    won: bool
    task_score: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/session/{session_id}/step", response_model=StepResponse)
def step(session_id: str, req: StepRequest) -> StepResponse:
    if session_id not in _INITIALIZED_SESSIONS:
        _POOL.init_session(session_id, task_id=req.task_id)
        _INITIALIZED_SESSIONS.add(session_id)

    try:
        observation, done, won, task_score = _POOL.step(session_id, req.action)
        return StepResponse(observation=observation, done=done, won=won, task_score=task_score)
    except Exception as e:  # a tool-usage error must surface as an observation, not a 500
        logger.warning("session=%s action=%r failed: %r", session_id, req.action, e)
        return StepResponse(observation=f"error:\n{e}", done=False, won=False, task_score=0.0)


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    """Explicit cleanup hook (not required for correctness -- see the module
    docstring -- but frees a live env proactively instead of waiting for
    process exit)."""
    _POOL.cleanup(session_id)
    _INITIALIZED_SESSIONS.discard(session_id)
    return {"status": "ok"}
