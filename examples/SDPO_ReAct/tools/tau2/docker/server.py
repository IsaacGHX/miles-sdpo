"""Sidecar server exposing a τ²-bench (tau2-bench) orchestration endpoint for
SDPO_ReAct's tau2 domain -- built directly on the ``tau2`` pip package
(github.com/sierra-research/tau2-bench, upstream, NOT the `dhh1995` async
fork -- see module docstring below for why upstream suffices).

Runs INSIDE its own container (tau2-bench requires Python>=3.12, the training
image is 3.10 -- same isolation rationale as every other sidecar in this
repo), one long-lived sidecar for the whole training job, exposing exactly
one HTTP port.

UNLIKE webshop/alfworld (which wrap a stateful step() env behind a session-
pooled HTTP API consumed by this repo's OWN <tool_call>-parsing loop,
miles.rollout.generate_hub.multi_turn.generate), tau2's Orchestrator class
drives the ENTIRE multi-turn agent<->user-simulator<->environment
conversation itself -- there is no seam to slot in our own tool-parsing loop.
This sidecar instead implements the "external agent server" contract that
examples/experimental/swe-agent-v2/ already established for exactly this
shape: miles.rollout.generate_hub.agentic_tool_call.generate hands a
--custom-agent-function-path an OpenAI-compatible `base_url` (a session-
server proxy that supports full `tools=`/`tool_calls` passthrough -- see
tests/e2e/sglang/utils/session_tool_agent.py) pointed at the live rollout
engine; that custom agent function (../agent_function.py) POSTs here with
that base_url, and THIS sidecar builds tau2's own LLMAgent pointed at it via
litellm's standard `api_base` custom-endpoint routing, runs the full
orchestrator+evaluator, and returns the resulting reward.

Why upstream tau2-bench (not the dhh1995/tau2-bench@dhh/async-and-custom-
completion fork AReaL's own training example depends on): confirmed via
reading tau2/utils/llm_utils.py that `LLMAgent(llm="openai/<name>",
llm_args={"api_base": ..., "api_key": ...})` reaches litellm's
`completion(model=model, messages=..., tools=..., **llm_args)` call with
zero interception -- litellm's own `api_base` kwarg is a standard way to
route ANY `openai/*` model string to an arbitrary self-hosted OpenAI-
compatible server. And `Orchestrator.run()` is fully synchronous (no
internal asyncio) -- confirmed via grep, zero matches for
async/await/asyncio in orchestrator.py -- so wrapping it in
``asyncio.to_thread`` is a safe, dependency-free way to run it from this
async FastAPI server without needing the fork's own async support. This
avoids depending on an unofficial, unmaintained fork branch for a capability
upstream already exposes through its own public extension points.

Per-task DB variant loading: AReaL-tau2-data ships ONE db.json/db.toml
VARIANT FILE PER TASK (not a shared default), because AReaL's own SEA data-
synthesis engine generates a fresh, consistent world state per task. Each
domain's ``get_environment(db=..., solo_mode=False)`` accepts a live DB
object to override the domain's own bundled default -- confirmed via reading
tau2/domains/{retail,airline,telecom}/environment.py. Critically,
``evaluate_simulation``'s ``env_kwargs`` parameter passes through to BOTH the
agent-replay AND the "gold" replay environment used for grading (confirmed
via tau2/evaluator/evaluator_env.py: both call
``environment_constructor(**env_kwargs)``) -- so ``env_kwargs={"db": ...}``
must be set to the SAME db variant the orchestrator ran against, or grading
silently replays against the WRONG starting state. This is a real gap in
AReaL's OWN training example (confirmed via reading their examples/tau2/
agent.py: it never passes `db`/`env_kwargs` at all, always using the domain's
default DB even though their own published dataset ships per-task variants)
-- this sidecar fixes it by reloading the SAME db_path fresh (a SEPARATE
load, not the same mutated live object -- tool calls mutate `db` in place
during orchestration, so reusing that exact object would make the "gold"
replay start from the wrong, already-mutated state) for the evaluator call.

Contract (see ../agent_function.py for the caller side):
    POST /run {
        "base_url": str,      # OpenAI-compatible endpoint the AGENT's LLM calls go to
        "api_key": str,       # forwarded as the agent's litellm api_key (session-server accepts any value)
        "model": str,         # model name the agent's litellm calls use (informational; api_base does the real routing)
        "domain": "retail" | "airline" | "telecom",
        "task": dict,         # a tau2 Task, as a plain dict (Task.model_validate-able; evaluation_criteria as a JSON STRING, not pre-parsed)
        "db_path": str,       # path (inside this container) to the task's db variant file
        "max_steps": int,     # default 100
        "user_llm": str,      # litellm model string for the user simulator, e.g. "gemini/gemini-2.5-flash-lite"
        "user_llm_args": dict,  # e.g. {"api_key": "..."} -- forwarded verbatim to UserSimulator's llm_args
    } -> {
        "reward": float,          # 0.0-1.0 (occasionally partial credit -- db-hash equality is 0/1 for retail/airline, but telecom's reward_breakdown can be fractional)
        "messages": list[dict],   # SimulationRun.messages, model_dump()'d
        "termination_reason": str,
        "task_id": str,
    }
    GET /health -> {"status": "ok"}
"""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import httpx
import litellm
from fastapi import FastAPI
from pydantic import BaseModel

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_environment as get_airline_environment
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import get_environment as get_retail_environment
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.environment import get_environment as get_telecom_environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import UserSimulator

# tau2.utils.llm_utils (imported transitively by every tau2.* import above)
# sets litellm.client_session/aclient_session to a SINGLE process-wide
# httpx.Client/AsyncClient at ITS OWN import time with max_connections=10,
# max_keepalive_connections=5 (upstream's own hardcoded default -- sized for
# a single-conversation CLI/notebook, not concurrent RL training). litellm's
# OpenAIChatCompletion._get_sync_http_client() checks `if litellm.
# client_session is not None: return litellm.client_session` BEFORE ever
# consulting api_base/cache-key logic, so this ONE object is reused for
# every agent AND user-simulator call across EVERY episode, regardless of
# each episode's distinct api_base -- confirmed live via py-spy thread dump
# against a real training run: 116 threads across ~85 concurrent episodes
# all blocked in httpcore's wait_for_connection on the exact same
# <ConnectionPool> object. This was the actual ceiling behind "GPU
# utilization sits near-0 with only occasional single-GPU spikes" --
# raising the to_thread executor to 256 (below) did nothing for THIS
# bottleneck on its own, since the episodes it freed to run concurrently all
# immediately queued back up on this global 10-connection cap. Must
# overwrite AFTER the tau2 imports above (which is when llm_utils.py's
# module-level assignment actually runs) or tau2's own import silently wins
# and this override never takes effect.
os.environ.setdefault("TAU2_HTTPX_MAX_CONNECTIONS", "256")
_httpx_limits = httpx.Limits(
    max_connections=int(os.environ["TAU2_HTTPX_MAX_CONNECTIONS"]),
    max_keepalive_connections=int(os.environ["TAU2_HTTPX_MAX_CONNECTIONS"]),
)
litellm.client_session = httpx.Client(limits=_httpx_limits)
litellm.aclient_session = httpx.AsyncClient(limits=_httpx_limits)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("tau2_sidecar")

app = FastAPI()

# asyncio.to_thread's own default executor caps at min(32, cpu_count()+4)
# (concurrent.futures.ThreadPoolExecutor's own hardcoded default) REGARDLESS
# of how many cores the container actually has -- confirmed live via py-spy
# thread dump against a real training run: exactly 31-32 threads sitting
# inside Orchestrator.run() at once, everything past that queued with zero
# thread to run on, while host CPU sat ~88% idle (each episode's real work is
# network-bound: waiting on OUR OWN sglang engine + the external user-
# simulator gateway, not local computation -- confirmed via per-message
# timestamp deltas in a real dump: environment.get_response() tool-execution
# steps take ~50ms, but tool->assistant/user->assistant turns average 7+s).
# That silent 32-episode ceiling meant most of an 8-GPU node's sglang engines
# had nothing queued to run at any given moment, regardless of how large
# --sglang-server-concurrency was set on the training side. 256 is a plain
# thread-count increase, not a resource ask -- these threads spend nearly all
# their time blocked on I/O, so raising the cap costs ~nothing on either CPU
# or memory (each idle thread's stack is a few KB, not competing for cores).
_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.environ.get("TAU2_SIDECAR_MAX_WORKERS", "256")), thread_name_prefix="tau2-run"
)


async def to_thread(func, /, *args, **kwargs):
    """``asyncio.to_thread`` against ``_EXECUTOR`` instead of the loop's
    tiny (cpu_count-capped-at-32) default -- see ``_EXECUTOR``'s comment."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXECUTOR, lambda: func(*args, **kwargs))

# One (DB loader, env constructor) pair per domain. Loading the DB is kept
# separate from constructing the Environment so the SAME loaded db object can
# be reused for both `get_environment(db=...)` (orchestration) and
# `evaluate_simulation(..., env_kwargs={"db": ...})` (grading) without
# building a throwaway second Environment just to reach back into its
# `.tools.db` -- a plain reload of the db file is both cheaper and clearer.
_DB_LOADERS = {"retail": RetailDB.load, "airline": FlightDB.load, "telecom": TelecomDB.load}
_ENV_BUILDERS = {
    "retail": get_retail_environment,
    "airline": get_airline_environment,
    # AReaL-tau2-data's tau2_telecom_db.toml only carries TelecomDB fields
    # (confirmed: parses to {plans, customers, lines, bills, devices}, no
    # user-device/surroundings fields) -- telecom's SEPARATE user_db (mock
    # phone state) is not part of the per-task variant, so it keeps the
    # domain's own bundled default there.
    "telecom": get_telecom_environment,
}


def _load_db(domain: str, db_path: str):
    loader = _DB_LOADERS.get(domain)
    if loader is None:
        raise ValueError(f"unknown tau2 domain '{domain}' (expected retail|airline|telecom)")
    return loader(db_path)


def _build_environment(domain: str, db):
    return _ENV_BUILDERS[domain](db=db)


class RunRequest(BaseModel):
    base_url: str
    api_key: str = "dummy"
    model: str = "policy"
    domain: str
    task: dict
    db_path: str
    max_steps: int = 100
    user_llm: str = "gemini/gemini-2.5-flash-lite"
    user_llm_args: dict = {}
    # Extra litellm kwargs merged into the agent's own llm_args (api_base/
    # api_key are always set from base_url/api_key above; this is for extras
    # a specific backend needs, e.g. reasoning_effort="none" for gpt-5* family
    # models when function tools are in play -- irrelevant for the real
    # training path, which points base_url at the session-server proxy in
    # front of OUR OWN sglang-served policy, not a hosted gateway).
    agent_llm_args: dict = {}


class RunResponse(BaseModel):
    reward: float
    messages: list[dict]
    termination_reason: str
    task_id: str
    # Per-RewardType score breakdown (DB/ENV_ASSERTION/ACTION/COMMUNICATE, +
    # NL_ASSERTION for telecom's NL-assertion checks) -- evaluate_simulation
    # multiplies these together into the single `reward` above, but each is
    # useful on its own (e.g. "did the agent take the right actions" vs "did
    # it communicate correctly" vs "is the DB in the right end state" fail
    # independently and look identical in the pooled reward alone). Keys are
    # RewardType enum VALUES (plain strings, e.g. "DB"/"ACTION"), not the
    # enum objects themselves (Pydantic/JSON can't serialize those directly).
    reward_breakdown: dict[str, float] = {}


def _run_task_sync(req: RunRequest) -> RunResponse:
    task_dict = deepcopy(req.task)
    # evaluation_criteria arrives as a JSON-encoded STRING in AReaL-tau2-data
    # (confirmed live: `type(row["evaluation_criteria"])` is `str`), but
    # Task.evaluation_criteria's field type is the nested EvaluationCriteria
    # model -- model_validate needs the parsed dict, not the raw string.
    ec = task_dict.get("evaluation_criteria")
    if isinstance(ec, str):
        task_dict["evaluation_criteria"] = json.loads(ec)
    task = Task.model_validate(task_dict)

    environment = _build_environment(req.domain, _load_db(req.domain, req.db_path))

    agent = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=f"openai/{req.model}",
        llm_args={
            "api_base": f"{req.base_url.rstrip('/')}/v1",
            "api_key": req.api_key,
            **req.agent_llm_args,
        },
    )
    # retail/airline have no user-side tools at all -- Environment.get_user_
    # tools() raises ValueError("User tools not available") rather than
    # returning [] (confirmed live) -- only telecom's phone-troubleshooting
    # domain has any. tau2's OWN canonical wiring (tau2/runner/build.py::
    # build_user) wraps this in exactly this try/except, falling back to
    # None -- mirror that rather than hand-rolling a domain allowlist.
    try:
        user_tools = environment.get_user_tools(include=task.user_tools) or None
    except ValueError:
        user_tools = None
    user = UserSimulator(
        llm=req.user_llm,
        instructions=str(task.user_scenario),
        tools=user_tools,
        llm_args=req.user_llm_args,
    )
    orchestrator = Orchestrator(
        domain=req.domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=req.max_steps,
    )
    simulation = orchestrator.run()

    # Grade against a FRESH load of the SAME db variant -- see module
    # docstring's "AReaL's own gap" note. Tool calls mutate `environment`'s
    # db object in place during orchestration; the gold-replay environment
    # evaluate_simulation builds internally must start from the ORIGINAL
    # (unmutated) state, not whatever the agent left it in.
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=req.domain,
        env_kwargs={"db": _load_db(req.domain, req.db_path)},
    )
    simulation.reward_info = reward_info

    reward_breakdown = {
        rt.value: float(v) for rt, v in (reward_info.reward_breakdown or {}).items()
    }

    return RunResponse(
        reward=float(reward_info.reward),
        # exclude={"raw_data"}: AssistantMessage/UserMessage both carry the
        # COMPLETE raw litellm/OpenAI API response object in this field
        # (ToolMessage has no such field -- pydantic's exclude is a no-op on
        # a model that lacks the key, confirmed live). Confirmed live against
        # a real training dump: one 31-message episode's raw_data fields
        # alone totaled ~8MB (vs ~130KB of actual message content), and the
        # cumulative agentic_traces dump for one training run reached 13GB
        # uncompressed -- pure serialization/disk overhead with zero
        # training value (nothing downstream reads raw_data: not the reward,
        # not the dashboard, not _sample_to_agentic_trace_record).
        messages=[m.model_dump(mode="json", exclude={"raw_data"}) for m in (simulation.messages or [])],
        termination_reason=str(simulation.termination_reason),
        task_id=task.id,
        reward_breakdown=reward_breakdown,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest) -> RunResponse:
    try:
        return await to_thread(_run_task_sync, req)
    except Exception as e:  # a run-usage error must surface as a 0-reward result, not a 500
        logger.warning("task=%s domain=%s failed: %r", req.task.get("id"), req.domain, e, exc_info=True)
        return RunResponse(reward=0.0, messages=[], termination_reason=f"error: {e}", task_id=req.task.get("id", ""))
