"""Custom agent function for ``miles.rollout.generate_hub.agentic_tool_call.
generate`` (``--custom-agent-function-path``, requires ``--use-session-
server``) -- dispatches ONE τ²-bench task to the tau2 sidecar
(docker/server.py, started by run_tau2_sidecar.sh) and returns its reward as
plain sample metadata.

Mirrors examples/experimental/swe-agent-v2/swe_agent_function.py's contract
exactly (same run(base_url, prompt, request_kwargs, metadata, **kwargs) ->
dict | None signature -- agentic_tool_call.generate calls this after
constructing an OpenAI-endpoint-tracing session, and merges whatever dict
this returns into every resulting Sample's metadata). Task-type analogue of
that module, but talking to the tau2 sidecar's /run endpoint instead of a
Harbor agent server.

WHY tau2 needs this path at all (not multi_turn.generate, like webshop/
alfworld): τ²-bench's own Orchestrator class drives the ENTIRE agent<->user-
simulator<->environment conversation itself -- there is no seam for this
repo's own <tool_call>-parsing loop to plug into. agentic_tool_call.generate
hands this function a `base_url` pointed at a session-server proxy that
fully supports OpenAI-native `tools=`/`tool_calls` passthrough (confirmed via
tests/e2e/sglang/utils/session_tool_agent.py) -- the tau2 sidecar's LLMAgent
then talks to THAT via litellm's own `api_base` custom-endpoint routing (see
docker/server.py's module docstring), so it never touches this repo's tool-
call parser at all.

Task assignment: UNLIKE webshop/alfworld (which pin a task id/game_file
that's forwarded on EVERY tool call within a trajectory, via
current_trajectory_metadata()), a tau2 episode makes exactly ONE outbound
call total (this function, once, at the start of generate()) -- so the full
task dict is passed directly in this one request body, no contextvar needed.
data/build_tau2_data.py stamps metadata["tau2_task"]/["tau2_domain"]/
["tau2_db_path"] onto every row for this to read.
"""

import asyncio
import logging
import os
from typing import Any

from miles.utils.http_utils import post

logger = logging.getLogger(__name__)

# os.environ.get's default only applies when the key is ABSENT -- the enroot
# launchers pass every optional override as `--env VAR="${VAR:-}"` (empty
# string, not unset, when the user didn't set it), so an `or` fallback is
# required here, not a dict .get default (confirmed live: an empty-string
# SDPO_REACT_TAU2_SIDECAR_URL collapsed this to "/run", a path with no host,
# which every httpx call then silently retried 60x with "Request URL is
# missing an 'http://' or 'https://' protocol").
TAU2_SIDECAR_URL = os.environ.get("SDPO_REACT_TAU2_SIDECAR_URL") or "http://127.0.0.1:8424"

# User-simulator model. Default: the Salesforce Research gateway's
# gpt-5.6-luna (SFT_GATEWAY_KEY/OPENAI_API_URL, loaded from ~/gitproj/apis/
# .env by enroot-run-sdpo-react.sh and forwarded to ray workers by the tau2
# run script -- SAME credentials run-qwen3-4B-sdpo-react-native.sh already
# uses for --sdpo-judge-model). This repo's own machine has no GEMINI_API_KEY/
# DEEPSEEK_API_KEY configured, so gemini/deepseek (examples/tau-bench's own
# TAU_USER_MODEL_PROVIDER convention) is NOT the default here -- override via
# TAU_USER_MODEL_PROVIDER=gemini|deepseek + the matching *_API_KEY if you
# have one and want to switch off the gateway.
_USER_MODEL_PROVIDER = os.environ.get("TAU_USER_MODEL_PROVIDER", "openai")
_USER_MODEL = os.environ.get("TAU_USER_MODEL", "gpt-5.6-luna")
_PROVIDER_KEY_ENV = {"gemini": "GEMINI_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "openai": "SFT_GATEWAY_KEY"}
# gpt-5* reasoning models reject `tools=` + a real reasoning_effort in one
# call ("Function tools with reasoning_effort are not supported ... set
# reasoning_effort to 'none'" -- confirmed live against this exact gateway
# during the port's own smoke test) -- litellm forwards unknown kwargs in
# llm_args straight into the request body, so this is the only extra needed.
#
# max_completion_tokens is ALSO required, not optional: tau2's own
# tau2/utils/llm_utils.py::generate() does `content =
# response_choice.message.content` with ZERO empty-content handling, and
# UserSimulator._generate_next_message builds `UserMessage(content=
# user_response, ...)` straight from that -- with no tool_calls (retail/
# airline's user has none), an empty content makes UserMessage.validate()
# raise ("UserMessage must have either content or tool_calls"), which
# aborts the whole episode at reward=0. Without an explicit budget, a
# reasoning model can spend the WHOLE default token allowance on <think>
# and leave nothing for the actual reply -- confirmed live: 2.5% of one
# ablation run's training episodes died on exactly this exception. A
# generous budget (user turns are short conversational replies, not
# multi-step reasoning) makes running out this way effectively impossible.
_GPT5_RESPONSE_KWARGS = {"reasoning_effort": "none", "max_completion_tokens": 2048}


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one τ²-bench task via the tau2 sidecar."""
    metadata = metadata or {}

    task = metadata.get("tau2_task")
    domain = metadata.get("tau2_domain")
    db_path = metadata.get("tau2_db_path")
    if task is None or domain is None or db_path is None:
        logger.error(
            "tau2 agent_function: sample.metadata missing tau2_task/tau2_domain/tau2_db_path "
            "(did the row come from data/build_tau2_data.py?)"
        )
        return None

    key_env = _PROVIDER_KEY_ENV.get(_USER_MODEL_PROVIDER, f"{_USER_MODEL_PROVIDER.upper()}_API_KEY")
    user_api_key = os.environ.get(key_env, "")
    if not user_api_key:
        logger.error(f"tau2 agent_function: {key_env} not set for user-simulator provider '{_USER_MODEL_PROVIDER}'")
        return None

    user_llm_args = {"api_key": user_api_key}
    if _USER_MODEL_PROVIDER == "openai":
        # Unlike gemini/deepseek (real litellm-recognized providers, api_key
        # alone is enough), "openai/<name>" must also be pointed at the
        # gateway's api_base -- litellm's own custom-endpoint routing, see
        # tools/tau2/docker/server.py's module docstring.
        gateway_url = os.environ.get("OPENAI_API_URL", "https://gateway.salesforceresearch.ai/openai/process/v1/")
        user_llm_args["api_base"] = gateway_url
    if _USER_MODEL.startswith("gpt-5"):
        user_llm_args.update(_GPT5_RESPONSE_KWARGS)

    # agent_llm_args stays empty here: the AGENT's model is always the
    # policy under training, served by our own sglang router through the
    # session-server proxy `base_url` -- never a gpt-5*/gateway model, so
    # the reasoning_effort fix above is user-simulator-only.
    request = {
        "base_url": base_url,
        "api_key": "dummy",  # session-server proxy does not check this
        "model": "policy",  # informational only -- api_base does the real routing
        "domain": domain,
        "task": task,
        "db_path": db_path,
        # tau2_max_steps: per-sample override (metadata_overrides in an eval
        # yaml, same override convention every other SDPO_ReAct domain's
        # generate_max_turns uses), else the run script's global
        # SDPO_REACT_TAU2_MAX_STEPS env var (forwarded to ray workers).
        "max_steps": int(metadata.get("tau2_max_steps", os.environ.get("SDPO_REACT_TAU2_MAX_STEPS", 30))),
        "user_llm": f"{_USER_MODEL_PROVIDER}/{_USER_MODEL}",
        "user_llm_args": user_llm_args,
    }

    try:
        response = await asyncio.wait_for(
            post(f"{TAU2_SIDECAR_URL}/run", request),
            timeout=1800,  # a full multi-turn conversation + user simulator calls can run long
        )
    except asyncio.TimeoutError:
        logger.error("tau2 sidecar call timed out after 1800s")
        return None
    except asyncio.CancelledError:
        logger.warning("tau2 sidecar call cancelled (sibling task failure?)")
        return None
    except Exception as e:
        logger.error(f"tau2 sidecar call failed: {e}")
        return None

    # Per-RewardType score breakdown (DB/ENV_ASSERTION/ACTION/COMMUNICATE, +
    # NL_ASSERTION on telecom) -- evaluate_simulation multiplies these into
    # the single scalar reward above, so a 0.0 doesn't say WHICH check(s)
    # failed. Stashed with a tau2_reward_ prefix per key (e.g.
    # tau2_reward_DB, tau2_reward_ACTION) so miles/ray/rollout/metrics.py's
    # per-domain eval aggregation (any metadata key -> mean) picks each up
    # as its own wandb series without any parsing on the metrics side.
    reward_breakdown = response.get("reward_breakdown") or {}
    breakdown_fields = {f"tau2_reward_{k}": float(v) for k, v in reward_breakdown.items()}
    messages = response.get("messages", [])

    return {
        # Constant "tau2" (not "tau2_retail" etc) for reward-dispatch routing
        # -- same pattern as alfworld, where metadata["domain"]=="alfworld"
        # stays constant across all 6 ALFRED task types and the specific
        # type lives in its OWN metadata["task_type"] field instead. Here
        # metadata["tau2_domain"] (stamped by the data builder, echoed back
        # by the sidecar) is that per-subdomain field.
        "reward": response.get("reward", 0.0),
        "domain": "tau2",
        "tau2_messages": messages,
        "tau2_termination_reason": response.get("termination_reason", ""),
        "tau2_task_id": response.get("task_id", ""),
        # examples/SDPO/sdpo.py's skill/pitfall generator reads
        # metadata["question"] (falling back to sample.prompt, the raw chat-
        # templated string, only when absent) to fill the "PROBLEM:" section
        # of the skill-distillation prompt. Without this key it fell back to
        # sample.prompt -- and _strip_chat_template there keeps only the
        # LAST <|im_start|>user...<|im_end|> block, which for tau2 is the
        # very FIRST real user turn (the whole conversation is one prompt,
        # tau2's own Orchestrator generates every later turn, so there is
        # only ever one user block in that raw string). Confirmed live: a
        # real trace's problem_text was "Hi, I need help changing the dates
        # on my reservation. My user ID is ..." -- the opening line only,
        # with the actual ask (which reservation, which new date, whether
        # other requests came up) still unresolved. Joining every REAL user
        # turn from tau2's own message log (the one that matters here, not
        # the raw prompt string) recovers what the user actually needed
        # across the whole conversation -- confirmed live on a real retail
        # trace: the opening line was just "I need to cancel my order for
        # the hiking boots", but by the end the user had also asked to swap
        # a T-shirt for sneakers on a SEPARATE order, which the opening line
        # gives no hint of at all.
        #
        # Each real user turn is labeled "round N:" rather than joined bare
        # -- multiple user turns run together with no separator read like
        # one continuous statement (e.g. "...the hiking boots.My email is
        # ..." with the turn boundary invisible), which obscures that later
        # turns are the user's RESPONSES to intervening agent/tool turns
        # (clarifications, corrections, new asks), not a single opening
        # statement continued.
        "question": "\n".join(
            f"round {i}: {m.get('content')}"
            for i, m in enumerate(
                (m for m in messages if m.get("role") == "user" and m.get("content")),
                start=1,
            )
        ),
        **breakdown_fields,
    }
