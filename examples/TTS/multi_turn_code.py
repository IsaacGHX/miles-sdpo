"""Multi-turn native tool-calling rollout for CODE tasks, driven by an
Engine.chat with OpenAI-style ``tools=`` (see engines.py) instead of miles'
SGLang-native ``/generate`` + FunctionCallParser path
(miles/rollout/generate_hub/multi_turn.py). Reused unchanged from
examples/SDPO_ReAct/: the ``code_interpreter`` tool's HTTP client
(tools/code/client.py -> the SAME already-running sandbox sidecar at
127.0.0.1:8420, no new container), and the tool-mandatory code grader
(examples/SDPO/reward.py::_grade_one_code via grading.grade_code).

Only ``code_interpreter`` is wired (the code domain's one required tool);
search/cli are not used here.

Loop, mirroring multi_turn.generate's shape:
    messages = [{"role":"system","content":sys_prompt}, {"role":"user","content":problem}]
    for turn in range(max_turns):
        result = await solver.chat(messages, tools=[CODE_INTERPRETER_SPEC])
        if no tool_calls: break  # final answer turn
        append assistant turn (with tool_calls)
        run each tool call concurrently -> append one {"role":"tool",...} turn each
    tool_trace = [{"tool_call": "code_interpreter(<json args>)", "observation": str}, ...]
    final_text = last assistant text (possibly "" if it never produced one)

Grading is TOOL-MANDATORY (examples/SDPO/reward.py's contract): the graded
candidate is the LAST code argument passed to code_interpreter in tool_trace,
never the final text's code fence -- a trajectory that never verifies its
solution via the tool is wrong regardless of what it claims.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from examples.SDPO_ReAct.sdpo_react import _TOOL_ERROR_PREFIXES
from examples.SDPO_ReAct.tools.code.client import run_code
from examples.TTS.engines import Engine

logger = logging.getLogger(__name__)

DEFAULT_TOOL_DESCRIPTION = (
    "Execute Python code in an isolated sandbox and return its stdout. "
    "Use this for calculations, symbolic math (sympy), or verifying a "
    "numeric answer before giving your final answer. The sandbox has "
    "sympy, numpy, and scipy preinstalled; it has no network access."
)


def code_interpreter_spec(tool_description: str = DEFAULT_TOOL_DESCRIPTION) -> dict:
    """The code_interpreter tool spec, with an overridable description -- the
    description is one half of the TextGrad-optimized SysPromptBundle
    (textgrad_prompts.py), so it must be swappable per round/critic step."""
    return {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Raw Python source code to execute (not wrapped in a markdown fence).",
                    },
                    "stdin": {
                        "type": "string",
                        "description": (
                            "Optional text piped to the program's real stdin. For a program that will be "
                            "GRADED, the LAST code_interpreter call's `code` is re-run verbatim against the "
                            "real hidden test input -- do not hardcode a stdin override in the source."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    }


CODE_INTERPRETER_SPEC = code_interpreter_spec()  # default spec, back-compat for direct callers


@dataclass
class Trajectory:
    """One multi-turn rollout: the full message transcript + the tool_trace
    grading needs + whether it was graded correct.

    ``completion_tokens``/``prompt_tokens`` and ``hit_max_turns``/``any_truncated``
    are the token/turn-budget diagnostics this harness can report in place of
    miles' GPU-side response_len/multi_turn panels (see textgrad_metrics.py) --
    populated from each turn's ChatResult.usage / finish_reason, the closest
    equivalent an external chat-completion API exposes."""

    problem: str
    messages: list[dict] = field(default_factory=list)  # full transcript incl. system
    tool_trace: list[dict] = field(default_factory=list)  # [{"tool_call","observation"}]
    final_text: str = ""
    n_turns: int = 0
    correct: bool = False
    completion_tokens: int = 0  # sum of assistant-turn completion_tokens across the trajectory
    prompt_tokens: int = 0  # last turn's prompt_tokens (running context size)
    any_truncated: bool = False  # any turn's finish_reason == "length" (max_tokens hit)
    hit_max_turns: bool = False  # loop exhausted max_turns without a final answer turn
    n_tool_calls: int = 0
    n_tool_errors: int = 0

    def transcript_text(self, max_chars: int = 6000) -> str:
        """A compact text rendering of the trajectory for a critic prompt:
        each assistant/tool turn, clipped. System/user turns are omitted (the
        critic already gets the system prompt and problem separately)."""
        parts = []
        for m in self.messages:
            role = m.get("role")
            if role == "system":
                continue
            if role == "user":
                continue  # the problem is passed separately to the critic
            if role == "assistant":
                text = m.get("content") or ""
                calls = m.get("tool_calls") or []
                call_str = ""
                if calls:
                    names = [c.get("function", {}).get("name", "?") for c in calls]
                    call_str = f" [called: {', '.join(names)}]"
                if text or call_str:
                    parts.append(f"ASSISTANT:{call_str} {text}".strip())
            elif role == "tool":
                parts.append(f"TOOL RESULT: {_clip(m.get('content') or '', 800)}")
        text = "\n".join(parts)
        return _clip(text, max_chars)


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + " ...[clipped]"


async def _execute_tool_call(call: dict) -> str:
    """Run one OpenAI-style tool call dict -> observation string. Only
    code_interpreter is wired; anything else returns an error observation
    (never raises, so one bad call can't crash a trajectory)."""
    fn = call.get("function", {}) or {}
    name = fn.get("name", "")
    raw_args = fn.get("arguments", "") or "{}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except json.JSONDecodeError:
        args = {"code": raw_args}
    if name != "code_interpreter":
        return f"error:\nunknown tool '{name}'"
    return await run_code(args.get("code", ""), stdin=args.get("stdin"))


async def run_trajectory(
    solver: Engine,
    system_prompt: str,
    problem: str,
    *,
    skill: str = "",
    tool_description: str = DEFAULT_TOOL_DESCRIPTION,
    max_turns: int = 8,
    max_tokens: int = 4096,
    temperature: float = 1.0,
) -> Trajectory:
    """Run ONE multi-turn code trajectory to completion (or max_turns). If
    ``skill`` is non-empty, it is prepended to the user turn (same splice as
    prompts.py::skill_augmented_user) -- the WITH-skill arm."""
    user_content = f"{skill}\n\n---\n\nNow solve this problem:\n\n{problem}" if skill.strip() else problem
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    traj = Trajectory(problem=problem, messages=messages)
    tool_spec = code_interpreter_spec(tool_description)

    for _turn in range(max_turns):
        try:
            res = await solver.chat(
                messages, tools=[tool_spec], temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            logger.warning(f"solver.chat failed mid-trajectory: {e!r}")
            break
        traj.n_turns += 1
        traj.completion_tokens += res.completion_tokens or 0
        traj.prompt_tokens = res.prompt_tokens or traj.prompt_tokens
        if res.finish_reason == "length":
            traj.any_truncated = True
        assistant_msg: dict = {"role": "assistant", "content": res.text or None}
        if res.tool_calls:
            assistant_msg["tool_calls"] = res.tool_calls
        messages.append(assistant_msg)

        if not res.tool_calls:
            traj.final_text = res.text or ""
            break

        async def _run_one(call: dict) -> tuple[dict, str]:
            obs = await _execute_tool_call(call)
            return call, obs

        results = await asyncio.gather(*(_run_one(c) for c in res.tool_calls))
        for call, obs in results:
            fn = call.get("function", {}) or {}
            traj.n_tool_calls += 1
            if obs.strip().lower().startswith(_TOOL_ERROR_PREFIXES):
                traj.n_tool_errors += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": fn.get("name", ""),
                    "content": obs,
                }
            )
            traj.tool_trace.append({"tool_call": f"{fn.get('name', '')}({fn.get('arguments', '')})", "observation": obs})
    else:
        # loop exhausted max_turns without a final (tool-call-free) turn
        traj.final_text = ""
        traj.hit_max_turns = True

    return traj


async def run_trajectory_and_grade(
    solver: Engine,
    system_prompt: str,
    problem: str,
    test_cases: list[dict],
    *,
    skill: str = "",
    tool_description: str = DEFAULT_TOOL_DESCRIPTION,
    max_turns: int = 8,
    max_tokens: int = 4096,
    temperature: float = 1.0,
) -> Trajectory:
    """run_trajectory + grade (tool-mandatory: the LAST code_interpreter call's
    code argument, against test_cases)."""
    from examples.TTS.grading import grade_code

    traj = await run_trajectory(
        solver, system_prompt, problem, skill=skill, tool_description=tool_description,
        max_turns=max_turns, max_tokens=max_tokens, temperature=temperature,
    )
    try:
        traj.correct = await grade_code(traj.tool_trace, test_cases)
    except Exception as e:
        logger.warning(f"grade_code failed: {e!r}")
        traj.correct = False
    return traj
