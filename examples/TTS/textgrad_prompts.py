"""Prompts for the TextGrad-style CODE training loop (textgrad_train.py).

Two DECOUPLED optimizable variables (per arxiv.org/abs/2406.07496's "backward
through text" idea -- a critic reads a trace + outcome and rewrites the text
that produced it, no gradients):

  1. ``SysPromptBundle`` -- the solver's full instruction block: the system
     prompt (solving guide) + the code_interpreter tool's description text.
     This governs the NO-skill AND WITH-skill trajectories alike (it is always
     in context); it is critiqued from WITH-skill trajectories, since that is
     the harness's actual deployed configuration.
  2. ``SkillPrompts`` (reused from prompts.py) -- the [Knowledge/Rule] and
     [Error/Rule/Example] skill-PREDICTION prompt pair (predict from the bare
     problem, no traces/answer -- same deployment-aligned protocol as
     examples/TTS/prompts.py's predict_knowledge_user/predict_pitfall_user,
     reused verbatim here). Critiqued from NO-skill trajectories + the skill
     that was predicted for that problem: does the skill target what the
     no-skill trace shows the solver actually gets wrong?

Each round proposes BOTH variables independently and CONCURRENTLY -- neither
critic sees the other's output -- so the two updates are decoupled, per the
task's explicit requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Variable 1: the solver's system prompt + tool description bundle
# --------------------------------------------------------------------------- #

SEED_SYSTEM_PROMPT = (
    "You solve the user's coding problem step by step. You have a code_interpreter "
    "tool available; call it as many times as you need, across as many turns as you "
    "need. Do NOT submit a solution you have not run: before giving your final answer, "
    "you MUST call code_interpreter with your full program and the sample stdin to "
    "VERIFY it produces the expected output. Only after the tool confirms your program "
    "works, give your final answer inside <answer> and </answer> tags -- put the "
    "ENTIRE solution program inside the tags, nothing else.\n\n"
    "Grading re-runs the LAST program you pass to code_interpreter (reading input from "
    "stdin, printing to stdout) against hidden test cases -- pass your program's code "
    "DIRECTLY as the `code` argument and the sample input via the `stdin` argument. "
    "Never wrap your program in your own verification harness (e.g. writing it to a "
    "temp file and invoking subprocess.run, or hardcoding a fake stdin/input() override "
    "inside the code): the same `code` you pass is re-run verbatim against the REAL "
    "hidden input at grading time, and any such wrapper or hardcoded input would "
    "replace it, making a logically correct program grade as wrong.\n\n"
    "Example: to solve 'read two integers a and b from stdin and print their sum', "
    "verify with code_interpreter(code=\"a, b = map(int, input().split())\\nprint(a + b)\", "
    "stdin=\"3 5\") -- NOT by wrapping it in subprocess/tempfile machinery."
)

SEED_TOOL_DESCRIPTION = (
    "Execute Python code in an isolated sandbox and return its stdout. Use this to "
    "run and verify your candidate solution program against the given stdin before "
    "committing to a final answer. The sandbox has sympy, numpy, and scipy "
    "preinstalled; it has no network access."
)


@dataclass
class SysPromptBundle:
    """The optimizable system-prompt variable: solving guide + tool description.
    Both are freeform text (no JSON-schema constraints), so both are safe for a
    critic to rewrite without breaking tool-call validity."""

    system_prompt: str = SEED_SYSTEM_PROMPT
    tool_description: str = SEED_TOOL_DESCRIPTION

    @staticmethod
    def seed() -> "SysPromptBundle":
        return SysPromptBundle()


SYS_PROMPT_CRITIC_SYSTEM = (
    "You are a PROMPT CRITIC in a TextGrad-style optimization loop (see "
    "arxiv.org/abs/2406.07496): you improve a CODING agent's SYSTEM PROMPT and "
    "its code_interpreter TOOL DESCRIPTION from observed multi-turn "
    "trajectories -- textual feedback standing in for a gradient.\n\n"
    "How the agent works: given the system prompt + tool description + a "
    "problem, it may call code_interpreter (Python, isolated sandbox) any "
    "number of times across turns, then must give a final answer inside "
    "<answer></answer> tags containing its full solution PROGRAM. Grading is "
    "TOOL-MANDATORY: only the code from the LAST code_interpreter call is "
    "graded (run against hidden tests) -- the <answer> text itself is never "
    "executed, so the prompt must make the agent actually RUN its final "
    "program through the tool before finishing, not just paste it.\n\n"
    "You are given the CURRENT system prompt, the CURRENT tool description, and "
    "a batch of trajectories -- each with the problem, a condensed transcript of "
    "the agent's turns/tool calls/tool results (the skill hint injected into "
    "context, if any, is included so you can see what the agent had available), "
    "and whether the trajectory was graded CORRECT. Diagnose recurring failure "
    "patterns (never verifying before answering? verifying with the WRONG input "
    "and then submitting something else? too many/few turns? ignoring the skill "
    "hint? malformed <answer> so grading can't find the program? giving up after "
    "one tool error?) and rewrite BOTH texts to fix them.\n\n"
    "Rules for your rewrite:\n"
    "- The system prompt must still end with a clear final-answer FORMAT "
    "contract (grading depends on <answer></answer> containing the exact "
    "program) and must still make verifying-before-answering a hard "
    "requirement (grading depends on the tool actually having been run).\n"
    "- Do NOT hard-code facts about the specific problems in this batch; both "
    "texts must generalize to unseen coding problems.\n"
    "- Make concrete, targeted changes justified by the failures you observed; "
    "do not rewrite wholesale if only one aspect is broken.\n"
    "- Output EXACTLY this structure and nothing else:\n"
    "  DIAGNOSIS: <2-4 sentences on the recurring failure pattern(s)>\n"
    "  ===SYSTEM===\n"
    "  <the full text of the improved system prompt>\n"
    "  ===TOOL===\n"
    "  <the full text of the improved tool description>\n"
)


def sys_prompt_critic_user(
    system_prompt: str,
    tool_description: str,
    examples: list[dict],
    max_examples: int = 16,
    max_transcript_chars: int = 2500,
) -> str:
    """``examples``: [{"problem","transcript","correct","n_turns"}, ...] from
    WITH-skill trajectories (the agent's real deployed configuration). Wrong
    trajectories are prioritized -- they carry the optimization signal."""

    def _clip(s: str, n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[:n] + " ...[clipped]"

    wrong = [e for e in examples if not e.get("correct")]
    right = [e for e in examples if e.get("correct")]
    chosen = wrong[:max_examples]
    if len(chosen) < max_examples:
        chosen += right[: max_examples - len(chosen)]

    lines = [
        "CURRENT SYSTEM PROMPT:\n<<<\n" + system_prompt.strip() + "\n>>>\n\n",
        "CURRENT TOOL DESCRIPTION:\n<<<\n" + tool_description.strip() + "\n>>>\n\n",
        f"WITH-SKILL TRAJECTORIES THIS ROUND ({len(right)} correct, {len(wrong)} incorrect; "
        f"sample below prioritizes INCORRECT ones):\n",
    ]
    for i, e in enumerate(chosen):
        lines.append(
            f"\n[{i + 1}] correct={e.get('correct')}  n_turns={e.get('n_turns')}\n"
            f"PROBLEM: {_clip(e.get('problem', ''), 400)}\n"
            f"TRANSCRIPT:\n{_clip(e.get('transcript', ''), max_transcript_chars)}\n"
        )
    lines.append(
        "\n\nNow diagnose and rewrite BOTH texts per your instructions. Remember the "
        "EXACT output structure: DIAGNOSIS, then ===SYSTEM===, then ===TOOL==="
    )
    return "".join(lines)


def parse_sys_prompt_critic_output(text: str, fallback: SysPromptBundle) -> tuple[str, SysPromptBundle]:
    """Split into (diagnosis, SysPromptBundle). Contract: 'DIAGNOSIS: ...\\n
    ===SYSTEM===\\n<system>\\n===TOOL===\\n<tool>'. Each section falls back to
    the current text if missing/implausibly short."""
    text = (text or "").strip()
    if not text:
        return "", fallback

    def _section(tag: str) -> str | None:
        marker = f"==={tag}==="
        idx = text.find(marker)
        if idx == -1:
            return None
        start = idx + len(marker)
        rest = text[start:]
        nxt = None
        for other in ("===SYSTEM===", "===TOOL==="):
            if other == marker:
                continue
            j = rest.find(other)
            if j != -1:
                nxt = j if nxt is None else min(nxt, j)
        body = rest[:nxt] if nxt is not None else rest
        return body.strip()

    diagnosis = ""
    head = text.split("===SYSTEM===", 1)[0]
    if "DIAGNOSIS" in head:
        diagnosis = head.split("DIAGNOSIS:", 1)[-1].strip() if "DIAGNOSIS:" in head else head.strip()

    system_p = _section("SYSTEM")
    tool_p = _section("TOOL")
    new = SysPromptBundle(
        system_prompt=system_p if (system_p and len(system_p) >= 40) else fallback.system_prompt,
        tool_description=tool_p if (tool_p and len(tool_p) >= 20) else fallback.tool_description,
    )
    return diagnosis, new


# --------------------------------------------------------------------------- #
# Variable 2: the skill-PREDICTION prompt pair (reused protocol from
# examples/TTS/prompts.py's predict_knowledge_user/predict_pitfall_user -- see
# that module for why "predict from the bare problem" is the deployment-aligned
# protocol, not distillation from traces/answers).
# --------------------------------------------------------------------------- #

SKILL_CRITIC_SYSTEM = (
    "You are a PROMPT CRITIC in a TextGrad-style optimization loop (see "
    "arxiv.org/abs/2406.07496): you improve TWO 'skill-prediction' prompts used "
    "by a coding agent -- one predicts [Knowledge/Rule] blocks, the other "
    "[Error/Rule/Example] pitfall blocks -- both PREDICTED from the bare "
    "problem statement alone (no solution, no traces, no ground truth; this "
    "is a deployment constraint, not a choice) and prepended to the problem "
    "before the agent (with tools) attempts it.\n\n"
    "You are given the CURRENT knowledge prompt, the CURRENT pitfall prompt, "
    "and a batch of diagnostic records. For each problem: the skill that was "
    "predicted this round, and -- critically -- a condensed transcript of the "
    "agent's trajectory WITHOUT any skill (its raw, unaided attempt) plus "
    "whether that unaided attempt was graded correct. Use the unaided "
    "trajectory as evidence of what the agent actually struggles with on this "
    "kind of problem (a wrong turn it takes, a case it misses, a verification "
    "it skips), then judge whether the CURRENT predicted skill would actually "
    "address that gap. If the skill is generic, misses the real difficulty, or "
    "states something not safely inferable from the bare problem (hallucinated "
    "or answer-leaking), diagnose why and rewrite BOTH prompts.\n\n"
    "Rules for your rewrite:\n"
    "- Both prompts must still output ONLY the structured skill blocks "
    "([Knowledge/Rule]/[Details/Examples] or [Error]/[Rule]/[Example]) and "
    "NEVER state or imply a final answer/program -- the model does not know "
    "it, and a wrong guess actively hurts the agent that trusts it.\n"
    "- Keep them GENERAL skill-PREDICTION instructions: do NOT hard-code facts "
    "about the specific problems in this batch; they must generalize.\n"
    "- Prioritize predicting what the unaided trajectories show is actually "
    "missing (a specific edge case, an off-by-one pattern, a data-structure "
    "choice, an I/O parsing subtlety) over generic textbook advice.\n"
    "- Output EXACTLY this structure and nothing else:\n"
    "  DIAGNOSIS: <2-4 sentences on what's wrong with the current prompts>\n"
    "  ===CORRECT===\n"
    "  <the full text of the improved knowledge-prediction prompt>\n"
    "  ===PITFALL===\n"
    "  <the full text of the improved pitfall-prediction prompt>\n"
)


def skill_critic_user(
    correct_prompt: str,
    pitfall_prompt: str,
    examples: list[dict],
    max_examples: int = 16,
    max_transcript_chars: int = 2000,
    max_skill_chars: int = 900,
) -> str:
    """``examples``: [{"problem","skill","no_skill_transcript","no_skill_correct",
    "with_skill_correct"}, ...]. Problems where the unaided (no-skill) attempt
    FAILED are prioritized -- they show the real gap the skill should address."""

    def _clip(s: str, n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[:n] + " ...[clipped]"

    struggled = [e for e in examples if not e.get("no_skill_correct")]
    solved = [e for e in examples if e.get("no_skill_correct")]
    chosen = struggled[:max_examples]
    if len(chosen) < max_examples:
        chosen += solved[: max_examples - len(chosen)]

    lines = [
        "CURRENT KNOWLEDGE-PREDICTION PROMPT:\n<<<\n" + correct_prompt.strip() + "\n>>>\n\n",
        "CURRENT PITFALL-PREDICTION PROMPT:\n<<<\n" + pitfall_prompt.strip() + "\n>>>\n\n",
        f"DIAGNOSTIC RECORDS THIS ROUND ({len(struggled)} where the unaided agent failed, "
        f"{len(solved)} where it already succeeded unaided; sample below prioritizes FAILURES):\n",
    ]
    for i, e in enumerate(chosen):
        lines.append(
            f"\n[{i + 1}] no_skill_correct={e.get('no_skill_correct')}  with_skill_correct={e.get('with_skill_correct')}\n"
            f"PROBLEM: {_clip(e.get('problem', ''), 400)}\n"
            f"UNAIDED (no-skill) TRANSCRIPT:\n{_clip(e.get('no_skill_transcript', ''), max_transcript_chars)}\n"
            f"PREDICTED SKILL THIS ROUND:\n{_clip(e.get('skill', ''), max_skill_chars)}\n"
        )
    lines.append(
        "\n\nNow diagnose and rewrite BOTH prompts per your instructions. Remember the "
        "EXACT output structure: DIAGNOSIS, then ===CORRECT===, then ===PITFALL==="
    )
    return "".join(lines)
