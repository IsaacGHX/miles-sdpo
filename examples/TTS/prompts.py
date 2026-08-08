"""Seed prompts for the TTS (test-time scaffolding) loop.

Skill-generation logic mirrors examples/SDPO exactly: ONE problem -> MULTIPLE
rollouts (a group) -> grade the group -> distill CORRECT KNOWLEDGE from the
correct traces AND PITFALLS from the incorrect traces -> combine into the skill.
So there are TWO layer-1 skill-gen prompts (the free variables the layer-2
optimizer rewrites each round):

  * CORRECT_SKILL_SYSTEM_SEED  -- distills [Knowledge/Rule]/[Details/Examples]
    from correct traces  (== SDPO's _SKILL_SYSTEM_PROMPT)
  * PITFALL_SKILL_SYSTEM_SEED  -- distills [Error]/[Rule]/[Example] pitfalls from
    failed traces        (== SDPO's _SKILL_SYSTEM_PROMPT_INCORRECT)

The skill injected into the solver is the CONCATENATION of the correct-knowledge
blocks and the pitfall blocks distilled from that problem's own group of
rollouts. This is the same signal SDPO builds (skill-source "all"): what the
correct peers knew + what the failed peers got wrong.

LAYER 2 (the meta-optimizer) reads the current prompt PAIR plus a sample of
(problem, generated-skill, group-accuracy) results and proposes an improved
PAIR. That is the "second-layer scaffold that captures the traces and generates
a new prompt" from the task; the only thing that changes across rounds is these
two prompt strings.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# LAYER 1 -- the two skill-generation prompts (the OPTIMIZED variables)
# --------------------------------------------------------------------------- #

# Correct-knowledge distiller. Verbatim from examples/SDPO/sdpo.py::
# _SKILL_SYSTEM_PROMPT.
CORRECT_SKILL_SYSTEM_SEED = (
    "You are given one or more CORRECT worked solutions to a problem. Distill the "
    "transferable KNOW-HOW they used into a list of tiny, self-contained SKILLS -- "
    "each a reusable knowledge/rule unit, NOT a step-by-step roadmap of this "
    "specific problem. Use this EXACT structured format, one block per distinct "
    "skill (1-3 blocks):\n\n"
    "[Knowledge/Rule]\n"
    "<a general principle / identity / method / theorem the solution relied on -- "
    "transferable, concrete, not vague>\n"
    "[Details/Examples]\n"
    "<a tiny concrete worked instance of the rule (small numbers / short snippet), "
    "NOT this problem's final answer>\n\n"
    "Good (specific):\n"
    "[Knowledge/Rule]\nExpanding a^2+b^2 keeps the cross term: a^2+b^2 = (a+b)^2 - 2ab.\n"
    "[Details/Examples]\nIf u+v=6 and uv=4 then u^2+v^2 = 36 - 8 = 28.\n\n"
    "Bad (vague / problem-specific roadmap, do NOT do this): 'First read the problem, "
    "then set up equations, then solve' / 'Be careful with algebra'.\n\n"
    "Hard constraints:\n"
    "- Use the literal [Knowledge/Rule]/[Details/Examples] headers for every block.\n"
    "- Each skill must be a TRANSFERABLE unit usable on OTHER problems, not a recipe "
    "specific to this one; the [Details/Examples] a self-contained mini instance.\n"
    "- Do NOT state this problem's final answer (no final letter/number/name, no "
    "'the answer is ...').\n"
    "- Output ONLY the [Knowledge/Rule]/[Details/Examples] blocks, nothing else."
)

# Pitfall distiller. Verbatim from examples/SDPO/sdpo.py::
# _SKILL_SYSTEM_PROMPT_INCORRECT.
PITFALL_SKILL_SYSTEM_SEED = (
    "You are given one or more FAILED attempts at a problem and the ground-truth "
    "answer. The attempts are WRONG. Do NOT solve the problem or write a correct "
    "solution -- you only learn from the failures. Extract the SPECIFIC mistake(s) "
    "as concrete, reusable lessons in this EXACT structured format (one block per "
    "distinct mistake, 1-3 blocks total):\n\n"
    "[Error]\n"
    "<the specific wrong step/assumption an attempt made -- concrete, not vague>\n"
    "[Rule]\n"
    "<the general principle/identity/method that would have avoided it>\n"
    "[Example]\n"
    "<a tiny concrete worked instance of the rule (small numbers / short snippet), "
    "NOT this problem's answer>\n\n"
    "Good (specific):\n"
    "[Error]\nDropped the coefficient 2 when expanding the identity.\n"
    "[Rule]\na^2+b^2 = (a+b)^2 - 2ab.\n"
    "[Example]\nIf u+v=6 and uv=4 then u^2+v^2 = 36 - 8 = 28.\n\n"
    "Bad (vague, do NOT do this): 'Be careful with the algebra' / 'Avoid mistakes "
    "in expansion'.\n\n"
    "Hard constraints:\n"
    "- Use the literal [Error]/[Rule]/[Example] headers for every block.\n"
    "- Each field must be SPECIFIC and concrete; the [Rule] must be a transferable "
    "principle, the [Example] a self-contained mini worked instance.\n"
    "- Never state this problem's final/ground-truth answer and never give its full "
    "worked solution.\n"
    "- Output ONLY the [Error]/[Rule]/[Example] blocks, nothing else."
)


@dataclass
class SkillPrompts:
    """The optimizable prompt PAIR. This bundle is what the layer-2 optimizer
    rewrites each round."""

    correct: str = CORRECT_SKILL_SYSTEM_SEED
    pitfall: str = PITFALL_SKILL_SYSTEM_SEED

    @staticmethod
    def seed() -> "SkillPrompts":
        return SkillPrompts()


def correct_skill_user(problem: str, solutions: list[str], max_chars: int = 4000) -> str:
    """User turn for correct-knowledge distillation from the CORRECT traces of a
    group (SDPO's _skill_user_prompt, generalized to >=1 solution)."""
    joined = "\n\n---\n\n".join(
        f"CORRECT SOLUTION {i + 1} (reference, do not echo):\n{_clip(s, max_chars)}"
        for i, s in enumerate(solutions)
    )
    return (
        f"PROBLEM:\n{problem}\n\n"
        f"{joined}\n\n"
        "Distill the transferable know-how into 1-3 [Knowledge/Rule]/[Details/Examples] "
        "tiny-skill blocks (each reusable on OTHER problems; do not state this "
        "problem's final answer)."
    )


def pitfall_skill_user(
    problem: str, attempts: list[str], ground_truth: str, max_chars: int = 4000
) -> str:
    """User turn for pitfall distillation from the FAILED traces of a group
    (SDPO's _skill_user_prompt_incorrect, generalized to >=1 attempt). The
    ground-truth answer is given ONLY to localize mistakes; the output must never
    state it."""
    joined = "\n\n---\n\n".join(
        f"FAILED ATTEMPT {i + 1} (wrong, do not echo):\n{_clip(a, max_chars)}"
        for i, a in enumerate(attempts)
    )
    gt = (ground_truth or "").strip()
    return (
        f"PROBLEM:\n{problem}\n\n"
        f"{joined}\n\n"
        f"GROUND-TRUTH ANSWER (for locating the mistakes only, do NOT put it in the output):\n{gt}\n\n"
        "Identify the specific mistake(s) and write them as [Error]/[Rule]/[Example] "
        "blocks (1-3 blocks, using the literal headers; specific and concrete; no "
        "solution, no answer)."
    )


# --------------------------------------------------------------------------- #
# TEST-TIME (deployment) skill PREDICTION -- no traces, no ground truth.
#
# At train time the skill is DISTILLED from a graded group (correct/failed
# traces + label). At deployment there is neither: the model sees only the bare
# problem and must PREDICT, up-front, the knowledge a correct solution will need
# and the pitfalls it is likely to hit. We reuse the SAME (optimized) system
# prompts -- they own the output format + quality bar -- and only swap the user
# turn from "distill from these traces" to "anticipate from this problem".
# --------------------------------------------------------------------------- #
def predict_knowledge_user(problem: str) -> str:
    """User turn: predict the knowledge a correct solution WILL need, from the
    bare problem (no solution shown). Same [Knowledge/Rule]/[Details/Examples]
    contract as correct_skill_user."""
    return (
        f"PROBLEM (you have NOT solved it yet; no solution is provided):\n{problem}\n\n"
        "Before solving, ANTICIPATE the transferable know-how a correct solution "
        "will most likely rely on. Write 1-3 [Knowledge/Rule]/[Details/Examples] "
        "tiny-skill blocks (each reusable on OTHER problems; the [Details/Examples] "
        "a self-contained mini instance; do NOT attempt to solve this problem or "
        "state any final answer)."
    )


def predict_pitfall_user(problem: str) -> str:
    """User turn: predict the mistakes a solver is LIKELY to make on this problem,
    from the bare problem (no failed attempts, no ground truth). Same
    [Error]/[Rule]/[Example] contract as pitfall_skill_user."""
    return (
        f"PROBLEM (you have NOT solved it yet; no attempts or answer are provided):\n{problem}\n\n"
        "Before solving, ANTICIPATE the specific mistakes a solver is LIKELY to make "
        "on THIS problem. Write 1-3 [Error]/[Rule]/[Example] blocks (using the literal "
        "headers; each mistake specific and concrete, the [Rule] a transferable "
        "principle, the [Example] a self-contained mini instance; do NOT solve the "
        "problem or state any final answer)."
    )


def combine_skill(correct_text: str, pitfall_text: str) -> str:
    """The skill injected into the solver = correct-knowledge blocks + pitfall
    blocks distilled from this problem's group (SDPO skill-source 'all')."""
    parts = []
    c = (correct_text or "").strip()
    p = (pitfall_text or "").strip()
    if c:
        parts.append("KNOWLEDGE / RULES (from solutions that worked):\n\n" + c)
    if p:
        parts.append("PITFALLS TO AVOID (from attempts that failed):\n\n" + p)
    return "\n\n".join(parts)


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " ...[clipped]"


# --------------------------------------------------------------------------- #
# LAYER 1 -- how the combined skill is spliced into the SOLVER's context
# --------------------------------------------------------------------------- #
_SKILL_HINT_TEMPLATE = (
    "Here is some relevant knowledge and common pitfalls that may help you solve "
    "the problem below:\n\n{skill}\n\n---\n\nNow solve this problem:\n\n{problem}"
)


def skill_augmented_user(problem: str, skill: str) -> str:
    """Solver user turn = skill hint + original problem (bare problem if the skill
    is empty, so the solver always gets a well-formed prompt)."""
    skill = (skill or "").strip()
    if not skill:
        return problem
    return _SKILL_HINT_TEMPLATE.format(skill=skill, problem=problem)


# --------------------------------------------------------------------------- #
# LAYER 2 -- the meta-optimizer that rewrites the skill-gen prompt PAIR
# --------------------------------------------------------------------------- #

OPTIMIZER_SYSTEM = (
    "You are a PROMPT OPTIMIZER. You improve the TWO 'skill-generation' prompts "
    "used inside a two-stage problem-solving scaffold.\n\n"
    "How the scaffold works:\n"
    "1. Each problem is attempted MULTIPLE times by a solver; the attempts are "
    "graded correct/incorrect against a ground-truth answer.\n"
    "2. A CORRECT-KNOWLEDGE prompt distills reusable [Knowledge/Rule]/[Details/"
    "Examples] blocks from the attempts that were CORRECT.\n"
    "3. A PITFALL prompt distills [Error]/[Rule]/[Example] blocks from the attempts "
    "that were WRONG.\n"
    "4. Those blocks are concatenated into a 'skill' shown to the solver, which "
    "then re-attempts the problem. The goal is to raise the solver's accuracy.\n\n"
    "You are given the CURRENT correct-knowledge prompt, the CURRENT pitfall "
    "prompt, and a batch of results from this round: for each problem, the skill "
    "that was generated and the solver's accuracy WITH that skill (and its baseline "
    "accuracy WITHOUT any skill). Diagnose WHY the current prompts produce skills "
    "that fail to help (too vague? leak the answer so the skill doesn't generalize? "
    "wrong kind of knowledge? pitfalls not actionable? bad format the solver "
    "ignores?), then rewrite BOTH prompts to fix it.\n\n"
    "Rules for your rewrite:\n"
    "- Both prompts must still output ONLY the structured skill blocks and NEVER "
    "state a problem's final answer (leaking the answer makes the skill useless -- "
    "it must generalize to unseen problems).\n"
    "- Keep them GENERAL skill-generation instructions: do NOT hard-code facts about "
    "the specific problems in this batch.\n"
    "- Make concrete, targeted changes justified by the failures you observed.\n"
    "- Output EXACTLY this structure and nothing else:\n"
    "  DIAGNOSIS: <2-4 sentences on what's wrong with the current prompts>\n"
    "  ===CORRECT===\n"
    "  <the full text of the improved correct-knowledge prompt>\n"
    "  ===PITFALL===\n"
    "  <the full text of the improved pitfall prompt>\n"
)


OPTIMIZER_SYSTEM_PREDICT = (
    "You are a PROMPT OPTIMIZER. You improve the TWO 'skill-generation' prompts "
    "used inside a two-stage problem-solving scaffold that runs at DEPLOYMENT "
    "time (no ground-truth answers available).\n\n"
    "How the scaffold works:\n"
    "1. Given ONLY a problem statement (unsolved, no answer), a KNOWLEDGE prompt "
    "asks a model to ANTICIPATE the [Knowledge/Rule]/[Details/Examples] a correct "
    "solution will need, and a PITFALL prompt asks it to ANTICIPATE the "
    "[Error]/[Rule]/[Example] mistakes a solver is likely to make.\n"
    "2. Those predicted blocks are concatenated into a 'skill' prepended to the "
    "problem, and a solver attempts it. The goal is to raise the solver's accuracy.\n\n"
    "The skills are PREDICTED from the bare problem -- the model has NOT seen a "
    "solution or the answer. So the prompts must elicit knowledge/pitfalls that "
    "are (a) reliably guessable from the problem statement alone, (b) genuinely "
    "useful for solving, and (c) never a hallucinated or answer-leaking claim (a "
    "confidently wrong predicted 'fact' actively hurts the solver).\n\n"
    "You are given the CURRENT knowledge prompt, the CURRENT pitfall prompt, and a "
    "batch of results: for each problem, the skill that was predicted and the "
    "solver's accuracy WITH it (and its baseline accuracy WITHOUT any skill). "
    "Diagnose WHY the current prompts produce predicted skills that fail to help "
    "(too vague? hallucinated facts the solver trusts? mis-anticipated the real "
    "difficulty? not actionable? bad format?), then rewrite BOTH prompts to fix "
    "it.\n\n"
    "Rules for your rewrite:\n"
    "- Both prompts must still output ONLY the structured skill blocks and NEVER "
    "state a final answer (the model doesn't know it, and guessing hurts).\n"
    "- Keep them GENERAL: do NOT hard-code facts about the specific problems in "
    "this batch; they must generalize to unseen problems.\n"
    "- Push for knowledge/pitfalls that are safely inferable from the problem "
    "statement and reduce solver errors; discourage speculative claims.\n"
    "- Make concrete, targeted changes justified by the failures you observed.\n"
    "- Output EXACTLY this structure and nothing else:\n"
    "  DIAGNOSIS: <2-4 sentences on what's wrong with the current prompts>\n"
    "  ===CORRECT===\n"
    "  <the full text of the improved knowledge prompt>\n"
    "  ===PITFALL===\n"
    "  <the full text of the improved pitfall prompt>\n"
)


def optimizer_user(
    correct_prompt: str,
    pitfall_prompt: str,
    examples: list[dict],
    max_examples: int = 20,
    max_skill_chars: int = 900,
) -> str:
    """Build the optimizer's user turn from this round's per-problem results.

    ``examples`` is a list of {"problem","skill","acc","baseline_acc"} dicts (one
    per problem). We prioritize problems where the skill HURT or failed to help
    (acc <= baseline_acc), since those carry the optimization signal."""

    def _clip(s: str, n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else s[:n] + " ...[clipped]"

    hurt = [e for e in examples if e.get("acc", 0.0) <= e.get("baseline_acc", 0.0)]
    helped = [e for e in examples if e.get("acc", 0.0) > e.get("baseline_acc", 0.0)]
    chosen = hurt[:max_examples]
    if len(chosen) < max_examples:
        chosen += helped[: max_examples - len(chosen)]

    lines = [
        "CURRENT CORRECT-KNOWLEDGE PROMPT:\n<<<\n" + correct_prompt.strip() + "\n>>>\n\n",
        "CURRENT PITFALL PROMPT:\n<<<\n" + pitfall_prompt.strip() + "\n>>>\n\n",
        f"RESULTS THIS ROUND ({len(helped)} problems where the skill helped, "
        f"{len(hurt)} where it did not; sample below prioritizes the ones it did NOT help):\n",
    ]
    for i, e in enumerate(chosen):
        lines.append(
            f"\n[{i + 1}] with-skill acc={e.get('acc', 0.0):.2f}  baseline acc={e.get('baseline_acc', 0.0):.2f}\n"
            f"PROBLEM: {_clip(e.get('problem', ''), 500)}\n"
            f"GENERATED SKILL:\n{_clip(e.get('skill', ''), max_skill_chars)}\n"
        )
    lines.append(
        "\n\nNow diagnose and rewrite BOTH prompts per your instructions. Remember "
        "the EXACT output structure: DIAGNOSIS, then ===CORRECT===, then ===PITFALL===."
    )
    return "".join(lines)


def parse_optimizer_output(
    text: str, fallback: SkillPrompts
) -> tuple[str, SkillPrompts]:
    """Split the optimizer output into (diagnosis, SkillPrompts). Contract:
    'DIAGNOSIS: ...\\n===CORRECT===\\n<correct>\\n===PITFALL===\\n<pitfall>'.
    Each section falls back to the current prompt if missing/implausibly short,
    so a malformed optimizer response never corrupts the loop."""
    text = (text or "").strip()
    if not text:
        return "", fallback

    def _section(tag: str) -> str | None:
        marker = f"==={tag}==="
        idx = text.find(marker)
        if idx == -1:
            return None
        start = idx + len(marker)
        # section ends at the next '===XXX===' marker or end of text
        rest = text[start:]
        nxt = None
        for other in ("===CORRECT===", "===PITFALL==="):
            if other == marker:
                continue
            j = rest.find(other)
            if j != -1:
                nxt = j if nxt is None else min(nxt, j)
        body = rest[:nxt] if nxt is not None else rest
        return body.strip()

    diagnosis = ""
    head = text.split("===CORRECT===", 1)[0]
    if "DIAGNOSIS" in head:
        diagnosis = head.split("DIAGNOSIS:", 1)[-1].strip() if "DIAGNOSIS:" in head else head.strip()

    correct = _section("CORRECT")
    pitfall = _section("PITFALL")
    new = SkillPrompts(
        correct=correct if (correct and len(correct) >= 40) else fallback.correct,
        pitfall=pitfall if (pitfall and len(pitfall) >= 40) else fallback.pitfall,
    )
    return diagnosis, new
