"""Minimal SDPO (prefix-conditioned self-distillation) reward function for Miles.

SDPO idea
---------
Rollout is exactly GRPO (n_samples_per_prompt traces per prompt). After the
group is generated we:

1. Grade every trace against the label to know which ones are *correct*.
2. For every trace, RANDOMLY pick one *other* correct trace in the group to use
   as a *prefix* (a "hint"); the prefix is never the trace itself.
3. Format that correct peer solution with the prefix template (see
   ``SOLUTION_TEMPLATE`` / ``PREFIX_INSTRUCTION`` below) and insert it between
   the prompt and this trace's response. Ask the teacher to score
   ``prompt + prefix + response`` and read its next-token behaviour over the
   response span. The teacher has seen a correct hint; the student never did.
4. The student signal is the ORIGINAL rollout one, conditioned on
   ``prompt + response`` with no prefix:
     - ``topk`` mode   -> the per-position top-k distribution captured during
       rollout into ``sample.metadata["opd_student_top_logprobs"]``, plus a
       single aggregated "tail" bucket for all remaining vocabulary mass.
     - ``sampled`` mode -> the per-token sampled log-prob in
       ``sample.rollout_log_probs``.
5. Compute a per-token divergence between teacher (with prefix) and student
   (without prefix) and store it in ``sample.opd_reverse_kl``. The framework
   subtracts ``opd_kl_coef * opd_reverse_kl`` from the GRPO advantages
   (see ``miles/backends/training_utils/loss_hub/opd.py``); no training-side
   change is needed.

Correct-trace policy (see ``sdpo_group_reward``)
    - 0 correct traces in the group -> no KL for anyone.
    - >= 1 correct trace            -> each trace draws a random correct peer
                                       (never itself) as its prefix; a trace
                                       that is the sole correct one gets no
                                       prefix (self-excluded pool is empty).

Wiring
------
Use this as a *group* reward model::

    --group-rm
    --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
    --rm-url http://<TEACHER_IP>:<TEACHER_PORT>/generate
    --use-opd --opd-type sglang --opd-log-prob-top-k 128
    --opd-kl-coef 1.0
    --sdpo-divergence jsd            # reverse_kl | forward_kl | jsd
    --sdpo-logprob-mode topk         # topk | sampled

Because ``--group-rm`` hands us the whole prompt group at once
(see ``sglang_rollout.generate_and_rm_group``), we can choose a prefix from
peer traces. This module only supports ``context_parallel_size == 1``: the
divergence is computed on the full, un-sharded response token sequence.
"""

import asyncio
import logging
import math
import os
import random
import re
import time
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from miles.rollout.rm_hub.math_dapo_utils import compute_score as _dapo_compute_score
from miles.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer
from miles.rollout.rm_hub.math_utils import grade_answer_verl
from miles.utils.http_utils import post  # miles' shared HTTP client: retries + shared pool
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# Per-phase wall-clock accumulators for one rollout's worth of SDPO scoring.
# These sum across all traces (concurrent), so they overcount vs wall time, but
# their RATIO tells us where time goes: HTTP teacher-scoring wait vs CPU prep vs
# vectorized divergence. Reset + logged per group-reward batch call.
_sdpo_timing = {"tokenize": 0.0, "student_maps": 0.0, "teacher_http": 0.0, "teacher_maps": 0.0, "divergence": 0.0}
_sdpo_calls = 0

# --------------------------------------------------------------------------- #
# prefix templates  (edit to taste -- this is the "prefix format")
# --------------------------------------------------------------------------- #
# The prefix is everything that follows {prompt} in a reprompt template: a
# correct peer solution plus an instruction. It is tokenized and inserted
# between the original prompt tokens and this trace's response tokens, so the
# response stays at the tail and per-position alignment is preserved.
SOLUTION_TEMPLATE = "\n\nCorrect solution:\n\n{successful_previous_attempt}"
PREFIX_INSTRUCTION = "\n\nCorrectly solve the original question.\n\n"
# Optional pitfalls block (group-aggregated warnings distilled from the group's
# INCORRECT traces). Inserted BEFORE the instruction, AFTER the correct-solution /
# skill prefix, so the teacher sees "here's the approach, and here are the mistakes
# to avoid". Kept as a clearly-labelled separate section so it is never confused
# with the correct solution.
PITFALLS_TEMPLATE = "\n\nCommon mistakes to avoid (seen in failed attempts):\n\n{pitfalls}"


def _strip_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from a response (for thinking models like Qwen3).
    Strips leading/trailing whitespace after removal so the peer solution stays clean."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return stripped.strip()


def _render_prefix(peer_response: str, remove_thinking: bool = False, pitfalls: str = "") -> str:
    content = _strip_thinking_blocks(peer_response) if remove_thinking else peer_response
    # Skip the "Correct solution:" section entirely when there is no base solution
    # (e.g. a failed trace in an all-wrong group gets a pitfalls-only prefix); an
    # empty "Correct solution:" heading would mislead the teacher.
    section = SOLUTION_TEMPLATE.format(successful_previous_attempt=content) if content and content.strip() else ""
    if pitfalls and pitfalls.strip():
        section += PITFALLS_TEMPLATE.format(pitfalls=pitfalls.strip())
    return section + PREFIX_INSTRUCTION


def _gen_prompt_suffix(tok) -> str:
    """The exact string the chat template appends AFTER the user content when
    add_generation_prompt=True — e.g. '<|im_end|>\\n<|im_start|>assistant\\n' for
    ChatML (Qwen2.5/Qwen3). Derived from the tokenizer so it is template-agnostic.

    We use this to splice the correct-peer solution into the USER turn (as context),
    NOT after the assistant marker. This matches lasgroup/SDPO, which builds the
    teacher input as apply_chat_template([system, {user: question + solution +
    instruction}], add_generation_prompt=True) then concatenates the response —
    i.e. the solution lives in the user turn, followed by a fresh assistant marker.
    Inserting it after '<|im_start|>assistant' instead (the old bug) pollutes the
    assistant turn and teaches the model to echo a pre-filled answer.
    """
    sentinel = " SDPO_SENTINEL "
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": sentinel}], tokenize=False, add_generation_prompt=True
    )
    return rendered.split(sentinel, 1)[1] if sentinel in rendered else ""


# Special/EOS tokens a rollout response may end with; they must be stripped before
# the response is embedded as text inside the teacher's USER turn, otherwise the
# stray <|im_end|> closes the user turn early and corrupts the chat structure.
_RESPONSE_EOS_MARKERS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>")


def _strip_response_eos(text: str) -> str:
    """Remove trailing chat/EOS markers (and whitespace) from a rollout response."""
    out = (text or "").rstrip()
    changed = True
    while changed:
        changed = False
        for marker in _RESPONSE_EOS_MARKERS:
            if out.endswith(marker):
                out = out[: -len(marker)].rstrip()
                changed = True
    return out


def _build_teacher_prompt_str(student_prompt: str, gen_suffix: str, peer_response: str, remove_thinking: bool = False, pitfalls: str = "") -> str:
    """Insert the correct-peer solution + instruction into the USER turn of the
    student's (already chat-templated) prompt, before the assistant generation
    marker. Returns the full teacher prompt string (system + user+solution +
    assistant marker).

    The peer response is stripped of its trailing <|im_end|>/EOS first — it was a
    full generated turn, and leaving that marker in mid-user-turn would close the
    user turn early and corrupt the teacher prompt (the model would then see the
    solution as a separate malformed turn instead of context)."""
    solution_section = _render_prefix(_strip_response_eos(peer_response), remove_thinking=remove_thinking, pitfalls=pitfalls)
    if gen_suffix and gen_suffix in student_prompt:
        idx = student_prompt.rfind(gen_suffix)
        return student_prompt[:idx] + solution_section + student_prompt[idx:]
    # Fallback (unknown template): append at the end. Not ideal but never crashes.
    return student_prompt + solution_section


def _build_failure_teacher_prompt_str(student_prompt: str, gen_suffix: str, failure_info: str) -> str:
    """Splice the group's per-trace failure skills into the USER turn of a
    problem-only pitfall-prediction prompt as PRIVILEGED info (pitfall-condense
    skill-KD teacher). Empty failure_info -> teacher == student (no privileged info,
    KD signal 0 for that sample). Mirrors _build_teacher_prompt_str's insert point."""
    if not (failure_info and failure_info.strip()):
        return student_prompt
    section = FAILURES_TEMPLATE.format(successful_previous_attempt=failure_info.strip())
    if gen_suffix and gen_suffix in student_prompt:
        idx = student_prompt.rfind(gen_suffix)
        return student_prompt[:idx] + section + student_prompt[idx:]
    return student_prompt + section


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #


def _divergence_mode(args: Namespace) -> str:
    mode = getattr(args, "sdpo_divergence", "jsd")
    if mode not in ("reverse_kl", "forward_kl", "jsd", "jeffrey", "jeffrey_jsd"):
        raise ValueError(
            f"Unknown --sdpo-divergence {mode!r}; use "
            "reverse_kl | forward_kl | jsd | jeffrey | jeffrey_jsd."
        )
    return mode


def _logprob_mode(args: Namespace) -> str:
    mode = getattr(args, "sdpo_logprob_mode", "topk")
    if mode not in ("topk", "sampled"):
        raise ValueError(f"Unknown --sdpo-logprob-mode {mode!r}; use topk | sampled.")
    return mode


def _prompt_len(sample: Sample) -> int:
    return len(sample.tokens) - sample.response_length


def _response_tokens(sample: Sample) -> list[int]:
    return sample.tokens[_prompt_len(sample) :]


def _extract_tagged_answer(text: str, tag: str = "answer") -> str | None:
    """Extract the content of the LAST <tag>...</tag> block, or None if absent.

    Open-ended SDPO asks the model to wrap its final answer in <answer>...</answer>
    (see examples/SDPO/build_sci_dataset.py). We take the last occurrence so a
    model that reasons and revises still yields its final answer. Returns None
    (not "") when the tag is missing so callers can treat "no answer" as wrong.
    """
    if not text:
        return None
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    # Tolerate an unclosed final tag ("<answer> foo" with no </answer>).
    m = re.search(rf"<{tag}>(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_answer(args: Namespace, sample: Sample) -> str | None:
    """The model's final answer: prefer the <answer> tag, fall back to \\boxed{}."""
    tag = getattr(args, "sdpo_answer_tag", "answer")
    tagged = _extract_tagged_answer(sample.response, tag)
    if tagged is not None:
        return tagged
    return extract_boxed_answer(sample.response)


def _is_correct(sample: Sample, args: Namespace | None = None) -> bool:
    """Grade a trace against its label with exact/heuristic matching (no LLM).

    The SciKnowEval dataset (see build_sci_dataset.py) is multiple choice: the
    label is the answer LETTER and the model outputs the letter inside
    <answer>...</answer>. We do a case-insensitive letter match on the extracted
    answer. Non-letter labels (if any) fall back to math-style grading.
    """
    # DAPO math dataset: integer answers in \boxed{}. Use DAPO's own grader with
    # strict_box_verify=True (the default minerva path expects an "Answer: X" line;
    # the strict-box path extracts the \boxed{} answer, which is what the models emit).
    if args is not None and getattr(args, "sdpo_grader", "mcq") == "dapo":
        label = (sample.label or "").strip()
        if not label:
            return False
        try:
            return bool(_dapo_compute_score(sample.response or "", label, strict_box_verify=True)["acc"])
        except Exception:
            return bool(grade_answer_verl(sample.response or "", label))

    tag = getattr(args, "sdpo_answer_tag", "answer") if args is not None else "answer"
    extracted = _extract_tagged_answer(sample.response, tag)
    if extracted is None:
        extracted = extract_boxed_answer(sample.response)
    label = (sample.label or "").strip()
    if not label:
        return False

    # Legacy single-letter labels (old MCQ dataset): case-insensitive letter match.
    if len(label) == 1 and label.isalpha():
        pred = (extracted or "").strip()
        if len(pred) > 1:  # e.g. "B." or "(B)" -> keep first alpha char
            pred = next((c for c in pred if c.isalpha()), pred)
        return pred.upper() == label.upper()

    # Open-ended: exact (normalized) match or math-style grading.
    pred = (extracted or "").strip()
    if pred and pred.lower() == label.lower():
        return True
    return bool(grade_answer_verl(extracted or sample.response, label))


# --------------------------------------------------------------------------- #
# LLM-as-judge grading  (defeats the MCQ letter-guessing reward hack)
# --------------------------------------------------------------------------- #

_JUDGE_SYSTEM = (
    "You are a strict grader for science exam answers. You are given the QUESTION, "
    "the REFERENCE ANSWER (ground truth), the model's FULL RESPONSE, and the model's "
    "EXTRACTED ANSWER. Decide whether the model's answer is scientifically correct "
    "and equivalent to the reference answer.\n\n"
    "Rules:\n"
    "- Judge correctness of the ANSWER's meaning, not its wording/format. Accept "
    "mathematically or chemically equivalent forms (e.g. same SMILES/quantity/name).\n"
    "- The extracted answer must actually answer the question. A blank, missing, or "
    "placeholder answer is INCORRECT even if the full response rambles near the topic.\n"
    "- Do NOT give credit for a guess with no supporting reasoning if it does not match "
    "the reference answer.\n"
    "Reply with EXACTLY one word on the final line: CORRECT or INCORRECT."
)


def _build_judge_prompt(args: Namespace, sample: Sample) -> tuple[str, str]:
    """Return (system, user) messages for the judge. Includes BOTH the full response
    and the extracted answer, per the requirement to give the judge both."""
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    question = meta.get("question") or (sample.prompt if isinstance(sample.prompt, str) else "")
    reference = (sample.label or "").strip()
    extracted = _extract_answer(args, sample)
    full = sample.response or ""
    # Cap the full response so the judge prompt stays bounded on very long traces.
    if len(full) > 8000:
        full = full[:4000] + "\n...[truncated]...\n" + full[-3000:]
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"MODEL FULL RESPONSE:\n{full}\n\n"
        f"MODEL EXTRACTED ANSWER:\n{extracted if extracted is not None else '(none — no <answer> tag found)'}\n\n"
        "Is the model's answer correct? Reply CORRECT or INCORRECT."
    )
    return _JUDGE_SYSTEM, user


# GLOBAL judge concurrency limiter. Each rollout spawns one generate_and_rm_group
# task PER GROUP (rollout_batch_size groups) and they all run concurrently, so a
# per-call semaphore would only bound the ~n_samples_per_prompt traces within one
# group — the real cap must be process-wide. This single semaphore is shared by
# every group's judge calls, so --sdpo-judge-max-concurrency bounds TOTAL in-flight
# judge requests to the gateway (else 32 groups x 8 traces = 256 at once).
_JUDGE_SEM: "asyncio.Semaphore | None" = None
_JUDGE_SEM_LIMIT: int | None = None


def _judge_semaphore(args: Namespace) -> asyncio.Semaphore:
    global _JUDGE_SEM, _JUDGE_SEM_LIMIT
    limit = int(getattr(args, "sdpo_judge_max_concurrency", 32))
    # Recreate if the limit changed (or first use). Safe: single event loop.
    if _JUDGE_SEM is None or _JUDGE_SEM_LIMIT != limit:
        _JUDGE_SEM = asyncio.Semaphore(limit)
        _JUDGE_SEM_LIMIT = limit
    return _JUDGE_SEM


def _parse_judge_verdict(text: str) -> bool:
    """Parse the judge's reply into a bool. Looks for the last CORRECT/INCORRECT."""
    if not text:
        return False
    up = text.upper()
    # INCORRECT contains CORRECT, so search for whole-word tokens and take the last.
    hits = re.findall(r"\b(INCORRECT|CORRECT)\b", up)
    if not hits:
        return False
    return hits[-1] == "CORRECT"


async def _llm_judge_correct(args: Namespace, sample: Sample) -> bool:
    """Grade one trace via the OpenAI-compatible LLM judge (SFR gateway).

    Falls back to deterministic _is_correct on any judge failure so a flaky
    gateway never stalls or crashes training.
    """
    system, user = _build_judge_prompt(args, sample)
    base_url = getattr(args, "sdpo_judge_base_url", "https://api.openai.com/v1").rstrip("/")
    model = getattr(args, "sdpo_judge_model", "gpt-5.4-mini")
    api_key = os.environ.get(getattr(args, "sdpo_judge_api_key_env", "OPENAI_API_KEY"), "") or "EMPTY"
    max_tokens = int(getattr(args, "sdpo_judge_max_tokens", 2048))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    # gpt-5*/o-series are reasoning models: use max_completion_tokens, ignore temperature.
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_completion_tokens"] = max_tokens
        payload["temperature"] = 0.0
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        # Low retry count: the judge is best-effort, fall back fast on failure.
        out = await post(f"{base_url}/chat/completions", payload, max_retries=3, headers=headers)
        content = out["choices"][0]["message"].get("content") or ""
        return _parse_judge_verdict(content)
    except Exception as e:
        logger.warning(f"LLM judge failed ({e!r}); falling back to deterministic grading.")
        return _is_correct(sample, args)


async def _grade_group(args: Namespace, group: list[Sample]) -> list[bool]:
    """Correctness for every trace in a group. Uses the LLM judge (bounded
    concurrency) when --sdpo-judge is set, else deterministic matching."""
    if not getattr(args, "sdpo_judge", False):
        return [_is_correct(s, args) for s in group]

    # Process-wide cap (shared across all concurrently-running groups), so the
    # judge calls are concurrent — both within a group (gather) and across groups
    # (each group is its own asyncio task) — without overloading the gateway.
    sem = _judge_semaphore(args)

    async def _one(s: Sample) -> bool:
        # Skip the judge for empty responses (cheap, obviously wrong).
        if not (s.response or "").strip():
            return False
        async with sem:
            return await _llm_judge_correct(args, s)

    return list(await asyncio.gather(*(_one(s) for s in group)))


# --------------------------------------------------------------------------- #
# Trace condensation / SkillOpt  (distill the correct peer trace into a SKILL)
# --------------------------------------------------------------------------- #
# When --sdpo-trace-condense is set, the correct-peer solution is first distilled
# into a short transferable SKILL (<=3 procedural bullets, no answer) by an LLM,
# and that skill — not the full trace — becomes the teacher prefix. Mirrors
# lasgroup/SDPO trace_condense (verl/trainer/ppo/trace_condense.py).

_SKILL_SYSTEM_PROMPT = (
    "You distill a worked solution into a SKILL: a concrete solution ROADMAP that a "
    "capable solver could follow to reach the correct answer to THIS problem on their "
    "own. It is grounded in this specific problem — name the key quantities, the "
    "governing relation/theorem to apply, the decisive facts or comparisons that "
    "discriminate the right choice from the wrong ones, and the key intermediate "
    "results along the way — but it stops one step short of the final answer.\n\n"
    "Hard constraints:\n"
    "- Be a clear numbered roadmap: 6-10 short steps, each an imperative instruction.\n"
    "- Instance-grounded: DO reference this problem's specific quantities, setup, and "
    "the critical intermediate values/comparisons needed to get the answer right.\n"
    "- Do NOT state the final answer itself (no final letter/number/name, no "
    "'the answer is ...'). Stop at the last step BEFORE committing to the answer, so "
    "the reader must still perform the final selection/computation themselves.\n"
    "- Output ONLY the numbered steps, nothing else."
)

# Incorrect-trace variant: the attempt is WRONG. A model that failed this problem
# CANNOT be trusted to rewrite a correct solution — asking it to "reach the right
# answer" just yields a hallucinated roadmap that would poison the KD target. So we
# do NOT distill know-how / a solution roadmap here. Instead we distill the ERROR
# PATTERN: identify the specific mistake(s) the attempt made and turn each into a
# concrete "avoid this" warning bullet. The ground-truth answer is provided ONLY so
# the model can localize where the attempt went wrong; the output is pitfalls, never
# a solution and never the answer.
_SKILL_SYSTEM_PROMPT_INCORRECT = (
    "You are given a FAILED attempt at a problem and the ground-truth answer. The "
    "attempt is WRONG. Do NOT try to solve the problem or write a correct solution — "
    "you only have a failed attempt to learn from. Instead, identify the SPECIFIC "
    "mistakes the attempt made and turn each into a concrete PITFALL WARNING: a "
    "'do not do X / watch out for Y' note grounded in this problem's setup, so a "
    "solver would avoid that same error on this kind of problem.\n\n"
    "Hard constraints:\n"
    "- Output 2-5 numbered pitfall bullets, each an imperative 'Avoid ...' / "
    "'Do not ...' / 'Watch out that ...' warning naming the concrete mistake.\n"
    "- Diagnose from the attempt; reference this problem's specific quantities/setup "
    "where it sharpens the warning. Do NOT provide the correct steps or method.\n"
    "- Never state the final/ground-truth answer (no letter/number/name, no "
    "'the answer is ...') and never give a worked solution.\n"
    "- Output ONLY the numbered pitfall warnings, nothing else."
)

# Second-stage aggregation: given the pitfalls distilled from EVERY failed trace in
# a group (each a small list of "avoid X" warnings), synthesize the COMMON failure
# lessons — the mistakes that recur across attempts on this problem — into one short
# shared list. This shared list (not the raw concatenation) is what gets spliced
# into the failed traces' teacher prefix, so the teacher sees a tight "here's how
# this group tends to fail" summary rather than a long noisy dump.
_PITFALL_SUMMARY_SYSTEM = (
    "You are given several sets of PITFALL WARNINGS, each distilled from a different "
    "failed attempt at the SAME problem. Synthesize the COMMON, recurring mistakes "
    "into one short shared list of pitfalls to avoid on this problem. Merge duplicates, "
    "keep only the mistakes that matter most, and drop one-off noise.\n\n"
    "Hard constraints:\n"
    "- Output 2-4 numbered pitfall bullets, each an imperative 'Avoid ...' / "
    "'Do not ...' / 'Watch out that ...' warning.\n"
    "- Be concise and general enough to cover the recurring errors; still grounded in "
    "this problem's setup where it sharpens the warning.\n"
    "- Never state the final/ground-truth answer and never give a worked solution.\n"
    "- Output ONLY the numbered pitfall warnings, nothing else."
)


def _pitfall_summary_user_prompt(problem: str, pitfall_sets: list[str]) -> str:
    blocks = "\n\n".join(f"FAILED ATTEMPT {k + 1} PITFALLS:\n{p}" for k, p in enumerate(pitfall_sets))
    return (
        f"PROBLEM:\n{_clean_problem_for_skill(problem)}\n\n"
        f"{blocks}\n\n"
        "Synthesize the common recurring pitfalls (2-4 numbered 'Avoid ...' bullets; "
        "no solution, no answer)."
    )


# --- pitfall-condense skill-KD (⑤): the skill's own OPD --------------------- #
# STUDENT (no privileged info): given ONLY the problem, predict the pitfalls a
# solver should avoid — a pure "foresee the traps" task with no failed attempt and
# no answer. TEACHER (privileged): the same problem-only prompt PLUS the group's
# actual per-trace failure skills spliced in, so it condenses what really went
# wrong. KD pulls the problem-only student toward the failure-informed teacher.
_PITFALL_PREDICT_SYSTEM = (
    "Given a problem (and NOTHING else — no attempt, no answer), predict the pitfalls "
    "a solver is most likely to fall into on this kind of problem, as concrete "
    "warnings to avoid.\n\n"
    "Hard constraints:\n"
    "- Output 2-5 numbered pitfall bullets, each an imperative 'Avoid ...' / "
    "'Do not ...' / 'Watch out that ...' warning grounded in this problem's setup.\n"
    "- Do NOT solve the problem or give the method/steps; only the traps to avoid.\n"
    "- Never state a final answer.\n"
    "- Output ONLY the numbered pitfall warnings, nothing else."
)

# Label for the privileged failure info spliced into the pitfall-condense TEACHER
# turn (distinct from "Correct solution:" — these are observed FAILURES, not a
# solution). Reuses the {successful_previous_attempt} field name for _render_prefix
# compatibility but is only ever fed the concatenated failure skills.
FAILURES_TEMPLATE = "\n\nObserved failed-attempt pitfalls (privileged, do not reveal):\n\n{successful_previous_attempt}"


def _pitfall_predict_user_prompt(problem: str) -> str:
    return (
        f"PROBLEM:\n{_clean_problem_for_skill(problem)}\n\n"
        "Predict the pitfalls to avoid (2-5 numbered 'Avoid ...' bullets; no solution, "
        "no answer)."
    )


_CONDENSE_SEM: "asyncio.Semaphore | None" = None
_CONDENSE_SEM_LIMIT: int | None = None


def _condense_semaphore(args: Namespace) -> asyncio.Semaphore:
    global _CONDENSE_SEM, _CONDENSE_SEM_LIMIT
    limit = int(getattr(args, "sdpo_condense_max_concurrency", 32))
    if _CONDENSE_SEM is None or _CONDENSE_SEM_LIMIT != limit:
        _CONDENSE_SEM = asyncio.Semaphore(limit)
        _CONDENSE_SEM_LIMIT = limit
    return _CONDENSE_SEM


# Answer-format scaffolding that datasets inject into the problem text (e.g. DAPO:
# "... The last line of your response should be of the form Answer: \boxed{$Answer}
# ..." and "Remember to put your answer on its own line after 'Answer:'."). If left
# in the skill-gen PROBLEM, the model dutifully appends "Answer: \boxed{...}" to the
# skill, leaking the answer into the skill-KD target. Strip these instruction lines.
_ANSWER_FORMAT_PATTERNS = [
    re.compile(r"Solve the following math problem step by step\.\s*", re.IGNORECASE),
    re.compile(r"The last line of your response should be of the form[^\n]*\n?", re.IGNORECASE),
    re.compile(r"Remember to put your answer[^\n]*\n?", re.IGNORECASE),
    re.compile(r"[Pp]ut your (?:final )?answer (?:in|inside)[^\n]*\\boxed\{\}[^\n]*\n?"),
]


# Chat-template scaffolding. A rollout sample.prompt for DAPO math is the FULL
# chat-templated string (system turn + user turn + assistant marker), NOT the raw
# question — SciKnowEval instead carries the raw question in metadata["question"].
# If we embed the whole templated string as the "PROBLEM" inside the skill-gen
# prompt, we NEST a chat template: the model sees a second <|im_start|>system turn
# (e.g. "You are a helpful function-calling assistant") whose instructions conflict
# with the skill/pitfall system prompt, and the two get confused (solution skills
# grow "Avoid ..." bullets, pitfall skills drop them). Strip the scaffolding down to
# the last user turn's content so the generator sees only the actual problem.
_CHAT_USER_BLOCK = re.compile(
    r"<\|im_start\|>\s*user\s*\n(.*?)<\|im_end\|>", re.DOTALL
)


def _strip_chat_template(text: str) -> str:
    """Recover the raw user-turn text from a chat-templated prompt. Returns the LAST
    user block's content if the template markers are present; otherwise returns the
    input unchanged (already raw, e.g. SciKnowEval's metadata['question'])."""
    if not text or "<|im_start|>" not in text:
        return text
    matches = _CHAT_USER_BLOCK.findall(text)
    if matches:
        return matches[-1].strip()
    # Markers present but no closed user block (unusual template): drop everything
    # up to a 'user' header and the trailing assistant marker as a best effort.
    return text


def _clean_problem_for_skill(problem: str) -> str:
    """Remove chat-template scaffolding AND answer-format instructions from the
    problem so the skill generator sees only the math — not a nested system turn
    (which confuses solution vs pitfall skills) or 'put your answer in \\boxed{}'
    (which leaks the answer format into the skill)."""
    out = _strip_chat_template(problem or "")
    for pat in _ANSWER_FORMAT_PATTERNS:
        out = pat.sub("", out)
    return out.strip()


def _skill_user_prompt(problem: str, solution: str) -> str:
    return (
        f"PROBLEM:\n{_clean_problem_for_skill(problem)}\n\n"
        f"WORKED SOLUTION (reference, do not echo):\n{solution}\n\n"
        "Write the solution roadmap (6-10 numbered steps, instance-grounded, "
        "stop one step before the final answer)."
    )


def _failure_kind(args: Namespace, sample: Sample) -> str:
    """Classify WHY a trace failed, so the pitfall generator can tailor its
    diagnosis. Three kinds, decided cheaply from the sample:
      - "truncated"    : the rollout hit the response-length limit (Status.TRUNCATED)
                         -> the attempt was cut off, not necessarily reasoned wrong.
      - "format"       : a complete response with no parseable answer (no <answer>
                         tag / no \\boxed) -> the reasoning may be fine but the output
                         format is broken.
      - "wrong"        : a complete, parseable answer that is simply incorrect.
    """
    try:
        if sample.status == Sample.Status.TRUNCATED:
            return "truncated"
    except Exception:
        pass
    if _extract_answer(args, sample) is None:
        return "format"
    return "wrong"


_FAILURE_NOTE = {
    "truncated": (
        "NOTE: this attempt was CUT OFF by the response-length limit before it "
        "finished. The reasoning may have been on track; the pitfall is more likely "
        "about efficiency/length (e.g. being too verbose, not reaching the answer in "
        "time) than a conceptual error. Judge accordingly."
    ),
    "format": (
        "NOTE: this attempt produced NO parseable final answer (missing/emptly answer "
        "tag). The reasoning may be fine but the OUTPUT FORMAT is broken. The pitfall "
        "should stress following the required answer format."
    ),
    "wrong": (
        "NOTE: this attempt gave a complete but INCORRECT answer. The pitfall should "
        "target the conceptual/computational mistake that led to the wrong answer."
    ),
}


def _skill_user_prompt_incorrect(problem: str, attempt: str, ground_truth: str, failure_kind: str = "wrong") -> str:
    """User prompt for the incorrect-trace skill: the attempt is wrong; give the
    ground-truth answer ONLY so the model can localize the mistake, then emit
    pitfall warnings (never a solution, never the answer). failure_kind tailors the
    diagnosis (truncated | format | wrong)."""
    gt = (ground_truth or "").strip()
    note = _FAILURE_NOTE.get(failure_kind, _FAILURE_NOTE["wrong"])
    return (
        f"PROBLEM:\n{_clean_problem_for_skill(problem)}\n\n"
        f"FAILED ATTEMPT (wrong, do not echo):\n{attempt}\n\n"
        f"GROUND-TRUTH ANSWER (for locating the mistake only, do NOT put it in the output):\n{gt}\n\n"
        f"{note}\n\n"
        "Identify the specific mistakes and write the pitfall warnings (2-5 numbered "
        "'Avoid ...'/'Do not ...' bullets; no solution, no answer)."
    )


async def _condense_trace_to_skill(args: Namespace, problem: str, solution: str) -> str:
    """Distill one worked solution into a short skill via the OpenAI-compatible LLM.
    Returns the skill string, or the original full solution on any failure."""
    base_url = getattr(args, "sdpo_condense_base_url", "https://api.openai.com/v1").rstrip("/")
    model = getattr(args, "sdpo_condense_model", "gpt-5.4-mini")
    api_key = os.environ.get(getattr(args, "sdpo_condense_api_key_env", "OPENAI_API_KEY"), "") or "EMPTY"
    max_tokens = int(getattr(args, "sdpo_condense_max_tokens", 2048))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SKILL_SYSTEM_PROMPT},
            {"role": "user", "content": _skill_user_prompt(problem, solution)},
        ],
    }
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_completion_tokens"] = max_tokens
        payload["temperature"] = 0.0
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        out = await post(f"{base_url}/chat/completions", payload, max_retries=3, headers=headers)
        skill = (out["choices"][0]["message"].get("content") or "").strip()
        return skill if skill else solution  # fall back to the full trace on empty
    except Exception as e:
        logger.warning(f"trace condense failed ({e!r}); falling back to full trace.")
        return solution


async def _external_llm_chat(args: Namespace, system: str, user: str) -> str | None:
    """Single (system, user) -> text completion via the OpenAI-compatible condenser
    endpoint. Returns the stripped content, or None on failure. Shared by
    trace-condense and the pitfall summary so both use one external-LLM path."""
    base_url = getattr(args, "sdpo_condense_base_url", "https://api.openai.com/v1").rstrip("/")
    model = getattr(args, "sdpo_condense_model", "gpt-5.4-mini")
    api_key = os.environ.get(getattr(args, "sdpo_condense_api_key_env", "OPENAI_API_KEY"), "") or "EMPTY"
    max_tokens = int(getattr(args, "sdpo_condense_max_tokens", 2048))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_completion_tokens"] = max_tokens
        payload["temperature"] = 0.0
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        out = await post(f"{base_url}/chat/completions", payload, max_retries=3, headers=headers)
        return (out["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        logger.warning(f"external LLM chat failed ({e!r}).")
        return None


async def _generate_skill_text(args: Namespace, system: str, user: str, backend: str) -> str:
    """Generate a skill/pitfall text from a (system, user) prompt using the selected
    backend: 'self' = the current policy over the rollout engine (on-policy, same
    generator class as self-skill), 'external' = the OpenAI-compatible LLM (same as
    trace-condense). Thinking is stripped on the self path. Returns "" on failure."""
    if backend == "external":
        return (await _external_llm_chat(args, system, user)) or ""
    # self / policy path: chat-template the (system, user), generate on the rollout
    # engine, strip thinking, decode.
    tok = _tokenizer(args)
    text = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tok.encode(text, add_special_tokens=False)
    res = await _self_generate_skill(args, prompt_ids)
    return res[0] if res else ""


async def _condense_solutions(args: Namespace, pairs: list[tuple[str, str]]) -> list[str]:
    """Condense a batch of (problem, solution) pairs into skills, concurrently and
    with a process-wide cap. Deduplicates identical (problem, solution) pairs so a
    trace shared by several traces in the group is condensed once."""
    sem = _condense_semaphore(args)
    # dedup
    uniq: dict[tuple[str, str], int] = {}
    for p in pairs:
        uniq.setdefault(p, len(uniq))
    keys = list(uniq.keys())

    async def _one(problem: str, solution: str) -> str:
        async with sem:
            return await _condense_trace_to_skill(args, problem, solution)

    skills = await asyncio.gather(*(_one(pr, sol) for pr, sol in keys))
    skill_of = {k: s for k, s in zip(keys, skills)}
    return [skill_of[p] for p in pairs]


# --------------------------------------------------------------------------- #
# Self-generated skill  (the current policy writes the skill during rollout)
# --------------------------------------------------------------------------- #


def _skill_gen_prompt_ids(
    args, tok, problem: str, solution: str, *, correct: bool = True, ground_truth: str = "", failure_kind: str = "wrong"
) -> list[int]:
    """Chat-templated skill-generation prompt (system=SKILL prompt, user=problem+
    solution + distill instruction), tokenized, with the assistant generation
    marker appended. This is the STUDENT context the skill is generated in.

    correct=False switches to the incorrect-trace framing: the attempt is wrong,
    the ground truth is supplied, and the model distills a pitfall-prevention skill
    instead of distilling know-how from a flawed solution. failure_kind (truncated |
    format | wrong) tailors the pitfall diagnosis.
    """
    solution = _strip_response_eos(solution)
    if correct:
        system = _SKILL_SYSTEM_PROMPT
        user = _skill_user_prompt(problem, solution)
    else:
        system = _SKILL_SYSTEM_PROMPT_INCORRECT
        user = _skill_user_prompt_incorrect(problem, solution, ground_truth, failure_kind=failure_kind)
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tok.encode(text, add_special_tokens=False)


def _strip_think_tokens(tok, tokens: list[int], logprobs: list[float]) -> tuple[list[int], list[float]]:
    """Drop everything up to and including the FIRST </think> from a token sequence,
    keeping tokens and their logprobs aligned. Thinking models (e.g. nemotron3,
    Qwen3) emit a leading <think>...</think> reasoning chain before the actual skill;
    the real skill is what follows that closing tag. We locate it at the TOKEN level
    (not by re-encoding the text) so the returned tokens/logprobs stay exactly the
    ones the policy generated — required for skill-KD (logprob alignment) and for a
    clean dump. Returns the sequence unchanged if no </think> is present."""
    if not tokens:
        return tokens, logprobs
    full = tok.decode(tokens)
    if "</think>" not in full:
        return tokens, logprobs
    # decode(tokens[:j]) containing </think> is monotonic in j (longer prefix, more
    # text), so binary-search the smallest j whose prefix already closes the tag.
    lo, hi = 1, len(tokens)
    while lo < hi:
        mid = (lo + hi) // 2
        if "</think>" in tok.decode(tokens[:mid]):
            hi = mid
        else:
            lo = mid + 1
    return tokens[lo:], logprobs[lo:]


async def _self_generate_skill(args: Namespace, prompt_ids: list[int]):
    """Have the current policy (rollout engine) generate a skill from the skill-gen
    prompt. Returns (skill_text, skill_token_ids, skill_logprobs) or None on failure."""
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "temperature": getattr(args, "rollout_temperature", 1.0),
            "max_new_tokens": int(getattr(args, "sdpo_skill_max_new_tokens", 512)),
            "skip_special_tokens": False,
        },
        "return_logprob": True,
    }
    try:
        out = await post(url, payload)
        meta = out.get("meta_info", {})
        otl = meta.get("output_token_logprobs")  # list of [logprob, token_id, ...]
        if not otl:
            return None
        skill_tokens = [int(x[1]) for x in otl]
        skill_logprobs = [float(x[0]) for x in otl]
        # Drop trailing EOS/stop tokens (natural stop appends e.g. <|endoftext|> /
        # <|im_end|>). Keep tokens+logprobs aligned so the skill (dump + skill-KD
        # target) is clean and doesn't end on a special token.
        tok = _tokenizer(args)
        stop_ids = {tok.eos_token_id}
        for s in ("<|endoftext|>", "<|im_end|>", "<|eot_id|>"):
            try:
                sid = tok.convert_tokens_to_ids(s)
                if isinstance(sid, int) and sid >= 0:
                    stop_ids.add(sid)
            except Exception:
                pass
        while skill_tokens and skill_tokens[-1] in stop_ids:
            skill_tokens.pop()
            skill_logprobs.pop()
        # Thinking models (nemotron3, Qwen3, ...) prepend a <think>...</think>
        # reasoning chain; the real skill is what follows </think>. Strip it at the
        # token level so the dumped/KD skill is the actual roadmap, not the model's
        # self-talk. No-op when --sdpo-remove-thinking-from-demonstration is off or
        # the output has no </think>.
        if getattr(args, "sdpo_remove_thinking_from_demonstration", False):
            skill_tokens, skill_logprobs = _strip_think_tokens(tok, skill_tokens, skill_logprobs)
        if not skill_tokens:
            return None
        return tok.decode(skill_tokens), skill_tokens, skill_logprobs
    except Exception as e:
        logger.warning(f"self-skill generation failed ({e!r}); falling back to full trace.")
        return None


def _tokenizer(args: Namespace):
    # GenerateState is a process-wide singleton shared with rollout; reuse its
    # tokenizer so we tokenize the prefix exactly like the rollout engine does.
    from miles.rollout.sglang_rollout import GenerateState

    return GenerateState(args).tokenizer


# --------------------------------------------------------------------------- #
# teacher scoring
# --------------------------------------------------------------------------- #


def _teacher_url(args: Namespace) -> str:
    """Where to send teacher scoring requests.

    True SDPO is *self*-distillation: the teacher is the current policy
    conditioned on a correct prefix, i.e. the rollout engine itself (which is
    re-synced to the latest student weights every rollout). With
    ``--sdpo-self-teacher`` (default) we score against that engine, so no
    separate teacher server is needed. Set ``--no-sdpo-self-teacher`` to use a
    fixed external teacher at ``--rm-url`` instead.
    """
    if getattr(args, "sdpo_self_teacher", True):
        return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    return args.rm_url


async def _teacher_score(args: Namespace, input_ids: list[int], token_ids: list[int] | None):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if token_ids:
        payload["token_ids_logprob"] = token_ids
    # Use miles' shared HTTP client (http_utils.post): it retries on transient
    # failures (max_retries=60), shares one global connection pool with rollout
    # generation (timeout=None, sized to server concurrency), and can dispatch
    # via Ray. This is what rollout generation and the official OPD scorer use;
    # a hand-rolled aiohttp session with a hard connect timeout instead crashed
    # the whole job whenever the router was momentarily saturated.
    return await post(_teacher_url(args), payload)


def _trim_to_response(values: list[Any], response_length: int) -> list[Any]:
    """Drop SGLang's leading placeholder position, then keep the response span.

    This is the alignment guarantee: position i of the returned list is the
    teacher's prediction for response token i, exactly matching the student.
    """
    if values is None:
        raise ValueError("Teacher response is missing an expected meta_info logprob field.")
    trimmed = values[1:][-response_length:] if response_length > 0 else []
    if len(trimmed) != response_length:
        raise ValueError(
            f"Teacher/response alignment mismatch: got {len(trimmed)} positions, expected {response_length}."
        )
    return trimmed


def _entries_to_map(entries: Any) -> dict[int, float]:
    if not entries:
        return {}
    return {int(e[1]): float(e[0]) for e in entries if e is not None}


# --------------------------------------------------------------------------- #
# divergences
# --------------------------------------------------------------------------- #


def _distribution_divergence(p_s: Sequence[float], p_t: Sequence[float], mode: str) -> float:
    eps = 1e-12
    if mode == "reverse_kl":
        return sum(s * math.log((s + eps) / (t + eps)) for s, t in zip(p_s, p_t, strict=True))
    if mode == "forward_kl":
        return sum(t * math.log((t + eps) / (s + eps)) for s, t in zip(p_s, p_t, strict=True))
    if mode == "jeffrey":  # forward KL + reverse KL
        return sum(
            s * math.log((s + eps) / (t + eps)) + t * math.log((t + eps) / (s + eps))
            for s, t in zip(p_s, p_t, strict=True)
        )
    if mode == "jeffrey_jsd":  # forward KL + JSD (reverse-KL half swapped for JSD)
        total = 0.0
        for s, t in zip(p_s, p_t, strict=True):
            m = 0.5 * (s + t)
            fkl = t * math.log((t + eps) / (s + eps))
            jsd = 0.5 * s * math.log((s + eps) / (m + eps)) + 0.5 * t * math.log((t + eps) / (m + eps))
            total += fkl + jsd
        return total
    total = 0.0  # jsd
    for s, t in zip(p_s, p_t, strict=True):
        m = 0.5 * (s + t)
        total += 0.5 * s * math.log((s + eps) / (m + eps)) + 0.5 * t * math.log((t + eps) / (m + eps))
    return total


def _probs_with_tail(logps: Sequence[float]) -> list[float]:
    """Turn true (full-vocab-normalised) log-probs over a token subset into a
    proper distribution by appending one aggregated tail bucket for all the
    remaining vocabulary mass. No renormalisation: exp(logp) are real probs.
    """
    probs = [math.exp(lp) for lp in logps]
    tail = max(0.0, 1.0 - math.fsum(probs))
    return probs + [tail]


def _sampled_divergence(student_logp: float, teacher_logp: float, mode: str) -> float:
    """Per-token divergence when only the sampled token's log-prob is available.

    Treats the token as a 2-point (sampled vs. rest) Bernoulli: reverse/forward
    KL reduce to the log-prob gap on the sampled outcome; JSD uses the full
    2-point split for a bounded, symmetric estimate.
    """
    if mode == "reverse_kl":
        return student_logp - teacher_logp
    if mode == "forward_kl":
        return teacher_logp - student_logp
    p_s = min(max(math.exp(student_logp), 0.0), 1.0)  # jsd over {sampled, rest}
    p_t = min(max(math.exp(teacher_logp), 0.0), 1.0)
    return _distribution_divergence([p_s, 1.0 - p_s], [p_t, 1.0 - p_t], "jsd")


# --------------------------------------------------------------------------- #
# per-sample KL computation
# --------------------------------------------------------------------------- #


def _topk_divergences_np(
    student_maps: list[dict[int, float]],
    teacher_maps: list[dict[int, float]],
    response_tokens: list[int],
    divergence_mode: str,
) -> tuple[list[float], list[float], list[float]]:
    """Numpy-vectorized replacement for the per-token divergence loop.

    Numerically matches the old scalar path: exp of true (full-vocab-normalised)
    top-k logprobs, one aggregated tail bucket = max(0, 1 - sum(probs)), missing
    teacher id -> logprob -100, same eps=1e-12. Rows are the student's top-k ids
    per position (assumed uniform width; ragged rows are handled per-position).

    Runs on CPU only (no GPU tensors) so it never touches rollout-engine memory.
    Returns (divergences, student_sampled_logps, teacher_sampled_logps).
    """
    n = len(student_maps)
    eps = 1e-12
    NEG = -100.0

    # Batch-collect ragged rows via list comprehension (fast), then one np.array.
    # Positions with no student ids are marked to be zeroed afterwards.
    widths = [len(m) for m in student_maps]
    k = max(widths, default=0)
    if n == 0 or k == 0:
        return [0.0] * n, [], []

    uniform = all(w == k for w in widths)
    student_sampled_logps: list[float] = []
    teacher_sampled_logps: list[float] = []

    if uniform:
        s_rows = [[student_maps[i][t] for t in student_maps[i]] for i in range(n)]
        t_rows = [[teacher_maps[i].get(t, NEG) for t in student_maps[i]] for i in range(n)]
        s = np.asarray(s_rows, dtype=np.float64)
        t = np.asarray(t_rows, dtype=np.float64)
        p_s = np.exp(s)
        p_t = np.exp(t)
        tail_s = np.clip(1.0 - p_s.sum(1), 0.0, None)[:, None]
        tail_t = np.clip(1.0 - p_t.sum(1), 0.0, None)[:, None]
        p_s = np.concatenate([p_s, tail_s], axis=1)
        p_t = np.concatenate([p_t, tail_t], axis=1)
        if divergence_mode == "reverse_kl":
            div = (p_s * np.log((p_s + eps) / (p_t + eps))).sum(1)
        elif divergence_mode == "forward_kl":
            div = (p_t * np.log((p_t + eps) / (p_s + eps))).sum(1)
        else:  # jsd
            m = 0.5 * (p_s + p_t)
            div = (0.5 * p_s * np.log((p_s + eps) / (m + eps)) + 0.5 * p_t * np.log((p_t + eps) / (m + eps))).sum(1)
        divergences = div.tolist()
    else:
        # Rare ragged case (some positions have < k ids): fall back per-position,
        # still vectorized within each position.
        divergences = []
        for i in range(n):
            sm = student_maps[i]
            if not sm:
                divergences.append(0.0)
                continue
            tm = teacher_maps[i]
            ids = list(sm.keys())
            p_s = np.exp(np.asarray([sm[t] for t in ids], dtype=np.float64))
            p_t = np.exp(np.asarray([tm.get(t, NEG) for t in ids], dtype=np.float64))
            p_s = np.append(p_s, max(0.0, 1.0 - p_s.sum()))
            p_t = np.append(p_t, max(0.0, 1.0 - p_t.sum()))
            if divergence_mode == "reverse_kl":
                divergences.append(float((p_s * np.log((p_s + eps) / (p_t + eps))).sum()))
            elif divergence_mode == "forward_kl":
                divergences.append(float((p_t * np.log((p_t + eps) / (p_s + eps))).sum()))
            else:
                m = 0.5 * (p_s + p_t)
                divergences.append(
                    float(
                        (0.5 * p_s * np.log((p_s + eps) / (m + eps)) + 0.5 * p_t * np.log((p_t + eps) / (m + eps))).sum()
                    )
                )

    # Sampled-token diagnostics (cheap scalar gather).
    for i in range(n):
        sm = student_maps[i]
        tok = response_tokens[i]
        if tok in sm:
            student_sampled_logps.append(sm[tok])
            teacher_sampled_logps.append(teacher_maps[i].get(tok, NEG))

    return divergences, student_sampled_logps, teacher_sampled_logps


async def _compute_kl_for_sample(
    args: Namespace,
    sample: Sample,
    prefix_sample: Sample,
    logprob_mode: str,
    divergence_mode: str,
) -> torch.Tensor:
    n = sample.response_length
    prompt_tokens = sample.tokens[: _prompt_len(sample)]
    response_tokens = _response_tokens(sample)

    # Format the correct peer solution as a prefix and insert it between the
    # prompt and this trace's response. Response stays at the tail -> aligned.
    # NOTE: this legacy sglang-teacher path splices the solution at the
    # prompt|response boundary (inside the assistant turn). The active megatron
    # path (sdpo_group_reward) instead rebuilds the teacher prompt with the
    # solution in the USER turn, matching lasgroup/SDPO. If this path is revived,
    # port _build_teacher_prompt_str here too.
    _t = time.perf_counter()
    prefix_text = _render_prefix(prefix_sample.response)
    prefix_tokens = _tokenizer(args).encode(prefix_text, add_special_tokens=False)
    teacher_input = prompt_tokens + prefix_tokens + response_tokens
    _sdpo_timing["tokenize"] += time.perf_counter() - _t

    if logprob_mode == "sampled":
        student_logps = sample.rollout_log_probs
        if student_logps is None or len(student_logps) != n:
            raise ValueError(
                f"sampled mode needs rollout_log_probs of length {n}, got "
                f"{None if student_logps is None else len(student_logps)}."
            )
        teacher = await _teacher_score(args, teacher_input, token_ids=None)
        teacher_entries = _trim_to_response(teacher["meta_info"]["input_token_logprobs"], n)
        divergences = []
        for i in range(n):
            # Alignment guarantee: the teacher's token at this position IS response[i].
            if int(teacher_entries[i][1]) != response_tokens[i]:
                raise ValueError(
                    f"Token misalignment at position {i}: teacher={teacher_entries[i][1]}, "
                    f"student={response_tokens[i]}."
                )
            divergences.append(_sampled_divergence(student_logps[i], float(teacher_entries[i][0]), divergence_mode))
        return torch.tensor(divergences, dtype=torch.float32)

    # topk: per-position distribution over the student's top-k tokens + a tail bucket.
    raw = sample.metadata.get("opd_student_top_logprobs")
    if raw is None:
        raise ValueError("topk mode needs student top-k logprobs; set --opd-log-prob-top-k > 0 (e.g. 128).")
    _t = time.perf_counter()
    student_maps = [_entries_to_map(pos) for pos in (raw[-n:] if n > 0 else [])]
    if len(student_maps) != n:
        raise ValueError(f"Student top-k length mismatch: got {len(student_maps)}, expected {n}.")
    # Query the teacher for exactly the student's top-k token ids at each position.
    union_ids = sorted({tid for pos in student_maps for tid in pos})
    _sdpo_timing["student_maps"] += time.perf_counter() - _t

    _t = time.perf_counter()
    teacher = await _teacher_score(args, teacher_input, token_ids=union_ids)
    _sdpo_timing["teacher_http"] += time.perf_counter() - _t

    _t = time.perf_counter()
    teacher_maps = [
        _entries_to_map(pos) for pos in _trim_to_response(teacher["meta_info"]["input_token_ids_logprobs"], n)
    ]
    _sdpo_timing["teacher_maps"] += time.perf_counter() - _t

    # Numpy-vectorized per-token divergence, run in a worker thread so this
    # CPU-bound work does not block the event loop (other groups' generation and
    # scoring keep progressing, keeping the GPUs busy). Replaces a per-token
    # Python exp/log loop that took ~0.9s per 16k trace and stalled everything.
    _t = time.perf_counter()
    divergences, student_sampled_logps, teacher_sampled_logps = await asyncio.to_thread(
        _topk_divergences_np, student_maps, teacher_maps, response_tokens, divergence_mode
    )
    _sdpo_timing["divergence"] += time.perf_counter() - _t

    # Stash per-sample scalar diagnostics for rollout logging (see
    # _compute_metrics_from_samples). Guarded to no-op if nothing was collected.
    if student_sampled_logps:
        s_mean = sum(student_sampled_logps) / len(student_sampled_logps)
        t_mean = sum(teacher_sampled_logps) / len(teacher_sampled_logps)
        if isinstance(sample.metadata, dict):
            sample.metadata["sdpo_student_logp_mean"] = s_mean
            sample.metadata["sdpo_teacher_logp_mean"] = t_mean
            sample.metadata["sdpo_logp_diff_mean"] = s_mean - t_mean

    return torch.tensor(divergences, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# entry point: group-level async reward model
# --------------------------------------------------------------------------- #


async def sdpo_group_reward(args: Namespace, group: list[Sample], **kwargs: Any) -> list[float]:
    """Group RM: returns the task reward per trace and, as a side effect, writes
    the per-token SDPO divergence into ``sample.opd_reverse_kl``.
    """
    logprob_mode = _logprob_mode(args)
    divergence_mode = _divergence_mode(args)

    # Correctness is always needed to choose which traces can serve as a correct
    # peer prefix, regardless of what reward we return to the estimator. With
    # --sdpo-judge this is an LLM-as-judge grade (open-ended answers, defeats the
    # MCQ letter-guess hack); otherwise deterministic matching.
    correctness = await _grade_group(args, group)
    correct_indices = [i for i, ok in enumerate(correctness) if ok]

    # Log the TRUE task success and perplexity per trace on metadata (see
    # _compute_metrics_from_samples). Under pure distill the returned reward is 0,
    # so success rate would otherwise be invisible; stash it here so it survives.
    for ok, s in zip(correctness, group, strict=True):
        if not isinstance(s.metadata, dict):
            continue
        s.metadata["sdpo_correct"] = 1.0 if ok else 0.0
        # PPL of the sampled response = exp(mean negative student log-prob).
        logps = s.rollout_log_probs
        if logps:
            nll = -sum(logps) / len(logps)
            s.metadata["sdpo_ppl"] = math.exp(min(nll, 20.0))  # clamp to avoid overflow

    # Pure distillation (default): return 0 task reward so the GRPO advantage is 0
    # and the training target is exactly -opd_kl_coef * divergence. Otherwise keep
    # the mixed GRPO(task reward) + distillation target.
    if getattr(args, "sdpo_pure_distill", True):
        rewards = [0.0 for _ in group]
    else:
        rewards = [1.0 if ok else 0.0 for ok in correctness]

    # Need >= 1 correct trace to have a valid peer prefix (matching lasgroup/SDPO:
    # _get_solution returns None only when the candidate pool is empty after self-
    # exclusion, not when there is exactly 1 correct trace). A single correct trace
    # can still serve as prefix for ALL incorrect traces in the group; the correct
    # trace itself gets an empty prefix (self-excluded -> empty pool -> no prefix).
    enable_kl = len(correct_indices) >= 1

    if getattr(args, "sdpo_teacher_backend", "sglang") == "megatron":
        # Megatron teacher path: DON'T score here. Just pick a correct peer and
        # stash its rendered+tokenized prefix on the sample. The training side
        # (megatron actor) then forwards prompt+prefix+response with the CURRENT
        # policy weights (self-teacher) to get teacher log-probs — a batched,
        # CUDA-graph'd forward, ~50x faster than SGLang eager full-seq-logprob
        # scoring. opd_reverse_kl is computed on the training side (opd.py).
        tok = _tokenizer(args)
        gen_suffix = _gen_prompt_suffix(tok)
        remove_thinking = getattr(args, "sdpo_remove_thinking_from_demonstration", False)
        condense = getattr(args, "sdpo_trace_condense", False)
        self_skill = getattr(args, "sdpo_self_skill", False)
        skill_kd = self_skill and getattr(args, "sdpo_skill_kd", False)
        skill_kd_mode = getattr(args, "sdpo_skill_kd_mode", "self-success")
        skill_source = getattr(args, "sdpo_skill_source", "correct")
        # --sdpo-self-skill (on-policy, trainable) and --sdpo-trace-condense (external
        # LLM) both produce a skill prefix; running both is ambiguous.
        assert not (self_skill and condense), "use only one of --sdpo-self-skill / --sdpo-trace-condense"
        # pitfall-condense (and the pitfall half of 'both') distils FAILED traces, so
        # skill-source must cover them. 'both' additionally does self-success on correct
        # traces, so it wants BOTH flavors -> require skill-source all.
        assert not (skill_kd and skill_kd_mode == "pitfall-condense" and skill_source not in ("incorrect", "all")), (
            "--sdpo-skill-kd-mode pitfall-condense requires --sdpo-skill-source incorrect|all"
        )
        assert not (skill_kd and skill_kd_mode == "both" and skill_source != "all"), (
            "--sdpo-skill-kd-mode both trains correct (self-success) AND failed "
            "(pitfall-condense) traces, so it requires --sdpo-skill-source all"
        )

        response_prefix = getattr(args, "sdpo_response_prefix", "trace")
        pitfall_backend = getattr(args, "sdpo_pitfall_summary_backend", "self")
        # Pitfall injection is active when self-skill distils failed traces. In that
        # mode the group's common failure lessons are spliced into FAILED traces'
        # teacher prefix (and a failed trace with no correct peer still gets a prefix
        # made of just those lessons).
        pitfall_active = self_skill and skill_source in ("incorrect", "all")

        # Pass 1: pick each trace's correct peer (self-excluded). prefix_text is the
        # peer's solution — either the full response, or (with --sdpo-trace-condense)
        # its distilled skill. peer_by_idx remembers the chosen peer so
        # --sdpo-response-prefix skill can later swap in that peer's skill. With
        # pitfall injection, FAILED traces that have no correct peer still enter
        # prefix_text_by_idx (empty base) so the shared pitfalls can be spliced in.
        prefix_text_by_idx: dict[int, str] = {}
        peer_by_idx: dict[int, int] = {}
        for i, sample in enumerate(group):
            if not isinstance(sample.metadata, dict):
                continue
            if sample.response_length == 0 or not enable_kl:
                sample.metadata["sdpo_teacher_prompt_tokens"] = []
                continue
            peers = [j for j in correct_indices if j != i]
            if not peers:
                # No correct peer. A FAILED trace under pitfall injection still gets a
                # prefix (base empty; shared pitfalls appended in pass 2). Everyone
                # else gets no prefix.
                self_ok = bool(correctness[i]) if i < len(correctness) else False
                if pitfall_active and not self_ok:
                    prefix_text_by_idx[i] = ""
                else:
                    sample.metadata["sdpo_teacher_prompt_tokens"] = []
                continue
            peer_j = random.choice(peers)
            peer_by_idx[i] = peer_j
            prefix_text_by_idx[i] = group[peer_j].response

        # Optional: the CURRENT policy self-generates a skill during rollout from a
        # trace's OWN response, and (for skill-KD) we run a second SDPO on the skill
        # tokens. --sdpo-skill-source gates WHICH traces get a skill; it does NOT
        # change the response teacher prefix (still a correct peer, above).
        if self_skill:

            def _problem_of(j: int) -> str:
                md = group[j].metadata if isinstance(group[j].metadata, dict) else {}
                q = md.get("question")
                if q:
                    return str(q)
                p = group[j].prompt
                return p if isinstance(p, str) else str(p)

            def _skill_eligible(i: int) -> bool:
                # env_feedback is a placeholder for a future env-feedback pipeline.
                if group[i].response_length == 0:
                    return False
                self_ok = bool(correctness[i]) if i < len(correctness) else False
                if skill_source == "correct":
                    return self_ok
                if skill_source == "incorrect":
                    return not self_ok
                if skill_source == "env_feedback":
                    return False  # no env-feedback traces yet
                return True  # "all"

            skill_idxs = [i for i in range(len(group)) if isinstance(group[i].metadata, dict) and _skill_eligible(i)]

            async def _gen_one(i: int):
                # Distill the trace's OWN response into a skill. For a correct trace
                # this is "extract the transferable procedure"; for an incorrect one
                # it flips to "diagnose the error -> pitfall warnings", tailored by WHY
                # it failed (truncated | format | wrong). The ground-truth answer is
                # passed so the model can localize where the attempt went wrong.
                problem = _problem_of(i)
                self_ok = bool(correctness[i]) if i < len(correctness) else False
                fkind = "wrong" if self_ok else _failure_kind(args, group[i])
                gen_prompt_ids = _skill_gen_prompt_ids(
                    args,
                    tok,
                    problem,
                    group[i].response,
                    correct=self_ok,
                    ground_truth=(group[i].label or "") if not self_ok else "",
                    failure_kind=fkind,
                )
                res = await _self_generate_skill(args, gen_prompt_ids)
                return i, gen_prompt_ids, res

            gen_results = await asyncio.gather(*(_gen_one(i) for i in skill_idxs))
            for i, gen_prompt_ids, res in gen_results:
                if res is None:
                    continue
                skill_text, skill_tokens, skill_logprobs = res
                md = group[i].metadata
                md["sdpo_skill"] = skill_text
                # Preserve the per-trace pitfall (failed traces only) under a stable
                # key: the pitfall-condense pass (⑤) later overwrites sdpo_skill with a
                # problem-only prediction, but the group-pitfall summary (stage 2) and
                # the ⑤ teacher's privileged info both need the ORIGINAL per-trace
                # failure pitfalls.
                self_ok_i = bool(correctness[i]) if i < len(correctness) else False
                if not self_ok_i:
                    md["sdpo_trace_pitfall"] = skill_text
                # rollout-side skill metrics: length + perplexity (from the skill's
                # own rollout logprobs). Surfaced as skill/* in _compute_metrics_from_samples.
                md["sdpo_skill_len"] = float(len(skill_tokens))
                if skill_logprobs:
                    _nll = -sum(skill_logprobs) / len(skill_logprobs)
                    md["sdpo_skill_ppl"] = math.exp(min(_nll, 20.0))
                # The skill's own tokens, the skill-gen prompt, and its rollout
                # logprobs exist for EVERY generated skill, independent of skill-KD.
                # Stash them unconditionally so the dump (skill_student_prompt_text /
                # skill_text) shows the skill and the prompt that produced it even
                # when skill-KD is off. The skill-KD *training* path is gated on
                # --sdpo-skill-kd at its call site (actor._append_sdpo_skill_samples),
                # not on these keys, so populating them here is dump-only and does not
                # turn skill-KD on.
                md["sdpo_skill_tokens"] = skill_tokens
                md["sdpo_skill_prompt_tokens"] = gen_prompt_ids
                md["sdpo_skill_rollout_logprobs"] = skill_logprobs
                # Skill-KD teacher hint (see DESIGN_self_skill.md), KD-only:
                #  self-success: teacher = skill-gen prompt + the sample's OWN trace
                #                as hint. (skill-source already restricts to correct.)
                #  problem-only: teacher = skill-gen prompt, NO hint.
                #  pitfall-condense: handled in a dedicated pass below (student is
                #                regenerated from a problem-only prompt).
                #  both: correct traces take the self-success teacher here; failed
                #                traces are handled by the pitfall-condense pass below.
                self_ok_kd = bool(correctness[i]) if i < len(correctness) else False
                use_self_success = skill_kd_mode == "self-success" or (skill_kd_mode == "both" and self_ok_kd)
                if skill_kd and (use_self_success or skill_kd_mode == "problem-only"):
                    if use_self_success:
                        gen_prompt_str = tok.decode(gen_prompt_ids)
                        skill_teacher_str = _build_teacher_prompt_str(
                            gen_prompt_str, gen_suffix, group[i].response, remove_thinking=remove_thinking
                        )
                        md["sdpo_skill_teacher_prompt_tokens"] = tok.encode(
                            skill_teacher_str, add_special_tokens=False
                        )
                    else:  # problem-only: no hint -> teacher context == student context
                        md["sdpo_skill_teacher_prompt_tokens"] = list(gen_prompt_ids)

            # pitfall-condense skill-KD (⑤): a SEPARATE skill OPD on failed traces.
            #  student = predict pitfalls from the PROBLEM ONLY (no attempt, no info);
            #  teacher = same problem-only prompt + the group's per-trace failure
            #            skills spliced in as privileged info.
            #  KD target = the student's own problem-only pitfall generation.
            # This regenerates the skill under the problem-only student context so the
            # KD'd tokens match that context (the earlier per-trace pitfalls were
            # generated with the failed attempt in context and are reused only as the
            # teacher's privileged info).
            if skill_kd and skill_kd_mode in ("pitfall-condense", "both"):
                # 'both' also runs self-success on correct traces (handled above); here
                # we only (re)build the FAILED traces' skill-KD via pitfall-condense.
                failed_idxs = [
                    i for i in skill_idxs
                    if not (bool(correctness[i]) if i < len(correctness) else False)
                    and isinstance(group[i].metadata, dict)
                    and (group[i].metadata.get("sdpo_trace_pitfall") or "").strip()
                ]
                # Privileged failure info = all failed traces' per-trace pitfalls.
                failure_info = "\n\n".join(
                    (group[j].metadata.get("sdpo_trace_pitfall") or "").strip() for j in failed_idxs
                )

                async def _gen_predict(i: int):
                    stu_text = tok.apply_chat_template(
                        [
                            {"role": "system", "content": _PITFALL_PREDICT_SYSTEM},
                            {"role": "user", "content": _pitfall_predict_user_prompt(_problem_of(i))},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    stu_ids = tok.encode(stu_text, add_special_tokens=False)
                    res2 = await _self_generate_skill(args, stu_ids)
                    return i, stu_ids, res2

                predict_results = await asyncio.gather(*(_gen_predict(i) for i in failed_idxs))
                for i, stu_ids, res2 in predict_results:
                    if res2 is None:
                        continue
                    p_text, p_tokens, p_logprobs = res2
                    md = group[i].metadata
                    # Overwrite the skill-KD payload with the problem-only student and
                    # the failure-informed teacher. The KD student/target is now the
                    # problem-only pitfall prediction.
                    md["sdpo_skill"] = p_text
                    md["sdpo_skill_len"] = float(len(p_tokens))
                    if p_logprobs:
                        _nll = -sum(p_logprobs) / len(p_logprobs)
                        md["sdpo_skill_ppl"] = math.exp(min(_nll, 20.0))
                    md["sdpo_skill_tokens"] = p_tokens
                    md["sdpo_skill_prompt_tokens"] = stu_ids
                    md["sdpo_skill_rollout_logprobs"] = p_logprobs
                    stu_prompt_str = tok.decode(stu_ids)
                    teacher_str = _build_failure_teacher_prompt_str(stu_prompt_str, gen_suffix, failure_info)
                    md["sdpo_skill_teacher_prompt_tokens"] = tok.encode(teacher_str, add_special_tokens=False)

        # Optional: distill each chosen peer trace into a transferable SKILL and use
        # that as the prefix instead of the full trace (SkillOpt / trace_condense).
        if condense and prefix_text_by_idx:
            idxs = list(prefix_text_by_idx.keys())

            def _problem_of(j: int) -> str:
                md = group[j].metadata if isinstance(group[j].metadata, dict) else {}
                q = md.get("question")
                if q:
                    return str(q)
                p = group[j].prompt
                return p if isinstance(p, str) else str(p)

            pairs = [(_problem_of(i), _strip_response_eos(prefix_text_by_idx[i])) for i in idxs]
            skills = await _condense_solutions(args, pairs)
            for i, skill in zip(idxs, skills):
                full_trace = prefix_text_by_idx[i]
                prefix_text_by_idx[i] = skill
                # Record the distilled skill (and the trace it replaced) so the
                # training-side dump can log it. condensed=False means the LLM
                # failed and we fell back to the full trace.
                if isinstance(group[i].metadata, dict):
                    group[i].metadata["sdpo_skill"] = skill
                    group[i].metadata["sdpo_skill_condensed"] = skill != full_trace

        # --sdpo-response-prefix skill: swap the response teacher prefix from the
        # peer's full trace to that peer's self-generated skill (fall back to the
        # trace if the peer has no skill). Requires self_skill (peers' skills exist).
        if response_prefix == "skill" and self_skill:
            for i in list(prefix_text_by_idx.keys()):
                peer_j = peer_by_idx.get(i)
                peer_md = group[peer_j].metadata if (peer_j is not None and isinstance(group[peer_j].metadata, dict)) else {}
                peer_skill = peer_md.get("sdpo_skill")
                # 1.0 if the response prefix used the peer's skill, 0.0 if it fell
                # back to the full trace (peer had no skill). Aggregated into the
                # skill/ panel as the response-prefix-is-skill fraction.
                if isinstance(group[i].metadata, dict):
                    group[i].metadata["sdpo_response_prefix_is_skill"] = 1.0 if peer_skill else 0.0
                if peer_skill:
                    prefix_text_by_idx[i] = peer_skill

        # Group-aggregated pitfalls (two stages), when self-skill covers INCORRECT
        # traces (--sdpo-skill-source incorrect|all):
        #   Stage 1 (above): every failed trace produced its OWN pitfall warnings.
        #   Stage 2 (here): feed ALL of those per-trace pitfalls back to the skill
        #     generator (self policy or external LLM, per --sdpo-pitfall-summary-
        #     backend) and synthesize the COMMON recurring failure lessons into ONE
        #     short shared list for the group.
        # The shared list — not the raw concatenation — is spliced ONLY into the
        # FAILED traces' teacher prefix (correct traces keep a clean correct-peer
        # prefix). A model that failed cannot rewrite the solution (that would
        # hallucinate a bad KD target), but it CAN flag concrete errors, so failed
        # traces contribute warnings, never solutions.
        group_pitfalls = ""
        if pitfall_active:
            def _problem_text(j: int) -> str:
                md_j = group[j].metadata if isinstance(group[j].metadata, dict) else {}
                q = md_j.get("question")
                if q:
                    return str(q)
                p = group[j].prompt
                return p if isinstance(p, str) else str(p)

            per_trace_pitfalls = []
            first_failed = None
            for i in range(len(group)):
                self_ok = bool(correctness[i]) if i < len(correctness) else False
                if self_ok:
                    continue
                md = group[i].metadata if isinstance(group[i].metadata, dict) else {}
                sk = md.get("sdpo_trace_pitfall")
                if sk and sk.strip():
                    per_trace_pitfalls.append(sk.strip())
                    if first_failed is None:
                        first_failed = i
            if len(per_trace_pitfalls) == 1:
                # Only one failed trace -> nothing to synthesize; use it directly.
                group_pitfalls = per_trace_pitfalls[0]
            elif per_trace_pitfalls:
                # Same problem across the group; use the first failed trace's text.
                problem = _problem_text(first_failed)
                summary = await _generate_skill_text(
                    args,
                    _PITFALL_SUMMARY_SYSTEM,
                    _pitfall_summary_user_prompt(problem, per_trace_pitfalls),
                    pitfall_backend,
                )
                group_pitfalls = summary.strip() if summary and summary.strip() else "\n\n".join(per_trace_pitfalls)

        # Pass 2: build the teacher prompt (peer solution/skill spliced into the USER
        # turn, before the assistant marker) and tokenize. The shared pitfalls go ONLY
        # into failed traces' prefix; correct traces keep the clean correct-peer prefix.
        for i, sample in enumerate(group):
            if i not in prefix_text_by_idx:
                continue
            self_ok = bool(correctness[i]) if i < len(correctness) else False
            pitfalls_for_i = group_pitfalls if (pitfall_active and not self_ok) else ""
            student_prompt = sample.prompt if isinstance(sample.prompt, str) else ""
            teacher_prompt_str = _build_teacher_prompt_str(
                student_prompt, gen_suffix, prefix_text_by_idx[i], remove_thinking=remove_thinking,
                pitfalls=pitfalls_for_i,
            )
            teacher_prompt_ids = tok.encode(teacher_prompt_str, add_special_tokens=False)
            # Training side builds teacher seq = teacher_prompt_ids + response_ids,
            # response kept at the tail so response-span outputs stay aligned.
            sample.metadata["sdpo_teacher_prompt_tokens"] = teacher_prompt_ids
            if isinstance(sample.metadata, dict):
                sample.metadata["sdpo_group_pitfalls"] = pitfalls_for_i
        return rewards

    # SGLang teacher path (original): score each trace against the rollout engine
    # over HTTP and write per-token opd_reverse_kl here. Concurrent to spread load.
    async def _score(i: int, sample: Sample) -> torch.Tensor:
        n = sample.response_length
        if n == 0 or not enable_kl:
            return torch.zeros((n,), dtype=torch.float32)
        peers = [j for j in correct_indices if j != i]
        prefix_sample = group[random.choice(peers)]
        return await _compute_kl_for_sample(args, sample, prefix_sample, logprob_mode, divergence_mode)

    global _sdpo_calls
    _wall = time.perf_counter()
    kls = await asyncio.gather(*(_score(i, s) for i, s in enumerate(group)))
    _wall = time.perf_counter() - _wall
    for sample, kl in zip(group, kls, strict=True):
        sample.opd_reverse_kl = kl

    # Log per-phase timing every 8 groups so we can see where a rollout's scoring
    # time actually goes (HTTP wait vs CPU prep vs divergence). Sums are across
    # concurrent traces, so compare RATIOS, not absolute vs wall.
    _sdpo_calls += 1
    if _sdpo_calls % 8 == 0:
        t = _sdpo_timing
        logger.info(
            "SDPO timing (cumulative over %d groups): wall_last_group=%.1fs | "
            "teacher_http=%.1fs student_maps=%.1fs teacher_maps=%.1fs tokenize=%.1fs divergence=%.1fs",
            _sdpo_calls,
            _wall,
            t["teacher_http"],
            t["student_maps"],
            t["teacher_maps"],
            t["tokenize"],
            t["divergence"],
        )

    return rewards


async def sdpo_eval_reward(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    """Per-sample eval RM for SDPO (--eval-custom-rm-path).

    Eval measures pass@1 and never uses the distillation signal, so it just needs
    the task reward. Uses the same grading as the group RM: the LLM judge when
    --sdpo-judge is set (open-ended answers), else deterministic matching.
    """
    if getattr(args, "sdpo_judge", False) and (sample.response or "").strip():
        # Eval fans out one sdpo_eval_reward coroutine per sample via asyncio.gather
        # upstream, so honor the SAME global concurrency cap to avoid flooding the
        # gateway during large evals.
        async with _judge_semaphore(args):
            ok = await _llm_judge_correct(args, sample)
    elif getattr(args, "sdpo_grader", "mcq") == "dapo":
        # Math eval (AIME-2025 = integers, Minerva Math = LaTeX). Use the general
        # math grader, which handles both, rather than the integer-only DAPO grader
        # used for training-trace correctness (Minerva answers are not integers).
        ok = bool(grade_answer_verl(sample.response or "", (sample.label or "").strip()))
    else:
        ok = _is_correct(sample, args)
    return 1.0 if ok else 0.0
