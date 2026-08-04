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

Correctness grading (deterministic matching, LLM-as-judge, code/search graders)
lives in ``reward.py`` next to this file and is imported below.
"""

import asyncio
import json
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

from miles.utils.http_utils import post  # miles' shared HTTP client: retries + shared pool
from miles.utils.types import Sample

# Correctness / grading subsystem, split out for size -- see reward.py's module
# docstring. Re-exported here (rather than only used internally) so external
# call sites that historically did `from examples.SDPO.sdpo import _grade_group`
# (e.g. examples/EPO/epo.py) keep working unchanged.
from examples.SDPO.reward import (
    _extract_answer,
    _grade_group,
    _grade_one_code,
    _grade_one_search,
    _is_correct,
    _judge_semaphore,
    _llm_judge_correct,
    _sample_domain,
)

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

# Skill-KD 'self-success' teacher privileged hint. The skill-gen STUDENT prompt
# already contains the full WORKED SOLUTION (see _skill_user_prompt) -- the
# teacher's own trace, same text -- so re-splicing SOLUTION_TEMPLATE (the
# response-SDPO template) here would just repeat it verbatim with zero new
# information, and PREFIX_INSTRUCTION's "Correctly solve the original question"
# is flatly wrong for a task whose job is "write a skill", not "answer the
# math problem". The only genuine privileged signal a self-success teacher has
# over the student is the CONFIRMATION that this solution is correct -- so hint
# with just that, not a restated solution.
SKILL_SELF_SUCCESS_HINT = (
    "\n\n(This worked solution has been verified CORRECT. Distill the skill with "
    "full confidence.)\n\n"
)


def _strip_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from a response (for thinking models like Qwen3).
    Strips leading/trailing whitespace after removal so the peer solution stays clean."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return stripped.strip()


# --- multi-turn native-trace reframing for the teacher prefix --------------- #
# A native tool-calling peer trace carries ChatML turn boundaries
# (<|im_end|><|im_start|>role) between the assistant's tool call and the tool
# response. Spliced verbatim into the teacher's USER turn, those RAW boundary
# tokens make the teacher see a malformed nested conversation and partly learn
# to emit control-token garbage (_strip_response_eos only removes the TRAILING
# <|im_end|>, not the interior ones).
#
# Per the user's spec: ONLY reframe the <|im_start|>/<|im_end|> turn boundaries
# -- replace each with a short NLP marker at the corresponding position (the
# assistant turn -> "Round N reasoning and tool call:", the tool/user turn ->
# "Observation:"). Leave the INNER semantic tags (<think>, <tool_call>,
# <tool_response>) completely intact -- they are the content, not the control
# structure. No-op for traces without ChatML boundaries (single-turn), so plain
# single-turn SDPO is unchanged. Gated by --sdpo-reframe-multiturn-prefix.
_IM_SPLIT_RE = re.compile(r"<\|im_end\|>\s*<\|im_start\|>\s*(assistant|user|system)\b[ \t]*\n?")
_IM_ANY_RE = re.compile(r"<\|im_(?:start|end)\|>[ \t]*(?:assistant|user|system)?[ \t]*\n?")


def _reframe_multiturn_trace(text: str) -> str:
    """Replace ChatML <|im_*|> turn boundaries with simple per-round NLP markers,
    keeping <think>/<tool_call>/<tool_response> content verbatim. Returns text
    unchanged if it has no ChatML boundaries (single-turn no-op)."""
    if not text or "<|im_" not in text:
        return text

    # Split on each "<|im_end|><|im_start|>role" boundary, remembering the role
    # that OPENS each subsequent segment. The first segment is whatever the trace
    # started mid-turn on (the assistant's first reasoning/tool-call turn).
    segments: list[tuple[str, str]] = []  # (opening_role, segment_text)
    last_end = 0
    role_for_next = "assistant"  # trace begins inside the assistant's turn
    for m in _IM_SPLIT_RE.finditer(text):
        segments.append((role_for_next, text[last_end : m.start()]))
        role_for_next = m.group(1)
        last_end = m.end()
    segments.append((role_for_next, text[last_end:]))

    def _clean(seg: str) -> str:
        # Only strip stray/standalone <|im_*|> tokens (e.g. leading/trailing);
        # do NOT touch <think>/<tool_call>/<tool_response>.
        return _IM_ANY_RE.sub("", seg).strip()

    parts: list[str] = []
    round_no = 0
    for role, seg in segments:
        seg = _clean(seg)
        if not seg:
            continue
        if role == "assistant":
            round_no += 1
            parts.append(f"Round {round_no} reasoning and tool call:\n{seg}")
        else:  # user/tool turn = the observation carrying <tool_response>
            parts.append(f"Observation:\n{seg}")
    return "\n\n".join(parts) if parts else _clean(text)


def _tool_call_args_str(arguments) -> str:
    """Normalize a tool call's arguments to a JSON string (native path stores a
    JSON string; some paths store a dict)."""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False)
    except Exception:
        return str(arguments)


def _render_tool_call(name: str, arguments, grammar: str = "qwen25") -> str:
    """Render ONE tool call in the target model's native grammar. The model must
    LEARN to emit exactly this, so it must byte-match what the rollout produces:
      - qwen25 (Qwen3-4B): JSON object inside <tool_call> tags.
      - qwen3_coder (Qwen3.5-4B): XML <function=NAME><parameter=P>value tags.
    Parameterised so ONE codebase distils either model family."""
    if grammar == "qwen3_coder":
        # XML function/parameter form. arguments -> dict of parameters.
        args_str = _tool_call_args_str(arguments)
        try:
            params = json.loads(args_str) if isinstance(args_str, str) else (arguments or {})
        except Exception:
            params = {}
        if not isinstance(params, dict):
            params = {}
        lines = [f"<function={name}>"]
        for k, v in params.items():
            vs = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            lines.append(f"<parameter={k}>\n{vs}\n</parameter>")
        lines.append("</function>")
        return "<tool_call>\n" + "\n".join(lines) + "\n</tool_call>"
    # default qwen25: JSON-in-tags
    call_json = f'{{"name": "{name}", "arguments": {_tool_call_args_str(arguments)}}}'
    return f"<tool_call>\n{call_json}\n</tool_call>"


def _render_tool_response(obs: str, grammar: str = "qwen25") -> str:
    """Observation grammar. Both qwen families wrap the tool result in
    <tool_response>...</tool_response>, so this is shared, but kept as a hook in
    case a future model family differs."""
    return f"<tool_response>\n{obs}\n</tool_response>"


def _reframe_messages_to_prose(messages: list, remove_thinking: bool = False, grammar: str = "qwen25") -> str:
    """NLP-ize a peer trace from its STRUCTURED message dict (the live record
    multi_turn.generate keeps on metadata["messages"]) into per-round text:

        Round N reasoning and tool call:
        <assistant reasoning>
        <tool_call> ...native tool-call grammar... </tool_call>

        Observation:
        <tool_response> <tool result> </tool_response>

    CRITICAL: only the ChatML TURN BOUNDARIES (<|im_start|>/<|im_end|>) are
    NLP-ized into "Round N.../Observation:" markers. The INNER tool grammar --
    the <tool_call>/<tool_response> blocks -- is rendered in the TARGET MODEL's
    native syntax (grammar arg: qwen25 JSON-in-tags for Qwen3-4B, qwen3_coder XML
    for Qwen3.5-4B), because that grammar is exactly what the student must LEARN
    to emit and it must byte-match the rollout. An earlier version rendered tool
    calls as prose ("calls web_search({...})"), teaching the student to NARRATE
    tool use instead of EMIT it -> tool-call rate collapsed 86%->16%. The tool
    tags are plain text tokens (NOT ChatML control tokens like <|im_start|>), so
    they splice safely into the teacher's USER turn without breaking student-
    response token alignment. Dict-native counterpart of _reframe_multiturn_trace."""
    parts: list[str] = []
    round_no = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content") or ""
        if role == "assistant":
            round_no += 1
            # multi_turn.generate splits Qwen3/3.5's raw output into
            # reasoning_content + content (the chat template bakes the OPENING
            # <think> into the generation prefix, so raw text only has the
            # closing tag -- see _split_reasoning_content). Re-wrap it here so
            # remove_thinking / _strip_thinking_blocks behave exactly as they
            # did when reasoning still lived inline in `content`.
            reasoning = (m.get("reasoning_content") or "").strip()
            if reasoning:
                content = f"<think>\n{reasoning}\n</think>\n\n{content}"
            if remove_thinking:
                content = _strip_thinking_blocks(content)
            seg = content.strip()
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "tool")
                seg = (seg + "\n" + _render_tool_call(name, fn.get("arguments", ""), grammar)).strip()
            if seg:
                parts.append(f"Round {round_no} reasoning and tool call:\n{seg}")
        elif role == "tool":
            obs = content.strip()
            if obs:
                parts.append(f"Observation:\n{_render_tool_response(obs, grammar)}")
        # system/user turns are the shared problem context, not part of the peer
        # trace prose -- skip (the student prompt already carries them).
    return "\n\n".join(parts)


def _render_prefix(peer_response: str, remove_thinking: bool = False, pitfalls: str = "") -> str:
    content = _strip_thinking_blocks(peer_response) if remove_thinking else peer_response
    # Skip the "Correct solution:" section entirely when there is no base solution
    # (e.g. a failed trace in an all-wrong group gets a pitfalls-only prefix); an
    # empty "Correct solution:" heading would mislead the teacher.
    section = SOLUTION_TEMPLATE.format(successful_previous_attempt=content) if content and content.strip() else ""
    if pitfalls and pitfalls.strip():
        section += PITFALLS_TEMPLATE.format(pitfalls=pitfalls.strip())
    return section + PREFIX_INSTRUCTION


def _gen_prompt_suffix(tok, chat_template_kwargs: dict | None = None) -> str:
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

    chat_template_kwargs (e.g. {"enable_thinking": False}) MUST match the run's,
    because the suffix DIFFERS by thinking mode: Qwen3.5 no-think appends
    '...assistant\\n<think>\\n\\n</think>\\n\\n', but the default (thinking) appends
    '...assistant\\n<think>\\n'. The student prompt was rendered with the run's
    kwargs, so the suffix must be derived the same way or the splice won't find it.
    """
    # Sentinel with NO surrounding spaces: some templates (Qwen3.5) strip leading/
    # trailing whitespace from user content, so a space-padded sentinel wouldn't
    # be found verbatim in the render (observed: " SDPO_SENTINEL " -> "SDPO_SENTINEL",
    # split returns "", breaking the teacher-prefix splice).
    sentinel = "SDPO_SENTINEL"
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": sentinel}], tokenize=False, add_generation_prompt=True,
        **(chat_template_kwargs or {}),
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


def _choose_peer(args: Namespace, group: list[Sample], peers: list[int]) -> int:
    """Pick one correct peer (already self-excluded by the caller) to serve as
    the teacher prefix. Uniform random by default; with --sdpo-prefer-tool-use-
    peer, prefer a peer that actually called a tool (sample.metadata
    ["tool_call_count"] > 0), falling back to uniform random over ALL peers
    when none did (or when no sample in the group carries that key at all --
    i.e. this is a no-op for non-agentic examples). See that flag's help text
    for why: uniform-random selection lets a no-tool-correct-trace majority
    silently become the KD teacher, teaching the student away from tool use
    over training even though task reward never penalizes it."""
    if not getattr(args, "sdpo_prefer_tool_use_peer", False):
        return random.choice(peers)
    tool_using_peers = [
        j
        for j in peers
        if isinstance(group[j].metadata, dict) and (group[j].metadata.get("tool_call_count") or 0) > 0
    ]
    return random.choice(tool_using_peers) if tool_using_peers else random.choice(peers)


def _build_teacher_prompt_str(student_prompt: str, gen_suffix: str, peer_response: str, remove_thinking: bool = False, pitfalls: str = "", reframe_multiturn: bool = False, peer_messages: list | None = None, grammar: str = "qwen25", max_prefix_chars: int = 0) -> str:
    """Insert the correct-peer solution + instruction into the USER turn of the
    student's (already chat-templated) prompt, before the assistant generation
    marker. Returns the full teacher prompt string (system + user+solution +
    assistant marker).

    The peer response is stripped of its trailing <|im_end|>/EOS first — it was a
    full generated turn, and leaving that marker in mid-user-turn would close the
    user turn early and corrupt the teacher prompt (the model would then see the
    solution as a separate malformed turn instead of context).

    reframe_multiturn (--sdpo-reframe-multiturn-prefix): for NATIVE multi-turn
    tool-calling peer traces, also strip the MID-trace control tokens
    (<tool_call>/<tool_response>/<|im_start|>/<|im_end|>) and re-template into
    clean per-round prose (see _reframe_multiturn_trace) — _strip_response_eos
    alone only removes the trailing marker, leaving the interior ones to pollute
    the teacher. No-op for single-turn traces.

    max_prefix_chars (--sdpo-max-prefix-chars, 0=off): cap the reframed prose to
    its LAST N chars before splicing. A peer trace with a long debugging loop
    (observed: code-domain traces up to ~65K chars) becomes the teacher prefix
    for every OTHER sample in its group; one such oversized sample can't be
    split across a dynamic-batch-size microbatch, so it OOMs the vocab-parallel
    forward alone. Keeping the TAIL (not head) preserves the final answer/
    conclusion, which matters more to a teacher-forced KD target than the
    early exploration."""
    # Prefer the dict-native prose when the peer's structured messages are
    # available (multi_turn.generate records them): rendered from the real
    # tool_calls field, can't mis-split on a stray marker. Fall back to the
    # raw-text reframe (legacy / when no messages recorded).
    if peer_messages:
        cleaned = _reframe_messages_to_prose(peer_messages, remove_thinking=remove_thinking, grammar=grammar)
    else:
        cleaned = _strip_response_eos(peer_response)
        if reframe_multiturn:
            cleaned = _reframe_multiturn_trace(cleaned)
    if max_prefix_chars and len(cleaned) > max_prefix_chars:
        cleaned = cleaned[-max_prefix_chars:]
    # remove_thinking already applied inside _reframe_messages_to_prose for the
    # dict path; _render_prefix re-applying it on already-clean prose is a no-op.
    solution_section = _render_prefix(cleaned, remove_thinking=remove_thinking and not peer_messages, pitfalls=pitfalls)
    if gen_suffix and gen_suffix in student_prompt:
        idx = student_prompt.rfind(gen_suffix)
        return student_prompt[:idx] + solution_section + student_prompt[idx:]
    # Fallback (unknown template): append at the end. Not ideal but never crashes.
    return student_prompt + solution_section


def _build_skill_self_success_teacher_prompt_str(student_prompt: str, gen_suffix: str) -> str:
    """Skill-KD 'self-success' teacher: the skill-gen prompt PLUS a privileged
    confirm-correct hint, inserted before the assistant marker (same insert
    point as _build_teacher_prompt_str/_build_failure_teacher_prompt_str).

    Does NOT reuse _build_teacher_prompt_str/_render_prefix (the response-SDPO
    template): that would re-splice SOLUTION_TEMPLATE's "Correct solution:
    {trace}" with the SAME worked solution already in the student prompt's own
    WORKED SOLUTION field (see _skill_user_prompt) -- a verbatim repeat, not
    new privileged info -- followed by PREFIX_INSTRUCTION's "Correctly solve
    the original question", which is simply the wrong instruction for a task
    whose job is "write a skill" (see SKILL_SELF_SUCCESS_HINT's docstring)."""
    if gen_suffix and gen_suffix in student_prompt:
        idx = student_prompt.rfind(gen_suffix)
        return student_prompt[:idx] + SKILL_SELF_SUCCESS_HINT + student_prompt[idx:]
    return student_prompt + SKILL_SELF_SUCCESS_HINT


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


# --------------------------------------------------------------------------- #
# Trace condensation / SkillOpt  (distill the correct peer trace into a SKILL)
# --------------------------------------------------------------------------- #
# When --sdpo-trace-condense is set, the correct-peer solution is first distilled
# into a short transferable SKILL (<=3 procedural bullets, no answer) by an LLM,
# and that skill — not the full trace — becomes the teacher prefix. Mirrors
# lasgroup/SDPO trace_condense (verl/trainer/ppo/trace_condense.py).

_SKILL_SYSTEM_PROMPT = (
    "You are given a CORRECT worked solution. Distill the transferable KNOW-HOW it "
    "used into a list of tiny, self-contained SKILLS — each a reusable knowledge/rule "
    "unit, NOT a step-by-step roadmap of this specific problem. Use this EXACT "
    "structured format, one block per distinct skill (1-3 blocks):\n\n"
    "[Knowledge/Rule]\n"
    "<a general principle / identity / method / theorem the solution relied on — "
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
    "attempt is WRONG. Do NOT solve the problem or write a correct solution — you "
    "only learn from the failure. Extract the SPECIFIC mistake(s) as concrete, "
    "reusable lessons in this EXACT structured format (one block per distinct "
    "mistake, 1-3 blocks total):\n\n"
    "[Error]\n"
    "<the specific wrong step/assumption the attempt made — concrete, not vague>\n"
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

# Second-stage aggregation: given the pitfalls distilled from EVERY failed trace in
# a group (each a small list of "avoid X" warnings), synthesize the COMMON failure
# lessons — the mistakes that recur across attempts on this problem — into one short
# shared list. This shared list (not the raw concatenation) is what gets spliced
# into the failed traces' teacher prefix, so the teacher sees a tight "here's how
# this group tends to fail" summary rather than a long noisy dump.
_PITFALL_SUMMARY_SYSTEM = (
    "You are given several sets of PITFALL LESSONS (each a list of [Error]/[Rule]/"
    "[Example] blocks) distilled from different failed attempts at the SAME problem. "
    "Synthesize the COMMON, recurring mistakes into one short shared list, merging "
    "duplicates and dropping one-off noise, KEEPING the same structured format.\n\n"
    "Output 1-3 blocks, each EXACTLY:\n"
    "[Error]\n<the specific recurring mistake>\n"
    "[Rule]\n<the general principle/identity/method that avoids it>\n"
    "[Example]\n<a tiny concrete worked instance, NOT this problem's answer>\n\n"
    "Hard constraints:\n"
    "- Use the literal [Error]/[Rule]/[Example] headers; keep each field SPECIFIC "
    "(no vague 'be careful' warnings).\n"
    "- Never state the final/ground-truth answer and never give a worked solution.\n"
    "- Output ONLY the [Error]/[Rule]/[Example] blocks, nothing else."
)


def _pitfall_summary_user_prompt(problem: str, pitfall_sets: list[str]) -> str:
    blocks = "\n\n".join(f"FAILED ATTEMPT {k + 1} PITFALLS:\n{p}" for k, p in enumerate(pitfall_sets))
    return (
        f"PROBLEM:\n{_clean_problem_for_skill(problem)}\n\n"
        f"{blocks}\n\n"
        "Synthesize the common recurring pitfalls into 1-3 [Error]/[Rule]/[Example] "
        "blocks (merge duplicates, drop one-off noise; no solution, no answer)."
    )


# --- pitfall-condense skill-KD (⑤): the skill's own OPD --------------------- #
# STUDENT (no privileged info): given ONLY the problem, predict the pitfalls a
# solver should avoid — a pure "foresee the traps" task with no failed attempt and
# no answer. TEACHER (privileged): the same problem-only prompt PLUS the group's
# actual per-trace failure skills spliced in, so it condenses what really went
# wrong. KD pulls the problem-only student toward the failure-informed teacher.
_PITFALL_PREDICT_SYSTEM = (
    "Given a problem (and NOTHING else — no attempt, no answer), predict the pitfalls "
    "a solver is most likely to fall into on this kind of problem. Output them as tiny "
    "self-contained skills in this EXACT format, one block per distinct pitfall "
    "(1-3 blocks):\n\n"
    "[Error]\n<the specific trap a solver is likely to fall into here — concrete>\n"
    "[Rule]\n<the general principle/method that avoids it>\n"
    "[Example]\n<a tiny concrete worked instance of the rule, NOT this problem's answer>\n\n"
    "Hard constraints:\n"
    "- Use the literal [Error]/[Rule]/[Example] headers for every block; keep each "
    "field SPECIFIC (no vague 'be careful' warnings).\n"
    "- Do NOT solve the problem or give its full method; only the traps + the rule "
    "that avoids each.\n"
    "- Never state a final answer.\n"
    "- Output ONLY the [Error]/[Rule]/[Example] blocks, nothing else."
)

# Label for the privileged failure info spliced into the pitfall-condense TEACHER
# turn (distinct from "Correct solution:" — these are observed FAILURES, not a
# solution). Reuses the {successful_previous_attempt} field name for _render_prefix
# compatibility but is only ever fed the concatenated failure skills.
FAILURES_TEMPLATE = "\n\nObserved failed-attempt pitfalls (privileged, do not reveal):\n\n{successful_previous_attempt}"


def _pitfall_predict_user_prompt(problem: str) -> str:
    return (
        f"PROBLEM:\n{_clean_problem_for_skill(problem)}\n\n"
        "Predict the pitfalls to avoid as 1-3 [Error]/[Rule]/[Example] tiny-skill "
        "blocks (no solution, no answer)."
    )


# --------------------------------------------------------------------------- #
# "blind-correct" skill-KD (--sdpo-skill-kd-mode blind-correct / both-blind):
# symmetric counterpart to pitfall-condense for CORRECT traces. self-success's
# student/teacher prompts differ by only one hint sentence (see
# SKILL_SELF_SUCCESS_HINT above) because the student prompt already contains
# the WORKED SOLUTION in its "You are given a CORRECT worked solution" framing
# -- there is barely any information asymmetry left for the KD divergence to
# measure. blind-correct instead regenerates the student from the PROBLEM
# ONLY (no solution, no attempt -- exactly like pitfall-condense's student),
# so the teacher's privileged info (this trace's actual correct solution) is a
# genuine, large information gap, matching pitfall-condense's asymmetry.
# --------------------------------------------------------------------------- #

_BLIND_PREDICT_SYSTEM = (
    "Given a problem (and NOTHING else — no solution, no answer), predict the "
    "general KNOWLEDGE/RULES a solver would need to solve this kind of problem. "
    "Output them as tiny self-contained skills in this EXACT format, one block "
    "per distinct skill (1-3 blocks):\n\n"
    "[Knowledge/Rule]\n<a general principle / identity / method / theorem likely "
    "needed here — transferable, concrete, not vague>\n"
    "[Details/Examples]\n<a tiny concrete worked instance of the rule (small "
    "numbers / short snippet), NOT this problem's answer>\n\n"
    "Hard constraints:\n"
    "- Use the literal [Knowledge/Rule]/[Details/Examples] headers for every block.\n"
    "- Each skill must be a TRANSFERABLE unit usable on OTHER problems, not a "
    "recipe specific to this one.\n"
    "- Do NOT solve the problem or give its full method; only the general "
    "knowledge it likely draws on.\n"
    "- Never state a final answer.\n"
    "- Output ONLY the [Knowledge/Rule]/[Details/Examples] blocks, nothing else."
)

# Label for the privileged correct-solution info spliced into the blind-correct
# TEACHER turn -- distinct from FAILURES_TEMPLATE/PITFALLS_TEMPLATE (these are a
# CORRECT solution, not observed mistakes). Reuses {successful_previous_attempt}
# for _render_prefix compatibility.
CORRECT_INFO_TEMPLATE = (
    "\n\nObserved correct solution (privileged, do not reveal):\n\n{successful_previous_attempt}"
)


def _blind_predict_user_prompt(problem: str) -> str:
    return (
        f"PROBLEM:\n{_clean_problem_for_skill(problem)}\n\n"
        "Predict the general knowledge/rules needed as 1-3 [Knowledge/Rule]/"
        "[Details/Examples] tiny-skill blocks (no solution, no answer)."
    )


def _build_blind_correct_teacher_prompt_str(student_prompt: str, gen_suffix: str, correct_info: str) -> str:
    """Splice the trace's own correct solution into the USER turn of a
    problem-only knowledge-prediction prompt as PRIVILEGED info (blind-correct
    skill-KD teacher). Empty correct_info -> teacher == student (no privileged
    info, KD signal 0 for that sample). Mirrors _build_failure_teacher_prompt_str's
    insert point (blind-correct's pitfall-condense analogue)."""
    if not (correct_info and correct_info.strip()):
        return student_prompt
    section = CORRECT_INFO_TEMPLATE.format(successful_previous_attempt=correct_info.strip())
    if gen_suffix and gen_suffix in student_prompt:
        idx = student_prompt.rfind(gen_suffix)
        return student_prompt[:idx] + section + student_prompt[idx:]
    return student_prompt + section


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
        "Distill the transferable know-how into 1-3 [Knowledge/Rule]/[Details/Examples] "
        "tiny-skill blocks (each reusable on OTHER problems; do not state this "
        "problem's final answer)."
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


def _skill_user_prompt_incorrect(
    problem: str, attempt: str, ground_truth: str, failure_kind: str = "wrong", env_feedback: str = ""
) -> str:
    """User prompt for the incorrect-trace skill: the attempt is wrong; give the
    ground-truth answer ONLY so the model can localize the mistake, then emit
    pitfall warnings (never a solution, never the answer). failure_kind tailors the
    diagnosis (truncated | format | wrong).

    env_feedback (optional): the trace's own tool-execution trace (code run +
    stdout/error seen during rollout, see --sdpo-skill-source env_feedback and
    examples/SDPO_ReAct/sdpo_react.py's tool_trace extraction), rendered as an
    extra section. This is the direct analogue of lasgroup/SDPO's
    "environment feedback (e.g. test errors)" reprompt mechanism: grounding the
    pitfall diagnosis in what the tools ACTUALLY returned, not just the final
    wrong answer.
    """
    gt = (ground_truth or "").strip()
    note = _FAILURE_NOTE.get(failure_kind, _FAILURE_NOTE["wrong"])
    env_section = f"TOOL EXECUTION FEEDBACK FROM THIS ATTEMPT:\n{env_feedback.strip()}\n\n" if env_feedback.strip() else ""
    return (
        f"PROBLEM:\n{_clean_problem_for_skill(problem)}\n\n"
        f"FAILED ATTEMPT (wrong, do not echo):\n{attempt}\n\n"
        f"GROUND-TRUTH ANSWER (for locating the mistake only, do NOT put it in the output):\n{gt}\n\n"
        f"{env_section}"
        f"{note}\n\n"
        "Identify the specific mistake(s) and write them as [Error]/[Rule]/[Example] "
        "blocks (1-3 blocks, using the literal headers; specific and concrete; no "
        "solution, no answer)."
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
        **_skill_gen_template_kwargs(),
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
    args,
    tok,
    problem: str,
    solution: str,
    *,
    correct: bool = True,
    ground_truth: str = "",
    failure_kind: str = "wrong",
    env_feedback: str = "",
) -> list[int]:
    """Chat-templated skill-generation prompt (system=SKILL prompt, user=problem+
    solution + distill instruction), tokenized, with the assistant generation
    marker appended. This is the STUDENT context the skill is generated in.

    correct=False switches to the incorrect-trace framing: the attempt is wrong,
    the ground truth is supplied, and the model distills a pitfall-prevention skill
    instead of distilling know-how from a flawed solution. failure_kind (truncated |
    format | wrong) tailors the pitfall diagnosis. env_feedback (incorrect-trace
    only) is the trace's own tool-execution trace, see --sdpo-skill-source
    env_feedback and _skill_user_prompt_incorrect.
    """
    solution = _strip_response_eos(solution)
    if correct:
        system = _SKILL_SYSTEM_PROMPT
        user = _skill_user_prompt(problem, solution)
    else:
        system = _SKILL_SYSTEM_PROMPT_INCORRECT
        user = _skill_user_prompt_incorrect(
            problem, solution, ground_truth, failure_kind=failure_kind, env_feedback=env_feedback
        )
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **_skill_gen_template_kwargs(),
    )
    return tok.encode(text, add_special_tokens=False)


def _skill_gen_template_kwargs() -> dict:
    """Chat-template kwargs for SKILL GENERATION. Force enable_thinking=False so a
    thinking model (Qwen3) does NOT spend its skill budget on a <think> chain that
    then dominates / leaks into the skill text used as the KD prefix (observed: the
    self-gen skill was mostly thinking prose, and the condensed skill became a
    tool-free reasoning roadmap that collapsed tool use). No-think makes the whole
    budget produce the actual skill directly. Harmless for non-thinking models
    whose template ignores the kwarg."""
    return {"enable_thinking": False}


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
        gen_suffix = _gen_prompt_suffix(tok, getattr(args, "apply_chat_template_kwargs", None))
        remove_thinking = getattr(args, "sdpo_remove_thinking_from_demonstration", False)
        reframe_multiturn = getattr(args, "sdpo_reframe_multiturn_prefix", False)
        tool_grammar = getattr(args, "sdpo_tool_grammar", "qwen25")  # qwen25 JSON | qwen3_coder XML
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
        assert not (
            skill_kd and skill_kd_mode in ("pitfall-condense", "both-blind") and skill_source not in ("incorrect", "all")
        ), "--sdpo-skill-kd-mode pitfall-condense|both-blind requires --sdpo-skill-source incorrect|all"
        assert not (skill_kd and skill_kd_mode == "both" and skill_source != "all"), (
            "--sdpo-skill-kd-mode both trains correct (self-success) AND failed "
            "(pitfall-condense) traces, so it requires --sdpo-skill-source all"
        )
        assert not (
            skill_kd and skill_kd_mode == "blind-correct" and skill_source not in ("correct", "all")
        ), "--sdpo-skill-kd-mode blind-correct requires --sdpo-skill-source correct|all"
        assert not (skill_kd and skill_kd_mode == "both-blind" and skill_source != "all"), (
            "--sdpo-skill-kd-mode both-blind trains correct (blind-correct) AND failed "
            "(pitfall-condense) traces, so it requires --sdpo-skill-source all"
        )

        response_prefix = getattr(args, "sdpo_response_prefix", "trace")
        pitfall_backend = getattr(args, "sdpo_pitfall_summary_backend", "self")
        # Pitfall injection is active when self-skill distils failed traces. In that
        # mode the group's common failure lessons are spliced into FAILED traces'
        # teacher prefix (and a failed trace with no correct peer still gets a prefix
        # made of just those lessons).
        pitfall_active = self_skill and skill_source in ("incorrect", "all")
        # enable_kl (above) gates the whole group on "at least 1 correct trace", which
        # was written for base SDPO (a correct-peer prefix needs a correct peer to
        # exist). Under pitfall injection a FAILED trace's prefix is built from OTHER
        # failed traces' pitfalls, not a correct peer, so an ALL-WRONG group (0 correct
        # traces) should still get pitfall-only prefixes for every failed trace --
        # enable_kl=False was skipping the per-sample loop below entirely before it
        # ever reached the pitfall_active branch, silently dropping the KD signal for
        # every sample in such groups (confirmed live: 10/32 groups, 80/256 samples in
        # one rollout had has_prefix=False despite --sdpo-skill-source incorrect).
        if pitfall_active:
            enable_kl = True

        # Pass 1: pick each trace's correct peer (self-excluded). prefix_text is the
        # peer's solution — either the full response, or (with --sdpo-trace-condense)
        # its distilled skill. peer_by_idx remembers the chosen peer so
        # --sdpo-response-prefix skill can later swap in that peer's skill. With
        # pitfall injection, FAILED traces that have no correct peer still enter
        # prefix_text_by_idx (empty base) so the shared pitfalls can be spliced in.
        prefix_text_by_idx: dict[int, str] = {}
        # Peer's STRUCTURED message dict (metadata["messages"] from
        # multi_turn.generate), when present -- the dict-native prefix source.
        # Preferred over the raw-text reframe: rendered from the real tool_calls
        # field, never mis-splits on a stray marker. See _reframe_messages_to_prose.
        prefix_messages_by_idx: dict[int, list] = {}
        peer_by_idx: dict[int, int] = {}
        # Traces that fell into the "no correct peer" branch below WHILE
        # self_ok=True -- i.e. this trace is the group's ONLY correct trace, so
        # there is no other correct peer's solution/skill to borrow. It still has
        # nothing of its OWN to diagnose (it didn't fail), but the group's OTHER
        # traces may have failed and produced pitfalls -- give it those instead of
        # leaving it with zero teacher signal. Tracked separately from the
        # "not self_ok" failed-trace case in pass 2 below, since both end up with
        # group_pitfalls as their only content but for a different reason.
        sole_correct_no_peer_idxs: set[int] = set()
        for i, sample in enumerate(group):
            if not isinstance(sample.metadata, dict):
                continue
            if sample.response_length == 0 or not enable_kl:
                sample.metadata["sdpo_teacher_prompt_tokens"] = []
                continue
            peers = [j for j in correct_indices if j != i]
            if not peers:
                # No correct peer. Under pitfall injection, BOTH a failed trace (no
                # solution to diagnose from, but the group's shared pitfalls still
                # apply) AND the group's sole correct trace (nothing of its own to
                # diagnose, but it can still see what tripped up the OTHER, failed
                # traces) get a prefix (base empty; shared pitfalls appended in pass
                # 2). Only when pitfall injection is off entirely does a no-peer
                # trace get no prefix at all.
                self_ok = bool(correctness[i]) if i < len(correctness) else False
                if pitfall_active:
                    prefix_text_by_idx[i] = ""
                    if self_ok:
                        sole_correct_no_peer_idxs.add(i)
                else:
                    sample.metadata["sdpo_teacher_prompt_tokens"] = []
                continue
            peer_j = _choose_peer(args, group, peers)
            peer_by_idx[i] = peer_j
            prefix_text_by_idx[i] = group[peer_j].response
            peer_md = group[peer_j].metadata if isinstance(group[peer_j].metadata, dict) else {}
            peer_msgs = peer_md.get("messages")
            if peer_msgs:
                prefix_messages_by_idx[i] = peer_msgs

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

            def _has_env_feedback(i: int) -> bool:
                md = group[i].metadata if isinstance(group[i].metadata, dict) else {}
                return bool(md.get("tool_trace"))

            def _skill_eligible(i: int) -> bool:
                if group[i].response_length == 0:
                    return False
                self_ok = bool(correctness[i]) if i < len(correctness) else False
                if skill_source == "correct":
                    return self_ok
                if skill_source == "incorrect":
                    return not self_ok
                if skill_source == "env_feedback":
                    # Grounded pitfall generation from the trace's OWN tool-execution
                    # trace (see examples/SDPO_ReAct/sdpo_react.py, which stashes
                    # sample.metadata["tool_trace"] = [{"tool_call":..., "observation":...}, ...]
                    # before calling into this group RM). Only meaningful for a FAILED
                    # trace that actually called a tool -- lasgroup/SDPO's "environment
                    # feedback (e.g. test errors)" reprompt idea, restricted to traces
                    # where such feedback exists.
                    return (not self_ok) and _has_env_feedback(i)
                return True  # "all"

            skill_idxs = [i for i in range(len(group)) if isinstance(group[i].metadata, dict) and _skill_eligible(i)]

            def _render_env_feedback(i: int) -> str:
                """Render trace i's tool_trace (see sdpo_react.py::_extract_tool_trace)
                as plain text for the pitfall prompt, truncated to
                --sdpo-env-feedback-max-chars (analogous to lasgroup/SDPO's
                max_reprompt_len/reprompt_truncation budget on the reprompt text).
                Returns "" when the trace has no tool_trace (e.g. non-tool rollouts) --
                _skill_user_prompt_incorrect treats "" as a no-op, so this never changes
                behavior for rollouts that never call a tool."""
                md = group[i].metadata if isinstance(group[i].metadata, dict) else {}
                trace = md.get("tool_trace") or []
                if not trace:
                    return ""
                max_chars = int(getattr(args, "sdpo_env_feedback_max_chars", 2000))
                parts = [f"[call {j + 1}] {t['tool_call']}\n[result {j + 1}] {t['observation']}" for j, t in enumerate(trace)]
                text = "\n\n".join(parts)
                return text if len(text) <= max_chars else text[-max_chars:]

            async def _gen_one(i: int):
                # Distill the trace's OWN response into a skill. For a correct trace
                # this is "extract the transferable procedure"; for an incorrect one
                # it flips to "diagnose the error -> pitfall warnings", tailored by WHY
                # it failed (truncated | format | wrong). The ground-truth answer is
                # passed so the model can localize where the attempt went wrong.
                # env_feedback grounds that diagnosis in the trace's actual tool calls
                # (see --sdpo-skill-source env_feedback above); it is "" for rollouts
                # that never call a tool, which is a no-op for the prompt.
                problem = _problem_of(i)
                self_ok = bool(correctness[i]) if i < len(correctness) else False
                fkind = "wrong" if self_ok else _failure_kind(args, group[i])
                # Skill is distilled from the trace. Prefer the dict-native prose
                # (from metadata["messages"]) over the raw ChatML response: the
                # skill generator then reads clean per-round "Round N.../
                # Observation:..." prose (tool calls from the real tool_calls
                # field) instead of scraping <tool_call>/<|im_*|> markers.
                _md_i = group[i].metadata if isinstance(group[i].metadata, dict) else {}
                _msgs_i = _md_i.get("messages")
                solution_i = _reframe_messages_to_prose(_msgs_i, grammar=tool_grammar) if _msgs_i else group[i].response
                gen_prompt_ids = _skill_gen_prompt_ids(
                    args,
                    tok,
                    problem,
                    solution_i,
                    correct=self_ok,
                    ground_truth=(group[i].label or "") if not self_ok else "",
                    failure_kind=fkind,
                    env_feedback=_render_env_feedback(i) if not self_ok else "",
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
                # Skill-KD teacher hint (see doc/DESIGN_self_skill.md), KD-only:
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
                        skill_teacher_str = _build_skill_self_success_teacher_prompt_str(gen_prompt_str, gen_suffix)
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
            if skill_kd and skill_kd_mode in ("pitfall-condense", "both", "both-blind"):
                # 'both'/'both-blind' also runs a correct-trace KD variant (self-success
                # or blind-correct, handled above/below); here we only (re)build the
                # FAILED traces' skill-KD via pitfall-condense.
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
                    # **_skill_gen_template_kwargs() (enable_thinking=False) to match
                    # every OTHER skill-generation call site (_generate_skill_text,
                    # _skill_gen_prompt_ids) -- without it this student render falls
                    # through to the run's real --apply-chat-template-kwargs (thinking
                    # ON for Qwen3), so the problem-only pitfall-prediction student
                    # would think freely while its sibling skill generations (self-
                    # success skill, per-trace pitfall, pitfall-summary) are all forced
                    # no-think, reproducing exactly the collapse _skill_gen_template_
                    # kwargs was added to prevent (see that function's docstring).
                    stu_text = tok.apply_chat_template(
                        [
                            {"role": "system", "content": _PITFALL_PREDICT_SYSTEM},
                            {"role": "user", "content": _pitfall_predict_user_prompt(_problem_of(i))},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        **_skill_gen_template_kwargs(),
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

            # blind-correct skill-KD: symmetric counterpart to pitfall-condense for
            # CORRECT traces (see the module-level comment above
            # _build_blind_correct_teacher_prompt_str for why self-success's KD signal
            # is weak and this fixes it).
            #  student = predict general knowledge from the PROBLEM ONLY (no solution,
            #            no attempt -- exactly mirrors pitfall-condense's student);
            #  teacher = same problem-only prompt + THIS trace's own correct solution
            #            as privileged info (per-trace, NOT group-shared -- unlike
            #            pitfall-condense's group_pitfalls, a correct trace's own
            #            solution is a self-contained privileged hint, no aggregation
            #            needed across peers).
            #  KD target = the student's own problem-only knowledge prediction.
            if skill_kd and skill_kd_mode in ("blind-correct", "both-blind"):
                correct_idxs = [
                    i for i in skill_idxs
                    if (bool(correctness[i]) if i < len(correctness) else False)
                ]

                async def _gen_blind(i: int):
                    stu_text = tok.apply_chat_template(
                        [
                            {"role": "system", "content": _BLIND_PREDICT_SYSTEM},
                            {"role": "user", "content": _blind_predict_user_prompt(_problem_of(i))},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        **_skill_gen_template_kwargs(),
                    )
                    stu_ids = tok.encode(stu_text, add_special_tokens=False)
                    res3 = await _self_generate_skill(args, stu_ids)
                    return i, stu_ids, res3

                blind_results = await asyncio.gather(*(_gen_blind(i) for i in correct_idxs))
                for i, stu_ids, res3 in blind_results:
                    if res3 is None:
                        continue
                    b_text, b_tokens, b_logprobs = res3
                    md = group[i].metadata
                    # Overwrite the skill-KD payload with the problem-only student and
                    # the correct-solution-informed teacher. The KD student/target is
                    # now the problem-only knowledge prediction.
                    md["sdpo_skill"] = b_text
                    md["sdpo_skill_len"] = float(len(b_tokens))
                    if b_logprobs:
                        _nll = -sum(b_logprobs) / len(b_logprobs)
                        md["sdpo_skill_ppl"] = math.exp(min(_nll, 20.0))
                    md["sdpo_skill_tokens"] = b_tokens
                    md["sdpo_skill_prompt_tokens"] = stu_ids
                    md["sdpo_skill_rollout_logprobs"] = b_logprobs
                    stu_prompt_str = tok.decode(stu_ids)
                    # Privileged info = THIS trace's own correct response (per-trace,
                    # not group-aggregated -- unlike pitfall-condense's failure_info).
                    # Prefer the dict-native prose (metadata["messages"]) over the raw
                    # ChatML response, same reasoning as the self-skill path above
                    # (sdpo.py:1593-1600): a peer's own reframed prose can't mis-split
                    # on a stray <|im_*|>/<tool_call> marker.
                    own_msgs = group[i].metadata.get("messages") if isinstance(group[i].metadata, dict) else None
                    correct_info = (
                        _reframe_messages_to_prose(own_msgs, grammar=tool_grammar)
                        if own_msgs
                        else _strip_response_eos(group[i].response)
                    )
                    teacher_str = _build_blind_correct_teacher_prompt_str(stu_prompt_str, gen_suffix, correct_info)
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

            # Prefer the peer's dict-native prose (metadata["messages"], tracked in
            # prefix_messages_by_idx alongside prefix_text_by_idx -- see pass 1
            # above) over the raw ChatML response text: the condenser LLM then
            # reads clean per-round "Round N.../Observation:..." prose instead of
            # scraping <tool_call>/<|im_*|> control tokens, same reasoning as the
            # self-skill path (sdpo.py:1593-1600) and the response-teacher-prefix
            # splice (_build_teacher_prompt_str, sdpo.py:403-412).
            def _peer_solution(j: int) -> str:
                peer_msgs = prefix_messages_by_idx.get(j)
                if peer_msgs:
                    return _reframe_messages_to_prose(peer_msgs, grammar=tool_grammar)
                cleaned = _strip_response_eos(prefix_text_by_idx[j])
                return _reframe_multiturn_trace(cleaned) if reframe_multiturn else cleaned

            pairs = [(_problem_of(i), _peer_solution(i)) for i in idxs]
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
        # Skip FAILED traces only under skill-source=incorrect: there, the peer
        # (always a correct_indices trace) never gets a self-generated skill
        # (_skill_eligible returns `not self_ok`), so the swap would be a silent
        # no-op anyway, and skill-source=incorrect's failed traces get their
        # prefix wiped to pitfalls-only right below regardless. Under
        # skill-source=all, a failed trace's peer IS eligible for a skill (every
        # trace is), so let the swap apply -- failed traces there keep BOTH the
        # peer's correct-solution skill AND the group pitfall summary (see the
        # pass-2 splice below), not pitfalls alone.
        if response_prefix == "skill" and self_skill:
            for i in list(prefix_text_by_idx.keys()):
                if pitfall_active and skill_source == "incorrect" and not (
                    bool(correctness[i]) if i < len(correctness) else False
                ):
                    continue
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
                    # BUG FIX: _build_teacher_prompt_str prefers peer_messages (dict-
                    # native structured trace) over peer_response WHENEVER peer_messages
                    # is truthy (see that function: `if peer_messages: cleaned =
                    # _reframe_messages_to_prose(peer_messages, ...)` -- peer_response is
                    # never even looked at in that branch). prefix_messages_by_idx[i] was
                    # set from the peer's ORIGINAL full trace at pass 1 (line ~1521) and
                    # never touched here, so every skill-prefix swap was being silently
                    # overridden back to the peer's full raw trace downstream -- confirmed
                    # live: 100% of a rollout's dumped teacher_prompt_text carried a full
                    # multi-thousand-char "Correct solution:" trace, never the ~300-char
                    # skill, despite skill/response_prefix_is_skill_frac reporting 1.0 (that
                    # metric only reflects THIS dict's bookkeeping, not what actually got
                    # spliced). Clear it so the skill (peer_response) branch is taken.
                    prefix_messages_by_idx.pop(i, None)

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

        # FAILED traces under skill-source=incorrect get ONLY the group-
        # summarized pitfall skill as their response-SDPO prefix -- there is no
        # correct-peer skill to keep here (skill-source=incorrect never
        # generates a skill for a correct trace, so the swap above was always a
        # no-op for these), so drop whatever raw-trace prefix pass 1 picked
        # (the correct peer's raw trace) and let pass 2's PITFALLS_TEMPLATE
        # splice become the ONLY content.
        #
        # FAILED traces under skill-source=all keep BOTH: the correct peer's
        # skill (or raw trace, picked in pass 1 / swapped in above) AND the
        # group pitfall summary -- _build_teacher_prompt_str/_render_prefix
        # APPENDS pitfalls after the solution section when both are non-empty,
        # it does not replace it. Only degrades to pitfalls-only when the group
        # has zero correct traces (prefix_text_by_idx[i] is already "" from
        # pass 1's "no peers" branch -- nothing to clear).
        if pitfall_active and skill_source == "incorrect":
            for i in list(prefix_text_by_idx.keys()):
                self_ok_i = bool(correctness[i]) if i < len(correctness) else False
                if not self_ok_i:
                    prefix_text_by_idx[i] = ""
                    prefix_messages_by_idx.pop(i, None)

        # Pass 2: build the teacher prompt (peer solution/skill spliced into the USER
        # turn, before the assistant marker) and tokenize. The shared pitfalls go ONLY
        # into failed traces' prefix (nothing of their own to show, but the group's
        # common mistakes still apply) PLUS the group's sole correct trace when it has
        # no peer of its own to borrow a solution/skill from (see
        # sole_correct_no_peer_idxs above) -- otherwise that trace would get ZERO
        # teacher signal despite having answered correctly. Every OTHER correct trace
        # keeps a clean correct-peer prefix with no pitfalls mixed in.
        for i, sample in enumerate(group):
            if i not in prefix_text_by_idx:
                continue
            self_ok = bool(correctness[i]) if i < len(correctness) else False
            pitfalls_for_i = (
                group_pitfalls if (pitfall_active and (not self_ok or i in sole_correct_no_peer_idxs)) else ""
            )
            student_prompt = sample.prompt if isinstance(sample.prompt, str) else ""
            teacher_prompt_str = _build_teacher_prompt_str(
                student_prompt, gen_suffix, prefix_text_by_idx[i], remove_thinking=remove_thinking,
                pitfalls=pitfalls_for_i, reframe_multiturn=reframe_multiturn,
                peer_messages=prefix_messages_by_idx.get(i), grammar=tool_grammar,
                max_prefix_chars=getattr(args, "sdpo_max_prefix_chars", 0),
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
        prefix_sample = group[_choose_peer(args, group, peers)]
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


async def plain_grpo_reward(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    """Single-sample reward for a plain-GRPO baseline (--custom-rm-path, NO
    --group-rm) -- the "no SDPO at all" arm of an ablation against
    sdpo_group_reward. Reuses _is_correct directly rather than --rm-type dapo/
    boxed_dapo: async_rm's "dapo" rm_type calls math_dapo_utils.compute_score
    with strict_box_verify defaulting to False (Minerva "Answer: X" line-
    matching), which fails on a bare \\boxed{...} response with no such line;
    the "boxed_" rm_type prefix pre-extracts the boxed answer but then feeds
    JUST that bare string back into the same Minerva-pattern grader, which
    also fails (there is no "Answer:" line left to match). _is_correct's own
    --sdpo-grader dapo branch is the only path that actually calls
    math_dapo_utils.compute_score with strict_box_verify=True (scans for the
    LAST \\boxed{...} occurrence directly) -- the same grading criterion every
    other arm's group reward uses, so a plain-GRPO baseline stays comparable."""
    return 1.0 if _is_correct(sample, args) else 0.0


# --------------------------------------------------------------------------- #
# EVAL-time skill augmentation (--sdpo-eval-skill-mode, wired via
# --custom-generate-function-path examples.SDPO.sdpo.sdpo_eval_generate):
# before the real eval rollout, self-predict a blind skill from the problem
# alone (the SAME self-predict prompts training uses for blind-correct/
# pitfall-condense skill-gen) and splice it into the eval prompt's user turn,
# so eval measures the model answering WITH its own self-predicted skill
# already in context. The mode is set MANUALLY (not auto-derived from
# --sdpo-skill-kd-mode) so it matches whichever skill type(s) a given training
# run actually trained -- e.g. skill-kd-mode 'both'/'both-blind' (skill-source
# all) trains BOTH correct- and pitfall-type skills -> eval-skill-mode 'all';
# skill-kd-mode 'self-success'/'blind-correct' (skill-source correct) only
# trains the correct-type skill -> eval-skill-mode 'correct'; a run whose
# skill-source is 'incorrect' only trains the pitfall-type skill ->
# eval-skill-mode 'pitfall'.
# --------------------------------------------------------------------------- #

EVAL_SKILL_CORRECT_TEMPLATE = "\n\nPredicted knowledge/rules for this problem:\n\n{skill}"
EVAL_SKILL_PITFALL_TEMPLATE = "\n\nPredicted pitfalls to avoid for this problem:\n\n{skill}"
EVAL_SKILL_INSTRUCTION = "\n\nNow solve the original problem above.\n\n"


async def sdpo_eval_generate(input: Any) -> Any:
    """--custom-generate-function-path for eval-time skill augmentation (see
    --sdpo-eval-skill-mode). No-op during TRAINING (evaluation=False) or when
    the mode is 'off' -- falls straight through to the stock generate(). During
    EVAL, self-predicts the configured skill type(s) from the problem alone
    (blind: no solution, no attempt) and splices them into the user turn
    before the assistant marker, then runs the real eval rollout on the
    skill-augmented prompt."""
    from miles.rollout.base_types import GenerateFnOutput
    from miles.rollout.sglang_rollout import generate

    args = input.args
    sample = input.sample
    mode = getattr(args, "sdpo_eval_skill_mode", "off")

    if not input.evaluation or mode == "off" or not isinstance(sample.prompt, str):
        sample = await generate(args, sample, input.sampling_params, evaluation=input.evaluation)
        return GenerateFnOutput(samples=sample)

    try:
        sections = []
        if mode in ("correct", "all"):
            skill = await _generate_skill_text(
                args, _BLIND_PREDICT_SYSTEM, _blind_predict_user_prompt(sample.prompt), "self"
            )
            if skill.strip():
                sections.append(EVAL_SKILL_CORRECT_TEMPLATE.format(skill=skill.strip()))
        if mode in ("pitfall", "all"):
            skill = await _generate_skill_text(
                args, _PITFALL_PREDICT_SYSTEM, _pitfall_predict_user_prompt(sample.prompt), "self"
            )
            if skill.strip():
                sections.append(EVAL_SKILL_PITFALL_TEMPLATE.format(skill=skill.strip()))

        if sections:
            tok = _tokenizer(args)
            gen_suffix = _gen_prompt_suffix(tok, getattr(args, "apply_chat_template_kwargs", None))
            section = "".join(sections) + EVAL_SKILL_INSTRUCTION
            if gen_suffix and gen_suffix in sample.prompt:
                idx = sample.prompt.rfind(gen_suffix)
                sample.prompt = sample.prompt[:idx] + section + sample.prompt[idx:]
            else:
                sample.prompt = sample.prompt + section
    except Exception as e:
        logger.warning(f"eval skill augmentation failed ({e!r}); evaluating on the unaugmented prompt.")

    sample = await generate(args, sample, input.sampling_params, evaluation=input.evaluation)
    return GenerateFnOutput(samples=sample)


async def sdpo_eval_reward(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    """Per-sample eval RM for SDPO (--eval-custom-rm-path).

    Eval measures pass@1 and never uses the distillation signal, so it just needs
    the task reward. Uses the same grading as the group RM: the LLM judge when
    --sdpo-judge is set (open-ended answers), else deterministic matching.
    """
    if _sample_domain(sample) == "code":
        # Code eval: run the program against its test cases (same judge as
        # training). Bounded by the shared concurrency cap like the LLM judge.
        async with _judge_semaphore(args):
            ok = await _grade_one_code(sample, args)
    elif _sample_domain(sample) == "search":
        # Search/QA eval: EM against golden answers, with the same optional
        # LLM-judge second opinion on an EM miss as training (reward.py's
        # _grade_one_search) -- bound by the shared concurrency cap whenever
        # that fallback might actually reach the judge gateway.
        async with _judge_semaphore(args):
            ok = await _grade_one_search(sample, args)
    elif getattr(args, "sdpo_judge", False) and (sample.response or "").strip():
        # Eval fans out one sdpo_eval_reward coroutine per sample via asyncio.gather
        # upstream, so honor the SAME global concurrency cap to avoid flooding the
        # gateway during large evals.
        async with _judge_semaphore(args):
            ok = await _llm_judge_correct(args, sample)
    elif getattr(args, "sdpo_grader", "mcq") == "dapo":
        # Math eval (AIME = integers, Minerva Math = LaTeX). Delegate to the SAME
        # grader the training side uses (_is_correct's dapo path): it EXTRACTS the
        # <answer> tag, wraps it in \boxed{}, and runs DAPO's scorer.
        # BUG FIX (2026-07-25): the old path called
        # grade_answer_verl(sample.response, label) on the WHOLE multi-turn
        # response with NO tag extraction and NO \boxed{} wrapping. grade_answer_verl
        # only matches when the prediction is \boxed{}-wrapped -- so even
        # <answer>73</answer> vs label 73 scored 0. Observed 143/552 AIME rows
        # correct-but-graded-0: the "math eval regression" was a grading artifact.
        # _is_correct(dapo) fixes both (tag extraction + \boxed{} wrap).
        ok = _is_correct(sample, args)
    else:
        ok = _is_correct(sample, args)
    return 1.0 if ok else 0.0
