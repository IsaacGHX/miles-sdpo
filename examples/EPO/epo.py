"""EPO (PMI-credit self-distillation) reward function for Miles.

EPO idea
--------
SDPO couples two things that arXiv:2603.24472 shows fight each other on math:
a DIRECTION (the teacher-vs-student KL direction, which drives epistemic
suppression) and a DENSITY/CREDIT (how much of that gradient each token gets).
EPO decouples them:

  credit_t  = |logp(y_t | x, f, y_<t>) - logp(y_t | x, y_<t>)|
              (the PMI a response token carries about the privileged context f
              — a correct peer's solution — under the SAME teacher weights)
  A(t)      = credit_t * (R(y) - baseline)
              (direction comes from the task OUTCOME reward, never from the
              teacher's distribution, so it cannot drive suppression)

Rollout is exactly GRPO (n_samples_per_prompt traces per prompt); after the
group is generated we:

1. Grade every trace against the label (same graders as SDPO: dapo | mcq |
   judge, see --sdpo-grader / --sdpo-judge).
2. For every trace, randomly pick one OTHER correct trace in the group as a
   privileged-context prefix (never itself) -- identical selection to SDPO's
   Megatron-teacher-backend Pass 1, but WITHOUT SDPO's pitfall/condense
   machinery. The prefix `f` is either the peer's raw response (default) or,
   with --epo-credit-skill, the peer's self-generated SKILL (a condensed
   solution roadmap, reusing SDPO's _skill_gen_prompt_ids/_self_generate_skill
   -- same mechanism as --sdpo-self-skill, but generated here directly rather
   than via SDPO's fuller skill-source/pitfall pipeline).
3. Splice that peer solution/skill into the prompt's user turn (same template
   as SDPO) and stash the tokenized teacher prompt on
   ``sample.metadata["sdpo_teacher_prompt_tokens"]`` -- the SAME key SDPO's
   Megatron self-teacher already knows how to consume
   (``MegatronTrainRayActor._build_sdpo_teacher_rollout_data`` /
   ``_compute_sdpo_teacher_log_probs``), so the training side needs no new
   prefix-selection logic, just a new credit computation gated by
   ``--epo-credit-loss`` (see actor.py:_compute_epo_credit). When
   --epo-credit-skill is set, the peer's skill text/tokens are also stashed on
   ``sdpo_skill``/``sdpo_skill_tokens``/``sdpo_skill_prompt_tokens`` purely so
   MegatronTrainRayActor._dump_sdpo_prompts's existing skill/ dump picks it up
   (EPO has no skill-KD objective -- those fields are dump-only here).
4. Unlike SDPO's default ``--sdpo-pure-distill``, EPO returns the REAL task
   reward (1.0 correct / 0.0 wrong): the credit weight only reshapes WHERE the
   GRPO advantage lands within a trace, it never replaces the reward.

Wiring
------
    --group-rm
    --custom-rm-path examples.EPO.epo.epo_group_reward
    --eval-custom-rm-path examples.EPO.epo.epo_eval_reward
    --advantage-estimator grpo
    --sdpo-grader dapo            # or mcq
    --sdpo-ema-teacher             # optional, reuses SDPO's EMA teacher snapshot
    --sdpo-teacher-backend megatron
    --epo-credit-loss
    --epo-credit-clip 5.0
    --epo-credit-normalize         # default on
    --epo-credit-mode abs_logp_diff  # or topk_divergence (see --epo-credit-divergence)
    --epo-credit-skill               # optional: peer SKILL instead of full trace

This module deliberately re-uses SDPO's grading / prompt-splicing / skill-
generation helpers (``examples.SDPO.sdpo``) instead of re-implementing them --
they are pure, side-effect-free w.r.t. SDPO's own reward bookkeeping (see
module docstring there), so importing them here does not couple EPO to SDPO's
KD-loss/skill-KD machinery.
"""

import asyncio
import logging
import random
from argparse import Namespace
from typing import Any

from examples.SDPO.sdpo import (
    _build_teacher_prompt_str,
    _gen_prompt_suffix,
    _grade_group,
    _self_generate_skill,
    _skill_gen_prompt_ids,
    _tokenizer,
)
from examples.SDPO.sdpo import sdpo_eval_reward as _sdpo_eval_reward
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _problem_of(sample: Sample) -> str:
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    q = md.get("question")
    if q:
        return str(q)
    p = sample.prompt
    return p if isinstance(p, str) else str(p)


async def _generate_peer_skills(args: Namespace, tok, group: list[Sample], correct_indices: list[int]) -> dict[int, str]:
    """Self-generate a solution-roadmap skill for every CORRECT trace (candidate
    peers), so --epo-credit-skill can splice a peer's skill instead of its full
    trace. Reuses SDPO's correct-trace skill-gen prompt (_skill_gen_prompt_ids
    with correct=True) and _self_generate_skill (an extra rollout-engine
    generation call per correct trace). Also stashes the skill text/tokens on
    each trace's own metadata (sdpo_skill / sdpo_skill_tokens /
    sdpo_skill_prompt_tokens) purely so the training-side dump
    (_dump_sdpo_prompts) can log it -- EPO never runs a KD objective on these
    tokens. Returns {trace_index: skill_text}; traces whose generation failed
    are simply absent (callers fall back to the full trace)."""

    async def _gen_one(i: int):
        sample = group[i]
        gen_prompt_ids = _skill_gen_prompt_ids(args, tok, _problem_of(sample), sample.response, correct=True)
        res = await _self_generate_skill(args, gen_prompt_ids)
        return i, gen_prompt_ids, res

    gen_results = await asyncio.gather(*(_gen_one(i) for i in correct_indices))
    skills: dict[int, str] = {}
    for i, gen_prompt_ids, res in gen_results:
        if res is None:
            continue
        skill_text, skill_tokens, _skill_logprobs = res
        skills[i] = skill_text
        md = group[i].metadata
        if isinstance(md, dict):
            md["sdpo_skill"] = skill_text
            md["sdpo_skill_tokens"] = skill_tokens
            md["sdpo_skill_prompt_tokens"] = gen_prompt_ids
    return skills


async def epo_group_reward(args: Namespace, group: list[Sample], **kwargs: Any) -> list[float]:
    """Group RM: grades the group, picks a correct-peer privileged-context prefix
    per trace (stashed for the training-side credit forward), and returns the
    REAL task reward (never zeroed out -- EPO's advantage direction comes from
    this reward, unlike SDPO's pure-distill mode)."""
    correctness = await _grade_group(args, group)
    correct_indices = [i for i, ok in enumerate(correctness) if ok]

    for ok, s in zip(correctness, group, strict=True):
        if isinstance(s.metadata, dict):
            s.metadata["sdpo_correct"] = 1.0 if ok else 0.0

    rewards = [1.0 if ok else 0.0 for ok in correctness]

    # Need >= 1 correct trace to have a valid privileged-context peer (matching
    # SDPO: a single correct trace still serves as prefix for every incorrect
    # trace; it self-excludes to an empty peer pool and gets no prefix itself).
    if not correct_indices:
        for s in group:
            if isinstance(s.metadata, dict):
                s.metadata["sdpo_teacher_prompt_tokens"] = []
        return rewards

    tok = _tokenizer(args)
    gen_suffix = _gen_prompt_suffix(tok)
    remove_thinking = getattr(args, "sdpo_remove_thinking_from_demonstration", False)
    use_skill = getattr(args, "epo_credit_skill", False)

    peer_skills: dict[int, str] = {}
    if use_skill:
        peer_skills = await _generate_peer_skills(args, tok, group, correct_indices)

    for i, sample in enumerate(group):
        if not isinstance(sample.metadata, dict):
            continue
        if sample.response_length == 0:
            sample.metadata["sdpo_teacher_prompt_tokens"] = []
            continue
        peers = [j for j in correct_indices if j != i]
        if not peers:
            sample.metadata["sdpo_teacher_prompt_tokens"] = []
            continue
        peer_j = random.choice(peers)
        # --epo-credit-skill: privileged context f = the peer's self-generated
        # skill instead of its raw trace; falls back to the full trace if that
        # peer's skill generation failed.
        peer_context = peer_skills.get(peer_j) if use_skill else None
        if not peer_context:
            peer_context = group[peer_j].response
        student_prompt = sample.prompt if isinstance(sample.prompt, str) else ""
        teacher_prompt_str = _build_teacher_prompt_str(
            student_prompt, gen_suffix, peer_context, remove_thinking=remove_thinking
        )
        sample.metadata["sdpo_teacher_prompt_tokens"] = tok.encode(teacher_prompt_str, add_special_tokens=False)
        # Dump-only bookkeeping: which flavor of privileged context this trace
        # actually got, mirroring SDPO's sdpo_response_prefix_is_skill metric.
        sample.metadata["sdpo_response_prefix_is_skill"] = 1.0 if (use_skill and peer_j in peer_skills) else 0.0

    return rewards


async def epo_eval_reward(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    """Per-sample eval RM for EPO (--eval-custom-rm-path). Purely a grading
    function -- identical to SDPO's (pass@1 grading never touches the credit /
    teacher-prefix machinery), so we delegate directly."""
    return await _sdpo_eval_reward(args, sample, **kwargs)
