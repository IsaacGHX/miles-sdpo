"""LAYER-1 scaffold: solve one problem SDPO-style -- a GROUP of N rollouts,
distill correct-knowledge + pitfalls from that group, then re-solve with the
combined skill in context.

This mirrors examples/SDPO exactly: one problem -> N rollouts -> grade -> from the
CORRECT traces distill [Knowledge/Rule] blocks, from the INCORRECT traces distill
[Error]/[Rule]/[Example] pitfalls -> the concatenation is the skill. The layer-2
optimizer (optimizer.py) rewrites the two distiller prompts each round.

Engines are model-agnostic and chosen per role (local or remote). Per problem:

    baseline: N solver rollouts WITHOUT skill -> grade  (the group to summarize)
    skill   : correct_writer(correct_prompt, problem, correct_traces)
              pitfall_writer(pitfall_prompt, problem, failed_traces, label)
              skill = combine(correct_blocks, pitfall_blocks)
    with    : N solver rollouts WITH skill -> grade

"how much it optimized" = with_acc - baseline_acc, per problem and overall.

The baseline group is generated ONCE (round 0) and cached: it does not depend on
the skill-gen prompt, so we never re-pay for it across rounds. Only the skill
distillation + the with-skill rollouts re-run each round.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from examples.TTS.engines import Engine
from examples.TTS.prompts import (
    SkillPrompts,
    combine_skill,
    correct_skill_user,
    pitfall_skill_user,
    predict_knowledge_user,
    predict_pitfall_user,
    skill_augmented_user,
)

logger = logging.getLogger(__name__)


@dataclass
class GroupResult:
    """One problem's group of rollouts + the skill distilled from it."""

    problem: str
    label: str
    domain: str = "math"
    # baseline (no skill) group
    baseline_responses: list[str] = field(default_factory=list)
    baseline_correct: list[bool] = field(default_factory=list)
    # skill distilled from the baseline group
    skill: str = ""
    correct_skill_text: str = ""
    pitfall_skill_text: str = ""
    # with-skill group
    skilled_responses: list[str] = field(default_factory=list)
    skilled_correct: list[bool] = field(default_factory=list)

    @property
    def baseline_acc(self) -> float:
        return _acc(self.baseline_correct)

    @property
    def skilled_acc(self) -> float:
        return _acc(self.skilled_correct)

    @property
    def delta(self) -> float:
        return self.skilled_acc - self.baseline_acc


def _acc(flags: list[bool]) -> float:
    return (sum(1 for f in flags if f) / len(flags)) if flags else 0.0


class Scaffold:
    def __init__(
        self,
        correct_writer: Engine,
        pitfall_writer: Engine,
        solver: Engine,
        solver_system_prompt: str,
        *,
        n_rollouts: int = 8,
        skill_max_tokens: int = 1024,
        solver_max_tokens: int = 4096,
        skill_temperature: float = 0.7,
        solver_temperature: float = 0.7,
        max_correct_traces: int = 3,
        max_failed_traces: int = 3,
    ):
        self.correct_writer = correct_writer
        self.pitfall_writer = pitfall_writer
        self.solver = solver
        self.solver_system_prompt = solver_system_prompt
        self.n_rollouts = n_rollouts
        self.skill_max_tokens = skill_max_tokens
        self.solver_max_tokens = solver_max_tokens
        self.skill_temperature = skill_temperature
        self.solver_temperature = solver_temperature
        self.max_correct_traces = max_correct_traces
        self.max_failed_traces = max_failed_traces

    # ---- solving ---------------------------------------------------------- #
    async def _solve_once(self, problem: str, skill: str) -> str:
        user = skill_augmented_user(problem, skill) if skill else problem
        try:
            res = await self.solver.chat(
                [
                    {"role": "system", "content": self.solver_system_prompt},
                    {"role": "user", "content": user},
                ],
                temperature=self.solver_temperature,
                max_tokens=self.solver_max_tokens,
            )
            return res.text
        except Exception as e:
            logger.warning(f"solve failed: {e!r}")
            return ""

    async def _solve_group(self, problem: str, skill: str) -> list[str]:
        return await asyncio.gather(*(self._solve_once(problem, skill) for _ in range(self.n_rollouts)))

    # ---- skill distillation ---------------------------------------------- #
    async def distill_skill(
        self, problem: str, label: str, correct_traces: list[str], failed_traces: list[str], prompts: SkillPrompts
    ) -> tuple[str, str, str]:
        """(combined_skill, correct_text, pitfall_text) from a graded group.
        Only distills the halves that have traces (a group can be all-correct or
        all-wrong)."""
        correct_traces = correct_traces[: self.max_correct_traces]
        failed_traces = failed_traces[: self.max_failed_traces]

        async def _correct() -> str:
            if not correct_traces:
                return ""
            try:
                return await self.correct_writer.complete(
                    prompts.correct,
                    correct_skill_user(problem, correct_traces),
                    temperature=self.skill_temperature,
                    max_tokens=self.skill_max_tokens,
                )
            except Exception as e:
                logger.warning(f"correct-skill distill failed: {e!r}")
                return ""

        async def _pitfall() -> str:
            if not failed_traces:
                return ""
            try:
                return await self.pitfall_writer.complete(
                    prompts.pitfall,
                    pitfall_skill_user(problem, failed_traces, label),
                    temperature=self.skill_temperature,
                    max_tokens=self.skill_max_tokens,
                )
            except Exception as e:
                logger.warning(f"pitfall-skill distill failed: {e!r}")
                return ""

        correct_text, pitfall_text = await asyncio.gather(_correct(), _pitfall())
        return combine_skill(correct_text, pitfall_text), correct_text, pitfall_text

    # ---- skill PREDICTION (deployment protocol) -------------------------- #
    async def predict_skill(self, problem: str, prompts: SkillPrompts) -> tuple[str, str, str]:
        """(combined_skill, knowledge_text, pitfall_text) predicted from the BARE
        problem -- no traces, no ground truth. This is the deployment protocol:
        the model anticipates the needed knowledge + likely pitfalls up front.
        Uses the SAME optimized system prompts as distillation; only the user turn
        differs (predict vs distill), so optimizing on this signal optimizes the
        prompts for exactly how they are used at test time."""

        async def _k() -> str:
            try:
                return await self.correct_writer.complete(
                    prompts.correct, predict_knowledge_user(problem),
                    temperature=self.skill_temperature, max_tokens=self.skill_max_tokens,
                )
            except Exception as e:
                logger.warning(f"knowledge predict failed: {e!r}")
                return ""

        async def _p() -> str:
            try:
                return await self.pitfall_writer.complete(
                    prompts.pitfall, predict_pitfall_user(problem),
                    temperature=self.skill_temperature, max_tokens=self.skill_max_tokens,
                )
            except Exception as e:
                logger.warning(f"pitfall predict failed: {e!r}")
                return ""

        knowledge, pitfall = await asyncio.gather(_k(), _p())
        return combine_skill(knowledge, pitfall), knowledge, pitfall

    async def run_with_predicted_skill(self, base: GroupResult, prompts: SkillPrompts, grade_fn) -> GroupResult:
        """Deployment-aligned round: PREDICT the skill from the bare problem (no
        traces/label), re-solve N times with it, grade. The baseline group is only
        used for its cached baseline_acc reference -- NOT as distillation input."""
        skill, knowledge, pitfall = await self.predict_skill(base.problem, prompts)
        responses = await self._solve_group(base.problem, skill)
        skilled_correct = [grade_fn(r, base.label, base.domain) for r in responses]
        return GroupResult(
            problem=base.problem, label=base.label, domain=base.domain,
            baseline_responses=base.baseline_responses, baseline_correct=base.baseline_correct,
            skill=skill, correct_skill_text=knowledge, pitfall_skill_text=pitfall,
            skilled_responses=responses, skilled_correct=skilled_correct,
        )

    async def run_with_predicted_skill_batch(
        self, bases: list[GroupResult], prompts: SkillPrompts, grade_fn, *, concurrency: int = 8
    ) -> list[GroupResult]:
        sem = asyncio.Semaphore(concurrency)

        async def _one(b: GroupResult) -> GroupResult:
            async with sem:
                return await self.run_with_predicted_skill(b, prompts, grade_fn)

        return await asyncio.gather(*(_one(b) for b in bases))

    # ---- full per-problem pipeline --------------------------------------- #
    async def run_baseline(self, item: dict, grade_fn) -> GroupResult:
        """Round-0 baseline group (no skill). ``grade_fn(response, label, domain)
        -> bool``. Cached across rounds by the caller (skill-independent)."""
        problem, label = item["problem"], item.get("label", "")
        domain = item.get("domain", "math")
        responses = await self._solve_group(problem, skill="")
        correct = [grade_fn(r, label, domain) for r in responses]
        return GroupResult(
            problem=problem, label=label, domain=domain,
            baseline_responses=responses, baseline_correct=correct,
        )

    async def run_with_skill(self, base: GroupResult, prompts: SkillPrompts, grade_fn) -> GroupResult:
        """Distill the skill from ``base``'s baseline group, re-solve N times with
        it, grade. Returns a NEW GroupResult (base is not mutated) so rounds stay
        independent and comparable."""
        correct_traces = [r for r, ok in zip(base.baseline_responses, base.baseline_correct) if ok]
        failed_traces = [r for r, ok in zip(base.baseline_responses, base.baseline_correct) if not ok]
        skill, correct_text, pitfall_text = await self.distill_skill(
            base.problem, base.label, correct_traces, failed_traces, prompts
        )
        responses = await self._solve_group(base.problem, skill)
        skilled_correct = [grade_fn(r, base.label, base.domain) for r in responses]
        return GroupResult(
            problem=base.problem, label=base.label, domain=base.domain,
            baseline_responses=base.baseline_responses, baseline_correct=base.baseline_correct,
            skill=skill, correct_skill_text=correct_text, pitfall_skill_text=pitfall_text,
            skilled_responses=responses, skilled_correct=skilled_correct,
        )

    # ---- batched over many problems -------------------------------------- #
    async def run_baseline_batch(self, items: list[dict], grade_fn, *, concurrency: int = 8) -> list[GroupResult]:
        sem = asyncio.Semaphore(concurrency)

        async def _one(it: dict) -> GroupResult:
            async with sem:
                return await self.run_baseline(it, grade_fn)

        return await asyncio.gather(*(_one(it) for it in items))

    async def run_with_skill_batch(
        self, bases: list[GroupResult], prompts: SkillPrompts, grade_fn, *, concurrency: int = 8
    ) -> list[GroupResult]:
        sem = asyncio.Semaphore(concurrency)

        async def _one(b: GroupResult) -> GroupResult:
            async with sem:
                return await self.run_with_skill(b, prompts, grade_fn)

        return await asyncio.gather(*(_one(b) for b in bases))
