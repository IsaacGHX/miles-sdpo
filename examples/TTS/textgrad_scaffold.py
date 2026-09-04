"""LAYER-1 scaffold for the TextGrad-style CODE training loop: per problem,
predict a skill (deployment protocol -- bare problem only, see prompts.py),
then run pass@k multi-turn trajectories under FOUR arms, mirroring the
single-turn TTS harness's arm split (examples/TTS/eval_tts.py's
baseline/knowledge_only/pitfall_only/combined) so "how much does the skill
help, and which half of it" is answered the same way for multi-turn code as
it is for single-turn math:

  * no_skill       -- bare problem, no skill prepended (the reference).
  * knowledge_only -- predicted [Knowledge/Rule] blocks only.
  * pitfall_only   -- predicted [Error]/[Rule]/[Example] blocks only.
  * combined       -- both (this is the "real deployed configuration" the
                       SysPromptCritic learns from -- see textgrad_train.py).

The skill is PREDICTED ONCE per problem (one knowledge call + one pitfall
call); the four arms just choose which half(s) to splice into the solver's
user turn, so arm count does not multiply skill-prediction cost -- only
solver rollout cost (4 arms x k trajectories instead of 2).

Two independent TextGrad variables are exercised here (see textgrad_prompts.py):
  * SysPromptBundle governs ALL FOUR arms (it's always in context).
  * SkillPrompts only governs what gets predicted/prepended in the three
    skill-bearing arms.

Each problem gets `k` trajectories PER ARM, run concurrently under one
semaphore (shared sandbox sidecar -- see multi_turn_code.py). pass@k for a
problem/arm = at least one of its k trajectories graded correct (the standard
"any of k" code-gen pass@k; with k modest by cost, not the combinatorial n>k
estimator).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from examples.TTS.engines import Engine
from examples.TTS.multi_turn_code import Trajectory, run_trajectory_and_grade
from examples.TTS.prompts import SkillPrompts, combine_skill, predict_knowledge_user, predict_pitfall_user
from examples.TTS.textgrad_prompts import SysPromptBundle

logger = logging.getLogger(__name__)

ARMS = ("no_skill", "knowledge_only", "pitfall_only", "combined")


@dataclass
class CodeRoundResult:
    problem: str
    test_cases: list[dict] = field(default_factory=list)
    difficulty: str = "unknown"
    knowledge: str = ""
    pitfall: str = ""
    no_skill_trajs: list[Trajectory] = field(default_factory=list)
    knowledge_only_trajs: list[Trajectory] = field(default_factory=list)
    pitfall_only_trajs: list[Trajectory] = field(default_factory=list)
    combined_trajs: list[Trajectory] = field(default_factory=list)

    @property
    def skill(self) -> str:
        """The combined skill text (both halves) -- what the "real deployed
        configuration" (the `combined` arm) actually saw."""
        return combine_skill(self.knowledge, self.pitfall)

    def trajs(self, arm: str) -> list[Trajectory]:
        return getattr(self, f"{arm}_trajs")

    def pass_at_k(self, arm: str) -> bool:
        return any(t.correct for t in self.trajs(arm))

    @property
    def no_skill_pass_at_k(self) -> bool:
        return self.pass_at_k("no_skill")

    @property
    def with_skill_pass_at_k(self) -> bool:
        """Back-compat alias for the `combined` arm (the prior 2-arm API's
        "with skill")."""
        return self.pass_at_k("combined")

    def representative(self, arm: str) -> Trajectory | None:
        """A WRONG trajectory if one exists (carries the optimization signal
        for a critic), else the first one."""
        trajs = self.trajs(arm)
        if not trajs:
            return None
        for t in trajs:
            if not t.correct:
                return t
        return trajs[0]


class CodeScaffold:
    def __init__(
        self,
        solver: Engine,
        skill_predictor: Engine,
        *,
        k: int = 4,
        max_turns: int = 8,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        skill_max_tokens: int = 2048,
        skill_temperature: float = 0.7,
        concurrency: int = 16,
    ):
        self.solver = solver
        self.skill_predictor = skill_predictor
        self.k = k
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.skill_max_tokens = skill_max_tokens
        self.skill_temperature = skill_temperature
        self.concurrency = concurrency
        self._sem = asyncio.Semaphore(concurrency)

    async def _predict_skill(self, problem: str, skill_prompts: SkillPrompts) -> tuple[str, str]:
        """Predict ONCE per problem: (knowledge_text, pitfall_text). The four
        arms below just choose which half(s) to splice in -- prediction cost
        does not scale with arm count."""

        async def _k() -> str:
            try:
                return await self.skill_predictor.complete(
                    skill_prompts.correct, predict_knowledge_user(problem),
                    temperature=self.skill_temperature, max_tokens=self.skill_max_tokens,
                )
            except Exception as e:
                logger.warning(f"knowledge predict failed: {e!r}")
                return ""

        async def _p() -> str:
            try:
                return await self.skill_predictor.complete(
                    skill_prompts.pitfall, predict_pitfall_user(problem),
                    temperature=self.skill_temperature, max_tokens=self.skill_max_tokens,
                )
            except Exception as e:
                logger.warning(f"pitfall predict failed: {e!r}")
                return ""

        return await asyncio.gather(_k(), _p())

    async def _run_k(self, sys_bundle: SysPromptBundle, problem: str, test_cases: list[dict], skill: str) -> list[Trajectory]:
        async def _one() -> Trajectory:
            async with self._sem:
                return await run_trajectory_and_grade(
                    self.solver, sys_bundle.system_prompt, problem, test_cases,
                    skill=skill, tool_description=sys_bundle.tool_description,
                    max_turns=self.max_turns, max_tokens=self.max_tokens, temperature=self.temperature,
                )

        return await asyncio.gather(*(_one() for _ in range(self.k)))

    async def run_problem(
        self, item: dict, sys_bundle: SysPromptBundle, skill_prompts: SkillPrompts
    ) -> CodeRoundResult:
        problem, test_cases = item["problem"], item["test_cases"]
        # gated by the SAME semaphore as solver trajectories: run_batch gathers
        # ALL problems at once, so without this, N problems -> 2N unbounded
        # concurrent HTTP calls fire the instant the batch starts (422 problems
        # -> 844 simultaneous connections), independent of `concurrency`. That
        # burst is what actually killed the process (not an idle-guard), not
        # the (properly capped) solver rollouts.
        async with self._sem:
            knowledge, pitfall = await self._predict_skill(problem, skill_prompts)
        skills = {
            "no_skill": "",
            "knowledge_only": combine_skill(knowledge, ""),
            "pitfall_only": combine_skill("", pitfall),
            "combined": combine_skill(knowledge, pitfall),
        }
        trajs_by_arm = await asyncio.gather(*(self._run_k(sys_bundle, problem, test_cases, skills[a]) for a in ARMS))
        return CodeRoundResult(
            problem=problem, test_cases=test_cases, difficulty=item.get("difficulty", "unknown") or "unknown",
            knowledge=knowledge, pitfall=pitfall,
            **{f"{a}_trajs": trajs for a, trajs in zip(ARMS, trajs_by_arm)},
        )

    async def run_batch(
        self, items: list[dict], sys_bundle: SysPromptBundle, skill_prompts: SkillPrompts
    ) -> list[CodeRoundResult]:
        return await asyncio.gather(*(self.run_problem(it, sys_bundle, skill_prompts) for it in items))
