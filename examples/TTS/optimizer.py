"""LAYER-2 scaffold: the meta-optimizer that rewrites the skill-generation
prompt PAIR (correct-knowledge + pitfall) from a round's group results.

It optimizes PROMPTS, not weights -- the whole point of test-time scaffolding.
One step:
    diagnosis, new_prompts = optimizer.step(current_prompts, group_results)
where ``group_results`` is the list of GroupResult from layer-1 (each with a
baseline group, a distilled skill, and a with-skill group). The optimizer engine
(any role, local or remote) reads the current prompt pair + a sample of
(problem, skill, with-skill acc, baseline acc) and proposes an improved pair,
used for the NEXT round's distillation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from examples.TTS.engines import Engine
from examples.TTS.prompts import (
    OPTIMIZER_SYSTEM,
    OPTIMIZER_SYSTEM_PREDICT,
    SkillPrompts,
    optimizer_user,
    parse_optimizer_output,
)
from examples.TTS.scaffold import GroupResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizerStep:
    old_prompts: SkillPrompts
    new_prompts: SkillPrompts
    diagnosis: str
    changed: bool
    mean_baseline_acc: float
    mean_skilled_acc: float

    @property
    def delta(self) -> float:
        return self.mean_skilled_acc - self.mean_baseline_acc


class PromptOptimizer:
    def __init__(self, engine: Engine, *, max_examples: int = 20, max_tokens: int = 8192,
                 skill_mode: str = "predict"):
        self.engine = engine
        self.max_examples = max_examples
        self.max_tokens = max_tokens
        # match the optimizer's mental model of the scaffold to how skills are built
        self.system = OPTIMIZER_SYSTEM_PREDICT if skill_mode == "predict" else OPTIMIZER_SYSTEM

    async def step(self, current: SkillPrompts, results: list[GroupResult]) -> OptimizerStep:
        mean_base = _mean(r.baseline_acc for r in results)
        mean_skill = _mean(r.skilled_acc for r in results)

        examples = [
            {
                "problem": r.problem,
                "skill": r.skill,
                "acc": r.skilled_acc,
                "baseline_acc": r.baseline_acc,
            }
            for r in results
        ]
        if not examples:
            logger.warning("optimizer.step: no results; keeping current prompts.")
            return OptimizerStep(current, current, "", False, mean_base, mean_skill)

        user = optimizer_user(current.correct, current.pitfall, examples, max_examples=self.max_examples)
        try:
            raw = await self.engine.complete(self.system, user, max_tokens=self.max_tokens, temperature=0.4)
        except Exception as e:
            logger.warning(f"optimizer engine call failed ({e!r}); keeping current prompts.")
            return OptimizerStep(current, current, "", False, mean_base, mean_skill)

        diagnosis, new = parse_optimizer_output(raw, fallback=current)
        changed = (new.correct.strip() != current.correct.strip()) or (new.pitfall.strip() != current.pitfall.strip())
        if not changed:
            logger.info("optimizer.step: proposed prompts identical/unusable; keeping current.")
        return OptimizerStep(current, new, diagnosis, changed, mean_base, mean_skill)


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
