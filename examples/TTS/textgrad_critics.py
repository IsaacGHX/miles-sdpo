"""The two DECOUPLED critics of the TextGrad-style code training loop (see
textgrad_prompts.py's module docstring). Each critic is a pure step:
(current_variable, this-round's evidence) -> (diagnosis, new_variable). Neither
critic is given the other's output, and both run CONCURRENTLY each round
(see textgrad_train.py) -- the two variables are optimized independently, as
required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from examples.TTS.engines import Engine
from examples.TTS.prompts import (
    OPTIMIZER_SYSTEM_PREDICT,
    SkillPrompts,
    optimizer_user,
    parse_optimizer_output,
)
from examples.TTS.textgrad_prompts import (
    SKILL_CRITIC_SYSTEM,
    SYS_PROMPT_CRITIC_SYSTEM,
    SysPromptBundle,
    parse_sys_prompt_critic_output,
    skill_critic_user,
    sys_prompt_critic_user,
)

logger = logging.getLogger(__name__)


@dataclass
class CriticStep:
    old: object
    new: object
    diagnosis: str
    changed: bool


class SysPromptCritic:
    """Reads WITH-skill trajectories (the agent's real deployed configuration)
    and rewrites the system prompt + tool description."""

    def __init__(self, engine: Engine, *, max_examples: int = 16, max_tokens: int = 8192):
        self.engine = engine
        self.max_examples = max_examples
        self.max_tokens = max_tokens

    async def step(self, current: SysPromptBundle, examples: list[dict]) -> CriticStep:
        if not examples:
            logger.warning("SysPromptCritic.step: no examples; keeping current bundle.")
            return CriticStep(current, current, "", False)
        user = sys_prompt_critic_user(
            current.system_prompt, current.tool_description, examples, max_examples=self.max_examples
        )
        try:
            raw = await self.engine.complete(
                SYS_PROMPT_CRITIC_SYSTEM, user, max_tokens=self.max_tokens, temperature=0.4
            )
        except Exception as e:
            logger.warning(f"SysPromptCritic engine call failed ({e!r}); keeping current bundle.")
            return CriticStep(current, current, "", False)
        diagnosis, new = parse_sys_prompt_critic_output(raw, fallback=current)
        changed = (new.system_prompt.strip() != current.system_prompt.strip()) or (
            new.tool_description.strip() != current.tool_description.strip()
        )
        return CriticStep(current, new, diagnosis, changed)


class SkillCritic:
    """Reads NO-skill trajectories (the agent's unaided attempt -- evidence of
    what it actually struggles with) + the skill predicted for each problem,
    and rewrites the skill-prediction prompt pair. Reuses prompts.py's
    predict-mode optimizer system prompt + ===CORRECT===/===PITFALL=== parser
    (identical output contract to the single-turn TTS loop)."""

    def __init__(self, engine: Engine, *, max_examples: int = 16, max_tokens: int = 8192):
        self.engine = engine
        self.max_examples = max_examples
        self.max_tokens = max_tokens

    async def step(self, current: SkillPrompts, examples: list[dict]) -> CriticStep:
        if not examples:
            logger.warning("SkillCritic.step: no examples; keeping current prompts.")
            return CriticStep(current, current, "", False)
        user = skill_critic_user(current.correct, current.pitfall, examples, max_examples=self.max_examples)
        try:
            raw = await self.engine.complete(
                SKILL_CRITIC_SYSTEM, user, max_tokens=self.max_tokens, temperature=0.4
            )
        except Exception as e:
            logger.warning(f"SkillCritic engine call failed ({e!r}); keeping current prompts.")
            return CriticStep(current, current, "", False)
        diagnosis, new = parse_optimizer_output(raw, fallback=current)
        changed = (new.correct.strip() != current.correct.strip()) or (new.pitfall.strip() != current.pitfall.strip())
        return CriticStep(current, new, diagnosis, changed)
