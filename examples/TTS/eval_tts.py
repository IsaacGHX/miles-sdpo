"""TEST-TIME evaluation of the optimized skill-generation prompts.

This is the deployment test: the skill-gen prompt PAIR was optimized on the DAPO
hard set (train); here we measure whether it GENERALIZES to a held-out set
(AIME25) under the REAL deployment protocol -- the model sees only the bare
problem, PREDICTS the skill up front (no rollouts, no ground-truth answer, since
deployment has neither), prepends it, and solves.

We report FOUR arms so you can see what each half of the skill buys:
  * baseline        -- no skill (bare problem)
  * knowledge_only  -- predicted [Knowledge/Rule] blocks prepended
  * pitfall_only    -- predicted [Error]/[Rule]/[Example] blocks prepended
  * combined        -- both

The skill-gen system prompts come from a prompts dir written by run_tts.py
(e.g. examples/TTS/logs/prompts_luna_solopt/BEST_correct.txt / BEST_pitfall.txt),
so we test EXACTLY the prompts the optimizer produced -- which only ever saw DAPO
training data, never AIME. Each arm is measured over n_eval rollouts/problem.

Usage:
    python -m examples.TTS.eval_tts \\
        --config examples/TTS/configs/single_turn_math_luna.yaml \\
        --eval-data examples/TTS/logs/aime25.jsonl \\
        --prompts-dir examples/TTS/logs/prompts_luna_solopt --prompts-tag BEST \\
        --n-eval 8 --out-dir examples/TTS/logs --run-name aime25_solopt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time

import yaml

from examples.TTS.data import load_many
from examples.TTS.engines import Engine, EngineClient, build_engines
from examples.TTS.grading import grade, make_grader_args
from examples.TTS.prompts import (
    SkillPrompts,
    combine_skill,
    predict_knowledge_user,
    predict_pitfall_user,
    skill_augmented_user,
)
from examples.TTS.run_tts import DEFAULT_SOLVER_SYSTEM, make_grade_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tts-eval")

ARMS = ("baseline", "knowledge_only", "pitfall_only", "combined")


def load_prompts_dir(prompts_dir: str, tag: str) -> SkillPrompts:
    """Load the optimized prompt pair (<tag>_correct.txt / <tag>_pitfall.txt),
    falling back to the seed for any half that is missing."""
    seed = SkillPrompts.seed()
    correct_p = os.path.join(prompts_dir, f"{tag}_correct.txt")
    pitfall_p = os.path.join(prompts_dir, f"{tag}_pitfall.txt")
    correct = open(correct_p).read().strip() if os.path.exists(correct_p) else seed.correct
    pitfall = open(pitfall_p).read().strip() if os.path.exists(pitfall_p) else seed.pitfall
    if not os.path.exists(correct_p):
        logger.warning("missing %s; using seed correct prompt", correct_p)
    if not os.path.exists(pitfall_p):
        logger.warning("missing %s; using seed pitfall prompt", pitfall_p)
    return SkillPrompts(correct=correct, pitfall=pitfall)


class Evaluator:
    def __init__(
        self,
        predictor: Engine,
        solver: Engine,
        prompts: SkillPrompts,
        solver_system_prompt: str,
        *,
        n_eval: int = 8,
        skill_max_tokens: int = 2048,
        solver_max_tokens: int = 8192,
        skill_temperature: float = 0.7,
        solver_temperature: float = 1.0,
    ):
        self.predictor = predictor
        self.solver = solver
        self.prompts = prompts
        self.solver_system_prompt = solver_system_prompt
        self.n_eval = n_eval
        self.skill_max_tokens = skill_max_tokens
        self.solver_max_tokens = solver_max_tokens
        self.skill_temperature = skill_temperature
        self.solver_temperature = solver_temperature

    async def _predict_skills(self, problem: str) -> tuple[str, str]:
        """Predict (knowledge_text, pitfall_text) from the bare problem, using the
        optimized system prompts + the predict-mode user turns."""

        async def _k() -> str:
            try:
                return await self.predictor.complete(
                    self.prompts.correct, predict_knowledge_user(problem),
                    temperature=self.skill_temperature, max_tokens=self.skill_max_tokens,
                )
            except Exception as e:
                logger.warning(f"knowledge predict failed: {e!r}")
                return ""

        async def _p() -> str:
            try:
                return await self.predictor.complete(
                    self.prompts.pitfall, predict_pitfall_user(problem),
                    temperature=self.skill_temperature, max_tokens=self.skill_max_tokens,
                )
            except Exception as e:
                logger.warning(f"pitfall predict failed: {e!r}")
                return ""

        return await asyncio.gather(_k(), _p())

    async def _solve_once(self, problem: str, skill: str) -> str:
        user = skill_augmented_user(problem, skill) if skill else problem
        try:
            res = await self.solver.chat(
                [{"role": "system", "content": self.solver_system_prompt},
                 {"role": "user", "content": user}],
                temperature=self.solver_temperature, max_tokens=self.solver_max_tokens,
            )
            return res.text
        except Exception as e:
            logger.warning(f"solve failed: {e!r}")
            return ""

    async def _acc(self, problem: str, skill: str, label: str, domain: str, grade_fn) -> float:
        responses = await asyncio.gather(*(self._solve_once(problem, skill) for _ in range(self.n_eval)))
        correct = [grade_fn(r, label, domain) for r in responses]
        return sum(1 for c in correct if c) / len(correct) if correct else 0.0

    async def eval_problem(self, item: dict, grade_fn) -> dict:
        problem, label = item["problem"], item.get("label", "")
        domain = item.get("domain", "math")
        knowledge, pitfall = await self._predict_skills(problem)
        skills = {
            "baseline": "",
            "knowledge_only": combine_skill(knowledge, ""),
            "pitfall_only": combine_skill("", pitfall),
            "combined": combine_skill(knowledge, pitfall),
        }
        accs = await asyncio.gather(
            *(self._acc(problem, skills[a], label, domain, grade_fn) for a in ARMS)
        )
        return {
            "problem": problem[:400], "label": label, "domain": domain,
            "predicted_knowledge": knowledge, "predicted_pitfall": pitfall,
            "acc": dict(zip(ARMS, accs)),
        }

    async def eval_batch(self, items: list[dict], grade_fn, *, concurrency: int = 8) -> list[dict]:
        sem = asyncio.Semaphore(concurrency)

        async def _one(it: dict) -> dict:
            async with sem:
                return await self.eval_problem(it, grade_fn)

        return await asyncio.gather(*(_one(it) for it in items))


async def run(cfg: dict, cli: argparse.Namespace) -> None:
    client = EngineClient(timeout=cfg.get("http_timeout", 600.0))
    engines = build_engines(cfg["engines"], client)
    for role in ("solver", "skill_writer"):
        if role not in engines:
            raise ValueError(f"config 'engines' must define role {role!r}; got {list(engines)}")
    # the skill PREDICTOR at test time = the skill_writer engine (same model that
    # distilled skills at train time; deployment just changes the user turn)
    predictor = engines["skill_writer"]
    solver = engines["solver"]
    logger.info("eval engines: solver=%s | predictor=%s", solver.name, predictor.name)

    prompts = load_prompts_dir(cli.prompts_dir, cli.prompts_tag)
    logger.info("loaded prompts from %s (tag=%s): correct=%dch pitfall=%dch",
                cli.prompts_dir, cli.prompts_tag, len(prompts.correct), len(prompts.pitfall))

    ev = Evaluator(
        predictor, solver, prompts,
        cfg.get("solver_system_prompt", DEFAULT_SOLVER_SYSTEM),
        n_eval=cli.n_eval,
        skill_max_tokens=cfg.get("skill_max_tokens", 2048),
        solver_max_tokens=cfg.get("solver_max_tokens", 8192),
        skill_temperature=cfg.get("skill_temperature", 0.7),
        solver_temperature=cfg.get("solver_temperature", 1.0),
    )
    grade_fn = make_grade_fn(cfg.get("grader", "dapo"))

    items = load_many(cli.eval_data, default_domain=cfg.get("default_domain", "math"), limit_each=cli.eval_limit)
    logger.info("loaded %d eval problems, n_eval=%d, arms=%s", len(items), cli.n_eval, list(ARMS))

    results = await ev.eval_batch(items, grade_fn, concurrency=cfg.get("concurrency", 8))

    means = {a: _mean(r["acc"][a] for r in results) for a in ARMS}
    logger.info("=== AIME eval (%d problems x %d rollouts) ===", len(results), cli.n_eval)
    for a in ARMS:
        logger.info("  %-16s acc=%.4f  (delta vs baseline %+.4f)", a, means[a], means[a] - means["baseline"])

    os.makedirs(cli.out_dir, exist_ok=True)
    stamp = cli.run_name or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out_path = os.path.join(cli.out_dir, f"eval_{stamp}.jsonl")
    with open(out_path, "w") as f:
        f.write(json.dumps({
            "phase": "eval_summary", "prompts_dir": cli.prompts_dir, "prompts_tag": cli.prompts_tag,
            "n_problems": len(results), "n_eval": cli.n_eval, "means": means,
        }, ensure_ascii=False) + "\n")
        for r in results:
            f.write(json.dumps({"phase": "eval_problem", **r}, ensure_ascii=False) + "\n")
    await client.aclose()
    logger.info("eval log: %s", out_path)


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="TTS test-time eval (predict skill from bare problem)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-data", nargs="+", required=True, help="held-out SDPO-format JSONL (e.g. aime25)")
    ap.add_argument("--prompts-dir", required=True, help="dir with <tag>_correct.txt / <tag>_pitfall.txt")
    ap.add_argument("--prompts-tag", default="BEST", help="which prompt pair (BEST, round1, round0_seed, ...)")
    ap.add_argument("--n-eval", type=int, default=8, help="rollouts per problem per arm")
    ap.add_argument("--eval-limit", type=int, default=0, help="cap problems (0=all)")
    ap.add_argument("--out-dir", default="examples/TTS/logs")
    ap.add_argument("--run-name", default="", help="log filename stamp (default: UTC timestamp)")
    cli = ap.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)
    asyncio.run(run(cfg, cli))


if __name__ == "__main__":
    main()
