"""TTS (Test-Time Scaffolding) -- single-turn, SDPO-style group loop.

Standalone (no Megatron / no gradients). "Training" = optimizing the layer-1
skill-generation prompt PAIR across rounds; "testing" = measuring the scaffold
on the same problems (baseline vs with-skill accuracy).

Two skill_mode protocols (config `skill_mode`, default 'predict'):
  * predict  (DEPLOYMENT-ALIGNED, default): the model sees only the BARE problem
    and PREDICTS the knowledge it will need + pitfalls it will hit -- no traces,
    no ground-truth answer. This is EXACTLY what eval_tts.py does at test time,
    so optimizing this signal tunes the prompts for how they are really used.
  * distill  (== examples/SDPO): one problem -> N rollouts -> grade -> summarize
    correct-knowledge from correct traces + pitfalls from failed traces. Needs
    the label; useful as an upper-bound reference, not the deployment protocol.

One ROUND (round r uses skill-gen prompt pair P_r):
  1. build each problem's skill with P_r (predict from bare problem, or distill
     from the cached baseline group)
  2. re-solve each problem N times WITH the skill -> grade
  3. layer-2 optimizer reads (problem, skill, with-skill acc, baseline acc) and
     rewrites the prompt pair -> P_{r+1}

The baseline (no-skill) group is generated ONCE and cached -- it is independent
of the skill-gen prompt, so "how much did the skill help" = with-skill acc minus
baseline acc, measured against a fixed reference each round.

Usage:
    python -m examples.TTS.run_tts --config examples/TTS/configs/single_turn_math_luna.yaml \\
        --train-data /root/math_eval/aime25.jsonl --out-dir examples/TTS/logs
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
from examples.TTS.engines import EngineClient, build_engines
from examples.TTS.grading import grade, make_grader_args
from examples.TTS.optimizer import PromptOptimizer
from examples.TTS.prompts import SkillPrompts
from examples.TTS.scaffold import Scaffold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tts")

DEFAULT_SOLVER_SYSTEM = (
    "You are a careful problem solver. Reason step by step, then give your final "
    "answer inside <answer> and </answer> tags, e.g. <answer>42</answer>. Put ONLY "
    "the final answer inside the tags (for multiple choice, only the option letter)."
)


def make_grade_fn(default_grader: str):
    """grade_fn(response, label, domain) -> bool, routing math->dapo, sci->mcq."""
    math_args = make_grader_args("dapo")
    mcq_args = make_grader_args("mcq")
    default_args = make_grader_args(default_grader)
    sci = {"chemistry", "biology", "physics", "materials", "material"}

    def _grade(response: str, label: str, domain: str) -> bool:
        if domain == "math":
            args = math_args
        elif domain in sci:
            args = mcq_args
        else:
            args = default_args
        return grade(response, label, args)

    return _grade


async def run(cfg: dict, cli: argparse.Namespace) -> None:
    client = EngineClient(timeout=cfg.get("http_timeout", 600.0))
    engines = build_engines(cfg["engines"], client)
    for role in ("solver", "skill_writer", "optimizer"):
        if role not in engines:
            raise ValueError(f"config 'engines' must define role {role!r}; got {list(engines)}")
    # correct/pitfall writers default to the skill_writer engine unless given
    correct_writer = engines.get("correct_writer", engines["skill_writer"])
    pitfall_writer = engines.get("pitfall_writer", engines["skill_writer"])
    logger.info(
        "engines: solver=%s | skill_writer=%s | optimizer=%s",
        engines["solver"].name, engines["skill_writer"].name, engines["optimizer"].name,
    )

    scaffold = Scaffold(
        correct_writer=correct_writer,
        pitfall_writer=pitfall_writer,
        solver=engines["solver"],
        solver_system_prompt=cfg.get("solver_system_prompt", DEFAULT_SOLVER_SYSTEM),
        n_rollouts=cfg.get("n_rollouts", 8),
        skill_max_tokens=cfg.get("skill_max_tokens", 1024),
        solver_max_tokens=cfg.get("solver_max_tokens", 8192),
        skill_temperature=cfg.get("skill_temperature", 0.7),
        solver_temperature=cfg.get("solver_temperature", 1.0),
        max_correct_traces=cfg.get("max_correct_traces", 3),
        max_failed_traces=cfg.get("max_failed_traces", 3),
    )
    optimizer = PromptOptimizer(
        engines["optimizer"],
        max_examples=cfg.get("optimizer_max_examples", 20),
        max_tokens=cfg.get("optimizer_max_tokens", 8192),
        skill_mode=cfg.get("skill_mode", "predict"),
    )
    grade_fn = make_grade_fn(cfg.get("grader", "dapo"))

    items = load_many(cli.train_data, default_domain=cfg.get("default_domain", "math"), limit_each=cli.train_limit)
    logger.info("loaded %d problems, n_rollouts=%d", len(items), scaffold.n_rollouts)

    concurrency = cfg.get("concurrency", 8)
    rounds = cli.rounds if cli.rounds is not None else cfg.get("rounds", 3)

    os.makedirs(cli.out_dir, exist_ok=True)
    stamp = cli.run_name or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    log_path = os.path.join(cli.out_dir, f"tts_{stamp}.jsonl")
    prompts_dir = os.path.join(cli.out_dir, f"prompts_{stamp}")
    os.makedirs(prompts_dir, exist_ok=True)
    log_f = open(log_path, "a")

    def _log(rec: dict):
        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_f.flush()

    def _dump_prompts(tag: str, sp: SkillPrompts):
        with open(os.path.join(prompts_dir, f"{tag}_correct.txt"), "w") as f:
            f.write(sp.correct)
        with open(os.path.join(prompts_dir, f"{tag}_pitfall.txt"), "w") as f:
            f.write(sp.pitfall)

    # ---- round 0: baseline groups (no skill), cached ----
    bases = await scaffold.run_baseline_batch(items, grade_fn, concurrency=concurrency)
    baseline_acc = _mean(b.baseline_acc for b in bases)
    logger.info("[baseline] no-skill acc=%.4f over %d problems (%d rollouts each)",
                baseline_acc, len(bases), scaffold.n_rollouts)
    _log({"phase": "baseline", "baseline_acc": baseline_acc, "n_problems": len(bases),
          "n_rollouts": scaffold.n_rollouts})

    current = SkillPrompts.seed()
    _dump_prompts("round0_seed", current)
    best = {"round": 0, "skilled_acc": baseline_acc, "delta": 0.0, "prompts": current}

    skill_mode = cfg.get("skill_mode", "predict")  # 'predict' (deployment-aligned) | 'distill'
    logger.info("skill_mode=%s (predict = anticipate skill from bare problem, matches test-time)", skill_mode)

    for rd in range(1, rounds + 1):
        # 1-2. build each problem's skill under the CHOSEN protocol + re-solve with it.
        #   predict: anticipate skill from the bare problem (no traces/label) -- SAME as
        #            eval_tts, so the optimizer tunes prompts for how they're really used.
        #   distill: summarize the cached baseline group's traces (needs label).
        if skill_mode == "predict":
            results = await scaffold.run_with_predicted_skill_batch(bases, current, grade_fn, concurrency=concurrency)
        else:
            results = await scaffold.run_with_skill_batch(bases, current, grade_fn, concurrency=concurrency)
        skilled_acc = _mean(r.skilled_acc for r in results)
        delta = skilled_acc - baseline_acc
        logger.info("[round %d] with-skill acc=%.4f (baseline=%.4f, delta=%+.4f)",
                    rd, skilled_acc, baseline_acc, delta)

        # 3. optimize the prompt pair
        step = await optimizer.step(current, results)
        logger.info("[round %d] optimizer changed=%s | diagnosis: %s",
                    rd, step.changed, step.diagnosis[:300])

        _log({
            "round": rd, "phase": "round",
            "baseline_acc": baseline_acc, "skilled_acc": skilled_acc, "delta": delta,
            "optimizer_changed": step.changed, "diagnosis": step.diagnosis,
            "prompts_used": {"correct": current.correct, "pitfall": current.pitfall},
            "next_prompts": {"correct": step.new_prompts.correct, "pitfall": step.new_prompts.pitfall},
            "per_problem": [
                {
                    "problem": r.problem[:400], "domain": r.domain,
                    "baseline_acc": r.baseline_acc, "skilled_acc": r.skilled_acc, "delta": r.delta,
                    "skill": r.skill[:1200],
                }
                for r in results
            ],
        })

        if skilled_acc > best["skilled_acc"]:
            best = {"round": rd, "skilled_acc": skilled_acc, "delta": delta, "prompts": current}

        current = step.new_prompts
        _dump_prompts(f"round{rd}", current)

    # ---- save the best prompt pair ----
    _dump_prompts("BEST", best["prompts"])
    logger.info("BEST: round=%d skilled_acc=%.4f delta=%+.4f (baseline=%.4f)",
                best["round"], best["skilled_acc"], best["delta"], baseline_acc)
    _log({"phase": "final_best", "round": best["round"], "skilled_acc": best["skilled_acc"],
          "delta": best["delta"], "baseline_acc": baseline_acc})
    log_f.close()
    await client.aclose()
    logger.info("logs: %s | prompts: %s", log_path, prompts_dir)


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="TTS single-turn SDPO-style group loop")
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-data", nargs="+", required=True, help="SDPO-format JSONL problem file(s)")
    ap.add_argument("--rounds", type=int, default=None, help="override rounds from config")
    ap.add_argument("--train-limit", type=int, default=0, help="cap problems per file (0=all)")
    ap.add_argument("--out-dir", default="examples/TTS/logs")
    ap.add_argument("--run-name", default="", help="log filename stamp (default: UTC timestamp)")
    cli = ap.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)
    asyncio.run(run(cfg, cli))


if __name__ == "__main__":
    main()
