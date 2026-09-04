"""TextGrad-style multi-turn CODE training loop (examples/TTS/).

Mirrors the miles training loop's SHAPE (wandb logging, per-round dumps
treated as "checkpoints") but is a pure-API harness: no Megatron, no
gradients, no GPU. What gets "trained" here are TWO TEXT variables (see
textgrad_prompts.py for the full rationale, following
arxiv.org/abs/2406.07496's "backward through text"):

  1. SysPromptBundle  -- the solver's system prompt + code_interpreter tool
     description. In context on EVERY trajectory, all four arms alike.
  2. SkillPrompts     -- the skill-PREDICTION prompt pair (deployment
     protocol: predict from the bare problem, no traces/answer -- see
     examples/TTS/prompts.py). Governs the three skill-bearing arms below.

FOUR arms per problem per round (mirrors the single-turn TTS harness's
baseline/knowledge_only/pitfall_only/combined split, see textgrad_scaffold.py):
  no_skill / knowledge_only / pitfall_only / combined -- the skill is
  predicted ONCE per problem (one knowledge call + one pitfall call); the arms
  just choose which half(s) to splice into the solver's user turn.

Each ROUND (round r uses variables SysPromptBundle_r, SkillPrompts_r):
  1. For every training problem: predict a skill, then run k multi-turn
     trajectories PER ARM (multi_turn_code.py), grade each (tool-mandatory:
     examples/SDPO/reward.py's contract via grading.grade_code).
  2. pass@k is computed for ALL FOUR arms -- "how much did the skill/prompts
     help, and via which half" = each skill-bearing arm's pass@k minus
     no_skill's.
  3. TWO critics propose the NEXT round's variables, CONCURRENTLY and
     independently (neither sees the other's output -- decoupled per the
     task's requirement):
       * SysPromptCritic reads `combined`-arm trajectories (the real deployed
         configuration) -> rewrites SysPromptBundle.
       * SkillCritic reads NO-skill trajectories (evidence of what the solver
         actually struggles with, unaided) + this round's predicted skill ->
         rewrites SkillPrompts.
  4. Both new variables + the full per-problem trace are dumped to disk (the
     "checkpoint": prompts_<run>/round<r>_{system,tool,skill_correct,skill_pitfall}.txt)
     and logged to wandb (if configured) under train/* -- pass@k, deltas,
     turn counts -- matching the miles trainer's rollout/eval namespacing
     spirit (see the "wandb logging conventions" investigation note).

Tool infra: reuses the ALREADY-RUNNING sdpo-react-sandbox sidecar
(127.0.0.1:8420, shared with whatever training job is using it -- this script
launches NO new container/port) via examples/SDPO_ReAct/tools/code/client.py +
judge.py, imported standalone (miles.utils.http_utils._http_client is set
manually here since no miles rollout process initializes it for us -- see the
"http client None breaks standalone tools" pitfall).

Usage:
    python -m examples.TTS.textgrad_train \\
        --config examples/TTS/configs/textgrad_code_luna.yaml \\
        --train-data /fsx/data/haoxiang.zhang/code_data/livecodebench_train.jsonl \\
        --eval-data /fsx/data/haoxiang.zhang/code_data_v6eval/livecodebench_eval.jsonl \\
        --rounds 3 --out-dir examples/TTS/logs --run-name textgrad_code_luna
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time

import httpx
import yaml

import miles.utils.http_utils as http_utils
from examples.TTS.data import load_code_dataset_jsonl
from examples.TTS.engines import EngineClient, build_engines
from examples.TTS.prompts import SkillPrompts
from examples.TTS.textgrad_critics import SkillCritic, SysPromptCritic
from examples.TTS.textgrad_metrics import compute_metrics, perf_metrics
from examples.TTS.textgrad_prompts import SysPromptBundle
from examples.TTS.textgrad_scaffold import ARMS, CodeRoundResult, CodeScaffold
from miles.utils.metric_utils import dict_add_prefix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("textgrad")


def _ensure_http_client(timeout: float = 120.0) -> None:
    """The SDPO_ReAct tool clients (run_code/grade_code) call
    miles.utils.http_utils.post, which reads a module-level _http_client that
    is normally set by init_http_client(args) inside a running miles rollout.
    This standalone script never runs that init, so set it directly -- same
    fix as examples/TTS/eval_tts.py's engines own their OWN httpx client for
    the API side; this is the separate client the SANDBOX/JUDGE tools need."""
    if http_utils._http_client is None:
        http_utils._http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))


def _dump_text(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def _round_examples_for_sys_critic(results: list[CodeRoundResult]) -> list[dict]:
    """Reads the `combined` arm -- the real deployed configuration (both skill
    halves in context) -- same as before the 4-arm split."""
    out = []
    for r in results:
        traj = r.representative("combined")
        if traj is None:
            continue
        out.append(
            {
                "problem": r.problem,
                "transcript": traj.transcript_text(),
                "correct": traj.correct,
                "n_turns": traj.n_turns,
            }
        )
    return out


def _round_examples_for_skill_critic(results: list[CodeRoundResult]) -> list[dict]:
    out = []
    for r in results:
        traj = r.representative("no_skill")
        if traj is None:
            continue
        out.append(
            {
                "problem": r.problem,
                "skill": r.skill,
                "no_skill_transcript": traj.transcript_text(),
                "no_skill_correct": r.no_skill_pass_at_k,
                "with_skill_correct": r.with_skill_pass_at_k,
            }
        )
    return out


async def run(cfg: dict, cli: argparse.Namespace) -> None:
    _ensure_http_client(timeout=cfg.get("sandbox_timeout", 120.0))

    client = EngineClient(timeout=cfg.get("http_timeout", 600.0))
    engines = build_engines(cfg["engines"], client)
    for role in ("solver", "skill_predictor", "sys_critic", "skill_critic"):
        if role not in engines:
            raise ValueError(f"config 'engines' must define role {role!r}; got {list(engines)}")
    logger.info(
        "engines: solver=%s | skill_predictor=%s | sys_critic=%s | skill_critic=%s",
        engines["solver"].name, engines["skill_predictor"].name,
        engines["sys_critic"].name, engines["skill_critic"].name,
    )

    scaffold = CodeScaffold(
        engines["solver"], engines["skill_predictor"],
        k=cli.k if cli.k is not None else cfg.get("k", 4),
        max_turns=cfg.get("max_turns", 8),
        max_tokens=cfg.get("solver_max_tokens", 4096),
        temperature=cfg.get("solver_temperature", 1.0),
        skill_max_tokens=cfg.get("skill_max_tokens", 2048),
        skill_temperature=cfg.get("skill_temperature", 0.7),
        concurrency=cli.concurrency if cli.concurrency is not None else cfg.get("concurrency", 16),
    )
    sys_critic = SysPromptCritic(
        engines["sys_critic"],
        max_examples=cfg.get("critic_max_examples", 16),
        max_tokens=cfg.get("sys_critic_max_tokens", 8192),
    )
    skill_critic = SkillCritic(
        engines["skill_critic"],
        max_examples=cfg.get("critic_max_examples", 16),
        max_tokens=cfg.get("skill_critic_max_tokens", 8192),
    )

    items = load_code_dataset_jsonl(cli.train_data, limit=cli.train_limit)
    logger.info("loaded %d training problems, k=%d, concurrency=%d", len(items), scaffold.k, scaffold.concurrency)

    rounds = cli.rounds if cli.rounds is not None else cfg.get("rounds", 3)

    os.makedirs(cli.out_dir, exist_ok=True)
    stamp = cli.run_name or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    log_path = os.path.join(cli.out_dir, f"textgrad_{stamp}.jsonl")
    prompts_dir = os.path.join(cli.out_dir, f"prompts_{stamp}")
    os.makedirs(prompts_dir, exist_ok=True)
    log_f = open(log_path, "a")

    def _log(rec: dict):
        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_f.flush()

    wandb_run = None
    if cfg.get("use_wandb", False):
        import wandb

        wandb_run = wandb.init(
            project=cfg.get("wandb_project", "miles-sdpo"),
            group=cfg.get("wandb_group", f"tts-textgrad-{stamp}"),
            name=stamp,
            config={"k": scaffold.k, "rounds": rounds, "n_problems": len(items), **cfg},
        )

    def _wandb_log(metrics: dict, step: int):
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)

    def _dump_round(tag: str, sys_bundle: SysPromptBundle, skill_prompts: SkillPrompts):
        _dump_text(os.path.join(prompts_dir, f"{tag}_system.txt"), sys_bundle.system_prompt)
        _dump_text(os.path.join(prompts_dir, f"{tag}_tool.txt"), sys_bundle.tool_description)
        _dump_text(os.path.join(prompts_dir, f"{tag}_skill_correct.txt"), skill_prompts.correct)
        _dump_text(os.path.join(prompts_dir, f"{tag}_skill_pitfall.txt"), skill_prompts.pitfall)

    sys_bundle = SysPromptBundle.seed()
    skill_prompts = SkillPrompts.seed()
    _dump_round("round0_seed", sys_bundle, skill_prompts)

    best = {"round": 0, "with_skill_pass_at_1": 0.0, "sys_bundle": sys_bundle, "skill_prompts": skill_prompts}

    for rd in range(1, rounds + 1):
        t0 = time.time()
        results = await scaffold.run_batch(items, sys_bundle, skill_prompts)
        elapsed = time.time() - t0
        metrics = compute_metrics(results, scaffold.k, prefix="train")
        metrics |= dict_add_prefix(perf_metrics(results, elapsed), "perf/train_")
        # pass@1 (unbiased estimate, not "did >=1 of k pass") is the headline
        # number -- same role as miles' val-core pass@1 (see metrics.py).
        # combined = both skill halves = the "real deployed configuration"
        # this harness optimizes for; report all 4 arms so knowledge-only and
        # pitfall-only's individual contributions stay visible (not just the
        # combined delta).
        p1 = {arm: metrics.get(f"train/{arm}_pass@1", 0.0) for arm in ARMS}
        with_skill_p1 = p1["combined"]
        delta_p1 = metrics.get("train/delta_combined_pass@1", with_skill_p1 - p1["no_skill"])
        logger.info(
            "[round %d] pass@1: no_skill=%.4f  knowledge_only=%.4f (%+.4f)  pitfall_only=%.4f (%+.4f)  "
            "combined=%.4f (%+.4f)  (k=%d, %.1fs, %d problems)",
            rd, p1["no_skill"],
            p1["knowledge_only"], p1["knowledge_only"] - p1["no_skill"],
            p1["pitfall_only"], p1["pitfall_only"] - p1["no_skill"],
            p1["combined"], delta_p1,
            scaffold.k, elapsed, len(results),
        )
        for key in sorted(k for k in metrics if k.startswith("domain/")):
            logger.info("    %s = %.4f", key, metrics[key])

        sys_examples = _round_examples_for_sys_critic(results)
        skill_examples = _round_examples_for_skill_critic(results)

        # the two critics run CONCURRENTLY and independently -- decoupled update
        sys_step, skill_step = await asyncio.gather(
            sys_critic.step(sys_bundle, sys_examples),
            skill_critic.step(skill_prompts, skill_examples),
        )
        logger.info("[round %d] sys_critic changed=%s | %s", rd, sys_step.changed, sys_step.diagnosis[:220])
        logger.info("[round %d] skill_critic changed=%s | %s", rd, skill_step.changed, skill_step.diagnosis[:220])

        _wandb_log(
            {
                **metrics,
                "train/sys_critic_changed": float(sys_step.changed),
                "train/skill_critic_changed": float(skill_step.changed),
            },
            step=rd,
        )

        _log(
            {
                "round": rd, "phase": "round",
                "metrics": metrics,
                "pass_at_1_by_arm": p1, "delta_combined_pass_at_1": delta_p1,
                "sys_critic_changed": sys_step.changed, "sys_diagnosis": sys_step.diagnosis,
                "skill_critic_changed": skill_step.changed, "skill_diagnosis": skill_step.diagnosis,
                "sys_prompt_used": sys_bundle.system_prompt, "tool_description_used": sys_bundle.tool_description,
                "skill_prompts_used": {"correct": skill_prompts.correct, "pitfall": skill_prompts.pitfall},
                "per_problem": [
                    {
                        "problem": r.problem[:300], "difficulty": r.difficulty,
                        "pass_at_k_by_arm": {a: r.pass_at_k(a) for a in ARMS},
                        "knowledge": r.knowledge[:800], "pitfall": r.pitfall[:800],
                        **{
                            f"{a}_trajs": [
                                {"correct": t.correct, "n_turns": t.n_turns, "final_text": t.final_text[:500],
                                 "tool_trace": t.tool_trace[:3]}
                                for t in r.trajs(a)
                            ]
                            for a in ARMS
                        },
                    }
                    for r in results
                ],
            }
        )

        if with_skill_p1 > best["with_skill_pass_at_1"]:
            best = {"round": rd, "with_skill_pass_at_1": with_skill_p1, "sys_bundle": sys_bundle, "skill_prompts": skill_prompts}

        sys_bundle, skill_prompts = sys_step.new, skill_step.new
        _dump_round(f"round{rd}", sys_bundle, skill_prompts)

    _dump_round("BEST", best["sys_bundle"], best["skill_prompts"])
    logger.info("BEST: round=%d with_skill_pass@1=%.4f", best["round"], best["with_skill_pass_at_1"])
    _log({"phase": "final_best", "round": best["round"], "with_skill_pass_at_1": best["with_skill_pass_at_1"]})

    if cli.eval_data:
        await _eval_held_out(scaffold, best, cli, cfg, _log, _wandb_log, rounds)

    log_f.close()
    await client.aclose()
    if wandb_run is not None:
        wandb_run.finish()
    logger.info("logs: %s | prompts: %s", log_path, prompts_dir)


async def _eval_held_out(scaffold: CodeScaffold, best: dict, cli, cfg: dict, _log, _wandb_log, final_step: int) -> None:
    """Held-out eval: BEST (seed vs optimized) sys/skill prompts, no-skill vs
    with-skill pass@k, on a dataset the training loop never saw."""
    eval_items = load_code_dataset_jsonl(cli.eval_data, limit=cli.eval_limit)
    logger.info("held-out eval: %d problems", len(eval_items))

    seed_bundle, seed_skill = SysPromptBundle.seed(), SkillPrompts.seed()
    best_bundle, best_skill = best["sys_bundle"], best["skill_prompts"]

    for tag, bundle, skill_prompts in (("seed", seed_bundle, seed_skill), ("best", best_bundle, best_skill)):
        t0 = time.time()
        results = await scaffold.run_batch(eval_items, bundle, skill_prompts)
        elapsed = time.time() - t0
        metrics = compute_metrics(results, scaffold.k, prefix=f"eval_{tag}")
        metrics |= dict_add_prefix(perf_metrics(results, elapsed), f"perf/eval_{tag}_")
        p1 = {arm: metrics.get(f"eval_{tag}/{arm}_pass@1", 0.0) for arm in ARMS}
        logger.info(
            "[held-out eval %s] pass@1: no_skill=%.4f  knowledge_only=%.4f (%+.4f)  pitfall_only=%.4f (%+.4f)  "
            "combined=%.4f (%+.4f)  (k=%d)",
            tag, p1["no_skill"],
            p1["knowledge_only"], p1["knowledge_only"] - p1["no_skill"],
            p1["pitfall_only"], p1["pitfall_only"] - p1["no_skill"],
            p1["combined"], p1["combined"] - p1["no_skill"],
            scaffold.k,
        )
        for key in sorted(k for k in metrics if k.startswith("domain/")):
            logger.info("    %s = %.4f", key, metrics[key])
        _wandb_log(metrics, step=final_step)
        _log({
            "phase": "held_out_eval", "tag": tag,
            "metrics": metrics,
            "pass_at_1_by_arm": p1,
            "n_problems": len(results),
            "per_problem": [
                {"problem": r.problem[:300], "difficulty": r.difficulty,
                 "pass_at_k_by_arm": {a: r.pass_at_k(a) for a in ARMS}}
                for r in results
            ],
        })


def main() -> None:
    ap = argparse.ArgumentParser(description="TextGrad-style multi-turn CODE training loop")
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--eval-data", default="", help="held-out code JSONL (optional)")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--k", type=int, default=None, help="rollouts per problem per arm (pass@k)")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--out-dir", default="examples/TTS/logs")
    ap.add_argument("--run-name", default="")
    cli = ap.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)
    asyncio.run(run(cfg, cli))


if __name__ == "__main__":
    main()
