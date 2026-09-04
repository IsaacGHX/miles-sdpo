"""pass@k + turn/token/error/truncation/repetition metrics for the TextGrad
code loop, matching miles' OWN wandb conventions 1:1 wherever a portable
equivalent exists (see the "textgrad wandb metric parity" investigation):

  * pass@k -- miles' UNBIASED estimator (Chen et al.'s 1 - C(n-c,k)/C(n,k)),
    via miles.utils.metric_utils.compute_pass_rate, computed for the full 2^i
    ladder up to group_size (k=4 -> pass@1, pass@2, pass@4), for EACH of the
    FOUR arms (no_skill/knowledge_only/pitfall_only/combined -- see
    textgrad_scaffold.ARMS), + each skill-bearing arm's delta vs no_skill.
    This is the ONE metric family also split by domain/<difficulty>/...
    (LiveCodeBench's own difficulty tag is this harness's sub-population axis,
    mirroring miles/ray/rollout/metrics.py::_compute_per_domain_metrics's
    domain/<name>/... convention) -- per the task's explicit instruction, the
    OTHER metrics below are reported globally only, not re-split per domain.
  * agentic/*        -- round_number (turns), tool_call_count, tool_error_rate,
    zero_tool_call_frac, hit_max_turns_frac. Same panel name + same underlying
    quantities as miles/ray/rollout/metrics.py::_compute_agentic_tool_metrics
    (populated there from Sample.metadata["round_number"/"tool_call_count"/
    "tool_error_count"], which multi_turn.generate/sdpo_react.py populate
    automatically for a REAL training run -- our Trajectory dataclass tracks
    the identical quantities directly, see multi_turn_code.py).
  * response_len/*   -- token counts (completion_tokens summed across
    assistant turns), matching metrics.py's response_len/mean|median|max|min
    via Sample.effective_response_length -- ChatResult.completion_tokens is
    the API equivalent (see engines.py).
  * rollout/truncated_ratio, rollout/repetition_frac -- same keys/semantics as
    metrics.py: truncated <=> any turn's finish_reason=="length" (the
    external-API analog of SGLang's finish_reason.type=="length"); repetition
    reuses miles.utils.metric_utils.has_repetition VERBATIM (pure string/zlib
    heuristic, no GPU dependency) applied to the trajectory's final text.
  * perf/*           -- wall-clock only (perf/rollout_time, tokens/sec). All
    GPU/Megatron-specific perf keys (tflops, weight_version, prefix_cache,
    logprob/entropy/advantage tensors) have NO portable equivalent for a
    pure-API harness and are intentionally omitted -- see the parity
    investigation note for the full list of what's NOT_APPLICABLE and why.
"""

from __future__ import annotations

from miles.utils.metric_utils import compute_pass_rate, compute_statistics, dict_add_prefix, has_repetition

from examples.TTS.textgrad_scaffold import ARMS

SKILL_ARMS = tuple(a for a in ARMS if a != "no_skill")


def _flat_correct(results, arm: str) -> list[float]:
    """Flatten [problem1's k trajs, problem2's k trajs, ...] -> [0/1, ...],
    preserving group order (required by compute_pass_rate's reshape)."""
    key = f"{arm}_trajs"
    out: list[float] = []
    for r in results:
        for t in getattr(r, key):
            out.append(1.0 if t.correct else 0.0)
    return out


def _pass_rate_for(results, arm: str, k: int) -> dict[str, float]:
    flat = _flat_correct(results, arm)
    if not flat:
        return {}
    return compute_pass_rate(flat_rewards=flat, group_size=k, num_groups=len(results))


def _all_trajs(results, arm: str):
    key = f"{arm}_trajs"
    return [t for r in results for t in getattr(r, key)]


def _agentic_metrics(trajs) -> dict[str, float]:
    """agentic/* -- mirrors _compute_agentic_tool_metrics's keys exactly
    (round_number_mean/max/min, tool_call_count_mean/max, zero_tool_call_frac,
    tool_error_rate, hit_max_turns_frac), computed from Trajectory fields that
    are the direct analog of the Sample.metadata keys that function reads."""
    if not trajs:
        return {}
    rounds = [t.n_turns for t in trajs]
    calls = [t.n_tool_calls for t in trajs]
    errors = [t.n_tool_errors for t in trajs]
    out = {
        "round_number_mean": sum(rounds) / len(rounds),
        "round_number_max": max(rounds),
        "round_number_min": min(rounds),
        "tool_call_count_mean": sum(calls) / len(calls),
        "tool_call_count_max": max(calls),
        "zero_tool_call_frac": sum(1 for c in calls if c == 0) / len(calls),
        "hit_max_turns_frac": sum(1 for t in trajs if t.hit_max_turns) / len(trajs),
    }
    total_calls = sum(calls)
    if total_calls > 0:
        out["tool_error_rate"] = sum(errors) / total_calls
    return out


def _response_len_metrics(trajs) -> dict[str, float]:
    """response_len/* -- mirrors metrics.py's response_len/mean|median|max|min
    (via Sample.effective_response_length) using completion_tokens summed
    across a trajectory's assistant turns as the API equivalent."""
    if not trajs:
        return {}
    lens = [float(t.completion_tokens) for t in trajs]
    return compute_statistics(lens)


def _rollout_metrics(trajs) -> dict[str, float]:
    """rollout/truncated_ratio + rollout/repetition_frac -- same keys/semantics
    as metrics.py (finish_reason==length analog; has_repetition reused as-is)."""
    if not trajs:
        return {}
    return {
        "truncated_ratio": sum(1 for t in trajs if t.any_truncated) / len(trajs),
        "repetition_frac": sum(1 for t in trajs if has_repetition(t.final_text)) / len(trajs),
    }


def compute_metrics(results, k: int, prefix: str = "train") -> dict[str, float]:
    """Full metric block for one round/eval-tag:
      * pass@k ladder for ALL FOUR arms (no_skill/knowledge_only/pitfall_only/
        combined) + each skill-bearing arm's delta vs no_skill, split by
        domain/<difficulty>/ too (the ONE family re-split per domain, per
        instruction).
      * agentic/*, response_len/*, rollout/* -- GLOBAL only (pooled across all
        difficulties), matching the real SDPO_ReAct run's panel names.
        Reported per-arm via a key prefix (these panels don't exist in a real
        run since there's no arm split there, but the harness's four arms are
        the one axis worth keeping distinct -- everything else is pooled).
    ``results``: list[CodeRoundResult] (see textgrad_scaffold.py)."""
    out: dict[str, float] = {}

    # ---- pass@k: overall (all difficulties pooled) ----
    pr_by_arm = {arm: _pass_rate_for(results, arm, k) for arm in ARMS}
    for arm in ARMS:
        out |= dict_add_prefix(pr_by_arm[arm], f"{prefix}/{arm}_")
    no_pr = pr_by_arm["no_skill"]
    for arm in SKILL_ARMS:
        for key in no_pr:
            if key in pr_by_arm[arm]:
                out[f"{prefix}/delta_{arm}_{key}"] = pr_by_arm[arm][key] - no_pr[key]

    # ---- pass@k: per-difficulty ("domain") split ----
    by_diff: dict[str, list] = {}
    for r in results:
        by_diff.setdefault(r.difficulty or "unknown", []).append(r)
    if len(by_diff) > 1:  # single-difficulty batches already covered by "overall" above
        for diff, subset in sorted(by_diff.items()):
            p = f"domain/{diff}/"
            out[p + "count"] = float(len(subset))
            out[p + "frac_of_batch"] = len(subset) / len(results) if results else 0.0
            sub_pr_by_arm = {arm: _pass_rate_for(subset, arm, k) for arm in ARMS}
            for arm in ARMS:
                out |= dict_add_prefix(sub_pr_by_arm[arm], p + f"{arm}_")
            sub_no_pr = sub_pr_by_arm["no_skill"]
            for arm in SKILL_ARMS:
                for key in sub_no_pr:
                    if key in sub_pr_by_arm[arm]:
                        out[p + f"delta_{arm}_{key}"] = sub_pr_by_arm[arm][key] - sub_no_pr[key]

    # ---- agentic/response_len/rollout: global, per-arm (not per-domain) ----
    # namespaced by `prefix` too (train vs eval_seed vs eval_best) -- these
    # panels are logged at possibly-overlapping wandb steps (e.g. the final
    # training round and both held-out eval tags), so an unprefixed key would
    # silently overwrite one call's numbers with another's at the same step.
    for arm in ARMS:
        trajs = _all_trajs(results, arm)
        out |= dict_add_prefix(_agentic_metrics(trajs), f"agentic/{prefix}_{arm}_")
        out |= dict_add_prefix(_response_len_metrics(trajs), f"response_len/{prefix}_{arm}_")
        out |= dict_add_prefix(_rollout_metrics(trajs), f"rollout/{prefix}_{arm}_")

    return out


def perf_metrics(results, elapsed_seconds: float) -> dict[str, float]:
    """perf/* -- WALL-CLOCK ONLY (perf/rollout_time, tokens/sec), the portable
    subset of metrics.py's perf panel. Every GPU/Megatron-specific perf key
    (tflops, weight_version, prefix_cache_hit_rate, per-token logprob/entropy/
    advantage tensors) is intentionally NOT reproduced here -- there is no
    equivalent when generation is an external chat-completion API call with no
    local forward pass; see the parity investigation note for the full list."""
    if elapsed_seconds <= 0:
        return {"rollout_time": elapsed_seconds}
    total_tokens = sum(t.completion_tokens for r in results for arm in ARMS for t in getattr(r, f"{arm}_trajs"))
    return {
        "rollout_time": elapsed_seconds,
        "tokens_per_sec": total_tokens / elapsed_seconds,
    }
