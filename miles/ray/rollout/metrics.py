import logging
from typing import Any

import numpy as np

from miles.utils import tracking_utils
from miles.utils.iter_utils import group_by
from miles.utils.metric_utils import (
    compute_pass_rate,
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
    has_repetition,
)
from miles.utils.misc import load_function
from miles.utils.types import Sample


logger = logging.getLogger(__name__)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if (x := args.custom_eval_rollout_log_function_path) is not None:
        custom_log_func = load_function(x)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    all_rewards: list[float] = []  # pooled across all eval datasets for eval/all
    per_dataset_acc: list[float] = []  # per-dataset acc for the equal-weight eval/mean
    for key in data.keys():
        rewards = data[key]["rewards"]
        all_rewards.extend(rewards)
        dataset_acc = sum(rewards) / len(rewards)
        per_dataset_acc.append(dataset_acc)
        log_dict[f"eval/{key}"] = dataset_acc
        # val-aux: per-domain accuracy (avg@N) lives with the auxiliary metrics.
        log_dict[f"val-aux/{key}_acc"] = dataset_acc
        if (samples := data[key].get("samples")) is not None:
            # response_len, truncated, repetition, etc. -> val-aux
            log_dict |= dict_add_prefix(_compute_metrics_from_samples(args, samples), f"val-aux/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"val-aux/{key}_truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            pr = compute_pass_rate(flat_rewards=rewards, group_size=args.n_samples_per_eval_prompt)
            # per-domain pass@1 -> val-core (the headline); pass@2/4/... -> val-aux
            if "pass@1" in pr:
                log_dict[f"val-core/{key}_pass@1"] = pr["pass@1"]
            log_dict |= dict_add_prefix({k: v for k, v in pr.items() if k != "pass@1"}, f"val-aux/{key}_")
            # keep the legacy eval/ keys too for backward compatibility
            log_dict |= dict_add_prefix(pr, f"eval/{key}-")

    # Two overall val scores across all domains:
    #   eval/all  = sample-pooled mean (domains with more samples weigh more)
    #   eval/mean = equal-weight mean of per-domain accuracies (each domain 1/N)
    if len(data) > 1 and all_rewards:
        log_dict["eval/all"] = sum(all_rewards) / len(all_rewards)
        log_dict["eval/mean"] = sum(per_dataset_acc) / len(per_dataset_acc)
        log_dict["val-aux/all_acc"] = log_dict["eval/all"]
        log_dict["val-aux/mean_acc"] = log_dict["eval/mean"]
        if args.log_passrate:
            pr_all = compute_pass_rate(flat_rewards=all_rewards, group_size=args.n_samples_per_eval_prompt)
            # merged pass@1 -> val-core (the single headline val score you want)
            if "pass@1" in pr_all:
                log_dict["val-core/all_pass@1"] = pr_all["pass@1"]
            log_dict |= dict_add_prefix({k: v for k, v in pr_all.items() if k != "pass@1"}, "val-aux/all_")
            log_dict |= dict_add_prefix(pr_all, "eval/all-")

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    tracking_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if (x := args.custom_rollout_log_function_path) is not None:
        custom_log_func = load_function(x)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    sample_metrics = _compute_metrics_from_samples(args, samples)
    log_dict = {**(rollout_extra_metrics or {})}
    # skill/* already carry their own panel prefix -> keep them top-level (NOT under
    # rollout/); everything else goes under rollout/.
    log_dict |= dict_add_prefix({k: v for k, v in sample_metrics.items() if not k.startswith("skill/")}, "rollout/")
    for k, v in sample_metrics.items():
        if k.startswith("skill/"):
            log_dict[k] = v
    log_dict |= dict_add_prefix(_compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    # Dedicated "response_len/" panel: all length-related metrics in one place.
    for k, v in sample_metrics.items():
        if k.startswith("response_len/") or k == "truncated_ratio":
            log_dict[f"response_len/{k[len('response_len/'):] if k.startswith('response_len/') else k}"] = v
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")


def _compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()

    oldest_versions = [s.oldest_weight_version for s in samples if s.oldest_weight_version is not None]
    if oldest_versions:
        log_dict |= dict_add_prefix(compute_statistics(oldest_versions), "weight_version/")
        mixed = sum(1 for s in samples if len(set(s.weight_versions)) > 1)
        log_dict["weight_version/mixed_version_ratio"] = mixed / len(samples)

    # SDPO diagnostics (examples/SDPO/sdpo.py): per-sample mean log-prob of the
    # sampled token under student vs teacher, and their diff (sampled-token
    # reverse KL). No-op for non-SDPO runs where these keys are absent.
    for mkey, out_key in (
        ("sdpo_student_logp_mean", "sdpo/student_logp"),
        ("sdpo_teacher_logp_mean", "sdpo/teacher_logp"),
        ("sdpo_logp_diff_mean", "sdpo/logp_diff"),
        # True task success rate (survives the zeroed reward under pure distill)
        # and response perplexity of the sampled rollout.
        ("sdpo_correct", "sdpo/success_rate"),
        ("sdpo_ppl", "sdpo/ppl"),
    ):
        vals = [s.metadata[mkey] for s in samples if isinstance(s.metadata, dict) and mkey in s.metadata]
        if vals:
            log_dict[out_key] = float(np.mean(vals))

    # SDPO self-skill (rollout-side): dedicated skill/ panel — skill length
    # min/max/mean and skill perplexity. Present only when --sdpo-self-skill is on
    # and at least one skill was generated this rollout.
    skill_lens = [s.metadata["sdpo_skill_len"] for s in samples if isinstance(s.metadata, dict) and "sdpo_skill_len" in s.metadata]
    if skill_lens:
        log_dict["skill/length_mean"] = float(np.mean(skill_lens))
        log_dict["skill/length_min"] = float(np.min(skill_lens))
        log_dict["skill/length_max"] = float(np.max(skill_lens))
        log_dict["skill/count"] = float(len(skill_lens))
    skill_ppls = [s.metadata["sdpo_skill_ppl"] for s in samples if isinstance(s.metadata, dict) and "sdpo_skill_ppl" in s.metadata]
    if skill_ppls:
        log_dict["skill/ppl"] = float(np.mean(skill_ppls))
    # --sdpo-response-prefix skill: fraction of response teacher prefixes that
    # actually used the peer's skill (vs falling back to the peer's full trace).
    rp_skill = [
        s.metadata["sdpo_response_prefix_is_skill"]
        for s in samples
        if isinstance(s.metadata, dict) and "sdpo_response_prefix_is_skill" in s.metadata
    ]
    if rp_skill:
        log_dict["skill/response_prefix_is_skill_frac"] = float(np.mean(rp_skill))

    tito_vals = [s.metadata.get("tito_session_mismatch") for s in samples]
    tito_vals = [v for v in tito_vals if v is not None]
    if tito_vals:
        log_dict["tito_session_mismatch_rate"] = np.mean([len(v) > 0 for v in tito_vals]).item()
        for mtype in ("special_token_count", "special_token_type", "non_assistant_text", "assistant_text"):
            log_dict[f"tito_session_mismatch_rate/{mtype}"] = np.mean(
                [any(m.get("type") == mtype for m in v) for v in tito_vals]
            ).item()
        if args.ci_test:
            for strict_type in ("special_token_count", "special_token_type", "non_assistant_text"):
                rate = log_dict.get(f"tito_session_mismatch_rate/{strict_type}", 0)
                assert rate == 0, (
                    f"tito_session_mismatch_rate/{strict_type}={rate:.4f} must be 0 — "
                    "this indicates a bug in the TITO algorithm or chat template. "
                    "Please check your tito model and chat template."
                )
            # assistant_text mismatch is non-critical: assistant tokens are inherited
            # from the pretokenized prefix and may differ from canonical tokenization.

    return log_dict


def _compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    counts = {reward: len(items) for reward, items in group_by(interesting_rewards).items()}
    log_dict = {f"zero_std/count_{reward}": count for reward, count in counts.items()}

    # Percentages over total groups, so "too hard" (all-0) and "too easy"
    # (all-1) rates are comparable across runs without needing to know the
    # rollout batch size.
    total_groups = len(all_sample_groups)
    if total_groups > 0:
        log_dict["zero_std/all_zero_percentage"] = counts.get("0.0", 0) / total_groups
        log_dict["zero_std/all_one_percentage"] = counts.get("1.0", 0) / total_groups

    return log_dict


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}
