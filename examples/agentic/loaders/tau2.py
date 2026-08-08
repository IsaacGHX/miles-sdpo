"""tau2 (tau2-bench)-specific loading on top of loaders/common.py's shared
jsonl readers -- filters by metadata['domain']=='tau2', computes success-rate
per rollout_id (overall + per-subdomain breakdown, since data/
build_tau2_data.py stamps metadata['tau2_domain'] on every row -- retail/
airline/telecom, analogous to loaders/alfworld.py's per-task-type
breakdown).
"""

import pandas as pd

from loaders.common import filter_by_domain, load_eval_rows

_SUBDOMAINS = ["retail", "airline", "telecom"]


def tau2_eval_rows(dump_root: str) -> pd.DataFrame:
    return filter_by_domain(load_eval_rows(dump_root), "tau2")


def success_rate_by_rollout(dump_root: str) -> pd.DataFrame:
    """One row per rollout_id: overall success rate + one column per tau2
    subdomain (NaN where that rollout_id has no samples of that subdomain).
    tau2's reward is continuous (db-hash equality is 0/1 for retail/airline,
    but telecom's reward_breakdown can land on partial credit) -- success
    here means reward >= 1.0 (a FULL match), same threshold
    examples/SDPO/reward.py's _grade_one_tau2 uses for correctness."""
    df = tau2_eval_rows(dump_root)
    if df.empty or "reward" not in df.columns:
        return pd.DataFrame(columns=["rollout_id", "success_rate", "n"] + _SUBDOMAINS)

    def _agg(group: pd.DataFrame) -> pd.Series:
        out = {"success_rate": (group["reward"] >= 1.0).mean(), "n": len(group)}
        if "tau2_domain" in group.columns:
            for sd in _SUBDOMAINS:
                sub = group[group["tau2_domain"] == sd]
                out[sd] = (sub["reward"] >= 1.0).mean() if len(sub) else float("nan")
        return pd.Series(out)

    grouped = df.groupby("rollout_id").apply(_agg, include_groups=False)
    return grouped.reset_index().sort_values("rollout_id")
