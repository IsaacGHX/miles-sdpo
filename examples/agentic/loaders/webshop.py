"""WebShop-specific loading on top of loaders/common.py's shared jsonl
readers -- filters by metadata['domain']=='webshop', computes success-rate
per rollout_id for the metrics chart.
"""

import pandas as pd

from loaders.common import filter_by_domain, load_eval_rows


def webshop_eval_rows(dump_root: str) -> pd.DataFrame:
    return filter_by_domain(load_eval_rows(dump_root), "webshop")


def success_rate_by_rollout(dump_root: str) -> pd.DataFrame:
    """One row per rollout_id: mean(reward > 0) across webshop eval samples
    at that checkpoint -- the success-rate-over-checkpoints curve."""
    df = webshop_eval_rows(dump_root)
    if df.empty or "reward" not in df.columns:
        return pd.DataFrame(columns=["rollout_id", "success_rate", "n"])
    grouped = df.groupby("rollout_id")["reward"].agg(
        success_rate=lambda s: (s > 0).mean(),
        n="count",
    )
    return grouped.reset_index().sort_values("rollout_id")
