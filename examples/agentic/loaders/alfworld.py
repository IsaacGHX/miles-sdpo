"""ALFWorld-specific loading on top of loaders/common.py's shared jsonl
readers -- filters by metadata['domain']=='alfworld', computes success-rate
per rollout_id (overall + per-task-type breakdown, since data/
build_alfworld_data.py stamps metadata['task_type'] on every row), and
locates an offline-rendered THOR manifest.json (see debug/
render_alfworld_thor.py) for a given episode if one has been generated.
"""

import json
import os

import pandas as pd

from loaders.common import filter_by_domain, load_eval_rows

_TASK_TYPES = [
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def alfworld_eval_rows(dump_root: str) -> pd.DataFrame:
    return filter_by_domain(load_eval_rows(dump_root), "alfworld")


def success_rate_by_rollout(dump_root: str) -> pd.DataFrame:
    """One row per rollout_id: overall success rate + one column per ALFRED
    task type (NaN where that rollout_id has no samples of that type)."""
    df = alfworld_eval_rows(dump_root)
    if df.empty or "reward" not in df.columns:
        return pd.DataFrame(columns=["rollout_id", "success_rate", "n"] + _TASK_TYPES)

    def _agg(group: pd.DataFrame) -> pd.Series:
        out = {"success_rate": (group["reward"] > 0).mean(), "n": len(group)}
        if "task_type" in group.columns:
            for tt in _TASK_TYPES:
                sub = group[group["task_type"] == tt]
                out[tt] = (sub["reward"] > 0).mean() if len(sub) else float("nan")
        return pd.Series(out)

    grouped = df.groupby("rollout_id").apply(_agg, include_groups=False)
    return grouped.reset_index().sort_values("rollout_id")


def find_thor_manifest(render_root: str, game_file: str) -> str | None:
    """Locate an offline-rendered THOR episode (debug/render_alfworld_thor.py
    output) whose manifest.json's game_file matches. render_root is a
    user-configurable directory of per-episode render output dirs (each
    containing numbered PNGs + manifest.json); this is a best-effort linear
    scan, fine for the handful of manually-rendered showcase episodes this is
    meant for, not a general-purpose index."""
    if not render_root or not os.path.isdir(render_root):
        return None
    for entry in sorted(os.listdir(render_root)):
        manifest_path = os.path.join(render_root, entry, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if manifest.get("game_file") == game_file:
            return manifest_path
    return None
