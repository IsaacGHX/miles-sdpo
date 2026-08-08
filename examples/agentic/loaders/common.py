"""Shared jsonl-reading helpers for the agentic dashboard's loaders.

Two dump sources exist per --dump-details root (see examples/SDPO_ReAct/
sdpo_react.py and miles/ray/rollout/debug_data.py):

  rollout_data/eval_{rollout_id}.jsonl   -- one row per eval sample, EVERY
      arm/domain, {rollout_id, prompt, response, label, reward,
      response_length, status, index, sdpo_correct, sdpo_ppl, domain}.
      Drives the success-rate-over-checkpoints chart.

  agentic_traces/{rollout_id}.jsonl      -- one row per TRAINING-side sample
      (arms 1.1+ via the group-RM path, arm 1 too since this port added a
      single-sample dump call -- see sdpo_react.py's _dump_agentic_trace_one),
      {messages, label, tool_call_count, tool_error_count, sdpo_correct,
      status, domain, episode_won, task_type, alfworld_game_file,
      webshop_task_id}. Drives the episode-replay view (full message trace).

Both are named by rollout_id, so they line up 1:1 by filename -- but
agentic_traces is a TRAINING dump, not eval, so its rollout_ids don't
necessarily match eval_*.jsonl's rollout_ids (eval fires every
--eval-interval steps; training dumps every step). Treat them as two
independent sources, not a joined table.
"""

import glob
import json
import os
import re

import pandas as pd
import streamlit as st


def _rollout_id_from_filename(path: str) -> int | None:
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None


@st.cache_data(show_spinner=False)
def list_dump_roots(base_dir: str) -> list[str]:
    """--dump-details roots directly under base_dir (e.g. /root/data/sdpo_dumps),
    each one an experiment run's own dump directory."""
    if not os.path.isdir(base_dir):
        return []
    return sorted(
        d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)
    )


@st.cache_data(show_spinner=False)
def load_eval_rows(dump_root: str) -> pd.DataFrame:
    """Every row from every rollout_data/eval_*.jsonl under dump_root, with a
    parsed `rollout_id` column (int) even though the raw file already has one
    -- re-derived from the filename as a fallback for older dumps that might
    predate that field."""
    rows = []
    for path in sorted(glob.glob(os.path.join(dump_root, "rollout_data", "eval_*.jsonl"))):
        file_rollout_id = _rollout_id_from_filename(path)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r.setdefault("rollout_id", file_rollout_id)
                rows.append(r)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_agentic_traces(dump_root: str) -> pd.DataFrame:
    """Every row from every agentic_traces/*.jsonl under dump_root, with a
    parsed `rollout_id` column (from the filename -- this dump doesn't stamp
    rollout_id per-row the way eval_*.jsonl does)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(dump_root, "agentic_traces", "*.jsonl"))):
        file_rollout_id = _rollout_id_from_filename(path)
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r["rollout_id"] = file_rollout_id
                r["_line"] = i
                r["_path"] = path
                rows.append(r)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def filter_by_domain(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    if df.empty or "domain" not in df.columns:
        return df.iloc[0:0]
    return df[df["domain"] == domain]
