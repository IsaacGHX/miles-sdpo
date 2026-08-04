"""Read examples/SDPO_ReAct/explog.jsonl (one row per launch, written by
run-qwen3-4B-sdpo-react-native.sh) and, for each experiment, pull the eval
curve (AIME24/25 pass@1 + tool-use) out of its wandb output log so config and
result sit side by side. Prints a compact table for analysis.

Usage:
    python -m examples.SDPO_ReAct.debug.analyze_explog
    python -m examples.SDPO_ReAct.debug.analyze_explog --explog path --wandb-dir wandb
"""

import argparse
import glob
import json
import os
import re


def _eval_curve(log_path: str):
    """Extract [(step, aime24_pass@1, aime25_pass@1, zero_tool_frac, tool_calls_mean), ...]
    from a wandb output log's `metrics.py:.. - eval N: {...}` lines."""
    pts = []
    if not log_path or not os.path.exists(log_path):
        return pts
    with open(log_path, errors="replace") as f:
        for ln in f:
            m = re.search(r"metrics\.py:\d+ - eval (\d+): (\{.*\})", ln)
            if not m:
                continue
            step = int(m.group(1))
            try:
                d = eval(m.group(2))  # noqa: S307 -- trusted local log, dict literal
            except Exception:
                continue
            pts.append(
                (
                    step,
                    d.get("val-core/aime24_pass@1"),
                    d.get("val-core/aime25_pass@1"),
                    d.get("val-aux/aime24/agentic/zero_tool_call_frac"),
                    d.get("val-aux/aime24/agentic/tool_call_count_mean"),
                )
            )
    return pts


def _find_wandb_log(wandb_dir: str, wandb_group: str, ts_utc: str):
    """Best-effort: find the wandb output log whose run started nearest the
    launch timestamp. Returns the path with the most eval lines matching the
    group (robust to the multi-log restart pattern)."""
    best, best_hits = None, -1
    for log in glob.glob(os.path.join(wandb_dir, "run-*", "files", "output_*.log")):
        try:
            with open(log, errors="replace") as f:
                head = f.read(4000)
        except Exception:
            continue
        if wandb_group and wandb_group not in head:
            continue
        hits = len(_eval_curve(log))
        if hits > best_hits:
            best, best_hits = log, hits
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explog", default=os.path.join(os.path.dirname(__file__), "explog.jsonl"))
    ap.add_argument("--wandb-dir", default="wandb")
    args = ap.parse_args()

    if not os.path.exists(args.explog):
        print(f"No explog at {args.explog} yet.")
        return

    rows = [json.loads(l) for l in open(args.explog) if l.strip()]
    print(f"{len(rows)} experiment(s) in {args.explog}\n")
    for r in rows:
        print("=" * 70)
        print(f"exp={r.get('exp')}  arm={r.get('arm')}  prompt={r.get('prompt_variant')}  toolset={r.get('toolset')}")
        print(f"  ts={r.get('ts_utc')}  git={r.get('git_commit')}/{r.get('git_state')}  note={r.get('note','')}")
        print(f"  train_data={r.get('train_data')}")
        curve = _eval_curve(_find_wandb_log(args.wandb_dir, r.get("wandb_group", ""), r.get("ts_utc", "")))
        if not curve:
            print("  (no eval curve found in wandb logs yet)")
            continue
        print("  step  aime24  aime25  zero_tool%  toolcalls/traj")
        for step, a24, a25, zt, tc in curve:
            def fmt(x):
                return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)

            print(f"  {step:>4}  {fmt(a24):>6}  {fmt(a25):>6}  {fmt(zt):>9}  {fmt(tc):>13}")


if __name__ == "__main__":
    main()
