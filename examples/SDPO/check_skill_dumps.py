"""Acceptance check for SDPO skill dumps.

Scans a run's skill/*.jsonl (produced when --dump-details is set with self-skill)
and reports health metrics so you can quickly verify a run after the skill-prompt /
entropy fixes:

  * solution vs pitfall counts, and whether skill_kind matches self_correct
  * CROSS-CONTAMINATION: solution skills that contain "avoid/do not/watch out"
    (should be ~0) and pitfall skills that DON'T (should be ~0). High rates mean
    the correct/incorrect skill prompts are being confused.
  * NESTED CHAT TEMPLATE: problem_text still carrying <|im_start|> scaffolding
    (should be 0 after the _strip_chat_template fix; a nested template is what
    caused the contamination in the first place).
  * skill length stats, and how many skills are empty.

Usage:
    python examples/SDPO/check_skill_dumps.py <dump_dir_or_skill_dir> [--examples N]

    # newest run under sdpo_dumps/:
    python examples/SDPO/check_skill_dumps.py sdpo_dumps/olmo3-7B-sdpo-math-colocate_20260716_013732
"""

import argparse
import glob
import json
import os
import re
import statistics

# Pitfall markers the incorrect-trace skill is instructed to emit.
_PITFALL_RE = re.compile(r"\b(avoid|do not|don't|watch out|never)\b", re.IGNORECASE)
# Solution-roadmap markers (imperative steps). Loose heuristic, only for reporting.
_STEP_RE = re.compile(r"(?m)^\s*\d+[\.\)]")


def _has_pitfall(text: str) -> bool:
    return bool(_PITFALL_RE.search(text or ""))


def _resolve_skill_dir(path: str) -> str:
    """Accept either the run dir (…/<exp>) or the skill dir (…/<exp>/skill)."""
    if os.path.basename(path.rstrip("/")) == "skill":
        return path
    cand = os.path.join(path, "skill")
    return cand if os.path.isdir(cand) else path


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", help="run dir (…/<exp>) or its skill/ subdir")
    ap.add_argument("--examples", type=int, default=2, help="print N example skills per kind")
    args = ap.parse_args()

    skill_dir = _resolve_skill_dir(args.dump_dir)
    files = sorted(glob.glob(os.path.join(skill_dir, "*.jsonl")))
    if not files:
        print(f"No skill/*.jsonl under {skill_dir!r}. "
              "Was the run launched with --dump-details and --sdpo-self-skill?")
        return

    rows = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    total = len(rows)
    sol = [r for r in rows if r.get("skill_kind") == "solution"]
    pit = [r for r in rows if r.get("skill_kind") == "pitfall"]
    unknown = [r for r in rows if r.get("skill_kind") not in ("solution", "pitfall")]

    # Label consistency: skill_kind must agree with self_correct.
    mislabeled = sum(
        1
        for r in rows
        if r.get("self_correct") is not None
        and (r.get("skill_kind") == "solution") != bool(r.get("self_correct"))
    )

    # Cross-contamination.
    sol_with_pitfall = sum(_has_pitfall(r.get("skill_text", "")) for r in sol)
    pit_without_pitfall = sum(not _has_pitfall(r.get("skill_text", "")) for r in pit)

    # Nested chat template (should be 0 after the fix).
    nested = sum("<|im_start|>" in (r.get("problem_text") or "") for r in rows)

    # Length / emptiness.
    lens = [int(r["skill_length"]) for r in rows if isinstance(r.get("skill_length"), (int, float))]
    empty = sum(not (r.get("skill_text") or "").strip() for r in rows)

    print(f"=== SDPO skill dump check: {skill_dir} ===")
    print(f"files={len(files)}  skills={total}  (solution={len(sol)} pitfall={len(pit)} "
          f"unknown={len(unknown)})")
    print()
    print("-- label consistency (skill_kind vs self_correct) --")
    print(f"  mislabeled: {mislabeled}/{total}  {_pct(mislabeled, total)}   [want 0]")
    print()
    print("-- cross-contamination (the bug _strip_chat_template fixes) --")
    print(f"  solution skills containing pitfall words: {sol_with_pitfall}/{len(sol)} "
          f"{_pct(sol_with_pitfall, len(sol))}   [want ~0]")
    print(f"  pitfall skills MISSING pitfall words:     {pit_without_pitfall}/{len(pit)} "
          f"{_pct(pit_without_pitfall, len(pit))}   [want ~0]")
    print()
    print("-- nested chat template in problem_text --")
    print(f"  problem_text with <|im_start|>: {nested}/{total} {_pct(nested, total)}   [want 0]")
    print()
    print("-- skill length (tokens) / emptiness --")
    if lens:
        print(f"  min={min(lens)} median={int(statistics.median(lens))} max={max(lens)}")
    print(f"  empty skills: {empty}/{total} {_pct(empty, total)}   [want 0]")

    # Verdict.
    ok = (
        mislabeled == 0
        and nested == 0
        and (not sol or sol_with_pitfall / len(sol) < 0.10)
        and (not pit or pit_without_pitfall / len(pit) < 0.10)
    )
    print()
    print("VERDICT:", "PASS ✓" if ok else "NEEDS ATTENTION ✗ (see the [want ...] targets above)")

    if args.examples > 0:
        for kind, group in (("solution", sol), ("pitfall", pit)):
            if not group:
                continue
            print(f"\n===== example {kind} skills =====")
            for r in group[: args.examples]:
                print(f"[idx={r.get('index')} self_correct={r.get('self_correct')} "
                      f"len={r.get('skill_length')}]")
                print((r.get("skill_text") or "").strip()[:500])
                print("-" * 40)


if __name__ == "__main__":
    main()
