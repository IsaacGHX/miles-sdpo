"""Combine per-domain SDPO_ReAct datasets into ONE shuffled multi-task train set
(proposal step 3: environment diversity ↑ -> ceiling ↑).

Each source jsonl is already in the native {prompt(msg list), label, tools,
metadata} shape. Grading is routed per-row by metadata['domain'] in
examples/SDPO/sdpo.py (math -> answer/boxed match; code -> test-case run;
search -> its own RM), so a mixed group grades each sample by its own domain --
no extra wiring. Math rows have no metadata (domain defaults to 'math'); this
tool stamps an explicit metadata['domain'] on every row so the mix is
unambiguous and analyzable.

Interleaves round-robin across domains then shuffles (seeded, deterministic --
Math.random is unavailable in this repo's tooling but plain random with a fixed
seed is fine here), so early rollout batches see all domains rather than one
domain's block first.

Usage (module, from repo root):
    python -m examples.SDPO_ReAct.build_multitask_data \\
        --out /root/data/multitask/train.jsonl \\
        --source math:/root/dapo-math-17k/dapo-math-17k-native-default.jsonl \\
        --source code:/root/data/code_data/livecodebench_train.jsonl \\
        --per-domain 400
"""

import argparse
import json
import random


def _load(path: str, domain: str, cap: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            md = r.get("metadata")
            if not isinstance(md, dict):
                md = {}
            md.setdefault("domain", domain)  # stamp domain if missing (math rows)
            r["metadata"] = md
            rows.append(r)
            if cap and len(rows) >= cap:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="DOMAIN:PATH",
        help="a domain-tagged source jsonl, e.g. code:/root/data/code_data/livecodebench_train.jsonl. Repeatable.",
    )
    ap.add_argument("--per-domain", type=int, default=0, help="cap rows per domain (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    per_domain = {}
    for spec in args.source:
        domain, path = spec.split(":", 1)
        rows = _load(path, domain, args.per_domain)
        per_domain[domain] = rows
        print(f"loaded {len(rows)} rows for domain '{domain}' <- {path}")

    # Balanced round-robin interleave so EVERY contiguous batch spans all domains
    # as uniformly as possible (e.g. rollout-batch 112 over 3 domains -> 38/37/37
    # per batch). We shuffle WITHIN each domain (variety across epochs' row
    # content) but do NOT global-shuffle afterwards -- a final global shuffle
    # would destroy the per-batch balance and leave batches with lopsided domain
    # mixes. Paired with --rollout-shuffle OFF in the launcher so the balanced
    # order is consumed as-is (data_source's per-epoch reshuffle would otherwise
    # re-randomize and unbalance each batch again).
    rng = random.Random(args.seed)
    lists = list(per_domain.values())
    for l in lists:
        rng.shuffle(l)
    interleaved = []
    i = 0
    while any(i < len(l) for l in lists):
        for l in lists:
            if i < len(l):
                interleaved.append(l[i])
        i += 1

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in interleaved:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {d: len(r) for d, r in per_domain.items()}
    print(f"wrote {len(interleaved)} shuffled rows -> {args.out}  (per-domain: {counts})")


if __name__ == "__main__":
    main()
