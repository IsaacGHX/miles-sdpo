"""Build AIME 2026 eval jsonl for native tool-calling SDPO_ReAct path.

AIME 2026 (MathArena/aime_2026): 30 problems, integer answers (0-999).

Usage:
    python -m examples.SDPO_ReAct.data.build_aime26 --out-dir /root/data/aime26
"""

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/data/aime26")
    ap.add_argument("--dataset", default="MathArena/aime_2026")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    from datasets import load_dataset

    from examples.SDPO_ReAct.native_prompt import build_native_messages
    from examples.SDPO_ReAct.tools.registry import all_tool_specs as tool_specs

    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    out_path = os.path.join(args.out_dir, "aime26_eval.jsonl")

    with open(out_path, "w") as f:
        for row in ds:
            record = {
                "prompt": build_native_messages(row["problem"]),
                "label": str(row["answer"]).strip(),
                "tools": tool_specs,
                "metadata": {
                    "domain": "math",
                    "question": row["problem"],
                    "problem_idx": row.get("problem_idx"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds)} rows -> {out_path}")


if __name__ == "__main__":
    main()
