"""Build AMO-Bench eval jsonl for the NATIVE tool-calling SDPO_ReAct path.

AMO-Bench (meituan-longcat/AMO-Bench) is 50 IMO-level math problems with
answer_type in {number, set, variable, description}. 39 problems are auto-
gradable via Math-Verify (answer_type != 'description'); the remaining 11
require LLM-judge grading.

Each output row:
    {"prompt": [system, user], "label": <answer>, "tools": [<spec>],
     "metadata": {"domain": "math", "answer_type": ..., "question_id": ...,
                  "solution": ..., "amo_use_judge": True/False}}

- domain="math" routes to the DAPO math grader (boxed matching) for
  answer_type in {number, set, variable}.
- answer_type="description" rows set amo_use_judge=True, so the eval
  reward function can route them to the LLM judge.

Usage:
    python -m examples.SDPO_ReAct.data.build_amo_bench --out-dir /root/data/amo_bench
"""

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/data/amo_bench")
    ap.add_argument("--dataset", default="meituan-longcat/AMO-Bench")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    from datasets import load_dataset

    from examples.SDPO_ReAct.native_prompt import build_native_messages
    from examples.SDPO_ReAct.tools.registry import all_tool_specs as tool_specs

    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    out_path = os.path.join(args.out_dir, "amo_bench_eval.jsonl")

    with open(out_path, "w") as f:
        for row in ds:
            question = row["prompt"]
            answer = row["answer"]
            answer_type = row.get("answer_type", "number")
            record = {
                "prompt": build_native_messages(question),
                "label": answer.strip(),
                "tools": tool_specs,
                "metadata": {
                    "domain": "math",
                    "question": question,
                    "answer_type": answer_type,
                    "question_id": row.get("question_id"),
                    "solution": row.get("solution", ""),
                    "amo_use_judge": answer_type == "description",
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_judge = sum(1 for row in ds if row.get("answer_type") == "description")
    print(f"Wrote {len(ds)} rows -> {out_path}")
    print(f"  {len(ds) - n_judge} auto-gradable (number/set/variable), {n_judge} LLM-judge (description)")


if __name__ == "__main__":
    main()
