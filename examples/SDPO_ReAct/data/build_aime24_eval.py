"""Build the AIME-2024 eval jsonl for SDPO_ReAct, in miles' rollout format
(--input-key prompt --label-key label). Adapts examples/SDPO/build_math_eval.py's
pattern to zhuzilin/aime-2024 -- the same dataset examples/retool_v2 already
uses for its AIME eval.

Runs the same react_prompt.build_react_messages the TRAINING dataset uses, so
eval prompts carry the SAME system prompt (<code>/<answer> tag instructions +
one-shot example) as training -- without it, eval samples have no tag-format
instructions at all and any tool usage the model shows is incidental, not
taught by this example's own prompt (a real bug found on an earlier run: this
script originally wrote zhuzilin/aime-2024's raw `prompt` verbatim, with no
system prompt).

Usage:
    python examples/SDPO_ReAct/build_aime24_eval.py --out-dir /root/math_eval
"""

import argparse
import json
import os

from examples.SDPO_ReAct.react_prompt import build_react_messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/math_eval")
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out_dir, exist_ok=True)

    aime = load_dataset("zhuzilin/aime-2024", split="train")
    out_path = os.path.join(args.out_dir, "aime24.jsonl")
    with open(out_path, "w") as f:
        for row in aime:
            # row["prompt"] is already [{"role": "user", "content": question}]
            # (zhuzilin/aime-2024's own format, no DAPO-style answer-format
            # wrapper to strip -- see react_prompt._strip_dapo_wrapper, a no-op
            # here).
            question = row["prompt"][0]["content"] if isinstance(row["prompt"], list) else row["prompt"]
            record = {"prompt": build_react_messages(question), "label": str(row["label"]).strip()}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(aime)} AIME-2024 -> {out_path}")


if __name__ == "__main__":
    main()
