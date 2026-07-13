"""Build AIME-2025 + Minerva Math eval jsonls for the DAPO-math SDPO run.

Both are written in miles' rollout format (--input-key prompt --label-key label
--apply-chat-template): prompt is a [{"role":"user","content": ...}] message list,
label is the ground-truth answer string.

  - zhuzilin/aime-2025:  already {prompt, label} (integer answers). Copied as-is.
  - math-ai/minervamath: raw {question, answer} -> wrap question as a user message,
    label = answer (LaTeX). Graded by the general math grader (grade_answer_verl),
    not the integer-only DAPO grader.

Usage:
    python examples/SDPO/build_math_eval.py --out-dir /root/math_eval
"""

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/math_eval")
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out_dir, exist_ok=True)

    # AIME 2025 — already in {prompt, label} form.
    aime = load_dataset("zhuzilin/aime-2025", split="train")
    aime_path = os.path.join(args.out_dir, "aime25.jsonl")
    with open(aime_path, "w") as f:
        for row in aime:
            f.write(json.dumps({"prompt": row["prompt"], "label": str(row["label"]).strip()}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(aime)} AIME-2025 -> {aime_path}")

    # Minerva Math — raw {question, answer} -> {prompt, label}.
    minerva = load_dataset("math-ai/minervamath", split="test")
    minerva_path = os.path.join(args.out_dir, "minerva_math.jsonl")
    n = 0
    with open(minerva_path, "w") as f:
        for row in minerva:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if not q or not a:
                continue
            rec = {"prompt": [{"role": "user", "content": q}], "label": a}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} Minerva Math -> {minerva_path}")


if __name__ == "__main__":
    main()
