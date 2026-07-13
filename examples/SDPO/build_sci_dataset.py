"""Build a mixed-domain science training set for the SDPO example.

Pulls SciKnowEval (4 domains: Biology, Chemistry, Physics, Materials) from the
Hugging Face hub, mixes every domain's train split into one jsonl for training,
and writes one val jsonl per domain for evaluation.

SciKnowEval ships only a single `test` split, so we deterministically slice each
domain into train/val by a fixed ratio (default 10% val), seeded for
reproducibility.

MULTIPLE CHOICE, ALIGNED WITH lasgroup/SDPO
-------------------------------------------
SciKnowEval mixes many question `type`s (mcq-4-choices, mcq-2-choices,
open-ended-qa, true_or_false, relation_extraction, filling). Most non-MCQ rows
are NOT self-contained (e.g. a context fragment ending "...M(i) = -aNc(i) + b."
with no actual question) or have numeric answers that are only selectable, not
derivable — training on them is hopeless. The official SDPO repo
(data/format/sciknoweval.py + data/load_dataset.py) therefore FILTERS to
`type in {mcq-4-choices, mcq-2-choices}` and `level == "L3"`, and keeps the
question AS multiple choice: it lists the A/B/C/D options and asks the model to
output the answer LETTER inside <answer>...</answer>.

We match that exactly:
  * filter to the same types + level,
  * render the choices into the prompt,
  * label = the answer LETTER (answerKey),
  * a system prompt instructs <reasoning>...</reasoning><answer>LETTER</answer>.

Grading (examples/SDPO/sdpo.py `_is_correct`) is a case-insensitive letter match
on the extracted <answer>.

Output format (one JSON object per line), matching the run script's
--input-key prompt --label-key label --apply-chat-template:

    {"prompt": [{"role": "system", "content": "<format instruction>"},
                {"role": "user", "content": "<question + choices>"}],
     "label": "B",
     "metadata": {"domain": "Chemistry", "type": "mcq-4-choices", "level": "L3",
                  "question": "<raw question>"}}

Usage:
    python examples/SDPO/build_sci_dataset.py --out-dir /root/sci --val-ratio 0.1
"""

import argparse
import json
import os
import random
import re


def _slug(name: str) -> str:
    """Filesystem-safe domain slug: lowercased, non-alphanumerics -> underscore."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"


# System prompt (mirrors lasgroup/SDPO data/format/sciknoweval.py SYSTEM_PROMPT):
# reason, then output ONLY the option letter inside <answer>...</answer>.
SYSTEM_PROMPT = (
    "Given a question and its options, select the correct answer. Respond in the "
    "following format:\n"
    "<reasoning>\n...\n</reasoning>\n"
    "<answer>\n...\n</answer>\n\n"
    "For the answer, output ONLY the letter of the correct option (e.g. A, B, C, or D) "
    "and nothing else. Do not restate the answer text. For example, if the answer is "
    '"A", output:\n<answer>\nA\n</answer>'
)

# The question types / difficulty level we keep, matching the official pipeline.
KEEP_TYPES = ("mcq-4-choices", "mcq-2-choices")
KEEP_LEVEL = "L3"


def _format_choices(choices) -> tuple[str, list[str]]:
    """Render "A: text" lines and return (rendered, valid_label_list)."""
    lines, labels = [], []
    if isinstance(choices, dict) and "text" in choices:
        labs = choices.get("label") or [chr(ord("A") + i) for i in range(len(choices["text"]))]
        for lab, txt in zip(labs, choices["text"], strict=False):
            lab = str(lab).strip()
            lines.append(f"{lab}: {str(txt).strip()}")
            labels.append(lab.upper())
    elif isinstance(choices, list):
        for i, c in enumerate(choices):
            lab = c.get("label", chr(ord("A") + i)) if isinstance(c, dict) else chr(ord("A") + i)
            txt = c.get("text", c) if isinstance(c, dict) else c
            lab = str(lab).strip()
            lines.append(f"{lab}: {str(txt).strip()}")
            labels.append(lab.upper())
    return "\n".join(lines), labels


def _level_of(row: dict) -> str | None:
    details = row.get("details") or {}
    return details.get("level") if isinstance(details, dict) else None


def _to_record(row: dict) -> dict | None:
    # Filter: keep only the well-formed multiple-choice types at the target level.
    if row.get("type") not in KEEP_TYPES:
        return None
    if _level_of(row) != KEEP_LEVEL:
        return None

    question = row.get("question")
    answer_key = row.get("answerKey")
    if not question or not answer_key:
        return None

    choices_str, valid_labels = _format_choices(row.get("choices"))
    key = str(answer_key).strip().upper()
    if not choices_str or key not in valid_labels:
        return None  # malformed row (answer not among the listed options)

    user = f"{question}\n\n{choices_str}\n\nPlease reason step by step."
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "label": key,  # the answer LETTER (deterministic letter-match grading)
        "metadata": {
            "domain": row.get("domain", "unknown"),
            "type": row.get("type", "unknown"),
            "level": KEEP_LEVEL,
            # Keep the raw question for any downstream inspection / judge use.
            "question": str(question).strip(),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/sci")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-dataset", default="hicai-zju/SciKnowEval")
    ap.add_argument("--max-per-domain", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset(args.hf_dataset, split="test")

    # group by domain (after type/level filtering in _to_record)
    by_domain: dict[str, list[dict]] = {}
    kept = 0
    for row in ds:
        rec = _to_record(row)
        if rec is None:
            continue
        kept += 1
        by_domain.setdefault(rec["metadata"]["domain"], []).append(rec)
    print(f"Kept {kept} rows after filter (types={KEEP_TYPES}, level={KEEP_LEVEL}) from {len(ds)} total")

    rng = random.Random(args.seed)
    train_all: list[dict] = []
    counts = {}
    for domain, recs in sorted(by_domain.items()):
        rng.shuffle(recs)
        if args.max_per_domain > 0:
            recs = recs[: args.max_per_domain]
        n_val = max(1, int(len(recs) * args.val_ratio))
        val, train = recs[:n_val], recs[n_val:]
        train_all.extend(train)

        val_path = os.path.join(args.out_dir, f"val_{_slug(domain)}.jsonl")
        with open(val_path, "w") as f:
            for r in val:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[domain] = {"train": len(train), "val": len(val)}

    # shuffle the mixed train set so domains are interleaved
    rng.shuffle(train_all)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    with open(train_path, "w") as f:
        for r in train_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_all)} mixed train examples -> {train_path}")
    for domain, c in counts.items():
        print(f"  {domain}: train={c['train']} val={c['val']} (val_{_slug(domain)}.jsonl)")


if __name__ == "__main__":
    main()
