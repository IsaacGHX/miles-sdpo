"""Build math eval jsonls (AIME-2024/2025) for the NATIVE tool-calling
SDPO_ReAct path, so eval prompts carry the SAME native system prompt + `tools`
field as training (see native_prompt.py). Sibling of build_aime24_eval.py,
which builds the plain-text-tag (react_prompt.py) variant for Qwen2.5.

Each row: {"prompt": [system, user], "label": <answer>, "tools": [<spec>]}.
Launched with `--eval-tool-key tools --apply-chat-template`, the `<tools>`
block is injected exactly as in training.

Also injects `metadata_overrides` support indirectly via the eval yaml (the
per-sample generate_max_turns override lives there, not here).

Usage:
    python -m examples.SDPO_ReAct.build_native_eval --out-dir /root/math_eval
"""

import argparse
import json
import os

from examples.SDPO_ReAct.native_prompt import build_native_messages
from examples.SDPO_ReAct.tools.registry import all_tool_specs as tool_specs  # Q1: all tools

# HF dataset id -> output filename. aime-2024/2025 in zhuzilin's mirror share the
# same {prompt:[{role:user,content}], label} shape build_aime24_eval.py assumes.
_EVAL_SETS = {
    "zhuzilin/aime-2024": "aime24_native.jsonl",
    "zhuzilin/aime-2025": "aime25_native.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/math_eval")
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=list(_EVAL_SETS.keys()),
        help="HF dataset ids to build; default AIME-2024 + AIME-2025.",
    )
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out_dir, exist_ok=True)

    for repo in args.datasets:
        out_name = _EVAL_SETS.get(repo)
        if out_name is None:
            # Derive a filename for an unlisted dataset rather than failing --
            # keeps this extensible to new math eval sets without editing code.
            out_name = repo.replace("/", "_") + "_native.jsonl"
        ds = load_dataset(repo, split="train")
        out_path = os.path.join(args.out_dir, out_name)
        with open(out_path, "w") as f:
            for row in ds:
                prompt = row["prompt"]
                question = prompt[0]["content"] if isinstance(prompt, list) else prompt
                record = {
                    "prompt": build_native_messages(question),
                    "label": str(row["label"]).strip(),
                    "tools": tool_specs,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(ds)} rows: {repo} -> {out_path}")


if __name__ == "__main__":
    main()
