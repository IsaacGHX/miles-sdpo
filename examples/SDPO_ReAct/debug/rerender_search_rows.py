"""Re-render already-FILTERED search rows to the new prompt/tool design.

The pass@k difficulty filter (data/passk_filter_search.py) selected WHICH search
questions to keep (the 15-75% learnable band). That selection is kept. But those
rows were built under the OLD design (verbose search-only system prompt + a
single web_search tool baked into each row). The redesign (Q1/Q2) makes every
row expose ALL tools with the shared minimal prompt -- so in the shuffled
multitask set the search rows must match math/code rows. This rewrites ONLY the
`prompt` (system turn) and `tools` fields, preserving the user question and
metadata/golden_answers (the difficulty selection) verbatim.

Usage (module, from repo root, inside the container):
  python -m examples.SDPO_ReAct.debug.rerender_search_rows \
    --in  /root/data/search_data/search_train_passk.jsonl \
    --out /root/data/search_data/search_train_passk.jsonl   # in place OK
"""

import argparse
import json

from examples.SDPO_ReAct.tools.registry import MINIMAL_SYSTEM_PROMPT, all_tool_specs


def _user_question(row: dict) -> str:
    """Recover the raw user question from a prebuilt row's prompt message list."""
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for m in prompt:
            if m.get("role") == "user":
                return m.get("content", "")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.infile) if l.strip()]
    out = []
    for r in rows:
        q = _user_question(r)
        out.append({
            "prompt": [
                {"role": "system", "content": MINIMAL_SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            "label": r.get("label", ""),
            "tools": all_tool_specs,
            "metadata": r.get("metadata", {"domain": "search"}),
        })
    with open(args.outfile, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"re-rendered {len(out)} search rows -> {args.outfile} "
          f"(tools={[t['function']['name'] for t in all_tool_specs]})")


if __name__ == "__main__":
    main()
