"""Build multi-hop QA (deepsearch) train/val jsonl for the SDPO_ReAct SEARCH
domain, from FlashRAG's HotpotQA + 2WikiMultiHopQA (clean {question,
golden_answers} jsonl, no dataset scripts).

Each row: {"prompt":[system(search prompt), user(question)], "label":<answer>,
"tools":[web_search spec + code_interpreter], "metadata":{"domain":"search",
"golden_answers":[...]}}. Grading is routed by metadata['domain']=='search' in
examples/SDPO/sdpo.py -> EM check against golden_answers (reuses
examples/search-r1/qa_em_format.py). The tool is web_search (local wiki-18 e5
retriever, torch_retrieval_server) -> genuinely tool-necessary + multi-hop
(needs several searches to chain facts), matching the proposal's "multi-hop ↑".

Usage:
    python -m examples.SDPO_ReAct.build_search_data --out-dir /root/data/search_data \
        --n-train-per 1000 --n-val-per 100
"""

import argparse
import json
import os

# Every row exposes ALL tools (Q1: let the model choose) with ONE shared minimal
# system prompt (Q2: question + answer-format only, no task-specific workflow) --
# both from the registry, the single source of truth.
from examples.SDPO_ReAct.tools.registry import MINIMAL_SYSTEM_PROMPT, all_tool_specs

SEARCH_SYSTEM_PROMPT = MINIMAL_SYSTEM_PROMPT
_SEARCH_TOOLS = all_tool_specs


def _load_flashrag(fname: str):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("RUC-NLPIR/FlashRAG_datasets", fname, repo_type="dataset")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_row(question: str, answers: list[str]) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "label": answers[0] if answers else "",
        "tools": _SEARCH_TOOLS,
        # "question" (the RAW question, not the rendered prompt string) lets
        # examples/SDPO/reward.py's --sdpo-search-judge-fallback build a clean
        # judge prompt -- without it the judge would see the full rendered
        # <tools>-block-and-system-prompt string instead of just the question.
        "metadata": {"domain": "search", "golden_answers": answers, "question": question},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/data/search_data")
    ap.add_argument("--n-train-per", type=int, default=1000, help="train rows per dataset")
    ap.add_argument("--n-val-per", type=int, default=100, help="val rows per dataset")
    ap.add_argument("--datasets", nargs="*", default=["hotpotqa", "2wikimultihopqa"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_rows, val_by_ds = [], {}
    for ds in args.datasets:
        tr = _load_flashrag(f"{ds}/train.jsonl")
        dv = _load_flashrag(f"{ds}/dev.jsonl")

        def _rows(raw, n):
            out = []
            for r in raw:
                q = r.get("question")
                ans = r.get("golden_answers") or ([r["answer"]] if r.get("answer") else [])
                if q and ans:
                    out.append(_build_row(q, [str(a) for a in ans]))
                if n and len(out) >= n:
                    break
            return out

        tr_rows = _rows(tr, args.n_train_per)
        val_rows = _rows(dv, args.n_val_per)
        train_rows += tr_rows
        val_by_ds[ds] = val_rows
        print(f"{ds}: {len(tr_rows)} train, {len(val_rows)} val")

    train_path = os.path.join(args.out_dir, "search_train.jsonl")
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(train_rows)} train -> {train_path}")
    # per-dataset val files (so eval reports hotpotqa vs 2wiki separately)
    for ds, rows in val_by_ds.items():
        vp = os.path.join(args.out_dir, f"{ds}_val.jsonl")
        with open(vp, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} val -> {vp}")


if __name__ == "__main__":
    main()
