"""Build τ²-bench (tau2) train/eval jsonl for the SDPO_ReAct TAU2 domain --
one row per task, downloaded from inclusionAI/AReaL-tau2-data (the SEA-
engine-scaled training set AReaL-SEA-235B-A22B was trained on: 1982 RL tasks
across retail/airline/telecom, vs τ²-bench's own tiny shipped task list --
see the port's design plan for why this dataset was chosen).

Each row: {"prompt":[system(placeholder), user(placeholder -- see below)],
"label":"", "tools":[], "metadata":{"domain":"tau2", "tau2_domain":<retail|
airline|telecom>, "tau2_task":<full Task dict, evaluation_criteria as a JSON
STRING exactly as AReaL-tau2-data ships it>, "tau2_db_path":<absolute path
INSIDE the tau2 sidecar container -- see tools/tau2/docker/Dockerfile, which
bakes these same filenames under /app/tau2_rl_database/>}}.

Why the prompt content is a placeholder, unlike every other SDPO_ReAct
domain: a tau2 episode's real conversation is driven ENTIRELY by tau2's own
Orchestrator (agent<->user-simulator<->environment) via
miles.rollout.generate_hub.agentic_tool_call.generate + --custom-agent-
function-path (see tools/tau2/agent_function.py) -- this repo's own rollout
loop never tokenizes/generates against sample.prompt's content at all for
this domain (agentic_tool_call.generate only reads sample.prompt to log it,
not to build a request). tools=[] for the same reason: tau2's tool schemas
come from environment.get_tools() inside the sidecar, not from this repo's
own --generate-tool-specs-path mechanism.

Grading is routed by metadata['domain']=='tau2' in examples/SDPO/reward.py ->
_grade_one_tau2, which reads metadata['reward'] (stamped by
tools/tau2/agent_function.py from the sidecar's evaluate_simulation() call).

Train/eval split: AReaL-tau2-data publishes only ONE file (tau2_rl_train.jsonl,
1982 rows) -- no official held-out split. Cuts a deterministic, seeded
per-domain tail slice for eval (last N rows per domain in a shuffled order),
mirroring how build_webshop_data.py improvises a split for a benchmark with
no built-in eval/test partition of its own.

Usage (runs in the MAIN training container -- unlike build_webshop_data.py/
build_alfworld_data.py, this only downloads+slices jsonl, no `tau2` package
import needed):
    python -m examples.SDPO_ReAct.data.build_tau2_data \\
        --out-dir /root/data/tau2_data --n-eval-per-domain 30
"""

import argparse
import json
import os
import random

_TAU2_DB_DIR_IN_SIDECAR = "/app/tau2_rl_database"
_DOMAINS = ("retail", "airline", "telecom")

# Placeholder prompt: never actually rendered/tokenized for this domain (see
# module docstring) -- present only because every SDPO_ReAct row needs a
# well-formed prompt field for Dataset/dump-tooling that assumes one exists.
_PLACEHOLDER_SYSTEM = "tau2-bench episode (conversation driven externally by tau2's own Orchestrator; see tools/tau2/agent_function.py)."
_PLACEHOLDER_USER = "(placeholder -- tau2's Orchestrator generates the real conversation)"


def _download_rl_train_jsonl() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download("inclusionAI/AReaL-tau2-data", "tau2_rl_train.jsonl", repo_type="dataset")


def _build_row(row: dict) -> dict:
    domain = row["id"].split("_")[0]
    db_filename = os.path.basename(row["db_path"])
    return {
        "prompt": [
            {"role": "system", "content": _PLACEHOLDER_SYSTEM},
            {"role": "user", "content": _PLACEHOLDER_USER},
        ],
        "label": "",
        "tools": [],
        "metadata": {
            "domain": "tau2",
            "tau2_domain": domain,
            "tau2_task": row,
            "tau2_db_path": os.path.join(_TAU2_DB_DIR_IN_SIDECAR, db_filename),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/data/tau2_data")
    ap.add_argument("--n-eval-per-domain", type=int, default=30, help="held-out rows per domain (0 = no eval split)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    path = _download_rl_train_jsonl()
    rows_by_domain: dict[str, list[dict]] = {d: [] for d in _DOMAINS}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            domain = r["id"].split("_")[0]
            if domain not in rows_by_domain:
                continue
            rows_by_domain[domain].append(r)

    rng = random.Random(args.seed)
    train_rows: list[dict] = []
    eval_paths: dict[str, str] = {}
    for domain, rows in rows_by_domain.items():
        rng.shuffle(rows)
        n_eval = min(args.n_eval_per_domain, len(rows))
        eval_rows, train_domain_rows = rows[:n_eval], rows[n_eval:]
        train_rows.extend(_build_row(r) for r in train_domain_rows)

        eval_path = os.path.join(args.out_dir, f"tau2_{domain}_eval.jsonl")
        with open(eval_path, "w") as f:
            for r in eval_rows:
                f.write(json.dumps(_build_row(r), ensure_ascii=False) + "\n")
        eval_paths[domain] = eval_path
        print(f"wrote {len(eval_rows)} {domain} eval rows -> {eval_path}")

    rng.shuffle(train_rows)
    train_path = os.path.join(args.out_dir, "tau2_train.jsonl")
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(train_rows)} train rows -> {train_path}")


if __name__ == "__main__":
    main()
