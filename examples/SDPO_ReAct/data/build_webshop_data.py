"""Build WebShop train/eval jsonl for the SDPO_ReAct WEBSHOP domain -- one row
per FIXED task index (WebShop's own "session" id), so the same shopping goal
replays across every arm/checkpoint's train/eval pass (see
tools/webshop/client.py's docstring for why this reproducibility matters, and
tools/webshop/docker/server.py's docstring for the random.seed(0) fix that
makes WebAgentTextEnv.reset(session=idx) fully deterministic -- not just the
goal-list order, but each goal's own price-threshold text too).

Each row: {"prompt":[system(minimal prompt + tool-format note), user(the
env's REAL initial observation text)], "label":"", "tools":[WEBSHOP_STEP_SPEC],
"metadata":{"domain":"webshop", "webshop_task_id":<int>}}.

Grading is routed by metadata['domain']=='webshop' in examples/SDPO/reward.py
-> _grade_one_webshop, which reads metadata['episode_won'] (stamped by
tools/webshop/client.py at a terminal step, not by this builder).

Train/eval split: reuses SDAR's own convention (task indices >=500 for train,
<500 held out for eval) for direct comparability with published SDAR numbers,
per this repo's own small (1000-product) WebShop dataset.

IMPORTANT: this MUST run inside the webshop sidecar's own container/env (see
tools/webshop/docker/Dockerfile) -- it needs to actually import
`web_agent_site.envs` and instantiate a real WebAgentTextEnv to capture each
task's REAL initial observation text (the instruction + starting search page).
It does NOT run inside the main training container, same reason the sidecar
itself can't (Python<=3.10 + old gym/pyserini/torch pins).

Usage (run inside the webshop sidecar image, e.g.
`docker run --rm -v $(pwd)/out:/out sdpo-react-webshop python build_webshop_data.py ...`):
    python build_webshop_data.py --out-dir /root/data/webshop_data \\
        --n-train 400 --n-eval 100
"""

import argparse
import json
import os
import random

import gym

# Registers WebAgentTextEnv-v0 with gym (see tools/webshop/docker/server.py's
# own import of this).
import web_agent_site.envs  # noqa: F401

# NOTE: standalone by design -- see build_alfworld_data.py's module docstring
# for why (this runs inside the webshop sidecar's own container, which has no
# miles/sglang installed and doesn't have the rest of the repo checked out).
WEBSHOP_STEP_SPEC = {
    "type": "function",
    "function": {
        "name": "webshop_step",
        "description": (
            "Take one action in the WebShop online-shopping environment. The observation "
            "returned after each call is the current page text (search results, product "
            "page, or product options) plus the list of currently clickable buttons -- "
            "choose your next action from exactly two forms: 'search[query]' to search "
            "for products, or 'click[button text]' to click a button/link shown on the "
            "current page (e.g. 'click[Buy Now]', 'click[< Prev]', a product's ASIN, or an "
            "option like a color/size). The episode ends when you buy a product or the "
            "turn budget runs out."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to take, e.g. 'search[wireless mouse]' or 'click[Buy Now]'.",
                },
            },
            "required": ["action"],
        },
    },
}
MINIMAL_SYSTEM_PROMPT = (
    "You solve the user's problem step by step. You have tools available "
    "(declared below); call any that help, as many times as you need (multiple "
    "tool calls in one turn run concurrently). Do NOT answer from memory alone: "
    "before you commit to a final answer, you MUST use a tool to VERIFY it -- "
    "run code to check a computation, or search to confirm a fact. Only after a "
    "tool has confirmed your reasoning, give your final answer inside <answer> "
    "and </answer> tags, e.g. <answer>42</answer>. Put ONLY the final answer "
    "inside the tags."
)
WEBSHOP_SYSTEM_PROMPT = (
    MINIMAL_SYSTEM_PROMPT
    + "\n\nThis is an online-shopping task: call webshop_step with a single action, "
    "either search[query] or click[button text]."
)
_WEBSHOP_TOOLS = [WEBSHOP_STEP_SPEC]


def _build_row(task_id: int, instruction_text: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": WEBSHOP_SYSTEM_PROMPT},
            {"role": "user", "content": instruction_text},
        ],
        "label": "",
        "tools": _WEBSHOP_TOOLS,
        "metadata": {"domain": "webshop", "webshop_task_id": task_id},
    }


def _rows_for_task_ids(task_ids: list[int]) -> list[dict]:
    rows = []
    for task_id in task_ids:
        # Same seed-before-gym.make() fix as the sidecar (see server.py's
        # docstring) -- without it, each row's captured instruction text
        # would carry an unseeded/non-reproducible price threshold, breaking
        # the very reproducibility this builder exists to establish.
        random.seed(0)
        env = gym.make("WebAgentTextEnv-v0", observation_mode="text")
        obs, _info = env.reset(session=task_id)
        env.close()
        rows.append(_build_row(task_id, obs))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/data/webshop_data")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument(
        "--eval-start",
        type=int,
        default=0,
        help="held-out eval task-id range starts here (SDAR convention: <500)",
    )
    ap.add_argument(
        "--train-start",
        type=int,
        default=500,
        help="train task-id range starts here (SDAR convention: >=500)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    eval_ids = list(range(args.eval_start, args.eval_start + args.n_eval))
    train_ids = list(range(args.train_start, args.train_start + args.n_train))

    train_rows = _rows_for_task_ids(train_ids)
    train_path = os.path.join(args.out_dir, "webshop_train.jsonl")
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(train_rows)} train rows -> {train_path}")

    eval_rows = _rows_for_task_ids(eval_ids)
    eval_path = os.path.join(args.out_dir, "webshop_eval.jsonl")
    with open(eval_path, "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(eval_rows)} eval rows -> {eval_path}")


if __name__ == "__main__":
    main()
