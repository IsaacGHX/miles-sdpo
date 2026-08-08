"""Build ALFWorld (TextWorld) train/eval jsonl for the SDPO_ReAct ALFWORLD
domain -- one row per FIXED game file, so the same task replays across every
arm/checkpoint's train/eval pass (see tools/alfworld/client.py's docstring
for why this reproducibility matters, and tools/alfworld/docker/server.py's
docstring for the game_files-filtering mechanism that pins a specific file).

Each row: {"prompt":[system(minimal prompt + tool-format note), user(the
env's REAL initial observation text)], "label":"", "tools":[ALFWORLD_STEP_SPEC],
"metadata":{"domain":"alfworld", "alfworld_game_file":<path>,
"alfworld_split":<train|eval_in_distribution|eval_out_of_distribution>,
"task_type":<one of the 6 ALFRED task types, parsed from the path>}}.

Grading is routed by metadata['domain']=='alfworld' in examples/SDPO/reward.py
-> _grade_one_alfworld, which reads metadata['episode_won'] (stamped by
tools/alfworld/client.py at a terminal step, not by this builder).

IMPORTANT: this MUST run inside the alfworld sidecar's own container/env (or
an equivalent local env with `alfworld` installed + `alfworld-download -f`
already run) -- it needs to actually import `alfworld.agents.environment` and
instantiate a real AlfredTWEnv to (a) enumerate game files per split and
(b) capture each game's REAL initial observation text (there is no static
"task description" separate from the live initial observation). It does NOT
run inside the main training container, same reason the sidecar itself can't
(see tools/alfworld/docker/Dockerfile's docstring).

Usage (run inside the alfworld sidecar image, e.g.
`docker run --rm -v $(pwd)/out:/out sdpo-react-alfworld python build_alfworld_data.py ...`):
    python build_alfworld_data.py --out-dir /root/data/alfworld_data \\
        --n-train 400 --n-eval-id 100 --n-eval-ood 100
"""

import argparse
import json
import os

import yaml
from alfworld.agents.environment import get_environment

# NOTE: this script is INTENTIONALLY standalone (no `examples.SDPO_ReAct...`
# imports) -- it runs inside the alfworld SIDECAR's own container (see module
# docstring), which only has server.py/config_tw.yaml copied in, not the
# whole repo (see tools/alfworld/docker/Dockerfile). ALFWORLD_STEP_SPEC and
# MINIMAL_SYSTEM_PROMPT are duplicated verbatim from tools/alfworld/spec.py
# and tools/registry.py respectively -- keep both copies in sync if either
# canonical source changes.
ALFWORLD_STEP_SPEC = {
    "type": "function",
    "function": {
        "name": "alfworld_step",
        "description": (
            "Take one action in the ALFWorld household task environment. The observation "
            "returned after each call includes the current room/object state and the list "
            "of currently admissible actions -- choose your next action from that list "
            "(free text, e.g. 'go to countertop 1', 'take apple 1 from countertop 1', "
            "'heat mug 1 with microwave 1'). The episode ends when the task is completed "
            "or the turn budget runs out."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to take, exactly as it appears in the admissible-actions list.",
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
ALFWORLD_SYSTEM_PROMPT = (
    MINIMAL_SYSTEM_PROMPT
    + "\n\nThis is a household task: call alfworld_step with a single free-text "
    "action from the admissible-actions list shown in each observation."
)
_ALFWORLD_TOOLS = [ALFWORLD_STEP_SPEC]

_TASK_TYPE_PREFIXES = [
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def _task_type_from_path(game_file: str) -> str:
    """The 6 ALFRED task types are encoded as a prefix of the game file's
    parent-parent directory name, e.g. '.../look_at_obj_in_light-Statue-None-
    DeskLamp-319/trial_.../game.tw-pddl' -> 'look_at_obj_in_light'."""
    task_dir = os.path.basename(os.path.dirname(os.path.dirname(game_file)))
    for prefix in _TASK_TYPE_PREFIXES:
        if task_dir.startswith(prefix):
            return prefix
    return "unknown"


def _build_row(game_file: str, split: str, instruction_text: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT},
            {"role": "user", "content": instruction_text},
        ],
        "label": "",
        "tools": _ALFWORLD_TOOLS,
        "metadata": {
            "domain": "alfworld",
            "alfworld_game_file": game_file,
            "alfworld_split": split,
            "task_type": _task_type_from_path(game_file),
        },
    }


def _rows_for_split(config: dict, split: str, n: int) -> list[dict]:
    if n <= 0:
        return []
    base_env = get_environment(config["env"]["type"])(config, train_eval=split)
    game_files = base_env.game_files[:n]
    rows = []
    for game_file in game_files:
        # Pin THIS env instance to exactly one game file (same technique as
        # the sidecar's own init_session -- see tools/alfworld/docker/
        # server.py's module docstring) so reset()'s initial observation is
        # for the file we're building a row for, not a random draw.
        base_env.game_files = [game_file]
        env = base_env.init_env(batch_size=1)
        obs, _infos = env.reset()
        instruction_text = obs[0]
        env.close()
        rows.append(_build_row(game_file, split, instruction_text))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("ALFWORLD_CONFIG", "/app/config_tw.yaml"))
    ap.add_argument("--out-dir", default="/root/data/alfworld_data")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-eval-id", type=int, default=100, help="eval_in_distribution rows")
    ap.add_argument("--n-eval-ood", type=int, default=100, help="eval_out_of_distribution rows")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    train_rows = _rows_for_split(config, "train", args.n_train)
    train_path = os.path.join(args.out_dir, "alfworld_train.jsonl")
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(train_rows)} train rows -> {train_path}")

    for split, n, fname in [
        ("eval_in_distribution", args.n_eval_id, "alfworld_eval_id.jsonl"),
        ("eval_out_of_distribution", args.n_eval_ood, "alfworld_eval_ood.jsonl"),
    ]:
        rows = _rows_for_split(config, split, n)
        path = os.path.join(args.out_dir, fname)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} {split} rows -> {path}")


if __name__ == "__main__":
    main()
