"""Offline THOR (visual) render of a completed ALFWorld episode, for the
examples/agentic/ dashboard's trajectory-view -- NOT part of training/eval,
never touches the TextWorld-only path this repo actually trains/evals on
(tools/alfworld/), and is the ONLY place in this whole port that ever touches
AlfredThorEnv.

Reads a dumped episode's action sequence + game file from
--dump-details/agentic_traces/{rollout_id}.jsonl (written by
examples/SDPO_ReAct/sdpo_react.py's _dump_agentic_traces /
_dump_agentic_trace_one), replays that SAME action sequence through THOR
(re-pinned to the SAME alfworld_game_file the text episode actually played --
see tools/alfworld/docker/server.py's module docstring for the game_files-
filtering pinning technique this script reuses), and saves one PNG per step
+ a manifest.json the dashboard can load without re-parsing the trace.

ENVIRONMENT: needs `ai2thor` + alfworld's THOR/vis extras (`pip install
alfworld[full]`), plus a GPU-backed X server for OpenGL rendering (see
alfworld's own README "Cloud Instance" section: `python
alfworld/docker/startx.py 0` then `export DISPLAY=:0`) -- a THIRD, separate
Python environment from both the training container and the TextWorld-only
sidecar (tools/alfworld/docker/, which never installs ai2thor at all). NOT
containerized here -- this is a one-off, manually-invoked, dashboard-support
script, same "environment-assuming standalone script" precedent as
debug/_debug_rollout.py.

Usage (run in a THOR-capable env, e.g. after `startx.py`/`export DISPLAY=:0`):
    python -m examples.SDPO_ReAct.debug.render_alfworld_thor \\
        --trace-path /root/data/sdpo_dumps/.../agentic_traces/40.jsonl \\
        --line 3 \\
        --out-dir /root/data/thor_renders/episode_3
"""

import argparse
import json
import os
import re


def _extract_alfworld_actions(record: dict) -> list[str]:
    """Pull the ordered alfworld_step action strings out of a dumped episode
    record's `messages` list (the OpenAI-standard trace _reconstruct_messages
    produces -- see sdpo_react.py). Each assistant turn's tool_calls entries
    named "alfworld_step" carry a JSON-or-plain `arguments` string with an
    `action` field."""
    actions = []
    for msg in record.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            if fn.get("name") != "alfworld_step":
                continue
            args = fn.get("arguments", "")
            action = None
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict):
                    action = parsed.get("action")
            except (json.JSONDecodeError, TypeError):
                pass
            if action is None:
                # Fallback: a bare/malformed arguments string -- best-effort
                # regex pull of an "action": "..." pair.
                m = re.search(r'"action"\s*:\s*"([^"]*)"', args)
                action = m.group(1) if m else args
            actions.append(action)
    return actions


def _load_record(trace_path: str, line: int) -> dict:
    with open(trace_path) as f:
        for i, raw in enumerate(f):
            if i == line:
                return json.loads(raw)
    raise IndexError(f"trace file {trace_path} has fewer than {line + 1} lines")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-path", required=True, help="agentic_traces/{rollout_id}.jsonl")
    ap.add_argument("--line", type=int, default=0, help="which record (0-indexed) in the trace file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None, help="alfworld config yaml (default: alfworld's own generic.load_config())")
    args = ap.parse_args()

    record = _load_record(args.trace_path, args.line)
    game_file = (record.get("metadata") or {}).get("alfworld_game_file")
    if not game_file:
        raise ValueError(
            f"record at line {args.line} has no metadata.alfworld_game_file -- "
            "is this really an alfworld episode's dump?"
        )
    actions = _extract_alfworld_actions(record)
    print(f"game_file: {game_file}")
    print(f"replaying {len(actions)} actions")

    # Imported lazily -- this whole module only works in a THOR-capable env
    # (ai2thor + X server), never the plain TextWorld sidecar/training path.
    import alfworld.agents.modules.generic as generic
    from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
    from PIL import Image

    config = generic.load_config(args.config) if args.config else generic.load_config()
    config["controller"]["type"] = "oracle"

    thor = AlfredThorEnv.Thor(queue=None, train_eval="eval_in_distribution")
    thor.init_env(config)

    os.makedirs(args.out_dir, exist_ok=True)
    thor.reset(game_file)
    frame = thor.get_last_frame()
    Image.fromarray(frame[:, :, ::-1]).save(os.path.join(args.out_dir, "0000.png"))  # undo alfworld's BGR flip

    won = False
    for step_idx, action in enumerate(actions, start=1):
        thor.step(action)
        _feedback, done, _admissible, step_won, _gc_sr, _expert = thor.get_results()
        frame = thor.get_last_frame()
        Image.fromarray(frame[:, :, ::-1]).save(os.path.join(args.out_dir, f"{step_idx:04d}.png"))
        if step_won:
            won = True
        if done:
            print(f"episode done at step {step_idx} (won={won})")
            break

    manifest = {"game_file": game_file, "actions": actions, "won": won}
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(actions) + 1} frames + manifest.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
