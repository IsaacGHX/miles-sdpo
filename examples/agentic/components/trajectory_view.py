"""Turn-by-turn message replay for a single episode record from
agentic_traces/*.jsonl (see loaders/common.py::load_agentic_traces) --
renders each assistant/tool turn in order, plus (for alfworld episodes with a
matching offline-rendered THOR manifest -- see loaders/alfworld.py's
find_thor_manifest) an image carousel synced to the action index.
"""

import json
import os

import streamlit as st

# One bubble per role, distinct from each other -- a trajectory alternates
# system(once) -> user(once, the initial task) -> assistant(picks an action,
# via tool_calls) -> tool(the env's response to that action) -> assistant ->
# tool -> ... (see miles/rollout/generate_hub/multi_turn.py's generate(),
# which is what actually builds this message list). Giving `tool` its own
# avatar instead of falling back to "assistant" is the whole point: without
# it, the model's decision and the environment's response were visually
# indistinguishable.
_ROLE_AVATARS = {
    "system": "⚙️",
    "user": "🧑",
    "assistant": "🤖",
    "tool": "🔧",
}


def render_trajectory(record: dict, thor_manifest_path: str | None = None) -> None:
    domain = record.get("domain") or "(unknown domain)"
    won = record.get("episode_won")
    st.caption(
        f"domain={domain} | episode_won={won} | tool_call_count={record.get('tool_call_count')} "
        f"| sdpo_correct={record.get('sdpo_correct')}"
    )

    messages = record.get("messages") or []
    for msg in messages:
        role = msg.get("role", "?")
        with st.chat_message(role, avatar=_ROLE_AVATARS.get(role, "❓")):
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                with st.expander("reasoning", expanded=False):
                    st.text(reasoning)
            content = msg.get("content", "")
            if content:
                st.write(content)
            for call in msg.get("tool_calls") or []:
                # Two shapes seen across domains: OpenAI-standard
                # {function: {name, arguments}} (webshop/alfworld, via
                # multi_turn.generate) vs tau2's own ToolCall.model_dump()
                # ({name, arguments} directly, no nested "function" key --
                # see tau2/data_model/message.py). Handle both rather than
                # normalizing tau2's shape at the loader layer, since this is
                # the only place that needs to know either exists.
                fn = call.get("function", call)
                st.code(f"{fn.get('name', '?')}({fn.get('arguments', '')})", language="json")

    if thor_manifest_path:
        _render_thor_carousel(thor_manifest_path)


def _render_thor_carousel(manifest_path: str) -> None:
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        st.warning(f"Could not load THOR manifest {manifest_path}: {e!r}")
        return

    render_dir = os.path.dirname(manifest_path)
    n_frames = len(manifest.get("actions", [])) + 1  # +1 for the initial reset frame
    frame_paths = [os.path.join(render_dir, f"{i:04d}.png") for i in range(n_frames)]
    frame_paths = [p for p in frame_paths if os.path.isfile(p)]
    if not frame_paths:
        st.info("No THOR-rendered frames found for this episode (run debug/render_alfworld_thor.py first).")
        return

    st.subheader("THOR visual replay")
    step = st.slider("step", 0, len(frame_paths) - 1, 0, key=f"thor_step_{manifest_path}")
    st.image(frame_paths[step], caption=f"step {step}")
