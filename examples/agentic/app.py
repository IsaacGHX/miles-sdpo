"""Streamlit dashboard for the agentic (webshop/alfworld, extensible to
tau3-bench later) SDPO_ReAct ablations -- reads --dump-details dump roots
directly, no separate database/index.

Run: `streamlit run app.py` from this directory (examples/agentic/), or
`streamlit run examples/agentic/app.py` from the repo root.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from components.metrics_panel import render_metrics_panel  # noqa: E402
from components.trajectory_view import render_trajectory  # noqa: E402
from loaders.alfworld import (  # noqa: E402
    alfworld_eval_rows,
    find_thor_manifest,
    success_rate_by_rollout as alfworld_success_rate,
)
from loaders.common import list_dump_roots, load_agentic_traces  # noqa: E402
from loaders.tau2 import (  # noqa: E402
    tau2_eval_rows,
    success_rate_by_rollout as tau2_success_rate,
)
from loaders.webshop import (  # noqa: E402
    success_rate_by_rollout as webshop_success_rate,
)

st.set_page_config(page_title="SDPO_ReAct Agentic Dashboard", layout="wide")
st.title("SDPO_ReAct Agentic Dashboard")

with st.sidebar:
    st.header("Data source")
    base_dir = st.text_input("dump roots base dir", value="/root/data/sdpo_dumps")
    all_roots = list_dump_roots(base_dir)
    if not all_roots:
        st.warning(f"No dump roots found under {base_dir!r}. Point this at your --dump-details parent dir.")
        st.stop()
    selected_roots = st.multiselect(
        "compare runs (select 1+ --dump-details roots)",
        options=all_roots,
        default=all_roots[:1],
        format_func=os.path.basename,
    )
    if not selected_roots:
        st.info("Select at least one run to inspect.")
        st.stop()

    st.header("THOR renders (optional)")
    thor_render_root = st.text_input(
        "debug/render_alfworld_thor.py output root",
        value="",
        help="Directory of per-episode render output dirs (each with numbered PNGs + manifest.json).",
    )

tab_webshop, tab_alfworld, tab_tau2 = st.tabs(["WebShop", "ALFWorld", "tau2"])

with tab_webshop:
    per_root = {os.path.basename(root): webshop_success_rate(root) for root in selected_roots}
    render_metrics_panel(per_root, "WebShop success rate over checkpoints")

    st.subheader("Episode replay")
    root_for_replay = st.selectbox("run", selected_roots, format_func=os.path.basename, key="webshop_replay_root")
    traces = load_agentic_traces(root_for_replay)
    webshop_traces = traces[traces.get("domain") == "webshop"] if not traces.empty else traces
    if webshop_traces.empty:
        st.info("No webshop episodes found in agentic_traces/ for this run.")
    else:
        idx = st.number_input("row index", min_value=0, max_value=len(webshop_traces) - 1, value=0, key="webshop_row")
        record = webshop_traces.iloc[int(idx)].to_dict()
        render_trajectory(record)

with tab_alfworld:
    per_root = {os.path.basename(root): alfworld_success_rate(root) for root in selected_roots}
    render_metrics_panel(per_root, "ALFWorld success rate over checkpoints")

    st.subheader("Per-task-type breakdown")
    root_for_breakdown = st.selectbox("run", selected_roots, format_func=os.path.basename, key="alfworld_breakdown_root")
    eval_rows = alfworld_eval_rows(root_for_breakdown)
    if eval_rows.empty or "task_type" not in eval_rows.columns:
        st.info("No per-task-type data available for this run.")
    else:
        breakdown = eval_rows.groupby("task_type")["reward"].apply(lambda s: (s > 0).mean())
        st.bar_chart(breakdown)

    st.subheader("Episode replay")
    root_for_replay = st.selectbox("run", selected_roots, format_func=os.path.basename, key="alfworld_replay_root")
    traces = load_agentic_traces(root_for_replay)
    alfworld_traces = traces[traces.get("domain") == "alfworld"] if not traces.empty else traces
    if alfworld_traces.empty:
        st.info("No alfworld episodes found in agentic_traces/ for this run.")
    else:
        idx = st.number_input("row index", min_value=0, max_value=len(alfworld_traces) - 1, value=0, key="alfworld_row")
        record = alfworld_traces.iloc[int(idx)].to_dict()
        game_file = record.get("alfworld_game_file")
        manifest_path = find_thor_manifest(thor_render_root, game_file) if game_file else None
        render_trajectory(record, thor_manifest_path=manifest_path)

with tab_tau2:
    per_root = {os.path.basename(root): tau2_success_rate(root) for root in selected_roots}
    render_metrics_panel(per_root, "tau2 success rate over checkpoints")

    st.subheader("Per-subdomain breakdown")
    root_for_breakdown = st.selectbox("run", selected_roots, format_func=os.path.basename, key="tau2_breakdown_root")
    eval_rows = tau2_eval_rows(root_for_breakdown)
    if eval_rows.empty or "tau2_domain" not in eval_rows.columns:
        st.info("No per-subdomain data available for this run.")
    else:
        breakdown = eval_rows.groupby("tau2_domain")["reward"].apply(lambda s: (s >= 1.0).mean())
        st.bar_chart(breakdown)

    st.subheader("Episode replay")
    root_for_replay = st.selectbox("run", selected_roots, format_func=os.path.basename, key="tau2_replay_root")
    traces = load_agentic_traces(root_for_replay)
    tau2_traces = traces[traces.get("domain") == "tau2"] if not traces.empty else traces
    if tau2_traces.empty:
        st.info("No tau2 episodes found in agentic_traces/ for this run.")
    else:
        idx = st.number_input("row index", min_value=0, max_value=len(tau2_traces) - 1, value=0, key="tau2_row")
        record = tau2_traces.iloc[int(idx)].to_dict()
        # sdpo_react.py's _sample_to_agentic_trace_record already routes tau2
        # episodes' own message list (metadata["tau2_messages"]) into this
        # record's "messages" field -- render_trajectory needs no aliasing.
        render_trajectory(record)
