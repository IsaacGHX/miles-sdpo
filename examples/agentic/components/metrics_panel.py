"""Success-rate-over-checkpoints line chart, shared across benchmark tabs.

One line per --dump-details root the user selects to compare (e.g. different
arms of the same ablation series) -- x=rollout_id, y=success_rate.
"""

import pandas as pd
import streamlit as st


def render_metrics_panel(per_root_df: dict[str, pd.DataFrame], title: str) -> None:
    """per_root_df: {dump_root_label: success_rate_by_rollout(...) DataFrame}."""
    st.subheader(title)
    non_empty = {label: df for label, df in per_root_df.items() if not df.empty}
    if not non_empty:
        st.info("No eval data found for this benchmark under the selected dump root(s).")
        return

    chart_df = pd.concat(
        [df.assign(run=label) for label, df in non_empty.items()],
        ignore_index=True,
    )
    pivot = chart_df.pivot_table(index="rollout_id", columns="run", values="success_rate")
    st.line_chart(pivot)

    with st.expander("Raw per-checkpoint numbers"):
        st.dataframe(chart_df.sort_values(["run", "rollout_id"]), width="stretch")
