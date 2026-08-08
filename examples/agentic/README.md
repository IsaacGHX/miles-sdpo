# Agentic benchmarks dashboard

Streamlit dashboard for visualizing SDPO_ReAct agentic-benchmark runs (WebShop,
ALFWorld today; extensible to future benchmarks like tau3-bench). Reads
directly from `--dump-details` output produced by
`examples/SDPO_ReAct/run-qwen3-4B-sdpo-react-agentic.sh` — no separate
database or index.

## Setup

```bash
cd examples/agentic
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

In the sidebar, point "dump roots base dir" at the parent directory that
contains one or more `--dump-details` output directories (each such
directory has `rollout_data/eval_*.jsonl` and `agentic_traces/*.jsonl`
subdirectories). Select one or more runs to compare success-rate curves
side by side, and use the per-benchmark tab to replay individual episode
transcripts turn by turn.

To include THOR-rendered visual replay for ALFWorld episodes, first run
`examples/SDPO_ReAct/debug/render_alfworld_thor.py` to produce a directory of
per-episode renders (numbered PNG frames + `manifest.json`), then point the
sidebar's "THOR renders" field at that directory's parent.

## Layout

- `app.py` — entrypoint: sidebar picks dump root(s) + benchmark tab.
- `loaders/` — jsonl readers (`common.py` shared, `webshop.py`/`alfworld.py`
  benchmark-specific), all `st.cache_data`-cached.
- `components/` — render functions (`metrics_panel.py` success-rate charts,
  `trajectory_view.py` turn-by-turn transcript + THOR carousel), taking
  already-loaded DataFrames/dicts — no data-loading logic of their own.
