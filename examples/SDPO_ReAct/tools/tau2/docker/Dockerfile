# SDPO_ReAct tau2 sidecar -- one long-lived container per training job,
# exposing exactly one HTTP port (see ../run_tau2_sidecar.sh), same pattern
# as ../../webshop/docker/ and ../../alfworld/docker/ (Python-version
# isolation).
#
# WHY A SEPARATE CONTAINER: tau2-bench (github.com/sierra-research/tau2-bench)
# requires Python>=3.12,<3.14 (see its own pyproject.toml) -- hard-conflicts
# with the training image's Python 3.10, same isolation rationale as every
# other sidecar in this repo. Its OWN dependency pins (litellm, pandas,
# fastapi -- see requirements.txt's comment) do not conflict with a modern
# stack; only the Python version itself does.
#
# Bakes in the AReaL-tau2-data per-task DB variant files (~a few MB total --
# cheap enough to bake at build time, same judgment call already made for
# WebShop's small search index vs sglang's own wiki-18 index, which is why
# THAT split retrieval into a separate always-running process instead).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# tau2-bench's PyPI/git wheel does NOT bundle its own `data/` directory
# (confirmed live: `tau2.utils.utils.SOURCE_DIR/data` doesn't exist after a
# plain pip install -- policy.md/task-set/pddl files are only in the GitHub
# repo's top-level data/ dir, not packaged). Without this, get_environment()
# fails with "No such file or directory: .../domains/retail/policy.md" the
# moment it tries to read a domain's policy -- clone it separately and point
# TAU2_DATA_DIR (env.utils.utils.py's own override hook) at it. Only
# data/tau2/domains/ + data/tau2/user_simulator/ are load-bearing for this
# sidecar's use (~140MB) -- data/tau2/results/ (577MB, pre-run benchmark
# results, unrelated to running NEW simulations) is dropped.
RUN git clone --depth 1 --filter=blob:none --sparse https://github.com/sierra-research/tau2-bench.git /tmp/tau2-src && \
    cd /tmp/tau2-src && git sparse-checkout set data/tau2/domains data/tau2/user_simulator && \
    mkdir -p /app/tau2_data/tau2 && \
    cp -r data/tau2/domains /app/tau2_data/tau2/domains && \
    cp -r data/tau2/user_simulator /app/tau2_data/tau2/user_simulator && \
    rm -rf /tmp/tau2-src
ENV TAU2_DATA_DIR=/app/tau2_data

# Per-task DB variants from inclusionAI/AReaL-tau2-data (HF dataset) -- one
# file per task-generation batch, NOT the domain's own bundled default (see
# server.py's module docstring for why per-task variants matter: AReaL's SEA
# data engine generates a fresh, consistent world state per task).
RUN mkdir -p /app/tau2_rl_database && cd /app/tau2_rl_database && \
    for f in \
        tau2_airline_db.json tau2_airline_new_db_1.json tau2_airline_new_db_2.json tau2_airline_new_db_3.json \
        tau2_retail_new_db_1.json tau2_retail_new_db_2.json tau2_retail_new_db_3.json tau2_retail_new_db_4.json \
        tau2_telecom_db.toml \
    ; do \
        curl -sfL -o "$f" "https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data/resolve/main/tau2_rl_database/$f"; \
    done

COPY server.py .

EXPOSE 8424
ENV TAU2_SIDECAR_PORT=8424
# shell form (not exec-form CMD) so $TAU2_SIDECAR_PORT expands -- lets
# run_tau2_sidecar.sh's PORT override reach the actual bind address under
# --network=host, where there's no -p remapping to fall back on.
CMD uvicorn server:app --host 0.0.0.0 --port ${TAU2_SIDECAR_PORT}
