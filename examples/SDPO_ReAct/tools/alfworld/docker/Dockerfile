# SDPO_ReAct alfworld_step sidecar -- one long-lived container per training
# job, exposing exactly one port (see ../run_alfworld_sidecar.sh), same
# pattern as ../../search/docker/ (Python-version isolation) and
# ../../docker/ (the code_interpreter sandbox).
#
# WHY A SEPARATE CONTAINER (not folded into the training image): the alfworld
# package pins gymnasium==0.29.1/stable-baselines3==2.6.0, independent of and
# likely conflicting with the training image's own torch/transformers stack
# -- same isolation rationale as every other sidecar in this repo.
#
# `alfworld-download -f` is baked in at BUILD time (bounded, one-time
# download of PDDL/grammar/game files -- see the alfworld package's own
# downloader) so this sidecar is self-contained and idempotent to start,
# mirroring how ../../webshop/docker/Dockerfile bakes in its own (smaller)
# product-database index rather than fetching it on every container start.
FROM python:3.10-slim

# textworld's jericho dependency builds a C++ frotz interpreter (calls `make`
# at pip-install time, not a prebuilt wheel) -- python:3.10-slim has no build
# toolchain by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# alfworld-download's default cache dir; ALFWORLD_DATA must point here at
# runtime too (set below) so config_tw.yaml's $ALFWORLD_DATA-relative paths
# resolve to what was actually downloaded at build time.
ENV ALFWORLD_DATA=/root/.cache/alfworld
RUN alfworld-download -f

COPY server.py config_tw.yaml .
ENV ALFWORLD_CONFIG=/app/config_tw.yaml

EXPOSE 8423
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8423"]
