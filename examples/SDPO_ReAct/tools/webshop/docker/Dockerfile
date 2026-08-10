# SDPO_ReAct webshop_step sidecar -- one long-lived container per training
# job, exposing exactly one port (see ../run_webshop_sidecar.sh), same
# pattern as ../../search/docker/ (Python-version isolation) and ../../docker/
# (the code_interpreter sandbox).
#
# WHY A SEPARATE CONTAINER: WebShop (github.com/princeton-nlp/WebShop) pins
# gym==0.24.0/pyserini==0.17.0/torch==1.11.0/spacy==3.3.0 and requires
# Python<=3.10 -- hard-conflicts with the training image's own stack, same
# isolation rationale as every other sidecar in this repo.
#
# INSIDE the container: run.py starts N worker.py PROCESSES (own GIL each)
# plus router.py, the one process actually bound to 8422 -- see router.py's
# docstring for why a single process couldn't use more than ~1 CPU under real
# concurrency (env.step()'s HTML/text processing and gym.make()'s Lucene/JVM
# index open are GIL-bound) and how the multi-process split fixes that while
# keeping the external HTTP contract (../client.py, ../run_webshop_sidecar.sh)
# unchanged.
#
# pyserini's LuceneSearcher (used to instantiate the env at ALL, not just for
# the search[query] action -- see web_agent_site/engine/engine.py) needs a
# JVM, hence openjdk below.
#
# Bakes in the SMALL (~1000-product) dataset + its Lucene search index at
# BUILD time -- cheap enough (~100MB) to bake, unlike wiki-18's index which is
# why the search tool split retrieval into a separate always-running process
# (see ../../search/run_retrieval.sh's own docstring for that asymmetry).
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git openjdk-21-jdk-headless curl \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# `python -m spacy download` shells out to a pip URL with an old-style egg
# fragment (`#egg=en_core_web_lg`) that pip>=25 rejects outright -- install
# the model wheels directly instead of going through spacy's own downloader.
# web_agent_site/engine/goal.py hard-loads en_core_web_sm (a SEPARATE, smaller
# model from en_core_web_lg -- both needed, not a duplicate).
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0-py3-none-any.whl

# Vendor WebShop verbatim (same discipline as ../../search/docker/browser.py's
# "copied VERBATIM" note) -- this is princeton-nlp/WebShop's own source, not
# reimplemented here.
RUN git clone --depth 1 https://github.com/princeton-nlp/WebShop.git /app/webshop

WORKDIR /app/webshop
# Small (1000-product) dataset + its human-instruction file. WebShop's own
# setup.sh fetches these via gdown from Google Drive, but that link is
# rate-limited/access-blocked in practice (confirmed: fails identically from
# both this container and the host, "Cannot retrieve the public link").
# WooooDyy/AgentGym vendors the IDENTICAL small-dataset files directly in
# their repo (verified: same byte sizes as WebShop's own docs describe,
# real JSON not an LFS pointer) -- use that mirror instead of gdown.
RUN mkdir -p data && cd data && \
    curl -sfL -o items_shuffle_1000.json "https://raw.githubusercontent.com/WooooDyy/AgentGym/main/agentenv-webshop/webshop/data/items_shuffle_1000.json" && \
    curl -sfL -o items_ins_v2_1000.json "https://raw.githubusercontent.com/WooooDyy/AgentGym/main/agentenv-webshop/webshop/data/items_ins_v2_1000.json" && \
    curl -sfL -o items_human_ins.json "https://raw.githubusercontent.com/WooooDyy/AgentGym/main/agentenv-webshop/webshop/data/items_human_ins.json"

# Build the Lucene index the env needs to even instantiate (not just to
# search) -- see requirements.txt's comment. convert_product_file_format.py
# writes into 4 resource-variant dirs without creating any of them first.
RUN cd search_engine && mkdir -p resources resources_100 resources_1k resources_100k && \
    python convert_product_file_format.py && bash run_indexing.sh

WORKDIR /app
COPY run.py router.py worker.py .
ENV PYTHONPATH=/app/webshop

EXPOSE 8422
CMD ["python3", "run.py"]
