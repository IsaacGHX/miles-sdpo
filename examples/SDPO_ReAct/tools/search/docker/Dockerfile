# SDPO_ReAct search/open/find sidecar -- one long-lived container per training
# job, exposing exactly one port (see ../run_search_sidecar.sh), same pattern
# as ../../docker/ (the code_interpreter sandbox).
#
# WHY A SEPARATE CONTAINER (not folded into the training image, not the code
# sandbox): gpt-oss (which vendors the search/open/find tool implementation
# this wraps, see server.py's module docstring) requires Python >=3.12. The
# training container runs a different Python -- rather than upgrading it (and
# risking the ABI breakage documented in ../../run_retrieval.sh's BM25
# section), this tool ships as its own isolated, disposable environment, using
# the exact same "one sidecar / one fixed port, reached over HTTP" pattern
# already established for the code sandbox and the retrieval server.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY browser.py server.py retrieval_adapter.py .

EXPOSE 8421
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8421"]
