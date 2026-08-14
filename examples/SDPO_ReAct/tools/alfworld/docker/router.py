"""Public entry point for the alfworld_step sidecar -- the ONLY process bound
to the container's exposed port (see run.py, which starts this alongside
N worker.py processes).

Why a router + N workers instead of one process: worker.py's actual work
(tatsu PDDL/grammar parsing, fast_downward grounding) is pure Python, so it
holds the GIL -- one process, however many threads, only ever runs ONE of
these at a time. Confirmed live on a 96-core host: a single worker.py process
maxed out around 1.1-1.3 CPUs under an 800-session eval load (100 unique
ALFWorld game files x n_samples_per_eval_prompt=8), regardless of thread pool
size (128 vs 512 made no real difference) -- the bottleneck was never thread
count, it was one interpreter's GIL. Splitting into N worker PROCESSES gives
each its own interpreter/GIL, so N game files now genuinely ground/parse in
parallel on N cores.

Routing key is GAME_FILE, not session_id: every session replaying the SAME
game_file (all n_samples_per_eval_prompt of them, per AlfworldPool's
game-file-pinning) hashes to the SAME worker, so that worker's
_PDDL2SAS_CACHE (see worker.py) still gets a cache hit for samples 2..N of
that file instead of a cross-process cache miss -- routing by session_id
instead would scatter those samples across workers and turn a single-flight
grounding win back into "repeat the grounding on every worker that happens to
get one of this file's sessions." Different game files hash to different
workers, which is exactly the parallelism this file exists to create.

DELETE has no game_file in its request (just a path param), so it can't be
routed by the SAME key /step used to create the session -- a session's
session_id and its game_file hash to different workers in general. Rather
than guess, DELETE broadcasts to every worker and returns the first non-404;
harmless (idempotent no-op on the N-1 workers that never had the session)
and simple, and cheap since ../client.py never actually calls DELETE (kept
only for contract parity with the pre-router single-process server).

Health: proxies to worker 0 (arbitrary pick; router.py's own /health should
reflect "the sidecar can serve requests", not aggregate every worker's
status -- if worker 0 is up the process pool as a whole is almost certainly
fine, and a full N-way health fan-out adds N HTTP round trips to every health
probe for no real signal, since workers don't have independent liveness
beyond process-alive, which the container's own health check already implies
by the port being open).
"""

import asyncio
import hashlib
import json
import os

import httpx
from fastapi import FastAPI, Request
from starlette.responses import Response

N_WORKERS = int(os.environ.get("ALFWORLD_N_WORKERS", "16"))
_WORKER_BASE_PORT = int(os.environ.get("ALFWORLD_WORKER_BASE_PORT", "9001"))
_WORKER_URLS = [f"http://127.0.0.1:{_WORKER_BASE_PORT + i}" for i in range(N_WORKERS)]

app = FastAPI()
_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))


def _worker_index(key: str) -> int:
    """Stable hash of ``key`` (a game_file or session_id) -> worker index.

    blake2b, not the builtin hash() (PYTHONHASHSEED-salted per process, so it
    would map the same key to a different worker on every restart -- fine for
    THIS process alone since a fresh container has no cross-restart session
    state to preserve, but blake2b costs nothing and removes the footgun for
    whoever copies this pattern into a context where it'd matter).
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % N_WORKERS


async def _proxy(worker_url: str, method: str, path: str, body: bytes | None) -> Response:
    try:
        resp = await _client.request(method, f"{worker_url}{path}", content=body)
    except httpx.HTTPError as e:
        return Response(
            content=json.dumps({"error": f"worker unreachable: {e}"}).encode(),
            status_code=503,
            media_type="application/json",
        )
    return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


@app.get("/health")
async def health() -> Response:
    return await _proxy(_WORKER_URLS[0], "GET", "/health", None)


@app.post("/session/{session_id}/step")
async def step(session_id: str, request: Request) -> Response:
    body = await request.body()
    # game_file (when present) is the routing key -- see module docstring for
    # why that, not session_id, is what keeps grounding cache hits local to
    # one worker. Falls back to session_id-hash only if the request has no
    # game_file (defensive -- every real alfworld_step call has one; see
    # ../client.py), so routing degrades to "some worker" rather than 400ing.
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        payload = {}
    key = payload.get("game_file") or session_id
    worker_url = _WORKER_URLS[_worker_index(key)]
    return await _proxy(worker_url, "POST", f"/session/{session_id}/step", body)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> Response:
    # No game_file in a DELETE -- can't route by the same key /step used to
    # create the session (see module docstring), so broadcast to every
    # worker instead of guessing. All but the one owning worker no-op
    # (cleanup() on an absent session_id is a plain dict.pop(..., None)).
    responses = await asyncio.gather(
        *(_proxy(url, "DELETE", f"/session/{session_id}", None) for url in _WORKER_URLS)
    )
    return next((r for r in responses if r.status_code == 200), responses[0])
