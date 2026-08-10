"""Public entry point for the webshop_step sidecar -- the ONLY process bound
to the container's exposed port (see run.py, which starts this alongside
N worker.py processes).

Why a router + N workers instead of one process: worker.py's actual work
(WebAgentTextEnv.step()'s HTML/text processing, gym.make()'s Lucene/JVM
index open) is pure-Python/JVM-bound in a way one process's GIL fully
serializes -- one process, however many threads, only ever makes progress on
ONE session's work at a time. Splitting into N worker PROCESSES gives each
its own interpreter/GIL, so N sessions now genuinely make progress in
parallel on N cores. Mirrors the identical fix (and its measured wall-clock
win) applied to ../../alfworld/docker/ -- see that router.py's docstring for
the numbers.

Routing key is TASK_ID, not session_id: every session replaying the SAME
task_id (all n_samples_per_eval_prompt of them, per WebAgentTextEnv.reset
(session=task_id)'s native task-pinning) hashes to the SAME worker. Unlike
ALFWorld, WebShop's worker.py has no cross-session grounding cache today, so
this doesn't win a cache hit the way ALFWorld's does -- but it's the same
low-cost choice (task_id is always present in a real /step call; see
../client.py) and keeps a future per-task_id cache (e.g. memoizing
get_goals()'s per-task price threshold) trivially local to one worker if one
is ever added. Sessions with no task_id (task_id is None) route by
session_id instead, since there's no task_id to hash.

DELETE has no task_id in its request (just a path param), so it can't be
routed by the SAME key /step used to create the session -- broadcasts to
every worker and returns the first non-404 (harmless idempotent no-op on the
N-1 workers that never had the session). ../client.py never actually calls
DELETE; kept only for contract parity with the pre-router single-process
server.

Health: proxies to worker 0 (arbitrary pick -- see ../../alfworld/docker/
router.py's docstring for why this, not an N-way fan-out, is the right
signal for "can the sidecar serve requests").
"""

import asyncio
import hashlib
import json
import os

import httpx
from fastapi import FastAPI, Request
from starlette.responses import Response

N_WORKERS = int(os.environ.get("WEBSHOP_N_WORKERS", "16"))
_WORKER_BASE_PORT = int(os.environ.get("WEBSHOP_WORKER_BASE_PORT", "9101"))
_WORKER_URLS = [f"http://127.0.0.1:{_WORKER_BASE_PORT + i}" for i in range(N_WORKERS)]

app = FastAPI()
_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))


def _worker_index(key: str) -> int:
    """Stable hash of ``key`` (a task_id or session_id) -> worker index.

    blake2b, not the builtin hash() (PYTHONHASHSEED-salted per process, so it
    would map the same key to a different worker on every restart).
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
    # task_id (when present) is the routing key -- see module docstring.
    # Falls back to session_id-hash only if the request has no task_id
    # (every real webshop_step call has one; see ../client.py).
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        payload = {}
    task_id = payload.get("task_id")
    key = str(task_id) if task_id is not None else session_id
    worker_url = _WORKER_URLS[_worker_index(key)]
    return await _proxy(worker_url, "POST", f"/session/{session_id}/step", body)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> Response:
    # No task_id in a DELETE -- can't route by the same key /step used to
    # create the session (see module docstring), so broadcast to every
    # worker instead of guessing. All but the one owning worker no-op
    # (cleanup() on an absent session_id is a plain dict.pop(..., None)).
    responses = await asyncio.gather(
        *(_proxy(url, "DELETE", f"/session/{session_id}", None) for url in _WORKER_URLS)
    )
    return next((r for r in responses if r.status_code == 200), responses[0])
