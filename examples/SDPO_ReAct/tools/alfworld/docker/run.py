"""Container entrypoint: starts N worker.py processes on internal ports, then
router.py (the process actually bound to the container's public port 8423)
in the foreground. See router.py's docstring for why the sidecar is split
into a router + N workers at all.

A plain multiprocessing.Process (not asyncio subprocesses) per worker, each
running its OWN uvicorn/FastAPI app -- these are full Python interpreters
with their own GIL, not coroutines sharing one, which is the entire point
(see router.py's docstring). Workers are spawned before router.py's uvicorn
takes over this process's foreground, so if a worker crashes at import time
(e.g. a bad ALFWORLD_CONFIG), that surfaces immediately rather than only once
the first request happens to hash to it.
"""

import multiprocessing
import os
import time

import httpx
import uvicorn

N_WORKERS = int(os.environ.get("ALFWORLD_N_WORKERS", "16"))
_WORKER_BASE_PORT = int(os.environ.get("ALFWORLD_WORKER_BASE_PORT", "9001"))


def _run_worker(port: int) -> None:
    uvicorn.run("worker:app", host="127.0.0.1", port=port, log_level="info")


def main() -> None:
    procs = []
    for i in range(N_WORKERS):
        port = _WORKER_BASE_PORT + i
        p = multiprocessing.Process(target=_run_worker, args=(port,), name=f"alfworld-worker-{i}", daemon=True)
        p.start()
        procs.append(p)

    # Wait for every worker's own /health before starting the router, so the
    # FIRST real request the router forwards doesn't race a worker that's
    # still importing alfworld/textworld (a multi-second cold start -- see
    # worker.py's module docstring).
    deadline = time.time() + 120
    for i, p in enumerate(procs):
        port = _WORKER_BASE_PORT + i
        while True:
            if not p.is_alive():
                raise RuntimeError(f"alfworld worker {i} (pid target port {port}) died during startup")
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                raise RuntimeError(f"alfworld worker {i} on port {port} did not become healthy within 120s")
            time.sleep(0.5)

    uvicorn.run("router:app", host="0.0.0.0", port=8423, log_level="info")


if __name__ == "__main__":
    main()
