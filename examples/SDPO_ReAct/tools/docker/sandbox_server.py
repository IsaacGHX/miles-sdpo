"""Sidecar sandbox server for the SDPO_ReAct code_interpreter tool.

Runs INSIDE the sandbox Docker container (not the training/rollout host), and
exposes exactly one HTTP endpoint on exactly one port for the whole training
job (see run_sandbox.sh) -- the server host this runs on has a hard limit on
how many ports a job may open, so every rollout worker/engine shares this same
sidecar instead of spawning its own container/port per call.

Execution model: each request runs the submitted code as a fresh `python3 -c`
subprocess *inside this container* (miles/backends never spawn the process
directly -- see tools/tool_client.py), so a runaway or malicious snippet can at
most exhaust this container's own resources, never the training host's. This
is real OS-process + container isolation, unlike AgentFlow's python_coder tool
(in-process exec()) and stricter than examples/retool_v2/tool_sandbox.py's
subprocess-on-the-host-process sandbox.

Concurrency: `execute()` uses `asyncio.create_subprocess_exec` (non-blocking)
rather than `subprocess.run` in a sync endpoint -- a sync `def` route makes
FastAPI hand it to Starlette's default thread pool (bounded, historically
~40 threads), which caps real parallelism well below what a --cpus=32 (see
run_sandbox.sh) container can actually execute concurrently. The async
version lets uvicorn's single event loop dispatch as many concurrent
subprocesses as the container's CPU/memory budget allows, with no artificial
thread-pool ceiling in between.

Contract (see tool_client.py for the caller side):
    POST /execute {"code": str, "timeout": float | None}
      -> {"stdout": str, "error": str | None, "timed_out": bool}
    GET  /stats -> execution-duration histogram (see StatsResponse) -- added to
      diagnose whether slow training throughput traces back to this sandbox's
      long tail (a handful of pathological snippets -- e.g. accidental large
      loops/sympy calls -- holding a worker for close to the timeout) or is
      genuinely just rollout/generation time elsewhere. Every call's duration
      and (if slow) a code preview are also logged to stdout (`docker logs
      sdpo-react-sandbox`) for the same reason.
"""

import ast
import asyncio
import logging
import sys
import time
from collections import deque

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("sandbox")

app = FastAPI()

DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_TIMEOUT_SECONDS = 60.0
MAX_OUTPUT_CHARS = 4000

# A call logged individually as "slow" (with a code preview) above this
# threshold -- well under the 10s timeout, so we see the buildup before a call
# actually times out, not just the timeouts themselves.
SLOW_LOG_THRESHOLD_SECONDS = 3.0
CODE_PREVIEW_CHARS = 200

# Rolling window for /stats -- bounded so a long-running sidecar doesn't grow
# this unboundedly; 5000 calls is far more than one rollout batch needs to
# show a representative tail.
_DURATIONS: deque[float] = deque(maxlen=5000)
_TIMED_OUT_COUNT = 0
_ERROR_COUNT = 0
_TOTAL_COUNT = 0


class ExecuteRequest(BaseModel):
    code: str
    timeout: float | None = None


class ExecuteResponse(BaseModel):
    stdout: str
    error: str | None
    timed_out: bool


class StatsResponse(BaseModel):
    count: int
    timed_out_count: int
    error_count: int
    min_seconds: float | None
    p50_seconds: float | None
    p95_seconds: float | None
    p99_seconds: float | None
    max_seconds: float | None


def _truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n...[truncated {len(text) - max_chars} chars]...\n{tail}"


def _percentile(sorted_values: list[float], pct: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = min(len(sorted_values) - 1, int(round(pct * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _auto_print_trailing_expr(code: str) -> str:
    """If the LAST top-level statement is a bare expression (no assignment,
    no print()), rewrite it to print(repr(...)) -- Jupyter/REPL auto-print
    behavior, which is what the model actually expects (it writes code like
    `k = 2\nm = 1\nk_plus_m = k + m\nk_plus_m` assuming the trailing
    `k_plus_m` gets echoed back). `python3 -c` never does this -- only
    explicit print() calls produce stdout -- so a huge fraction of
    code_interpreter calls came back as "(no output)" even though the
    computation succeeded and the model was staring right at the answer,
    one line away. On any parse/rewrite failure, return the code UNCHANGED
    (never let this best-effort rewrite break real execution)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    last = tree.body[-1]
    if not isinstance(last, ast.Expr):
        return code
    if isinstance(last.value, ast.Call) and isinstance(last.value.func, ast.Name) and last.value.func.id == "print":
        # Already a print(...) call -- wrapping would become
        # print(repr(print(...))), which prints "None" (repr of print's own
        # return value) INSTEAD OF what the call already printed.
        return code
    expr_src = ast.get_source_segment(code, last)
    if expr_src is None or "#" in expr_src:
        # get_source_segment can fail to recover exact source for some node
        # shapes; a trailing "#" would also break the print(repr(...)) wrap
        # (a same-line comment swallows the closing parens). Either way, skip
        # the rewrite rather than risk turning working code into a
        # SyntaxError -- never let this best-effort optimization break real
        # execution.
        return code
    lines = code.splitlines()
    start, end = last.lineno - 1, last.end_lineno
    lines[start:end] = [f"print(repr({expr_src}))"]
    return "\n".join(lines)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    values = sorted(_DURATIONS)
    if not values:
        return StatsResponse(
            count=_TOTAL_COUNT,
            timed_out_count=_TIMED_OUT_COUNT,
            error_count=_ERROR_COUNT,
            min_seconds=None,
            p50_seconds=None,
            p95_seconds=None,
            p99_seconds=None,
            max_seconds=None,
        )
    return StatsResponse(
        count=_TOTAL_COUNT,
        timed_out_count=_TIMED_OUT_COUNT,
        error_count=_ERROR_COUNT,
        min_seconds=values[0],
        p50_seconds=_percentile(values, 0.50),
        p95_seconds=_percentile(values, 0.95),
        p99_seconds=_percentile(values, 0.99),
        max_seconds=values[-1],
    )


@app.post("/execute", response_model=ExecuteResponse)
async def execute(req: ExecuteRequest) -> ExecuteResponse:
    global _TIMED_OUT_COUNT, _ERROR_COUNT, _TOTAL_COUNT

    timeout = min(req.timeout, MAX_TIMEOUT_SECONDS) if req.timeout else DEFAULT_TIMEOUT_SECONDS
    start = time.monotonic()
    _TOTAL_COUNT += 1

    def _log(elapsed: float, outcome: str) -> None:
        _DURATIONS.append(elapsed)
        if elapsed >= SLOW_LOG_THRESHOLD_SECONDS:
            preview = req.code[:CODE_PREVIEW_CHARS].replace("\n", "\\n")
            logger.warning("slow execute: %.2fs (%s) code=%r", elapsed, outcome, preview)
        else:
            logger.info("execute: %.2fs (%s)", elapsed, outcome)

    code = _auto_print_trailing_expr(req.code)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            _TIMED_OUT_COUNT += 1
            _log(time.monotonic() - start, "timed_out")
            return ExecuteResponse(stdout="", error=f"Execution timed out after {timeout:.0f} seconds", timed_out=True)
    except Exception as e:  # defensive: never let a malformed request crash the sidecar
        _ERROR_COUNT += 1
        _log(time.monotonic() - start, "spawn_error")
        return ExecuteResponse(stdout="", error=f"Failed to execute code: {e}", timed_out=False)

    elapsed = time.monotonic() - start
    stdout = _truncate(stdout_bytes.decode(errors="replace"))
    error = _truncate(stderr_bytes.decode(errors="replace")) if proc.returncode != 0 else None
    if error:
        _ERROR_COUNT += 1
    _log(elapsed, "error" if error else "ok")
    return ExecuteResponse(stdout=stdout, error=error, timed_out=False)
