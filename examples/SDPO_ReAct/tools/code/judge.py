"""LiveCodeBench-style code judge for SDPO_ReAct.

Grades a candidate program against a set of stdin/stdout test cases by running
it in the SAME Docker sandbox the code_interpreter tool uses (tools/docker/,
port 8420) -- no new sidecar/port. The sandbox's /execute takes only {code},
no stdin, so we WRAP the candidate + all its test cases into ONE self-contained
harness program per submission: the harness embeds the candidate source and the
test I/O, runs the candidate once per test case (feeding the test input via a
patched sys.stdin, capturing stdout), compares against expected output, and
prints a JSON verdict the caller parses back.

Reward contract (used by the SDPO code grader, see sdpo.py domain dispatch):
  grade_code(code, test_cases) -> float in [0,1] = fraction of tests passed.
  A trace is "correct" for SDPO if this is 1.0 (all tests pass) -- same
  all-or-nothing notion of correctness math uses, so the group's correct-peer
  selection is consistent across domains.

Test-case shape (LiveCodeBench-style): a list of {"input": str, "output": str}.
Comparison is whitespace-normalized per line (strip trailing spaces + trailing
newlines), matching LiveCodeBench's own lenient stdout comparison.

Debug from repo root (needs the sandbox up: bash tools/run_sandbox.sh):
    python -m examples.SDPO_ReAct.tools.code.judge --demo
"""

import json

from miles.utils.http_utils import post

SANDBOX_URL = None  # set lazily from env so imports don't require it


def _sandbox_url() -> str:
    import os

    return os.environ.get("SDPO_REACT_SANDBOX_URL", "http://127.0.0.1:8420")


# Per-test wall-clock budget inside the harness (the sandbox's own /execute
# timeout must be >= n_tests * this; we pass a generous overall timeout below).
_PER_TEST_TIMEOUT = 6
_MAX_TESTS = 15  # cap tests actually run so one pathological problem can't hang the batch


def _extract_code_block(response: str) -> str:
    """Pull the candidate program out of a model response. Prefer a fenced
    ```python block; else the <answer> tag; else the whole response. The model
    is prompted (code system prompt) to put its final program in ```python."""
    import re

    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m[-1].strip()
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return response.strip()


def _build_harness(candidate: str, test_cases: list[dict]) -> str:
    """A self-contained program: run `candidate` (as a fresh subprocess-like
    exec with patched stdin) against each test case, print JSON {passed,total}.
    Runs entirely inside the sandbox via one /execute call."""
    tests = [{"input": t.get("input", ""), "output": t.get("output", "")} for t in test_cases[:_MAX_TESTS]]
    return (
        "import sys, io, json, contextlib, multiprocessing as mp\n"
        f"CANDIDATE = {candidate!r}\n"
        f"TESTS = {json.dumps(tests)}\n"
        f"PER_TEST_TIMEOUT = {_PER_TEST_TIMEOUT}\n"
        "def _run(inp, q):\n"
        "    buf = io.StringIO()\n"
        "    sys.stdin = io.StringIO(inp)\n"
        "    try:\n"
        "        with contextlib.redirect_stdout(buf):\n"
        "            exec(compile(CANDIDATE, '<candidate>', 'exec'), {'__name__': '__main__'})\n"
        "        q.put(buf.getvalue())\n"
        "    except Exception as e:\n"
        "        q.put('__ERROR__' + repr(e))\n"
        "def _norm(s):\n"
        "    return '\\n'.join(line.rstrip() for line in s.strip().splitlines())\n"
        "passed = 0\n"
        "for t in TESTS:\n"
        "    q = mp.Queue()\n"
        "    p = mp.Process(target=_run, args=(t['input'], q))\n"
        "    p.start(); p.join(PER_TEST_TIMEOUT)\n"
        "    if p.is_alive():\n"
        "        p.terminate(); p.join(); continue\n"
        "    try:\n"
        "        out = q.get_nowait()\n"
        "    except Exception:\n"
        "        continue\n"
        "    if isinstance(out, str) and not out.startswith('__ERROR__') and _norm(out) == _norm(t['output']):\n"
        "        passed += 1\n"
        "print(json.dumps({'passed': passed, 'total': len(TESTS)}))\n"
    )


async def grade_code(response: str, test_cases: list[dict]) -> float:
    """Fraction of test cases the response's program passes, in [0,1]. Returns
    0.0 on any sandbox/parse failure (a broken submission is simply wrong)."""
    if not test_cases:
        return 0.0
    candidate = _extract_code_block(response)
    if not candidate.strip():
        return 0.0
    harness = _build_harness(candidate, test_cases)
    overall_timeout = min(60.0, _PER_TEST_TIMEOUT * min(len(test_cases), _MAX_TESTS) + 5)
    try:
        payload = await post(
            f"{_sandbox_url()}/execute",
            {"code": harness, "timeout": overall_timeout},
            max_retries=3,
            action="post",
        )
    except Exception:
        return 0.0
    if payload.get("timed_out"):
        return 0.0
    stdout = (payload.get("stdout") or "").strip()
    # the harness prints exactly one JSON line last
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                v = json.loads(line)
                total = v.get("total", 0)
                return (v.get("passed", 0) / total) if total else 0.0
            except Exception:
                return 0.0
    return 0.0


def main() -> None:
    import argparse
    import asyncio

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.parse_args()

    # toy: read two ints, print their sum
    good = "```python\na,b=map(int,input().split())\nprint(a+b)\n```"
    bad = "```python\na,b=map(int,input().split())\nprint(a*b)\n```"
    tests = [{"input": "2 3\n", "output": "5"}, {"input": "10 20\n", "output": "30"}]
    print("good ->", asyncio.run(grade_code(good, tests)), "(expect 1.0)")
    print("bad  ->", asyncio.run(grade_code(bad, tests)), "(expect 0.0)")


if __name__ == "__main__":
    main()
