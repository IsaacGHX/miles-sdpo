"""Build LiveCodeBench train/eval jsonl for the SDPO_ReAct CODE domain.

Each row (same schema as native_prompt.py's math rows, plus code fields):
    {
      "prompt": [ {system: code prompt}, {user: problem statement} ],
      "label":  "" (unused for code; correctness comes from test cases),
      "tools":  [code_interpreter spec],   # native <tools> injection
      "metadata": {"domain": "code", "test_cases": [{"input","output"}, ...]},
    }

Correctness is decided by examples/SDPO/sdpo.py's code grader (domain=="code"
-> examples/SDPO_ReAct/tools/code/judge.py: run the program against test_cases
in the sandbox, all-or-nothing). The code system prompt tells the model to use
code_interpreter to develop/verify, then put its FINAL program in a ```python
fence (code_judge extracts that).

Initial data = plain LiveCodeBench (user's choice: validate code single-domain
first). Scale later per the DeepCoder curation (TACO Verified + SYNTHETIC-1 +
LiveCodeBench'23-24) -- see the deepcoder-code-data-reference memory.

Usage (module, from repo root):
    python -m examples.SDPO_ReAct.build_code_data --out-dir /root/code_data \
        --n-train 2000 --n-eval 100
"""

import argparse
import json
import os

from examples.SDPO_ReAct.tools.registry import MINIMAL_SYSTEM_PROMPT, all_tool_specs

tool_specs = all_tool_specs  # Q1: every row exposes all tools; model chooses.

# Code system prompt = shared minimal prompt + the grading-critical contract
# for code (the graded artifact is the last PROGRAM actually RUN through
# code_interpreter, reading stdin, printing stdout) + a ONE-SHOT worked example.
#
# The example exists to fix two failure modes observed live in rollouts:
#   1. The model decomposes the problem into steps and VERIFIES its final
#      program with the tool before submitting, instead of dumping an
#      unverified guess straight into the answer.
#   2. The model feeds test input through code_interpreter's `stdin` param,
#      NOT by hardcoding `sys.stdin = io.StringIO(...)` into the source. The
#      hardcode is fatal at grading time: the LAST code_interpreter call's
#      code is re-run verbatim against the REAL hidden test input, so a
#      hardcoded stdin override silently replaces that real input and the
#      submission fails regardless of its logic (observed: every trace that
#      did this scored 0, ~15% of code-domain traces).
CODE_SYSTEM_PROMPT = (
    MINIMAL_SYSTEM_PROMPT
    + (
        "\n\nFor programming problems, your solution is graded by running the LAST "
        "program you pass to code_interpreter (reading input from stdin, printing to "
        "stdout) against hidden test cases -- code you only write in text but never "
        "run does not count. Break the problem into steps, and before submitting, "
        "VERIFY your program by running it through code_interpreter with a sample "
        "input passed via the `stdin` parameter -- never hardcode a fake stdin "
        "inside the code, since the same code is re-run against the real hidden "
        "input at grading time and a hardcoded override would replace it."
    )
    + (
        "\n\nExample:\n"
        "User: Read two integers a and b from stdin (space-separated on one line) "
        "and print their sum.\n"
        "Assistant: Step 1: read one line, split on whitespace, parse two ints. "
        "Step 2: print their sum. Let me verify with a sample input before "
        "finalizing.\n"
        "<tool_call>code_interpreter(code=\"a, b = map(int, input().split())\\n"
        "print(a + b)\", stdin=\"3 5\")</tool_call>\n"
        "Tool result: 8\n"
        "Assistant: Verified -- 3 + 5 = 8 is correct. This is my final program.\n"
        "<answer>a, b = map(int, input().split())\nprint(a + b)</answer>"
    )
)


# Same shape as CODE_SYSTEM_PROMPT, but for FUNCTIONAL (leetcode-style)
# problems, whose graded artifact is a CLASS + METHOD, not a stdin/stdout
# program: judge.py's functional harness execs the candidate and calls
# Solution().<func_name>(*args) directly (no stdin at all). The two things this
# must get right, mirroring the stdin prompt's two observed failure modes:
#   1. The final class/method signature must match the starter code EXACTLY
#      (the harness looks the method up by the dataset's func_name).
#   2. The graded code is still the LAST code RUN through code_interpreter
#      (--sdpo-code-require-tool), so the class definition must be present in
#      that call -- a driver `print(Solution().f(...))` alongside it is fine
#      (the harness discards the candidate's own stdout), but a call that only
#      contains the test driver and imports the class from nowhere is not.
FUNCTIONAL_SYSTEM_PROMPT = (
    MINIMAL_SYSTEM_PROMPT
    + (
        "\n\nFor programming problems given as a function signature (a `class Solution` "
        "starter), your solution is graded by importing the LAST code you pass to "
        "code_interpreter and CALLING the required method directly with the test "
        "arguments -- code you only write in text but never run does not count, and "
        "nothing is read from stdin. Requirements: (a) define `class Solution` with the "
        "method name and parameter order EXACTLY as given in the starter code, (b) "
        "RETURN the answer from that method (do not print it), (c) make sure that same "
        "code_interpreter call contains the full class definition. Break the problem "
        "into steps, and before submitting, VERIFY your solution by calling it on the "
        "example inputs inside the same code_interpreter call and printing the result "
        "(extra prints are ignored at grading time)."
    )
    + (
        "\n\nExample:\n"
        "User: Given a list of integers nums and an integer target, return the indices "
        "of the two numbers that add up to target.\n"
        "```python\nclass Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
        "        \n```\n"
        "Assistant: Step 1: one pass, keep a value -> index map. Step 2: for each value, "
        "check whether target - value was already seen. Let me verify on the example "
        "before finalizing.\n"
        '<tool_call>code_interpreter(code="class Solution:\\n    def twoSum(self, nums: '
        "List[int], target: int) -> List[int]:\\n        seen = {}\\n        for i, v in "
        "enumerate(nums):\\n            if target - v in seen:\\n                return "
        "[seen[target - v], i]\\n            seen[v] = i\\n        return []\\n\\n"
        'print(Solution().twoSum([2, 7, 11, 15], 9))")</tool_call>\n'
        "Tool result: [0, 1]\n"
        "Assistant: Verified -- indices [0, 1] give 2 + 7 = 9. This is my final solution.\n"
        "<answer>class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
        "        seen = {}\n        for i, v in enumerate(nums):\n"
        "            if target - v in seen:\n                return [seen[target - v], i]\n"
        "            seen[v] = i\n        return []</answer>"
    )
)


def _decode_private_tests(val) -> list:
    """LiveCodeBench's private_test_cases: either a plain JSON string (older
    files) or base64(zlib(pickle(json_string))) (current files) -- the same
    two-shape decode LiveCodeBench's own loader does. [] on anything else."""
    if isinstance(val, list):
        return val
    if not isinstance(val, str) or not val:
        return []
    try:
        return json.loads(val)
    except Exception:
        pass
    try:
        import base64
        import pickle
        import zlib

        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(val.encode("utf-8")))))
    except Exception:
        return []


def _normalize_tests(row: dict, include_private: bool = False, max_test_chars: int = 0) -> list[dict]:
    """Extract test cases from a LiveCodeBench-style row into
    [{"input","output","testtype"[,"fn_name"]}]. LiveCodeBench stores
    public_test_cases / private_test_cases as JSON strings of
    [{input, output, testtype}]. Be tolerant of a few shapes so this survives
    minor dataset schema drift.

    include_private (default OFF, so every existing build reproduces byte-for-
    byte): also pull private_test_cases. Public tests alone were enough for the
    initial stdin validation, but leetcode/functional problems ship only ~2
    public tests -- far too few to call a hard problem solved -- while private
    adds ~40. Public tests come FIRST so a later --max-tests cap keeps them.

    max_test_chars (0 = no limit): drop test cases whose input+output exceeds
    it. Needed once private tests are in: a few have multi-MB inputs, and the
    judge embeds the tests in the harness SOURCE, which the sandbox runs via
    `python3 -c` (ARG_MAX ~2MB) -- one such test would fail the whole problem.
    """
    tests: list[dict] = []
    keys = ("public_test_cases", "test_cases", "tests")
    if include_private:
        keys = keys + ("private_test_cases",)
    # LeetCode-style rows carry the graded entrypoint in metadata.func_name;
    # stamp it on each test case so judge.py's functional harness can find the
    # method (test_cases is the only thing the reward path passes it).
    md = row.get("metadata")
    if isinstance(md, str):
        try:
            md = json.loads(md)
        except Exception:
            md = {}
    fn_name = str((md or {}).get("func_name") or "") if isinstance(md, dict) else ""
    for key in keys:
        val = row.get(key)
        if not val:
            continue
        if key == "private_test_cases":
            val = _decode_private_tests(val)
        elif isinstance(val, str):
            try:
                val = json.loads(val)
            except Exception:
                continue
        if isinstance(val, list):
            for t in val:
                if isinstance(t, dict) and "input" in t and "output" in t:
                    inp, out = str(t["input"]), str(t["output"])
                    if max_test_chars and len(inp) + len(out) > max_test_chars:
                        continue
                    # Keep testtype (stdin | functional) -- code_judge grades the
                    # two differently (stdin/stdout harness vs LeetCode function
                    # call). Default stdin for older rows without the field.
                    tc = {"input": inp, "output": out, "testtype": t.get("testtype", "stdin")}
                    if tc["testtype"] == "functional" and fn_name:
                        tc["fn_name"] = fn_name
                    tests.append(tc)
    # dedup while preserving order
    seen = set()
    uniq = []
    for t in tests:
        k = (t["input"], t["output"])
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


def _question(row: dict) -> str:
    for key in ("question_content", "question", "problem", "prompt", "content"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            q = v.strip()
            # Functional (leetcode) problems are only answerable against the
            # exact signature the judge calls -- append the dataset's starter
            # code, as LiveCodeBench's own prompt does.
            starter = row.get("starter_code")
            if isinstance(starter, str) and starter.strip():
                q += (
                    "\n\nComplete the following starter code (keep the class and method "
                    "signature exactly as given, and RETURN the answer):\n"
                    f"```python\n{starter.rstrip()}\n```"
                )
            return q
    return ""


def _build_row(question: str, tests: list[dict], system_prompt: str = CODE_SYSTEM_PROMPT) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "label": "",  # code correctness is from test_cases, not a label string
        "tools": tool_specs,
        "metadata": {"domain": "code", "test_cases": tests},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/code_data")
    ap.add_argument("--hf-repo", default="livecodebench/code_generation_lite")
    ap.add_argument(
        "--jsonl-files",
        nargs="*",
        default=["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
        help="jsonl files in the repo to load directly (bypasses the no-longer-supported dataset "
        "script). test6.jsonl = 2025-01..2025-04 (LCB v6).",
    )
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--max-tests", type=int, default=15, help="cap test cases kept per problem")
    ap.add_argument(
        "--include-private-tests",
        action="store_true",
        help="also use LiveCodeBench's private_test_cases (base64+zlib+pickle-encoded). OFF by "
        "default so existing builds reproduce exactly. Turn ON for functional/leetcode problems: "
        "they ship only ~2 public tests (vs ~40 private), too few to call a hard problem solved.",
    )
    ap.add_argument(
        "--max-test-chars",
        type=int,
        default=0,
        help="drop test cases whose input+output exceeds this many chars (0 = no limit). Use with "
        "--include-private-tests: the judge embeds tests in the harness source, which the sandbox "
        "runs via `python3 -c` (ARG_MAX ~2MB), and a handful of private tests have multi-MB inputs.",
    )
    ap.add_argument(
        "--testtype",
        default="stdin",
        choices=["stdin", "functional", "both"],
        help="which LiveCodeBench problem type to keep. 'stdin' (default) = codeforces-style "
        "stdin/stdout, graded by code_judge's stdin harness (the validated path). 'functional' "
        "= LeetCode-style function-call problems (needs the functional harness). 'both' = all.",
    )
    ap.add_argument(
        "--difficulty",
        default="medium,hard",
        help="comma-separated LiveCodeBench difficulties to keep (easy|medium|hard). Default "
        "'medium,hard': the easy subset is trivially one-shot by Qwen3-4B (zero_tool_call_frac "
        "was 1.0 -- tool never needed, no room for SDPO gains), so exclude it. Use 'easy,medium,hard' "
        "for all.",
    )
    ap.add_argument(
        "--min-date",
        default="",
        help="keep only problems with contest_date >= this YYYY-MM (e.g. 2025-02 for LCB v6, "
        "which post-dates Qwen3-4B's training -> less memorization, genuinely harder). Empty = no filter.",
    )
    args = ap.parse_args()
    keep_diff = {d.strip().lower() for d in args.difficulty.split(",") if d.strip()}

    from huggingface_hub import hf_hub_download

    os.makedirs(args.out_dir, exist_ok=True)

    # Load the repo's jsonl files DIRECTLY. livecodebench/code_generation_lite
    # ships a dataset script (code_generation_lite.py) which newer `datasets`
    # refuses ("Dataset scripts are no longer supported"), but the underlying
    # test*.jsonl files load fine. Each row has question_content +
    # public_test_cases (a JSON string of [{input,output,testtype}]).
    ds = []
    for fname in args.jsonl_files:
        try:
            path = hf_hub_download(args.hf_repo, fname, repo_type="dataset")
        except Exception as e:
            print(f"skip {fname}: {e!r}")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ds.append(json.loads(line))
    print(f"loaded {len(ds)} raw problems from {args.hf_repo}")

    rows = []
    kept_type = {"stdin": 0, "functional": 0}
    kept_diff = {}
    for row in ds:
        q = _question(row)
        tests = _normalize_tests(row, include_private=args.include_private_tests, max_test_chars=args.max_test_chars)
        if not (q and tests):
            continue
        ttype = tests[0].get("testtype", "stdin")
        if args.testtype != "both" and ttype != args.testtype:
            continue
        diff = (row.get("difficulty") or "unknown").strip().lower()
        if diff not in keep_diff:
            continue
        if args.min_date and (row.get("contest_date", "")[:7] < args.min_date):
            continue
        kept_type[ttype] = kept_type.get(ttype, 0) + 1
        kept_diff[diff] = kept_diff.get(diff, 0) + 1
        # functional rows are graded by a function CALL, not stdin/stdout -> the
        # grading contract in the system prompt has to match (see
        # FUNCTIONAL_SYSTEM_PROMPT). 'both' mode mixes the two per row.
        prompt = FUNCTIONAL_SYSTEM_PROMPT if ttype == "functional" else CODE_SYSTEM_PROMPT
        # carry difficulty in metadata for later analysis / stratified eval
        r = _build_row(q, tests[: args.max_tests], system_prompt=prompt)
        r["metadata"]["difficulty"] = diff
        # provenance, so a future contamination check can join eval rows back to
        # the upstream problem without re-deriving them from the question text
        r["metadata"]["testtype"] = ttype
        for k in ("question_id", "platform", "contest_date"):
            if row.get(k):
                r["metadata"][k] = str(row[k])
        rows.append(r)
    print(f"kept {len(rows)} problems (testtype={args.testtype}, difficulty={sorted(keep_diff)}): types={kept_type} diffs={kept_diff}")

    # When building an EVAL-ONLY set (n_train==0, e.g. a held-out v6 window),
    # take up to n_eval rows directly. Otherwise cap eval at 20% so train keeps
    # the bulk.
    n_eval = min(args.n_eval, len(rows)) if args.n_train == 0 else min(args.n_eval, len(rows) // 5)
    eval_rows, train_rows = rows[:n_eval], rows[n_eval : n_eval + args.n_train]

    train_path = os.path.join(args.out_dir, "livecodebench_train.jsonl")
    eval_path = os.path.join(args.out_dir, "livecodebench_eval.jsonl")
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(eval_path, "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_rows)} train -> {train_path}")
    print(f"Wrote {len(eval_rows)} eval  -> {eval_path}")
    if train_rows:
        ex = train_rows[0]
        print(f"example: {len(ex['metadata']['test_cases'])} test cases, question {len(ex['prompt'][1]['content'])} chars")


if __name__ == "__main__":
    main()
