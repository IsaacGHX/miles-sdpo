"""Dataset loading for the TTS harness.

Reuses the datasets the SDPO example already builds -- their JSONL rows are
``{"prompt": [{"role","content"}...], "label": str, "metadata": {"domain": ...}}``
(see examples/SDPO/build_math_eval.py and build_sci_dataset.py). This loader
extracts the BARE question (the user turn, minus SDPO's own system prompt and
answer-format boilerplate, since this harness supplies its OWN solver system
prompt and answer contract) plus the label and domain, into the flat
``{"problem", "label", "domain"}`` dicts the scaffold consumes.

Build the source data with SDPO's builders (they need `datasets` installed):
    python examples/SDPO/build_math_eval.py   --out-dir /root/math_eval
    python examples/SDPO/build_sci_dataset.py --out-dir /root/sci
Then point --train-data / --eval-data at the resulting .jsonl files. Any JSONL
in the same {prompt,label,metadata} shape works.
"""

from __future__ import annotations

import json
import re

# SDPO's own answer-format instruction lines that we strip from the question,
# because THIS harness's solver system prompt owns the answer contract. Mirrors
# the spirit of examples/SDPO/sdpo.py::_ANSWER_FORMAT_PATTERNS and
# examples/SDPO_ReAct/native_prompt.py::_strip_dapo_wrapper.
_STRIP_PATTERNS = [
    re.compile(r"\n*Please reason step by step\.?\s*$", re.IGNORECASE),
    re.compile(r"\n*Let's think step by step\.?\s*$", re.IGNORECASE),
]
# The DAPO-math-17k wrapper (prefix + suffix) that conflicts with our <answer>
# contract -- strip it so the ONLY final-answer directive is the solver system
# prompt's. Same regexes examples/SDPO_ReAct/native_prompt.py uses.
_DAPO_PREFIX_RE = re.compile(
    r"^Solve the following math problem step by step\. The last line of your response "
    r"should be of the form Answer: \\boxed\{\$Answer\} where \$Answer is the answer to "
    r"the problem\.\n\n",
)
_DAPO_SUFFIX_RE = re.compile(r'\n\nRemember to put your answer on its own line after "Answer:"\.\s*$')


def _extract_question(prompt) -> str:
    """The user question from an SDPO-format prompt (message list or string)."""
    if isinstance(prompt, str):
        text = prompt
    elif isinstance(prompt, list):
        # last user turn (SDPO puts the question there; system holds format rules)
        user_turns = [m.get("content", "") for m in prompt if m.get("role") == "user"]
        text = user_turns[-1] if user_turns else ""
    else:
        text = str(prompt)
    text = text.strip()
    text = _DAPO_PREFIX_RE.sub("", text)
    text = _DAPO_SUFFIX_RE.sub("", text)
    for pat in _STRIP_PATTERNS:
        text = pat.sub("", text).strip()
    return text.strip()


def load_dataset_jsonl(path: str, *, default_domain: str = "math", limit: int = 0) -> list[dict]:
    """Load an SDPO-format JSONL into flat {"problem","label","domain"} items."""
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            problem = _extract_question(row.get("prompt", ""))
            if not problem:
                continue
            md = row.get("metadata") or {}
            items.append(
                {
                    "problem": problem,
                    "label": str(row.get("label", "") or "").strip(),
                    "domain": (md.get("domain") or default_domain).strip().lower(),
                }
            )
            if limit and len(items) >= limit:
                break
    return items


def load_many(paths: list[str], *, default_domain: str = "math", limit_each: int = 0) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        out.extend(load_dataset_jsonl(p, default_domain=default_domain, limit=limit_each))
    return out


def load_code_dataset_jsonl(path: str, *, limit: int = 0) -> list[dict]:
    """Load an SDPO_ReAct code-domain JSONL (examples/SDPO_ReAct/data/build_code_data.py's
    row shape: {"prompt":[{"role","content"}...], "metadata":{"domain":"code","test_cases":[...]}})
    into flat {"problem","test_cases","difficulty"} dicts for multi_turn_code.py.
    Unlike load_dataset_jsonl, the bare question is NOT stripped of any solver
    system prompt boilerplate -- this harness supplies its OWN system prompt
    (the optimized variable) and only needs the raw user question + tests."""
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt", [])
            user_turns = [m.get("content", "") for m in prompt if m.get("role") == "user"]
            problem = (user_turns[-1] if user_turns else "").strip()
            md = row.get("metadata") or {}
            tests = md.get("test_cases") or []
            if not problem or not tests:
                continue
            items.append({"problem": problem, "test_cases": tests, "difficulty": md.get("difficulty", "")})
            if limit and len(items) >= limit:
                break
    return items
