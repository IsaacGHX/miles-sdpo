"""ReAct system prompt for SDPO_ReAct: plain-text <code>/<answer> tags, in the
SAME style as examples/search-r1's <search>/<answer> tags -- NOT chat-template
native tool-calling. See generate_with_tools.py's module docstring for why
(chat-template `tools=` injection silently never fires once the dataset is
pre-rendered to a plain string, which is what caused the base version's
near-zero tool-call rate).

The one-shot is a single worked example given as PLAIN TEXT inside the system
prompt -- not a fake multi-turn chat history nested in separate
user/assistant/tool messages. It shows the exact tag format directly:
`<code>...</code>` -> `<output>...</output>` -> `<answer>...</answer>`. This
mirrors how examples/search-r1 leaves example construction to its dataset's
own prompt template (the Search-R1 upstream data already teaches the tag
format via instructions, no fake conversation), rather than manufacturing
several extra `apply_chat_template` role-turns just to demonstrate a plain
text pattern.

Usage: run as a one-shot PREPROCESSING step over a plain {prompt, label}
dataset (e.g. dapo-math-17k.jsonl), producing a new jsonl whose `prompt` field
is [{"role": "system", ...}, {"role": "user", "content": question}] --
still rendered ONCE via --apply-chat-template at the Dataset level, same as
any other example; generate_with_tools.py then does plain string
concatenation ("prompt_text + response") for the actual multi-turn loop, so no
further chat-template calls happen after this.

    python -m examples.SDPO_ReAct.react_prompt \
        --in /root/dapo-math-17k/dapo-math-17k.jsonl \
        --out /root/dapo-math-17k/dapo-math-17k-react.jsonl
"""

import argparse
import json
import re

# DAPO-math-17k's own prompt wraps the bare question in ITS OWN answer-format
# instruction ("...Answer: \boxed{$Answer}...", "Remember to put your answer
# on its own line after 'Answer:'."). That wrapper contradicts our system
# prompt's <answer>...</answer> tag instruction (the model sees two
# conflicting answer-format directives at once), so it must be stripped,
# leaving only the bare math question -- react_prompt.py's own system prompt
# is the ONLY place answer-format instructions should come from. AIME-2024's
# prompt has no such wrapper (bare question already), so this is a no-op there.
_DAPO_PREFIX_RE = re.compile(
    r"^Solve the following math problem step by step\. The last line of your response "
    r"should be of the form Answer: \\boxed\{\$Answer\} where \$Answer is the answer to "
    r"the problem\.\n\n",
)
_DAPO_SUFFIX_RE = re.compile(r'\n\nRemember to put your answer on its own line after "Answer:"\.\s*$')


def _strip_dapo_wrapper(question: str) -> str:
    question = _DAPO_PREFIX_RE.sub("", question)
    question = _DAPO_SUFFIX_RE.sub("", question)
    return question

REACT_SYSTEM_PROMPT = """You are a careful math problem solver with access to one tool:
- code_interpreter: runs a Python code snippet and returns its printed output. Available: sympy, numpy, scipy, math, and the standard library. No network access, no file access.

IMPORTANT -- each code_interpreter call is a FRESH, ISOLATED Python process:
- Variables, imports, and function definitions do NOT persist between calls. A name defined in one <code> block is gone by the next one.
- If a later step needs a value from an earlier step, RECOMPUTE it (or print it and copy the number into the next block) -- do not assume it is still in scope.
- Each snippet must print() everything it needs you to see; nothing is returned except what is printed.

To use the tool, put your Python code between <code> and </code>. Its output will be given back to you between <output> and </output>. When you are ready to give your final answer, put it between <answer> and </answer>.

Example:
<code>
print(sum(range(1, 11)))
</code>

<output>
55
</output>

The code confirms the sum is 55.

<answer>55</answer>

Always use the tool to verify any calculation before you answer -- do not rely on mental arithmetic or algebra alone, even when you feel confident. Double-check your result (recompute it a different way, or plug it back into the original problem) and confirm it is actually correct before giving your final answer. You may call the tool multiple times across multiple turns. Always end your response with either a <code> block or an <answer> block -- never both, and never neither."""


def build_react_messages(question: str) -> list[dict]:
    """Full prompt for one training/eval sample: system (tag format + inline
    one-shot example) + the real question. Passed as the dataset's `prompt`
    field (a message list), rendered once by --apply-chat-template
    (miles/utils/data.py::Dataset)."""
    return [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def _extract_question(prompt) -> str:
    if isinstance(prompt, str):
        return _strip_dapo_wrapper(prompt)
    if isinstance(prompt, list):
        # dapo-math-17k / aime-2024 jsonls store `prompt` as a single-user-turn
        # message list already; take its content as the real question text.
        for message in prompt:
            if message.get("role") == "user":
                return _strip_dapo_wrapper(message["content"])
    raise ValueError(f"Unrecognized prompt format: {prompt!r}")


def build_react_dataset(in_path: str, out_path: str) -> int:
    n = 0
    with open(in_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = _extract_question(row["prompt"])
            row["prompt"] = build_react_messages(question)
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    n = build_react_dataset(args.in_path, args.out_path)
    print(f"Wrote {n} ReAct-prefixed rows -> {args.out_path}")


if __name__ == "__main__":
    main()
