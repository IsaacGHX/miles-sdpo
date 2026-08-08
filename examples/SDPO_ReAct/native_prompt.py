"""Native chat-template tool-calling prompt + data prep for SDPO_ReAct on
Qwen3 (and any model whose chat template natively renders a `<tools>` block).

Why a SEPARATE module from react_prompt.py: react_prompt.py teaches plain-text
`<code>/<answer>` tags because Qwen2.5's chat template silently drops the
`tools=` schema once the dataset is pre-rendered to a string (measured
~98-100% zero-tool-call). Qwen3 is the opposite -- it was trained on native
`<tool_call>` grammar and its template DOES inject a `<tools>` block from
`apply_chat_template(tools=...)` (verified on Qwen3-4B). So on Qwen3 we use the
model's OWN tool-calling grammar (`miles.rollout.generate_hub.multi_turn.
generate` + `--generate-tool-call-parser qwen25`, which parses the `<tool_call>`
XML Qwen3 emits) instead of hand-taught text tags.

Integration with SDPO's skill-prefix splice (see the module memory note
"sdpo-react-native-integration-design"): each row is written as a message list
PLUS a top-level `tools` field. Launched with `--apply-chat-template --tool-key
tools`, miles/utils/data.py renders `apply_chat_template(tools=...)` ONCE at
load time, baking the `<tools>` block into the STRING prompt. That string then
(a) satisfies multi_turn.generate (a string prompt is not re-injected, so no
double `<tools>` block) and (b) satisfies examples/SDPO/sdpo.py's teacher-prefix
splice, which requires `isinstance(sample.prompt, str)`. No core code is
forked.

Final-answer contract: the model gives its final answer inside
`<answer>...</answer>` (graded by examples/SDPO/sdpo.py with
`--sdpo-answer-tag answer`; `_extract_tagged_answer` scans the WHOLE response
for the last such tag, which is robust for a long multi-turn trajectory --
unlike the raw `\\boxed{}` path that only scans the last 100 chars and would
miss an answer followed by trailing prose).

Usage (data prep, run as a module from REPO_ROOT so the package import works):
    python -m examples.SDPO_ReAct.native_prompt \\
        --in  /root/dapo-math-17k/dapo-math-17k.jsonl \\
        --out /root/dapo-math-17k/dapo-math-17k-native.jsonl
"""

import argparse
import json
import re

from examples.SDPO_ReAct.tools.registry import MINIMAL_SYSTEM_PROMPT, all_tool_specs

tool_specs = all_tool_specs  # Q1: every row exposes all tools; model chooses.

# DAPO-math-17k wraps each question in its own "Answer: \boxed{$Answer}"
# instruction, which conflicts with our <answer>...</answer> contract. Strip it
# (same regex react_prompt.py uses) so the ONLY final-answer directive is ours.
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


# System prompt: describe the ONE tool in prose (the actual callable schema is
# injected separately by apply_chat_template(tools=...) as a real `<tools>`
# block, so we do NOT hand-write tool signatures here) + the final-answer
# contract. Deliberately NOT a plain-text-tag one-shot (that is react_prompt.py's
# job) -- Qwen3 already knows the `<tool_call>` grammar from its own template.
NATIVE_SYSTEM_PROMPT = """You are a careful problem solver. You have access to a code_interpreter tool that runs Python in an isolated sandbox (sympy, numpy, scipy, and the standard library available; no network, no filesystem).

Each code_interpreter call is a FRESH, ISOLATED Python process: variables, imports, and function definitions do NOT persist between calls. If a later step needs a value from an earlier one, recompute it or print it and reuse the number. Each snippet must print() everything you need to see -- nothing is returned except stdout.

Use the tool to verify any non-trivial calculation before you commit to it -- do not rely on mental arithmetic or algebra alone. You may call it as many times as you need, across as many turns as you need. When you are certain, give your final answer inside <answer> and </answer> tags, e.g. <answer>42</answer>. Put ONLY the final answer inside the tags."""

# force_tool variant: the first native run showed the model learning to SKIP the
# tool (tool-use collapsed, held-out acc dropped). This prompt makes tool use
# MANDATORY -- the model MUST run and show code before answering. Pairs with the
# "tool is necessary" hypothesis (SDPO then can't take the answer-directly
# shortcut). Selected at data-prep time via SDPO_REACT_PROMPT=force_tool.
NATIVE_SYSTEM_PROMPT_FORCE_TOOL = """You are a careful problem solver with access to a code_interpreter tool that runs Python in an isolated sandbox (sympy, numpy, scipy, and the standard library available; no network, no filesystem).

Each code_interpreter call is a FRESH, ISOLATED Python process: variables, imports, and function definitions do NOT persist between calls. If a later step needs a value from an earlier one, recompute it or print it and reuse the number. Each snippet must print() everything you need to see -- nothing is returned except stdout.

MANDATORY WORKFLOW -- you MUST follow this:
1. You MUST use code_interpreter to derive and verify your answer. Do NOT answer from mental math or algebra alone -- an answer given without having run code that produces it is not acceptable.
2. Write code that actually computes the final numeric answer and print()s it, then read the tool output.
3. Only after the tool has printed a result you trust, give your final answer inside <answer> and </answer> tags, e.g. <answer>42</answer>. Put ONLY the final answer inside the tags.

You may call the tool as many times as you need across multiple turns. Always run code before answering."""

# "minimal" (Q2, the new default): shared question+format-only prompt, all tools
# exposed, no task-specific workflow. The verbose "default"/"force_tool" variants
# are kept selectable via SDPO_REACT_PROMPT for ablation/back-compat.
_PROMPTS = {
    "minimal": MINIMAL_SYSTEM_PROMPT,
    "default": NATIVE_SYSTEM_PROMPT,
    "force_tool": NATIVE_SYSTEM_PROMPT_FORCE_TOOL,
}


def _system_prompt() -> str:
    """Which system prompt to bake in, chosen by env SDPO_REACT_PROMPT
    (minimal | default | force_tool). Read at data-prep time so a single dataset
    row carries the chosen prompt verbatim -- rollout does no further templating.
    Defaults to the minimal question+format-only prompt (Q2)."""
    import os

    return _PROMPTS.get(os.environ.get("SDPO_REACT_PROMPT", "minimal").strip().lower(), MINIMAL_SYSTEM_PROMPT)


def build_native_messages(question: str) -> list[dict]:
    """Prompt as a message list (system + user). Written to the dataset's
    `prompt` field; rendered once by `--apply-chat-template --tool-key tools`,
    which also injects the `<tools>` schema block from the row's `tools` field.
    System prompt variant is chosen by SDPO_REACT_PROMPT (see _system_prompt)."""
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": question},
    ]


def _extract_question(prompt) -> str:
    if isinstance(prompt, str):
        return _strip_dapo_wrapper(prompt)
    if isinstance(prompt, list):
        for message in prompt:
            if message.get("role") == "user":
                return _strip_dapo_wrapper(message["content"])
    raise ValueError(f"Unrecognized prompt format: {prompt!r}")


def build_native_dataset(in_path: str, out_path: str) -> int:
    n = 0
    with open(in_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = _extract_question(row["prompt"])
            row["prompt"] = build_native_messages(question)
            # The `tools` field --tool-key points at; apply_chat_template reads
            # it to inject the model-native <tools> block. Same spec list the
            # rollout parser (--generate-tool-specs-path) uses, so declared and
            # parsed tools can never drift.
            row["tools"] = tool_specs
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args()

    n = build_native_dataset(args.in_path, args.out_path)
    print(f"Wrote {n} native-tool-calling rows -> {args.out_path}")


if __name__ == "__main__":
    main()
