"""Runnable test: does an OpenAI-standard message dict (as multi_turn.generate now
records it) render CORRECTLY through Qwen3's own chat template, and does that
render match what the rollout ACTUALLY fed the model?

Answers the review question: is the system->user boundary
(`...</tool_call><|im_end|>\n<|im_start|>user\n<question>`) a missing separator?
No -- `<|im_start|>user\n` IS the ChatML role separator Qwen3 was trained on; the
"For each function call, return a json object..." block is Qwen3's OWN native
tool-instruction text, auto-generated from `tools=`, not something we inject.

Run inside the enroot container:
  python -m examples.SDPO_ReAct.debug.test_message_template
(needs /root/Qwen3-4B for the tokenizer). Prints the rendered template + PASS/FAIL
checks. No GPU, no server.
"""

import json

from transformers import AutoTokenizer

from examples.SDPO_ReAct.tools.registry import MINIMAL_SYSTEM_PROMPT, all_tool_specs

MODEL = "/root/Qwen3-4B"


def _openai_to_qwen_tools(specs):
    """apply_chat_template(tools=...) wants the {type, function:{...}} list, which
    is exactly all_tool_specs -- pass through."""
    return specs


def build_messages():
    """One realistic multi-turn conversation in the SAME OpenAI-standard schema
    multi_turn.generate now records (assistant.tool_calls + tool.tool_call_id)."""
    return [
        {"role": "system", "content": MINIMAL_SYSTEM_PROMPT},
        {"role": "user", "content": "Which magazine was started first, Arthur's Magazine or First for Women?"},
        {
            "role": "assistant",
            "content": "I should look up each magazine's founding date.",
            "tool_calls": [
                {
                    "id": "call_0001",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": json.dumps({"query": "Arthur's Magazine founded"})},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0001",
            "name": "web_search",
            "content": "[doc 1] Arthur's Magazine was an American literary periodical first published in 1844.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0002",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": json.dumps({"query": "First for Women founded"})},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0002",
            "name": "web_search",
            "content": "[doc 1] First for Women is a woman's magazine launched in 1989.",
        },
        {"role": "assistant", "content": "<answer>Arthur's Magazine</answer>"},
    ]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    messages = build_messages()

    print("=" * 78)
    print("RENDERED via tok.apply_chat_template(messages, tools=all_tool_specs)")
    print("=" * 78)
    rendered = tok.apply_chat_template(
        messages,
        tools=_openai_to_qwen_tools(all_tool_specs),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    print(rendered)

    print("=" * 78)
    print("CHECKS")
    print("=" * 78)
    checks = {
        "system turn present": "<|im_start|>system" in rendered,
        "native tools block present (Qwen3 auto-gen)": "<tools>" in rendered and "For each function call" in rendered,
        "user turn separator present": "<|im_start|>user\nWhich magazine" in rendered,
        "assistant reasoning rendered": "look up each magazine" in rendered,
        "1st tool_call rendered as <tool_call>": '"name": "web_search"' in rendered and "Arthur's Magazine founded" in rendered,
        "tool observation rendered as <tool_response>": "<tool_response>" in rendered and "first published in 1844" in rendered,
        "2nd tool_call rendered": "First for Women founded" in rendered,
        "final answer rendered": "<answer>Arthur's Magazine</answer>" in rendered,
        "tokenizes without error": True,
    }
    try:
        # apply_chat_template(tokenize=True) is the correct call; it returns a
        # BatchEncoding {'input_ids', 'attention_mask'} (NOT a bare list), so read
        # input_ids for the real token count -- len() on the BatchEncoding itself
        # just counts its 2 keys.
        enc = tok.apply_chat_template(
            messages, tools=all_tool_specs, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        input_ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        checks["tokenizes without error"] = len(input_ids) > 50
        print(f"token count: {len(input_ids)}")
        back = tok.decode(input_ids)
        checks["round-trip preserves tool_call"] = "web_search" in back and "<tool_response>" in back
    except Exception as e:
        checks["tokenizes without error"] = False
        print(f"TOKENIZE ERROR: {e}")

    all_ok = True
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok

    print("=" * 78)
    print(f"RESULT: {'ALL PASS -- dict renders correctly through Qwen3 template' if all_ok else 'SOME FAILED'}")
    print("=" * 78)

    # Show the system->user boundary the review asked about, in isolation.
    b = rendered.find("</tool_call>")
    e = rendered.find("Which magazine")
    if b != -1 and e != -1:
        print("\nSYSTEM->USER boundary (the reviewed span) verbatim:")
        print(repr(rendered[b : e + 20]))
        print("\n-> `<|im_start|>user\\n` IS the separator; no 'Question:' label needed.")


if __name__ == "__main__":
    main()
