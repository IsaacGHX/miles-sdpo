"""CPU-only tests for examples/SDPO_ReAct.

Covers, in the order the user asked to verify things in:
  1. template / tool-call round-trip -- the react_prompt one-shot renders as
     genuine chat-template turns, and code_interpreter's spec + a canned
     <tool_call> completion round-trip through SGLang's qwen25 FunctionCallParser
     and tool_call_utils' append-only-prefix tokenization without raising.
  2. sdpo_react's tool-call bookkeeping (_count_tool_calls / _extract_tool_trace),
     which examples/SDPO/sdpo.py's env_feedback skill source depends on.

No GPU, no Docker, no network beyond the one-time HF tokenizer download shared
with tests/fast/rollout/generate_hub/test_tool_call_utils.py.
"""

import json

import pytest
from tests.ci.ci_register import register_cpu_ci

from examples.SDPO_ReAct.react_prompt import REACT_SYSTEM_PROMPT, build_react_messages
from examples.SDPO_ReAct.sdpo_react import (
    _count_tool_calls,
    _count_tool_errors,
    _extract_tool_trace,
    _reconstruct_messages,
)
from examples.SDPO_ReAct.tools.tool_specs import CODE_INTERPRETER_SPEC, tool_specs
from miles.rollout.generate_utils.tool_call_utils import create_tool_call_parser
from miles.utils.types import Sample

register_cpu_ci(est_time=30, suite="stage-b-cpu", labels=[])

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # qwen25 tool-call parser's own reference family


class TestReactPrompt:
    def test_one_shot_is_real_chat_messages_not_string_tags(self):
        """The one-shot example must be genuine role-tagged messages (a real
        `tool_calls` field, a real `tool` role) -- never a hand-written
        <tool_call> string spliced into content -- so it always renders through
        whatever chat template the installed tokenizer defines."""
        messages = build_react_messages("What is 2 + 2?")

        assert messages[0] == {"role": "system", "content": REACT_SYSTEM_PROMPT}
        roles = [m["role"] for m in messages]
        # Two worked examples (single-line code, then multi-statement code --
        # the latter guards against the model emitting unescaped multi-line
        # Python as the JSON `code` argument), then the real question.
        assert roles == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "user",
            "assistant",
            "tool",
            "assistant",
            "user",
        ]

        tool_call_message = messages[2]
        assert "tool_calls" in tool_call_message
        assert "<tool_call>" not in tool_call_message["content"]  # no hand-written tag string
        call = tool_call_message["tool_calls"][0]
        assert call["function"]["name"] == "code_interpreter"
        json.loads(call["function"]["arguments"])  # must be valid JSON, matching the real spec's contract

        tool_response_message = messages[3]
        assert tool_response_message["role"] == "tool"
        assert tool_response_message["tool_call_id"] == call["id"]

        assert messages[-1] == {"role": "user", "content": "What is 2 + 2?"}

    def test_code_interpreter_spec_matches_arguments_the_one_shot_sends(self):
        """The one-shot's tool_calls arguments must satisfy the spec's own
        required-parameters contract, else the worked example would teach the
        model an invalid call shape."""
        [spec] = tool_specs
        assert spec == CODE_INTERPRETER_SPEC
        required = spec["function"]["parameters"]["required"]

        one_shot_call = build_react_messages("x")[2]["tool_calls"][0]
        args = json.loads(one_shot_call["function"]["arguments"])
        for key in required:
            assert key in args


class TestToolCallRoundTrip:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        from miles.utils.processing_utils import load_tokenizer

        return load_tokenizer(MODEL_NAME, trust_remote_code=True)

    def test_tool_specs_render_via_apply_chat_template(self, tokenizer):
        """Tool defs must be injected via the model's OWN template (tools=...),
        never a hand-written XML/tag block -- this is what lets tool_specs.py
        stay valid if the model is swapped for a different --tito-model."""
        messages = [{"role": "user", "content": "compute something"}]
        rendered = tokenizer.apply_chat_template(
            messages, tools=tool_specs, tokenize=False, add_generation_prompt=True
        )
        assert "code_interpreter" in rendered

    def test_canned_tool_call_parses_via_qwen25_parser(self, tokenizer):
        """A model completion containing a code_interpreter call must parse via
        SGLang's native qwen25 FunctionCallParser -- the mechanism
        --generate-tool-call-parser qwen25 relies on during real rollout."""
        parser = create_tool_call_parser(tool_specs, "qwen25")
        completion = (
            "Let me verify with code.\n"
            "<tool_call>\n"
            '{"name": "code_interpreter", "arguments": {"code": "print(2 + 2)"}}\n'
            "</tool_call>"
        )
        _normal_text, tool_calls = parser.parse_non_stream(completion)

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "code_interpreter"
        parsed_args = json.loads(tool_calls[0].parameters)
        assert parsed_args == {"code": "print(2 + 2)"}

    def test_tool_response_tokenization_is_append_only(self, tokenizer):
        """tool_call_utils._tokenize_postfix_messages asserts the "with tool
        response" token prefix matches the "without" prefix -- this is the
        TITO/append-only invariant docs/user-guide/agentic-chat-template.md
        documents. Must not raise for our tool's response shape."""
        from miles.rollout.generate_utils.tool_call_utils import tokenize_tool_responses

        tool_messages = [{"role": "tool", "tool_call_id": "call_1", "content": "4", "name": "code_interpreter"}]
        token_ids = tokenize_tool_responses(tool_messages, tokenizer=tokenizer)
        assert len(token_ids) > 0


class TestToolCallBookkeeping:
    def test_counts_tool_calls_in_response(self):
        response = (
            "thinking\n"
            '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(1)"}}\n</tool_call>\n'
            "saw 1\n"
            '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(2)"}}\n</tool_call>\n'
            "saw 2. \\boxed{2}"
        )
        sample = Sample(response=response)
        assert _count_tool_calls(sample) == 2

    def test_zero_tool_calls_for_direct_answer(self):
        sample = Sample(response="No tool needed. \\boxed{42}")
        assert _count_tool_calls(sample) == 0

    def test_extracts_tool_call_and_observation_pairs(self):
        # Realistic shape: the observation is wrapped in <tool_response> (as
        # tokenize_tool_responses actually renders it, see test_snapshot in
        # tests/fast/rollout/generate_hub/test_tool_call_utils.py), with chat-
        # template role-turn scaffolding around it -- NOT bare text directly
        # after </tool_call>. A single-tool-call trace with unwrapped text
        # would swallow the model's whole final answer into "observation";
        # this shape is what guards against that regression.
        response = (
            '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(2+2)"}}\n</tool_call>'
            "<|im_end|>\n<|im_start|>user\n<tool_response>\nresult was 4\n</tool_response>"
            "<|im_end|>\n<|im_start|>assistant\n"
            'Now the second one.\n<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(1)"}}\n</tool_call>'
            "<|im_end|>\n<|im_start|>user\n<tool_response>\n1\n</tool_response>"
            "<|im_end|>\n<|im_start|>assistant\n"
            "done. \\boxed{4}"
        )
        sample = Sample(response=response)
        trace = _extract_tool_trace(sample)

        assert len(trace) == 2
        assert "print(2+2)" in trace[0]["tool_call"]
        assert trace[0]["observation"] == "result was 4"
        assert "print(1)" in trace[1]["tool_call"]
        assert trace[1]["observation"] == "1"

    def test_empty_trace_for_no_tool_calls(self):
        sample = Sample(response="direct answer, no tools. \\boxed{1}")
        assert _extract_tool_trace(sample) == []

    def test_counts_errors_from_observation_prefix(self):
        response = (
            '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "1/0"}}\n</tool_call>'
            "<|im_end|>\n<|im_start|>user\n<tool_response>\nerror:\nZeroDivisionError\n</tool_response>"
            "<|im_end|>\n<|im_start|>assistant\n\\boxed{err}"
        )
        sample = Sample(response=response)
        trace = _extract_tool_trace(sample)
        assert _count_tool_errors(trace) == 1

    def test_no_errors_for_successful_observation(self):
        response = (
            '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(4)"}}\n</tool_call>'
            "<|im_end|>\n<|im_start|>user\n<tool_response>\n4\n</tool_response>"
            "<|im_end|>\n<|im_start|>assistant\n\\boxed{4}"
        )
        sample = Sample(response=response)
        trace = _extract_tool_trace(sample)
        assert _count_tool_errors(trace) == 0


class TestReconstructMessages:
    def test_single_tool_call_final_answer_is_not_swallowed(self):
        """Regression test: a naive "everything after <tool_call> up to the
        next one, or end-of-string" split would fold the model's ENTIRE final
        answer into the tool observation for a single-tool-call trace, and no
        final assistant turn would ever appear. Must reconstruct a real
        trailing assistant turn with the actual answer text."""
        response = (
            "Let me compute.\n"
            '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(2+2)"}}\n</tool_call>'
            "<|im_end|>\n<|im_start|>user\n<tool_response>\nresult was 4\n</tool_response>"
            "<|im_end|>\n<|im_start|>assistant\n"
            "The result is 4.\n\\boxed{4}"
        )
        sample = Sample(prompt="What is 2+2?", response=response)
        messages = _reconstruct_messages(sample)

        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
        assert messages[2]["content"] == "result was 4"
        assert messages[-1]["content"] == "The result is 4.\n\\boxed{4}"
        # No leftover chat-template control tokens in the human-readable dump.
        assert "<|im_end|>" not in messages[-1]["content"]
        assert "<|im_start|>" not in messages[-1]["content"]

    def test_zero_tool_calls_produces_single_assistant_turn(self):
        sample = Sample(prompt="2+2?", response="Just 4. \\boxed{4}")
        messages = _reconstruct_messages(sample)
        assert messages == [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "Just 4. \\boxed{4}"},
        ]

    def test_multi_turn_tool_calls_alternate_correctly(self):
        response = (
            "Step 1.\n"
            '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(1)"}}\n</tool_call>'
            "<|im_end|>\n<|im_start|>user\n<tool_response>\n1\n</tool_response>"
            "<|im_end|>\n<|im_start|>assistant\n"
            "Step 2.\n"
            '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(2)"}}\n</tool_call>'
            "<|im_end|>\n<|im_start|>user\n<tool_response>\n2\n</tool_response>"
            "<|im_end|>\n<|im_start|>assistant\n"
            "Done. \\boxed{3}"
        )
        sample = Sample(prompt="sum 1 and 2", response=response)
        messages = _reconstruct_messages(sample)

        assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
        assert messages[1]["content"] == "Step 1."
        assert messages[2]["content"] == "1"
        assert messages[3]["content"] == "Step 2."
        assert messages[4]["content"] == "2"
        assert messages[5]["content"] == "Done. \\boxed{3}"
