"""SDPO group-RM wrapper for multi-turn tool-calling rollouts (SDPO_ReAct).

Design: SDPO's own reward/prefix machinery (grading, correct-peer prefix
selection, KD loss, EMA teacher, ...) lives in ``examples.SDPO.sdpo`` and
operates purely on ``Sample.response`` (full decoded text) / ``Sample.tokens``
(with the response as the tail span) / ``Sample.metadata``. None of that
assumes single-turn rollout, so a multi-turn trajectory produced by
``generate_with_tools.generate`` (see that module -- one Sample per
trajectory, tool turns loss_mask=0) flows through it completely unchanged.

This module does NOT fork or reimplement any of that logic -- it only adds
the tool-call bookkeeping SDPO_ReAct needs on top (a message-dict trace dump
for post-hoc inspection) and then delegates to ``sdpo_group_reward`` /
``sdpo_eval_reward`` verbatim. Base version has NO skill/skill-KD wiring --
that machinery is orthogonal to getting tool-calling itself working and is
deliberately left off here (--sdpo-self-skill is not set in the launcher).

Wiring (mirrors examples/EPO/epo.py's wiring table):
    --group-rm
    --custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_group_reward
    --eval-custom-rm-path examples.SDPO_ReAct.sdpo_react.sdpo_react_eval_reward
    --custom-generate-function-path examples.SDPO_ReAct.generate_with_tools.generate
    --sdpo-grader dapo
    --sdpo-teacher-backend megatron

Note: no MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 needed -- generate_with_tools.
generate uses the legacy 3-arg generate(args, sample, sampling_params) style
(see that module's docstring for why), so this runs on the plain
miles/rollout/sglang_rollout.py rollout path.
"""

import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

from examples.SDPO.sdpo import sdpo_eval_reward as _sdpo_eval_reward
from examples.SDPO.sdpo import sdpo_group_reward as _sdpo_group_reward
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# tool_client.py's error surface (see tools/tool_client.py::_format_result /
# execute_tool's NotImplementedError branch): every failure mode -- sandbox
# unreachable, code raised, timeout, unknown tool -- renders the observation
# text starting with one of these prefixes. Used only for the post-hoc
# tool_error_count diagnostic (miles.ray.rollout.metrics.py's agentic/
# tool_error_rate panel); it never affects training (loss_mask already zeros
# tool-observation tokens regardless of error/success).
_TOOL_ERROR_PREFIXES = ("error:", "[timeout]")


def _extract_tool_trace(sample: Sample) -> list[dict[str, str]]:
    """(code, output) pairs, straight from generate_with_tools.generate's own
    turn-by-turn bookkeeping (sample.metadata["turns"]) -- NOT re-derived by
    regex-matching <code>/<output> tags out of the final decoded response
    text. Regex reconstruction is unsound here: the "invalid tag" retry nudge
    (see generate_with_tools.py::execute_predictions) is plain English that
    itself CONTAINS the literal substrings <code>/</code>/<answer>/</answer>
    as instructional prose, so a naive re.findall over the full text matches
    inside the nudge too, corrupting the reconstruction (observed live:
    role=tool entries with content="and" from splitting the nudge sentence
    mid-word). The generation loop already knows, unambiguously, which turns
    were real tool calls -- use that."""
    turns = sample.metadata.get("turns") if isinstance(sample.metadata, dict) else None
    if not turns:
        return []
    pairs = []
    for i, turn in enumerate(turns):
        if turn.get("role") == "assistant" and turn.get("action") == "code":
            output = turns[i + 1]["content"] if i + 1 < len(turns) and turns[i + 1].get("role") == "tool" else ""
            pairs.append({"code": turn.get("content", ""), "output": output})
    return pairs


def _count_tool_errors(tool_trace: list[dict[str, str]]) -> int:
    return sum(1 for t in tool_trace if t["output"].strip().lower().startswith(_TOOL_ERROR_PREFIXES))


def _reconstruct_messages(sample: Sample) -> list[dict[str, Any]]:
    """Message-dict trace for post-hoc inspection, built directly from
    sample.metadata["turns"] (the generation loop's own live record of what
    was model output vs. tool output) rather than re-parsed from text -- see
    _extract_tool_trace's docstring for why regex reconstruction is unsound
    here."""
    problem = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
    messages: list[dict[str, Any]] = [{"role": "user", "content": problem}]

    turns = sample.metadata.get("turns") if isinstance(sample.metadata, dict) else None
    if not turns:
        # No metadata (e.g. truncated/aborted before the loop recorded
        # anything) -- fall back to the raw decoded response as a single turn.
        messages.append({"role": "assistant", "content": sample.response or ""})
        return messages

    for turn in turns:
        if turn.get("role") == "tool":
            messages.append({"role": "tool", "content": turn.get("content", "")})
        else:
            messages.append({"role": "assistant", "content": turn.get("content", "")})

    return messages


def _dump_agentic_traces(args: Namespace, group: list[Sample]) -> None:
    """Dump a message-dict trace per sample to
    --dump-details/agentic_traces/{rollout_id}.jsonl for post-hoc inspection
    -- what did the model reason, what code did it run, what came back, what
    was the final answer, was it graded correct.

    Filename is the REAL rollout_id (see generate_with_tools.py, which reads
    it off the GenerateState singleton that sglang_rollout.py's
    generate_rollout_async/eval_rollout stamp) -- one file per training/eval
    step, matching rollout_data/{rollout_id}.jsonl's own numbering, so the two
    dumps line up 1:1 and can be cross-referenced by filename. This function
    is called once per GROUP (n_samples_per_prompt traces sharing one
    prompt), and a rollout step has rollout_batch_size such groups all
    resolving concurrently on the same asyncio event loop -- hence append
    mode, not overwrite; the write itself never awaits mid-write, so
    concurrent groups for the same rollout_id cannot interleave a partial
    line onto each other.

    Mirrors MegatronTrainRayActor._dump_sdpo_prompts's own dump conventions
    (same --dump-details root, same non-fatal try/except so a dump bug never
    breaks rollout), but at the ROLLOUT side (this runs inside the group RM,
    which already has the full group and sample.metadata) rather than the
    training side, and in message-dict form rather than decoded-text form --
    the two dumps are complementary, not a replacement for each other.
    """
    dump_dir = getattr(args, "dump_details", None)
    if dump_dir is None:
        return
    try:
        rollout_id = None
        for sample in group:
            if isinstance(sample.metadata, dict) and sample.metadata.get("rollout_id") is not None:
                rollout_id = sample.metadata["rollout_id"]
                break
        records = []
        for sample in group:
            md = sample.metadata if isinstance(sample.metadata, dict) else {}
            records.append(
                {
                    "messages": _reconstruct_messages(sample),
                    "label": sample.label,
                    "tool_call_count": md.get("tool_call_count"),
                    "tool_error_count": md.get("tool_error_count"),
                    "sdpo_correct": md.get("sdpo_correct"),
                    "status": sample.status.value if sample.status is not None else None,
                }
            )
        path = Path(dump_dir) / "agentic_traces" / f"{rollout_id if rollout_id is not None else 'unknown'}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except Exception as e:  # dumping must never break rollout
        logger.warning(f"SDPO_ReAct agentic trace dump failed (non-fatal): {e!r}")


async def sdpo_react_group_reward(args: Namespace, group: list[Sample], **kwargs: Any) -> list[float]:
    for sample in group:
        if not isinstance(sample.metadata, dict):
            continue
        # generate_with_tools.generate already stamps round_number /
        # tool_call_count directly from the tag matches (source of truth);
        # this only adds tool_error_count, which needs the paired output text.
        sample.metadata["tool_error_count"] = _count_tool_errors(_extract_tool_trace(sample))

    rewards = await _sdpo_group_reward(args, group, **kwargs)
    # sdpo_correct is stamped by _sdpo_group_reward above; dump AFTER it runs
    # so the trace records include the grading result.
    _dump_agentic_traces(args, group)
    return rewards


async def sdpo_react_eval_reward(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    """Per-sample eval RM (--eval-custom-rm-path). Grading is identical to
    SDPO's (pass@1 never touches the prefix/distillation machinery), so we
    delegate directly -- same pattern as examples/EPO/epo.py::epo_eval_reward."""
    return await _sdpo_eval_reward(args, sample, **kwargs)
