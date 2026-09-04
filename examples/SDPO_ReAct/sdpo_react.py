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

from examples.SDPO.reward import (
    _grade_one_alfworld,
    _grade_one_code,
    _grade_one_search,
    _grade_one_tau2,
    _grade_one_webshop,
    _is_correct,
    _llm_judge_correct,
    _sample_domain,
)
from examples.SDPO.sdpo import _BLIND_PREDICT_SYSTEM
from examples.SDPO.sdpo import _PITFALL_PREDICT_SYSTEM
from examples.SDPO.sdpo import EVAL_SKILL_CORRECT_TEMPLATE
from examples.SDPO.sdpo import EVAL_SKILL_INSTRUCTION
from examples.SDPO.sdpo import EVAL_SKILL_PITFALL_TEMPLATE
from examples.SDPO.sdpo import _blind_predict_user_prompt
from examples.SDPO.sdpo import _gen_prompt_suffix
from examples.SDPO.sdpo import _generate_skill_text
from examples.SDPO.sdpo import _pitfall_predict_user_prompt
from examples.SDPO.sdpo import _tokenizer
from examples.SDPO.sdpo import sdpo_eval_reward as _sdpo_eval_reward
from examples.SDPO.sdpo import sdpo_group_reward as _sdpo_group_reward
from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.generate_hub.multi_turn import generate as _multi_turn_generate
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
    """(tool_call, observation) pairs for this trajectory. Two rollout paths
    feed this wrapper, so we normalize BOTH to the {"tool_call", "observation"}
    schema examples/SDPO/sdpo.py::_render_env_feedback consumes:

    - NATIVE tool-calling (miles.rollout.generate_hub.multi_turn.generate, used
      by the Qwen3 launcher): that loop already records the ground-truth
      call/observation pairs live in sample.metadata["tool_trace"] (see the
      tool_trace bookkeeping added there). Prefer it verbatim.
    - LEGACY plain-text tags (generate_with_tools.generate, used by the
      Qwen2.5 launcher): reconstruct from sample.metadata["turns"] (the loop's
      own {role, action, content} record) -- NOT by regex-matching <code>/
      <output> tags out of the decoded text, which is unsound (the "invalid
      tag" nudge is English prose that itself CONTAINS those tag substrings, so
      re.findall matches inside it; observed live as role=tool content="and").

    Both keys are always present in the returned dicts, so _count_tool_errors /
    _render_env_feedback / the trace dump never need to know which path ran."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    native = md.get("tool_trace")
    if native:
        # multi_turn.generate already emits the canonical schema.
        return [
            {"tool_call": t.get("tool_call", ""), "observation": t.get("observation", "")}
            for t in native
            if isinstance(t, dict)
        ]
    turns = md.get("turns")
    if not turns:
        return []
    pairs = []
    for i, turn in enumerate(turns):
        if turn.get("role") == "assistant" and turn.get("action") == "code":
            output = turns[i + 1]["content"] if i + 1 < len(turns) and turns[i + 1].get("role") == "tool" else ""
            pairs.append({"tool_call": turn.get("content", ""), "observation": output})
    return pairs


def _count_tool_errors(tool_trace: list[dict[str, str]]) -> int:
    return sum(1 for t in tool_trace if t["observation"].strip().lower().startswith(_TOOL_ERROR_PREFIXES))


def _prompt_to_messages(prompt: str) -> list[dict[str, Any]]:
    """Split the chat-templated prompt string (the rollout's baked-in
    system+user turns, e.g. "<|im_start|>system\\n...<|im_end|><|im_start|>user
    \\n...<|im_end|>") back into clean {role, content} messages. The <tools>
    block lives inside the system turn (native tool injection) and is kept there
    verbatim. Falls back to a single user message if no ChatML markers."""
    if not isinstance(prompt, str) or "<|im_start|>" not in prompt:
        return [{"role": "user", "content": prompt or ""}]
    import re
    msgs: list[dict[str, Any]] = []
    for m in re.finditer(r"<\|im_start\|>(\w+)\s*\n(.*?)(?:<\|im_end\|>|$)", prompt, re.DOTALL):
        role, content = m.group(1), m.group(2).strip()
        # Drop the trailing assistant GENERATION PROMPT (no closing <|im_end|>):
        # it's the empty scaffold the model continues from (e.g. "" or an empty
        # "<think>\n\n</think>" in no-thinking mode), not a real turn -- the live
        # messages carry the actual assistant content.
        if role == "assistant" and re.sub(r"</?think>", "", content).strip() == "":
            continue
        msgs.append({"role": role, "content": content})
    return msgs or [{"role": "user", "content": prompt}]


def _reconstruct_messages(sample: Sample) -> list[dict[str, Any]]:
    """OpenAI-standard message-dict trace for post-hoc inspection: the baked-in
    prompt split into system+user turns, then the LIVE conversation the
    generation loop recorded (assistant turns with a structured `tool_calls`
    field, tool turns with `tool_call_id`) -- the same schema as standard
    tool-calling trajectory logs (e.g. i-DeepSearch observation-masking), so it
    can be re-sent through any chat template / OpenAI client without re-parsing.

    Correct by construction from metadata["messages"] (native multi_turn.generate)
    or metadata["turns"] (legacy generate_with_tools); falls back to the raw
    response as one assistant turn only when no live record exists (e.g.
    truncated before the loop recorded anything)."""
    prompt = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    head = _prompt_to_messages(prompt)

    # Native path: multi_turn.generate stores the running conversation directly
    # (already OpenAI-standard: assistant.tool_calls + tool.tool_call_id).
    live = md.get("messages")
    if live:
        return [*head, *live]

    # Legacy path: generate_with_tools records {role, content} turns.
    turns = md.get("turns")
    if turns:
        msgs = list(head)
        for turn in turns:
            role = "tool" if turn.get("role") == "tool" else "assistant"
            msgs.append({"role": role, "content": turn.get("content", "")})
        return msgs

    return [*head, {"role": "assistant", "content": sample.response or ""}]


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
        records = [_sample_to_agentic_trace_record(sample) for sample in group]
        _append_agentic_trace_records(dump_dir, rollout_id, records)
    except Exception as e:  # dumping must never break rollout
        logger.warning(f"SDPO_ReAct agentic trace dump failed (non-fatal): {e!r}")


def _sample_to_agentic_trace_record(sample: Sample) -> dict[str, Any]:
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    # tau2 samples come from agentic_tool_call.generate (an external
    # Orchestrator drives the whole conversation, see tools/tau2/docker/
    # server.py's module docstring), not multi_turn.generate -- there is no
    # sample.prompt/metadata["messages"] to reconstruct FROM the way
    # _reconstruct_messages expects; the tau2 sidecar's own message list
    # (agent_function.py's metadata["tau2_messages"], already {role,
    # content, tool_calls}-shaped) is the correct-by-construction record.
    messages = md.get("tau2_messages") if md.get("domain") == "tau2" else None
    if messages is None:
        messages = _reconstruct_messages(sample)
    return {
        "messages": messages,
        "label": sample.label,
        "tool_call_count": md.get("tool_call_count"),
        "tool_error_count": md.get("tool_error_count"),
        "sdpo_correct": md.get("sdpo_correct"),
        "status": sample.status.value if sample.status is not None else None,
        # domain/episode_won/task_type/game_file: let examples/agentic/'s
        # dashboard tell webshop from alfworld episodes and break down success
        # by task type without re-deriving anything from `messages`.
        "domain": md.get("domain"),
        "episode_won": md.get("episode_won"),
        "task_type": md.get("task_type"),
        "alfworld_game_file": md.get("alfworld_game_file"),
        "webshop_task_id": md.get("webshop_task_id"),
        "tau2_domain": md.get("tau2_domain"),
        "tau2_termination_reason": md.get("tau2_termination_reason"),
        "reward": md.get("reward"),
    }


def _append_agentic_trace_records(dump_dir: str, rollout_id: Any, records: list[dict[str, Any]]) -> None:
    path = Path(dump_dir) / "agentic_traces" / f"{rollout_id if rollout_id is not None else 'unknown'}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _dump_agentic_trace_one(args: Namespace, sample: Sample) -> None:
    """Single-sample counterpart to ``_dump_agentic_traces``, for reward
    paths that never see the full group (e.g. arm 1's plain-GRPO baseline,
    --custom-rm-path without --group-rm -- see sdpo_react_plain_grpo_reward).
    Appends ONE record to the same --dump-details/agentic_traces/
    {rollout_id}.jsonl file the group path writes, so arm-1 baseline runs are
    inspectable in the same dashboard/tooling as every other arm (previously
    a real gap: arm 1 produced NO agentic_traces dump at all)."""
    dump_dir = getattr(args, "dump_details", None)
    if dump_dir is None:
        return
    try:
        md = sample.metadata if isinstance(sample.metadata, dict) else {}
        rollout_id = md.get("rollout_id")
        _append_agentic_trace_records(dump_dir, rollout_id, [_sample_to_agentic_trace_record(sample)])
    except Exception as e:  # dumping must never break rollout
        logger.warning(f"SDPO_ReAct agentic trace dump failed (non-fatal): {e!r}")


async def sdpo_react_group_reward(args: Namespace, group: list[Sample], **kwargs: Any) -> list[float]:
    for sample in group:
        if not isinstance(sample.metadata, dict):
            continue
        # Normalize the trajectory's tool calls to the canonical
        # {"tool_call", "observation"} schema and stamp it back on metadata so
        # examples/SDPO/sdpo.py's env_feedback skill path (_has_env_feedback /
        # _render_env_feedback) works uniformly regardless of which rollout
        # path produced the trace: multi_turn.generate already writes this
        # schema (so this is idempotent there), while the legacy plain-text
        # generate_with_tools.generate only records metadata["turns"] (so this
        # is the ONE place it gets converted). tool_error_count needs the
        # paired observation text, computed from the same normalized trace.
        tool_trace = _extract_tool_trace(sample)
        sample.metadata["tool_trace"] = tool_trace
        sample.metadata["tool_error_count"] = _count_tool_errors(tool_trace)

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


async def sdpo_react_plain_grpo_reward(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    """Single-sample reward for the plain-GRPO baseline arm (--custom-rm-path,
    NO --group-rm): the "no SDPO at all" control. examples/SDPO/sdpo.py's own
    plain_grpo_reward always uses the math/dapo grader (_is_correct) -- wrong
    for a multitask (math+code+search) rollout, where code/search samples need
    their own graders (test-case execution / EM-against-golden-answers). This
    domain-routes per sample the same way _grade_group does for the KD arms,
    so arm 1's task-reward criterion is the SAME grader every other arm's
    sdpo_react_group_reward uses -- the ablation isolates the SDPO/skill
    machinery, not a grading-rule difference."""
    if not isinstance(sample.metadata, dict):
        return 1.0 if _is_correct(sample, args) else 0.0
    sample.metadata["tool_trace"] = _extract_tool_trace(sample)
    domain = _sample_domain(sample)
    if domain == "code":
        ok = await _grade_one_code(sample, args)
    elif domain == "search":
        ok = await _grade_one_search(sample, args)
    elif domain == "webshop":
        ok = _grade_one_webshop(sample, args)
    elif domain == "alfworld":
        ok = _grade_one_alfworld(sample, args)
    elif domain == "tau2":
        ok = _grade_one_tau2(sample, args)
    elif sample.metadata.get("amo_use_judge") and getattr(args, "sdpo_judge", False) and (sample.response or "").strip():
        ok = await _llm_judge_correct(args, sample)
    else:
        ok = _is_correct(sample, args)
    # Stamp so the trace dump below (and any other consumer expecting the
    # same key the group-reward path sets, see sdpo.py:1408) reflects the
    # grading result, even though arm 1 has no SDPO group machinery at all.
    sample.metadata["sdpo_correct"] = 1.0 if ok else 0.0
    _dump_agentic_trace_one(args, sample)
    return 1.0 if ok else 0.0


# --------------------------------------------------------------------------- #
# EVAL-time skill augmentation for AGENTIC (tool-calling) domains.
#
# examples/SDPO/sdpo.py::sdpo_eval_generate already implements "self-predict a
# blind skill from the problem alone, splice it into the prompt, then run the
# real eval rollout" (--sdpo-eval-skill-mode) -- but it hardcodes the second
# pass to miles.rollout.sglang_rollout.generate, a SINGLE-TURN generate with
# no tool-calling loop. Wiring it directly to webshop/alfworld eval would
# silently drop every webshop_step/alfworld_step call: the model would emit
# one text completion and stop, never touching the sidecar, so reward would
# never reflect a real episode. This function reuses sdpo.py's skill-gen/
# splice logic VERBATIM (same self-predict prompts, same splice-before-
# gen-suffix point) but dispatches the augmented prompt to
# miles.rollout.generate_hub.multi_turn.generate instead, so the second pass
# still runs the full multi-turn tool-calling loop.
# --------------------------------------------------------------------------- #


async def sdpo_react_eval_generate_with_skill(input: Any) -> Any:
    """--custom-generate-function-path for an EVAL-only dataset entry (see
    e.g. eval_agentic.yaml's *_skill datasets) that measures the model
    answering WITH its own self-predicted skill already in context, on top of
    the SAME multi-turn tool-calling loop normal eval uses. No-op during
    TRAINING (evaluation=False) or when --sdpo-eval-skill-mode is 'off' --
    falls straight through to multi_turn.generate. Mirrors
    examples.SDPO.sdpo.sdpo_eval_generate; see that function's docstring for
    the skill-splice mechanics this reuses."""
    args = input.args
    sample = input.sample
    mode = getattr(args, "sdpo_eval_skill_mode", "off")

    if not input.evaluation or mode == "off" or not isinstance(sample.prompt, str):
        return await _multi_turn_generate(input)

    try:
        sections = []
        if mode in ("correct", "all"):
            skill = await _generate_skill_text(
                args, _BLIND_PREDICT_SYSTEM, _blind_predict_user_prompt(sample.prompt), "self"
            )
            if skill.strip():
                sections.append(EVAL_SKILL_CORRECT_TEMPLATE.format(skill=skill.strip()))
        if mode in ("pitfall", "all"):
            skill = await _generate_skill_text(
                args, _PITFALL_PREDICT_SYSTEM, _pitfall_predict_user_prompt(sample.prompt), "self"
            )
            if skill.strip():
                sections.append(EVAL_SKILL_PITFALL_TEMPLATE.format(skill=skill.strip()))

        if sections:
            tok = _tokenizer(args)
            gen_suffix = _gen_prompt_suffix(tok, getattr(args, "apply_chat_template_kwargs", None))
            section = "".join(sections) + EVAL_SKILL_INSTRUCTION
            if gen_suffix and gen_suffix in sample.prompt:
                idx = sample.prompt.rfind(gen_suffix)
                sample.prompt = sample.prompt[:idx] + section + sample.prompt[idx:]
            else:
                sample.prompt = sample.prompt + section
    except Exception as e:
        logger.warning(f"eval skill augmentation failed ({e!r}); evaluating on the unaugmented prompt.")

    output = await _multi_turn_generate(input)
    return output if isinstance(output, GenerateFnOutput) else GenerateFnOutput(samples=output)
