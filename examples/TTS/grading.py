"""Grading for the TTS harness -- reuses examples/SDPO/reward.py verbatim.

We do NOT reimplement answer extraction / math equivalence / MCQ-letter
matching: examples/SDPO/reward.py::_is_correct already handles all of them
(DAPO boxed-answer math via --sdpo-grader dapo, single-letter MCQ match, and
open-ended normalized/sympy math grading) and is a PURE function of
(Sample, args) -- no engine, no network, no training state. So we build a
throwaway ``Sample`` and a tiny ``args`` namespace and call straight into it.
This guarantees the TTS scaffold is graded by the EXACT same criterion SDPO
trains against, so numbers are comparable across the two examples.

``_is_correct`` reads only two args attributes: ``sdpo_grader`` ("dapo" for
math, anything else -> MCQ/open-ended path) and ``sdpo_answer_tag`` (default
"answer"). Everything else on the namespace is irrelevant to it.
"""

from __future__ import annotations

from argparse import Namespace

from examples.SDPO.reward import _grade_one_code, _is_correct
from miles.utils.types import Sample


def make_grader_args(grader: str = "dapo", answer_tag: str = "answer") -> Namespace:
    """The minimal args namespace ``_is_correct`` needs. ``grader='dapo'`` for
    math (boxed / numeric), ``grader='mcq'`` for letter-answer science MCQ."""
    return Namespace(sdpo_grader=grader, sdpo_answer_tag=answer_tag)


def grade(response: str, label: str, args: Namespace) -> bool:
    """Is ``response`` a correct answer for ``label`` under ``args``' grader?

    The solver's raw text goes in ``response``; SDPO's extractor scans it for
    the last <answer>...</answer> tag (or a \\boxed{}), so the solver just needs
    to follow the answer-format contract in its system prompt."""
    sample = Sample(response=response or "", label=label or "")
    return bool(_is_correct(sample, args))


async def grade_code(tool_trace: list[dict], test_cases: list[dict]) -> bool:
    """Is the LAST code the model actually ran via code_interpreter (per
    ``tool_trace``) correct against ``test_cases``? TOOL-MANDATORY, same
    contract as examples/SDPO/reward.py::_grade_one_code -- a trajectory that
    never runs its solution through the tool has no candidate and is wrong
    regardless of what its final text claims."""
    sample = Sample(metadata={"tool_trace": tool_trace, "test_cases": test_cases})
    args = Namespace(sdpo_code_require_tool=True)
    return bool(await _grade_one_code(sample, args))
