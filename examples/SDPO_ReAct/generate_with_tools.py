"""Custom multi-turn generate function for SDPO_ReAct, in the SAME style as
examples/search-r1/generate_with_search.py: plain regex-tag detection on the
raw generated text (<code>...</code> -> run in the sandbox -> <output>...
</output> spliced back in), NOT the model's native chat-template tool-calling
grammar.

Why not chat-template tools=: miles.rollout.generate_utils.
generate_endpoint_utils.compute_prompt_ids_from_sample only calls
tokenizer.apply_chat_template(..., tools=tools) when sample.prompt is STILL a
message list. Our dataset is already rendered into a plain string by
react_prompt.py + --apply-chat-template at the Dataset level (needed so the
one-shot example's tags are part of the literal prompt text), so that branch
never fires and the <tools>...</tools> schema block is silently never
injected -- the model then has no real tool-call grammar in context, only an
English description + a text pattern to imitate, which is why the base
version measured ~98-100% zero_tool_call_frac. search-r1's plain-tag approach
in this repo makes the SAME choice deliberately (see its generate_with_search.
py) and needs no schema injection at all -- the tag syntax IS the entire
contract, taught by the system prompt + one-shot text.

Also runs on the LEGACY rollout path (no MILES_EXPERIMENTAL_ROLLOUT_REFACTOR
needed): --custom-generate-function-path here takes the plain
`async def generate(args, sample, sampling_params)` signature exactly like
examples/search-r1/generate_with_search.py, auto-adapted by
LegacyGenerateFnAdapter. This sidesteps every experimental-refactor-only issue
hit by the multi_turn.generate/tools= path (the --sglang-router-policy
incompatibility, and the missing --eval-custom-rm-path support under
--group-rm for eval, which crashed training entirely) -- the legacy
eval_rollout_single_dataset in miles/rollout/sglang_rollout.py already
supports --eval-custom-rm-path natively.
"""

import asyncio
import re

from examples.SDPO_ReAct.tools.tool_client import execute_tool
from miles.rollout.sglang_rollout import GenerateState
from miles.utils.http_utils import post
from miles.utils.types import Sample

MAX_TURNS = 5
CODE_CONCURRENCY = 64

# Without an explicit stop sequence, SGLang has no reason to stop generating
# right after </code> or </answer> -- the model can (and, observed live on a
# real training run, DOES) keep going past its own closing tag within the
# SAME /generate call, hallucinating a FAKE <output>...</output> AND a fake
# <answer> before our code ever gets a chance to run the real tool and splice
# in the real result. By the time we append the real <output>, the model has
# already "seen" (generated) its own made-up one and moved on -- the real
# result lands in the transcript as dead text with no effect on the answer.
# Stopping generation exactly at the closing tag is what makes the tool
# actually execute BETWEEN the model's code and its next token, instead of
# after the model has already imagined what it would say.
_STOP_SEQUENCES = ["</code>", "</answer>"]

_SEMAPHORE = asyncio.Semaphore(CODE_CONCURRENCY)

# <code>...</code> is the ONE tag this base version teaches (see
# react_prompt.py's system prompt + one-shot). Matches search-r1's
# postprocess_predictions pattern shape (<tag>content</tag>), generalized to
# any of the recognized action tags.
_ACTION_RE = re.compile(r"<(code|answer)>(.*?)</\1>", re.DOTALL)


def postprocess_predictions(prediction: str):
    match = _ACTION_RE.search(prediction)
    if match:
        return match.group(1), match.group(2).strip()
    return None, ""


async def execute_predictions(code: str | None, action: str) -> tuple[str, str | None, bool]:
    """Returns (next_obs_text_to_splice_into_response, raw_observation_text_for_the_turns_trace, done).

    The raw (unwrapped, untagged) text is threaded straight into
    ``sample.metadata["turns"]`` below -- it is the ONE place the actual
    turn/role structure is known for certain (this loop built it), so nothing
    downstream (sdpo_react.py's trace dump) needs to re-derive it by
    regex-matching tags out of the final decoded text. That regex approach is
    what corrupted earlier dumps: the "invalid tag" nudge below is plain
    English that itself CONTAINS the literal substrings <code></code>/
    <answer></answer> as instructional prose, so a naive re.findall over the
    full response matched inside the nudge's own text too.
    """
    if action == "code":
        async with _SEMAPHORE:
            result = (await execute_tool("code_interpreter", {"code": code})).strip()
        next_obs = f"\n\n<output>{result}</output>\n\n"
        return next_obs, result, False
    else:
        # action == "answer" OR no valid tag at all -- both end the trajectory
        # here, never retry. A no-tag turn only happens when the model has
        # already hit its own natural stop (finish_reason wasn't "length" --
        # checked by the caller before this is reached -- and neither </code>
        # nor </answer> matched), i.e. the model emitted <|im_end|> with no
        # tag. Retrying used to splice a "try again" nudge onto raw text
        # immediately after that <|im_end|> with no <|im_start|> reopening --
        # asking the model to keep talking past its own EOS with no role
        # framing, which is undefined continuation behavior, not a real turn.
        # It also doesn't help: a trace with no <answer> tag is already
        # graded wrong regardless of how many extra turns run, so nothing is
        # gained by continuing. Just stop; grading treats "no answer tag" as
        # incorrect either way (see sdpo.py's _grade_group).
        return "", None, True


async def generate(args, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for this function."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_text = sample.prompt
    prompt_tokens_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []

    # Per-sample override so eval can run more turns than training (e.g.
    # eval_aime24.yaml's metadata_overrides: {generate_max_turns: 20} while
    # MAX_TURNS=5 governs training) without a second global constant --
    # EvalDatasetConfig.inject_metadata (called for every eval sample in both
    # the legacy and experimental rollout paths) writes this key.
    max_turns = MAX_TURNS
    if isinstance(sample.metadata, dict) and "generate_max_turns" in sample.metadata:
        max_turns = sample.metadata["generate_max_turns"]

    max_context_len = getattr(args, "rollout_max_context_len", None) or 32768
    final_status = Sample.Status.COMPLETED
    rounds_used = 0
    tool_calls_executed = 0
    # Ground-truth turn structure, recorded live as each round happens --
    # stashed in sample.metadata so sdpo_react.py's trace dump can render
    # {"role": "assistant"/"tool", ...} messages directly instead of
    # re-deriving them by regex-matching <code>/<output> tags out of the
    # final decoded response text (which breaks whenever a turn's own prose,
    # e.g. the "invalid tag" nudge below, happens to CONTAIN those tag
    # strings as instructional text rather than a real tool invocation).
    turns: list[dict] = []

    for _turn_idx in range(max_turns):
        # Clamp max_new_tokens to the model's context window MINUS what the
        # prompt+response-so-far already occupies. Without this, a fixed
        # sampling_params["max_new_tokens"] (e.g. --eval-max-response-len
        # 16384) gets requested on EVERY turn regardless of how much context
        # multi-turn tool observations have already consumed -- by turn N the
        # request (input + max_new_tokens) can exceed the model's max context
        # length (32768 for Qwen2.5-7B), which the router rejects with a 400
        # on every retry (never recovers), and enough concurrent 400s
        # eventually degrade the whole router into a sustained run of 503s
        # that crashes the job. This was observed during the FIRST real eval
        # (--generate-max-turns 20) of this rollout.
        turn_sampling_params = dict(sampling_params)
        # `no_stop_trim=True` is set globally (miles/rollout/sglang_rollout.py's
        # base_sampling_params), so the matched stop string is kept in the
        # returned text -- required for postprocess_predictions' regex below to
        # still see the closing </code>/</answer> tag it stops on.
        turn_sampling_params["stop"] = _STOP_SEQUENCES
        cur_prompt_tokens = state.tokenizer(prompt_text + response, add_special_tokens=False)["input_ids"]
        remaining = max_context_len - len(cur_prompt_tokens)
        if remaining <= 0:
            final_status = Sample.Status.TRUNCATED
            break
        turn_sampling_params["max_new_tokens"] = min(
            turn_sampling_params.get("max_new_tokens", remaining), remaining
        )

        payload = {
            "text": prompt_text + response,
            "sampling_params": turn_sampling_params,
            "return_logprob": True,
        }
        output = await post(url, payload)
        rounds_used += 1

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        if "output_token_logprobs" not in output["meta_info"]:
            raise RuntimeError(
                "output_token_logprobs missing from /generate response; "
                "return_logprob=True must be honored by the rollout engine."
            )
        cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        cur_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)
        rollout_log_probs += cur_response_log_probs

        if output["meta_info"]["finish_reason"]["type"] == "length":
            final_status = Sample.Status.TRUNCATED
            turns.append({"role": "assistant", "content": cur_response, "action": "truncated"})
            break

        action, content = postprocess_predictions(cur_response)
        if action == "code":
            tool_calls_executed += 1
        next_obs, raw_obs, done = await execute_predictions(content, action)
        turns.append({"role": "assistant", "content": cur_response, "action": action})
        if done:
            final_status = Sample.Status.COMPLETED
            break

        assert next_obs != "", "Next observation should not be empty."
        turns.append({"role": "tool", "content": raw_obs})
        obs_token_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_token_ids
        loss_mask += [0] * len(obs_token_ids)
        rollout_log_probs += [0.0] * len(obs_token_ids)

        assert len(response_token_ids) == len(rollout_log_probs), (
            f"Token/logp length mismatch: {len(response_token_ids)} tokens vs "
            f"{len(rollout_log_probs)} logps"
        )
    else:
        # Loop exhausted max_turns while every turn kept calling code_
        # interpreter (never hit `done`, `length`, or ran out of context) --
        # the model just never gave a final <answer> within its turn budget.
        # An untagged turn can't land here: execute_predictions now treats
        # "no valid tag" as `done` immediately (see that function's
        # docstring), so it always breaks the loop rather than looping
        # around to consume another turn. Grading still treats this trace as
        # wrong (no <answer> tag found), consistent with search-r1's own
        # tag-validity scoring.
        final_status = Sample.Status.COMPLETED

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text
    sample.rollout_log_probs = rollout_log_probs or None
    sample.status = final_status

    # Bookkeeping for the agentic/* wandb panel (miles.ray.rollout.metrics.py
    # ::_compute_agentic_tool_metrics) -- counted from the LOOP'S OWN COUNTERS
    # (rounds_used / tool_calls_executed), not by re.findall-ing tags out of
    # the final concatenated response text. A regex count over the full text
    # overcounts whenever a single turn's generated text merely MENTIONS a
    # <code>/<output> tag as illustrative prose (e.g. the model explaining
    # "for example <code>...</code>" without actually invoking the tool that
    # turn) -- observed as round_number_max=37 / tool_call_count_max=22 on a
    # run capped at --generate-max-turns 20, which is impossible if counted
    # from real /generate round-trips.
    if isinstance(sample.metadata, dict):
        sample.metadata["round_number"] = rounds_used
        sample.metadata["tool_call_count"] = tool_calls_executed
        sample.metadata["turns"] = turns
        # The real train/eval step id (see sglang_rollout.py's
        # generate_rollout_async/eval_rollout, which stamp this on the same
        # GenerateState singleton) -- lets sdpo_react.py's trace dump group
        # by actual rollout_id instead of a local per-process call counter,
        # so agentic_traces/{rollout_id}.jsonl lines up 1:1 with
        # rollout_data/{rollout_id}.jsonl for easy cross-reference.
        sample.metadata["rollout_id"] = state.rollout_id

    return sample
