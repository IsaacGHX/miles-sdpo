"""Per-model chat-template verification + dump helper for SDPO_ReAct.

react_prompt.py/generate_with_tools.py are already model-agnostic BY
CONSTRUCTION: build_react_messages() returns a plain [{"role": ...}, ...]
list, --apply-chat-template renders it ONCE via the tokenizer's own
chat_template.jinja (loaded from --hf-checkpoint), and generate_with_tools.py
never touches control tokens again -- it does plain string concatenation on
top of whatever that render produced. So swapping models (e.g. Qwen2.5-7B-
Instruct -> Olmo-3-7B-Instruct) needs NO code change in either of those files.

What *does* need checking every time a new model is swapped in is the
ASSUMPTION that plain-string design quietly depends on:
  1. The model's EOS token is what generate_with_tools.py's stop sequences
     (</code>, </answer>) and the "no tag -> terminate" logic expect to sit
     next to -- if a model's template renders a totally different turn
     boundary, the loop's every-turn "prompt_text + response" concatenation
     could silently drift out of template.
  2. sdpo.py's _gen_prompt_suffix (used to splice the KD teacher's correct-
     peer prefix into the USER turn) actually derives a non-empty,
     structurally sane suffix from the new tokenizer.
  3. The rendered system+user prompt actually contains the literal one-shot
     example text (i.e. nothing in the template escaped/mangled our tags).

This module is a STANDALONE CHECK, run once per model swap (see each
run-<model>.sh's data-prep section) -- not imported by the rollout hot path.
It fails loudly (raises) rather than silently producing a subtly-wrong
prompt, and writes a full dump of what it rendered so a human can eyeball it.

Usage:
    python -m examples.SDPO_ReAct.model_template_check \
        --hf-checkpoint /root/Olmo-3-7B-Instruct \
        --dump-path /root/miles/sdpo_dumps/olmo3_template_check.json
"""

import argparse
import json

from examples.SDPO_ReAct.react_prompt import build_react_messages


def check_model_template(hf_checkpoint: str) -> dict:
    """Render the ReAct one-shot prompt through hf_checkpoint's OWN tokenizer
    and assert the invariants generate_with_tools.py / sdpo.py's KD-prefix
    splicing depend on. Returns a dict of what was checked (for dumping);
    raises AssertionError with a specific message on any violation."""
    from miles.utils.processing_utils import load_tokenizer

    tok = load_tokenizer(hf_checkpoint, trust_remote_code=True)

    messages = build_react_messages("What is 2 + 2?")
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    assert rendered, f"{hf_checkpoint}: apply_chat_template returned empty string"

    # The one-shot example's exact tag text must survive rendering verbatim --
    # if the template HTML/XML-escapes content or otherwise mangles it, the
    # model sees a corrupted worked example instead of the real tag syntax.
    for tag_text in ("<code>", "</code>", "<output>", "</output>", "<answer>", "</answer>"):
        assert tag_text in rendered, (
            f"{hf_checkpoint}: one-shot tag {tag_text!r} missing from the rendered prompt -- "
            "the chat template likely escaped or stripped it. Inspect the raw render before training."
        )

    # sdpo.py's _gen_prompt_suffix: the exact string appended after user content
    # when add_generation_prompt=True. Must be non-empty (an empty suffix means
    # apply_chat_template's sentinel round-trip found nothing -- see that
    # function's docstring in examples/SDPO/sdpo.py) or the KD teacher prefix
    # splice point is undefined for this model.
    sentinel = " SDPO_SENTINEL "
    suffix_probe = tok.apply_chat_template(
        [{"role": "user", "content": sentinel}], tokenize=False, add_generation_prompt=True
    )
    assert sentinel in suffix_probe, (
        f"{hf_checkpoint}: sentinel round-trip failed -- apply_chat_template did not preserve "
        "the literal user-content string. _gen_prompt_suffix (sdpo.py) cannot derive a splice "
        "point for this model's template."
    )
    gen_suffix = suffix_probe.split(sentinel, 1)[1]
    assert gen_suffix, (
        f"{hf_checkpoint}: _gen_prompt_suffix would be empty -- add_generation_prompt=True adds "
        "nothing after user content for this template. KD teacher-prefix splicing needs a real "
        "turn boundary (e.g. '<|im_end|>\\n<|im_start|>assistant\\n') to insert before."
    )

    # EOS token(s): generate_with_tools.py's "no valid tag -> terminate" path
    # assumes an untagged turn ends because the model hit its OWN natural EOS
    # (checked upstream via finish_reason != "length"). tokenizer.eos_token is
    # only ONE default -- some models (Olmo-3-7B-Instruct: eos_token is the
    # generic <|endoftext|>) list MULTIPLE real stop tokens in
    # generation_config.json's eos_token_id (Olmo-3: [<|im_end|>,
    # <|endoftext|>]), and SGLang reads that full list when it loads the HF
    # checkpoint -- so <|im_end|>, the token that actually closes each
    # chat-template turn, IS a real stop condition even though it is not
    # tokenizer.eos_token. Surface BOTH here so a human reviewing the dump can
    # confirm the turn-closing token is covered, rather than trusting the
    # single-token default silently.
    eos = tok.eos_token
    assert eos, f"{hf_checkpoint}: tokenizer has no eos_token set"

    gen_config_eos_ids = []
    try:
        import json as _json
        import os as _os

        gc_path = _os.path.join(hf_checkpoint, "generation_config.json")
        if _os.path.exists(gc_path):
            with open(gc_path) as f:
                gen_config_eos_ids = _json.load(f).get("eos_token_id") or []
            if not isinstance(gen_config_eos_ids, list):
                gen_config_eos_ids = [gen_config_eos_ids]
    except Exception:
        pass  # best-effort: generation_config.json may not exist for every checkpoint layout
    gen_config_eos_tokens = [tok.decode([tid]) for tid in gen_config_eos_ids]

    return {
        "hf_checkpoint": hf_checkpoint,
        "rendered_one_shot_prompt": rendered,
        "gen_prompt_suffix": gen_suffix,
        "eos_token": eos,
        "generation_config_eos_tokens": gen_config_eos_tokens,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-checkpoint", required=True)
    ap.add_argument("--dump-path", default=None, help="If set, write the check result here as JSON.")
    args = ap.parse_args()

    result = check_model_template(args.hf_checkpoint)
    print(f"Template check OK for {args.hf_checkpoint}")
    print(f"  eos_token: {result['eos_token']!r}")
    print(f"  generation_config eos tokens: {result['generation_config_eos_tokens']!r}")
    print(f"  gen_prompt_suffix: {result['gen_prompt_suffix']!r}")

    if args.dump_path:
        with open(args.dump_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  full render dumped to {args.dump_path}")


if __name__ == "__main__":
    main()
