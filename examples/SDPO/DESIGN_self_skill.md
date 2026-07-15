# Self-generated skill + skill-SDPO (design)

Extends trace-condense: instead of an external LLM, the **current policy** generates
the skill during rollout, and (optionally) we run a **second SDPO objective on the
skill's own tokens**.

## 1. Self-skill generation (rollout phase)

For each rollout sample that will receive a teacher prefix (i.e. a correct peer
exists), during rollout we issue an extra generation against the SGLang rollout
engine (the current policy) with a **skill-generation prompt**:

    skill_gen_prompt(problem, solution) = chat_template([
        {system: SKILL_SYSTEM_PROMPT},
        {user: "PROBLEM:\n{problem}\n\nWORKED SOLUTION:\n{solution}\n\n
                Write the solution roadmap (6-10 numbered steps, instance-grounded,
                stop one step before the final answer)."},
    ], add_generation_prompt=True)

**Skill format — instance-grounded roadmap, NOT a generic ≤3-bullet abstraction.**
The skill's whole job as a teacher prefix is to let the teacher (policy conditioned
on the skill) *confidently reach the correct answer*, so its per-token distribution
over the response is a clean, low-entropy target for KD. A maximally-abstract
"3 generic procedural bullets, no instance values" skill fails at this on MCQ:
the teacher, seeing only vague procedure, is about as unsure as the prefix-free
student, so the KD signal (teacher − student) is weak and val acc lags the full
trace. Measured on real dumps: the full-trace prefix contained the answer in
~100% of samples (median ~2–4k chars), the old 3-bullet skill in ~1–13% (median
~220 chars). We therefore make the skill a concrete **solution roadmap**: 6–10
steps that DO name this problem's key quantities, governing relation, decisive
comparisons and key intermediate results — but stop one step before stating the
final letter/number, so it is not a raw answer leak. See `_SKILL_SYSTEM_PROMPT`
in `sdpo.py`. (`--sdpo-skill-max-new-tokens` is bumped 512→1024 accordingly, since
the roadmap is ~3–5× longer than the old bullets and 512 truncated its tail.)

- The generated `skill_text` (+ its tokens + rollout logprobs) is captured, exactly
  like a normal rollout response.
- `skill_text` replaces the full peer trace as the teacher prefix for the normal
  response-SDPO (same as trace-condense, but self-generated).
- Because the policy generated it, we have `skill_tokens` and `skill_rollout_logprobs`
  — required to run SDPO on the skill tokens (§2).

`--sdpo-self-skill` turns this on (mutually exclusive with the external
`--sdpo-trace-condense`: both produce a skill prefix, but self-skill is on-policy
and trainable).

## 2. Skill-SDPO (optional second objective)

`--sdpo-skill-kd` adds a SECOND KD loss, computed on the **skill's own tokens**,
generated under the skill-gen prompt. Same divergence + IS machinery as the
response KD, just a different (prompt, response, teacher-prefix) triple.

Student and teacher for the skill, mirroring the response SDPO:

- **student**: the skill distribution under `skill_gen_prompt(problem, solution)`
  with NO extra hint — i.e. exactly the context the skill was generated in.
- **teacher**: the skill distribution under `skill_gen_prompt` PLUS a hint spliced
  into the user turn. **The hint is the sample's OWN correct trace** (the response
  this same sample produced), NOT another peer's trace. (Key difference vs the
  response SDPO, which uses a *peer's* correct trace.)

KD target: `D(student_skill ‖ teacher_skill)` over the skill tokens, added to the
loss as a separate term weighted by `--sdpo-skill-kd-coef` (independent of the
response `--sdpo-kd-coef`), so it can be tuned / monitored separately.

### Two eligibility modes (`--sdpo-skill-kd-mode`)

Which samples' skills get the skill-SDPO, and what the teacher hint is:

- **`self-success`**: only when the sample itself answered correctly (self-success).
  Teacher hint = the sample's own correct trace. "Given I solved it, distill my
  skill toward the skill I'd write if I re-read my own correct solution."
- **`problem-only`**: for the problem regardless of self-correctness. Teacher =
  skill-gen prompt with NO hint (student == teacher context) — degenerates to a
  self-consistency/regularization target on the skill tokens; the signal comes only
  from the EMA-teacher lag, not from a correct-answer hint.
- **`pitfall-condense`**: for FAILED traces (requires `--sdpo-skill-source
  incorrect|all`). A *separate* skill OPD that mirrors the response-SDPO condense
  idea but for failure knowledge:
  - **student** = predict the pitfalls to avoid from the PROBLEM ONLY (no attempt,
    no answer, no privileged info). Regenerated during rollout so the KD'd tokens
    match this problem-only context.
  - **teacher** = the same problem-only prompt PLUS the group's per-trace failure
    skills spliced in as privileged info ("here's how attempts actually failed").
  - KD pulls the problem-only student toward the failure-informed teacher, i.e. the
    policy learns to foresee this problem's traps without having seen the failures.

## Data threaded rollout -> train (per sample)

- `sdpo_skill_tokens`      : token ids of the generated skill
- `sdpo_skill_prompt_tokens`: skill-gen prompt token ids (student context)
- `sdpo_skill_teacher_prompt_tokens`: skill-gen prompt + own-trace-hint (teacher context)
- `sdpo_skill_rollout_logprobs`: per-token old logprobs of the skill (for IS)
- `sdpo_skill` (text)      : for dumping

Teacher-side (megatron actor) computes the skill teacher top-k the same way as the
response teacher top-k (batched forward on `sdpo_skill_teacher_prompt_tokens +
skill_tokens`), and `policy_loss_function` adds the skill KD term.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--sdpo-self-skill` | off | Policy self-generates a skill during rollout (from a trace's own response). On by itself it only produces the skill (for skill-KD / response-prefix); it does NOT change the response prefix unless `--sdpo-response-prefix skill`. |
| `--sdpo-skill-kd` | off | Add the second KD objective on the skill's own tokens. |
| `--sdpo-skill-kd-coef` | 1.0 | Weight of the skill KD term (independent of response `--sdpo-kd-coef`). |
| `--sdpo-skill-kd-mode` | self-success | `self-success` (own-trace hint) / `problem-only` (no hint) / `pitfall-condense` (problem-only student vs failure-informed teacher; needs skill-source incorrect\|all). |
| `--sdpo-skill-max-new-tokens` | 512 | max_new_tokens for the self-skill generation call. |
| `--sdpo-skill-source` | correct | Which traces get a skill: `correct` / `incorrect` / `env_feedback` (placeholder) / `all`. The skill-gen prompt adapts per trace: a CORRECT trace is distilled with the "worked solution -> transferable procedure" prompt; an INCORRECT trace uses a pitfall prompt that tells the model the attempt is wrong, supplies the ground-truth answer (for diagnosis only, never emitted), tags WHY it failed (truncated / format / wrong), and asks for pitfall warnings (never a solution, never the answer). |
| `--sdpo-pitfall-summary-backend` | self | Second-stage aggregation of a group's per-trace pitfalls into one shared "common failure lessons" list: `self` (current policy) / `external` (OpenAI-compatible LLM). Used when skill-source covers incorrect traces. |
| `--sdpo-response-prefix` | trace | RESPONSE-SDPO teacher prefix: `trace` = correct peer's full solution (base SDPO); `skill` = that peer's self-generated skill (needs `--sdpo-self-skill` + peer to have a skill; falls back to the trace otherwise). |

Note: with `--sdpo-response-prefix skill`, the peers are correct traces, so
`--sdpo-skill-source` must include them (`correct` or `all`) or every prefix
silently falls back to the trace.

## 3. Group failure pitfalls (response-SDPO prefix)

When `--sdpo-skill-source` covers incorrect traces (`incorrect` | `all`), failure
knowledge is folded into the RESPONSE-SDPO teacher prefix in two stages:

1. **Per-trace pitfalls** — every failed trace distils its own "mistakes to avoid"
   warnings, tailored by why it failed: `truncated` (hit the length limit — likely
   a verbosity/efficiency pitfall), `format` (no parseable answer — output-format
   pitfall), or `wrong` (a complete but incorrect answer — conceptual pitfall).
2. **Common lessons** — all of the group's per-trace pitfalls are fed back to the
   skill generator (`--sdpo-pitfall-summary-backend`) and synthesized into ONE short
   shared list of the recurring mistakes.

That shared list is spliced **only into the FAILED traces' teacher prefix** (under a
"Common mistakes to avoid" block, after the correct-peer solution/skill); CORRECT
traces keep a clean correct-peer prefix. A failed trace with no correct peer (an
all-wrong group) still gets a pitfalls-only prefix. Rationale: a model that failed
cannot be trusted to rewrite a correct solution (that would hallucinate a bad KD
target), but it *can* flag concrete errors — so failed traces contribute warnings,
never solutions.

## Panels

All in one top-level `skill/` panel:
- rollout-side (step=rollout/step): `skill/length_{mean,min,max}`, `skill/count`, `skill/ppl`, `skill/response_prefix_is_skill_frac`.
- train-side (step=train/step): `skill/kl` (= skill KD loss), `skill/entropy` (skill-token entropy). Also `loss/sdpo_skill_kd_loss`.
