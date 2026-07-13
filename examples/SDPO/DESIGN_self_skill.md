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
                Distill the transferable skill (<=3 terse procedural bullets)."},
    ], add_generation_prompt=True)

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
| `--sdpo-skill-kd-mode` | self-success | `self-success` (own-trace hint) / `problem-only` (no hint). |
| `--sdpo-skill-max-new-tokens` | 512 | max_new_tokens for the self-skill generation call. |
| `--sdpo-skill-source` | correct | Which traces get a skill: `correct` / `incorrect` / `env_feedback` (placeholder) / `all`. The skill-gen prompt adapts per trace: a CORRECT trace is distilled with the "worked solution -> transferable procedure" prompt; an INCORRECT trace uses a different prompt that tells the model the attempt is wrong, supplies the ground-truth answer (for diagnosis only, never emitted), and asks for a pitfall-prevention / error-recovery skill. Both stay generic and never reveal the answer. |
| `--sdpo-response-prefix` | trace | RESPONSE-SDPO teacher prefix: `trace` = correct peer's full solution (base SDPO); `skill` = that peer's self-generated skill (needs `--sdpo-self-skill` + peer to have a skill; falls back to the trace otherwise). |

Note: with `--sdpo-response-prefix skill`, the peers are correct traces, so
`--sdpo-skill-source` must include them (`correct` or `all`) or every prefix
silently falls back to the trace.

## Panels

All in one top-level `skill/` panel:
- rollout-side (step=rollout/step): `skill/length_{mean,min,max}`, `skill/count`, `skill/ppl`, `skill/response_prefix_is_skill_frac`.
- train-side (step=train/step): `skill/kl` (= skill KD loss), `skill/entropy` (skill-token entropy). Also `loss/sdpo_skill_kd_loss`.
