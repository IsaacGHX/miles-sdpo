# SDPO options reference

All SDPO-specific CLI flags (defined in `miles/utils/arguments.py`, consumed by
`examples/SDPO/sdpo.py` and the megatron actor). Grouped by feature.

## Core distillation

| Flag | Default | Meaning |
|---|---|---|
| `--sdpo-kd-loss` | off | Use a real distribution-level KD **loss** (student‖teacher divergence), not an advantage/REINFORCE hook. THE objective. |
| `--sdpo-kd-coef` | 1.0 | Weight of the KD loss. Under pure distill this is the whole loss, so 1.0. |
| `--sdpo-kd-max-tokens` | 0 | Cap the KD loss to the first N response tokens (0 = whole response). Bounds full-vocab log-softmax memory. |
| `--sdpo-divergence` | jsd | `forward_kl` (α=0) / `reverse_kl` (α=1) / `jsd` (α=0.5, symmetric) / `jeffrey`. |
| `--sdpo-logprob-mode` | topk | `topk` (per-position top-k distribution + tail bucket) / `sampled` (sampled-token logprob only). |
| `--opd-log-prob-top-k` | — | k for the teacher top-k distribution (e.g. 100, matches lasgroup/SDPO `distillation_topk`). |
| `--sdpo-distillation-add-tail` | on | Add an aggregated tail bucket to the top-k distributions before the divergence. |
| `--sdpo-is-clip` | 2.0 | IS-ratio clip for off-policy/async: per token KD is scaled by `exp(student_logp - old_logp).clamp(max=is_clip)`. `<=0` disables. |
| `--sdpo-pure-distill` | on | Task reward = 0 (GRPO advantage = 0), so the KD loss is the entire objective. |
| `--sdpo-self-teacher` | on | Teacher = current policy (self-distillation); no external teacher server. |
| `--sdpo-teacher-backend` | sglang | `megatron` (batched CUDA-graph teacher forward on the training actor, ~50x faster) / `sglang` (HTTP eager scoring). |

## Teacher regularization (anti-collapse)

| Flag | Default | Meaning |
|---|---|---|
| `--sdpo-ema-teacher` | off | Teacher is a slow EMA copy of the policy (`teacher = (1-rate)*teacher + rate*student` each step), not the live policy. Prevents the self-reinforcing "emit the answer immediately" collapse. |
| `--sdpo-ema-teacher-rate` | 0.05 | EMA update rate (paper uses 0.05; smaller = slower/more stable teacher). |
| `--sdpo-kd-clip-cov-frac` | 0.0 | KD Clip-Cov (arXiv:2505.22617, adapted): detach the gradient of the top-fraction of response tokens with the largest KD divergence (batch-wide). 0.0 = off. |

## Prefix construction

| Flag | Default | Meaning |
|---|---|---|
| `--sdpo-remove-thinking-from-demonstration` | off | Strip `<think>...</think>` from the correct-peer response before it becomes the teacher prefix. Required for thinking models (Qwen3). |
| `--sdpo-answer-tag` | answer | XML tag the model wraps its final answer in (`<answer>...</answer>`). |

## Trace condensation / SkillOpt

Distill the correct-peer trace into a short transferable **skill** and use that as
the teacher prefix instead of the full trace (matches lasgroup/SDPO `trace_condense`).

| Flag | Default | Meaning |
|---|---|---|
| `--sdpo-trace-condense` | off | Enable condensing the peer trace into a skill (≤3 procedural bullets, no answer). Falls back to the full trace on failure. |
| `--sdpo-condense-base-url` | `https://api.openai.com/v1` | OpenAI-compatible base URL for the condenser. |
| `--sdpo-condense-model` | gpt-5.4-mini | Model that distills traces into skills. |
| `--sdpo-condense-api-key-env` | OPENAI_API_KEY | Env var holding the condenser API key. |
| `--sdpo-condense-max-tokens` | 2048 | max_completion_tokens for the condenser call. |
| `--sdpo-condense-max-concurrency` | 32 | Process-wide cap on concurrent condenser requests. |

When enabled, `sdpo_dumps/<exp>/sdpo_prompts/<id>_rank<r>.jsonl` gets a `"skill"`
field per record (the distilled skill spliced into `teacher_prompt_text`).

## Self-generated skill + skill-SDPO

The policy self-generates the skill on-policy (trainable), and optionally runs a
second SDPO objective on the skill tokens. Full design: `DESIGN_self_skill.md`.

| Flag | Default | Meaning |
|---|---|---|
| `--sdpo-self-skill` | off | Policy self-generates a skill during rollout (from a trace's own response). Trainable (unlike external `--sdpo-trace-condense`); mutually exclusive with it. |
| `--sdpo-skill-kd` | off | Second KD objective on the skill's own tokens (needs `--sdpo-self-skill`). |
| `--sdpo-skill-kd-coef` | 1.0 | Weight of the skill KD term (independent of `--sdpo-kd-coef`). |
| `--sdpo-skill-kd-mode` | self-success | Skill teacher hint: `self-success` (the sample's own trace) / `problem-only` (no hint). |
| `--sdpo-skill-source` | correct | Which traces get a skill: `correct` / `incorrect` / `env_feedback` (placeholder) / `all`. |
| `--sdpo-skill-max-new-tokens` | 512 | max_new_tokens for the self-skill generation call. |
| `--sdpo-response-prefix` | trace | RESPONSE-SDPO teacher prefix: `trace` (correct peer's full solution) or `skill` (that peer's self-generated skill; needs `--sdpo-self-skill` + a skill on the peer). |

Skill dumps go to a SEPARATE `sdpo_dumps/<exp>/skill/<id>_rank<r>.jsonl` (skill_text,
problem_text, skill_student/teacher_prompt_text, skill_student/teacher_text).
Metrics (top-level `skill/` panel): `skill/{length_*,count,ppl,response_prefix_is_skill_frac}`
(rollout/step) and `skill/{kl,entropy}` + `loss/sdpo_skill_kd_loss` (train/step).

## Grading (correct-peer selection & eval)

| Flag | Default | Meaning |
|---|---|---|
| `--sdpo-grader` | mcq | How the group RM grades trace correctness. `mcq`: `<answer>`-letter / math-verl (SciKnowEval). `dapo`: DAPO integer-boxed math grader (dapo-math-17k). |

## LLM-as-judge grading (optional, defeats MCQ guess-hack)

| Flag | Default | Meaning |
|---|---|---|
| `--sdpo-judge` | off | Grade trace correctness with an LLM judge (sees question + reference + full response + extracted answer). |
| `--sdpo-judge-base-url` | `https://api.openai.com/v1` | Judge base URL. |
| `--sdpo-judge-model` | gpt-5.4-mini | Judge model. |
| `--sdpo-judge-api-key-env` | OPENAI_API_KEY | Judge API key env var. |
| `--sdpo-judge-max-tokens` | 2048 | Judge max_completion_tokens (gpt-5* are reasoning models). |
| `--sdpo-judge-max-concurrency` | 32 | Process-wide cap on concurrent judge requests. |
