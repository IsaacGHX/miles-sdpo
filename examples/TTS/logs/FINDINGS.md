# TTS findings — is the current (SDPO seed) skill-gen prompt already near-optimal?

**Short answer: yes.** Across every scaffold-search run here, the round-1 prompt —
which *is* the SDPO seed prompt (`prompts.py::CORRECT_SKILL_SYSTEM_SEED` /
`PITFALL_SKILL_SYSTEM_SEED`, `md5 correct=ded96cb5`) — is the best or
statistically-tied-best skill-generation prompt. The layer-2 meta-optimizer
(luna / sol) rewrites it every round and **never durably beats it**: later rounds
regress or wobble inside noise. This holds both for **distilling a skill from
correct traces** and for the deployment-aligned **predict-skill-from-the-bare-problem**
protocol.

Setup: solver + skill-writer = `gpt-5.6-luna`, optimizer = luna or `gpt-5.6-sol`,
grader = SDPO's DAPO grader. Optimization set = 20 hard AIME-style problems
(`logs/hard_set.jsonl`), 8 rollouts/problem. Held-out set = AIME25, 30 problems ×
8 (`logs/aime25.jsonl`). All three optimization runs share the identical seed at
round 1, so round 1 is a clean "seed prompt" measurement.

---

## 1. Optimization runs — seed (round 1) vs optimizer-rewritten (rounds 2–3)

`skilled_acc` = accuracy with the skill spliced in; `baseline` = same solver, no
skill. Round 1 uses the SDPO seed; rounds 2–3 use the optimizer's rewrite.

| Run (`logs/…jsonl`) | baseline | **R1 = seed** | R2 (opt) | R3 (opt) | best round |
|---|---|---|---|---|---|
| `tts_luna_solopt` (luna solve, sol optimize) | 0.4375 | **0.6687** | 0.6125 | 0.5875 | **1 (seed)** |
| `tts_luna_hard20` (luna solve+optimize) | 0.3688 | **0.6188** | 0.5437 | 0.5750 | **1 (seed)** |
| `tts_predict_solopt` (predict-mode, sol optimize) | 0.3750 | 0.6062 | 0.5437 | **0.6250** | 3 (+0.019 over seed) |

- Two of three runs: **the seed prompt (round 1) is the single best round**, and every
  optimizer rewrite is *worse*.
- The third run (`predict_solopt`) has round 3 nominally best, but only **+1.9 pts**
  over its own seed (0.625 vs 0.606) — well inside 20-problem × 8-rollout noise
  (±~3.5 pts), i.e. not a real improvement.
- The seed lifts skilled-acc **+17 to +25 pts over the no-skill baseline** — the
  skill mechanism works; it's the *prompt* that's already saturated.

The optimizer's own diagnoses (it keeps proposing "prioritize the bottleneck
method / add uniqueness+format checks / tighten leakage") describe plausible
edits that nonetheless **fail to move held-out accuracy** — evidence the seed
already captures what matters and further prompt surgery mostly adds length or
over-constrains.

---

## 2. Held-out AIME25 — seed prompt vs BEST optimized prompt (deployment-aligned "predict" protocol)

This is the decisive test: predict the skill from the **bare problem only** (no
traces, no answer — exactly how it's used at test time), then measure accuracy.
`prompts_predict_solopt`, 30 problems × 8, from `eval_aime25_{seed,best}.jsonl`.

| arm | seed prompt | BEST optimized | Δ |
|---|---|---|---|
| **baseline** (no skill) | 0.8167 | 0.8250 | — |
| **knowledge_only** (predicted `[Knowledge/Rule]`) | 0.8042 | 0.8083 | +0.004 |
| **pitfall_only** (predicted `[Error]/[Rule]/[Example]`) | 0.8125 | 0.8458 | +0.033 |
| **combined** | 0.8125 | 0.8500 | +0.038 |

Reading this:

- **Predicted knowledge is already at/below baseline** — for both the seed and the
  best-optimized prompt (0.804 / 0.808 vs baseline 0.817 / 0.825). On a strong
  solver, adding a predicted-from-scratch "knowledge" skill does **not** help and
  slightly hurts; optimizing the prompt does not fix this (+0.4 pt).
- The only positive signal is **pitfall/combined**, and even the *best-optimized*
  prompt beats the seed by just **+3.3 / +3.8 pts** on 240 samples — small, and
  the seed's combined arm (0.8125) is essentially at baseline too.
- Net: the seed skill-gen prompt is **already near the ceiling** of what prompt
  optimization can extract here, whether the skill is *distilled from correct
  traces* (§1) or *predicted from the bare problem* (§2).

---

## 3. Takeaway

- **The current prompt is essentially optimal.** Meta-optimizing the
  skill-generation prompt — the whole premise of this TTS harness — yields no
  reliable gain over the SDPO seed on either protocol. Round 1 (seed) wins or ties
  in every run; held-out gains are ≤4 pts and within noise.
- **"Predict skill without a trace" is already as good as it gets, and its main
  value is pitfalls, not knowledge.** Predicted knowledge sits at/below a strong
  solver's baseline; the only (marginal) lift comes from predicted pitfalls. This
  matches the SDPO-side observation that on capable solvers the knowledge/skill
  prefix approaches a no-op while failure-pitfall signal is where the headroom is.
- **Implication for SDPO:** since the fixed seed prompt is already near-ceiling for
  skill *generation*, effort is better spent on the *training* signal (which traces
  become teacher prefixes, pitfall weighting) than on further prompt engineering of
  the skill-gen prompt.

> Caveats: 20-problem optimization set and 30-problem held-out set → per-cell 95% CI
> ≈ ±3–4 pts; a strong solver (luna) sits near the AIME25 ceiling (~0.82 baseline),
> which compresses the room any skill has to help. The qualitative conclusion —
> seed ≈ best, knowledge ≈ no-op, pitfalls carry the small remaining signal — is
> consistent across all three runs and both protocols.

Data: `logs/tts_{luna_solopt,luna_hard20,predict_solopt}.jsonl` (per-round),
`logs/eval_aime25_{seed,best}.jsonl` (held-out), prompts in
`logs/prompts_*/` (`round0_seed_*` vs `BEST_*`).
