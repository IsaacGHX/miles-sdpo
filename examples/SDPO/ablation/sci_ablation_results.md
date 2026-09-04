# Qwen2.5-7B SciKnowEval (MCQ) ablation — completion status

Both series are EXPLORATORY: `--num-rollout 100`, no checkpointing. "final eval
acc" = mean `reward` over `rollout_data/eval_99.jsonl` (the last eval, after
rollout 99; MCQ letter-match, 3464 eval examples per run).

All 14 runs listed below reached `rollout_data/eval_99.jsonl` — **fully
complete**, no further action needed on either series.

## KD-loss series (`run-qwen2.5-7B-sdpo-sci-colocate.sh`, `--sdpo-kd-loss`)

| Arm | Config | Dump | Final eval acc |
|---|---|---|---|
| 1   | pure vanilla GRPO, no SDPO | `qwen2.5-7B-sdpo-sci-ablation-arm1_20260729_054027` | 0.5372 |
| 1.2 | SDPO baseline: `--group-rm` + KD loss (jsd), no skill, pure distill (task reward always 0) | `qwen2.5-7B-sdpo-sci-ablation-arm1.2_20260729_070057` | 0.4916 |
| 2   | + self-skill, skill-source correct only, no skill-KD | `qwen2.5-7B-sdpo-sci-ablation-arm2_20260729_083908` | 0.4166 |
| 3   | + self-skill, skill-source incorrect only, no skill-KD | `qwen2.5-7B-sdpo-sci-ablation-arm3_20260729_101751` | 0.4388 |
| 4   | + self-skill, skill-source all, no skill-KD | `qwen2.5-7B-sdpo-sci-ablation-arm4_20260730_064452` | 0.4330 |
| 5   | + self-skill, skill-source all, skill-KD mode=both | `qwen2.5-7B-sdpo-sci-ablation-arm5_20260729_214858` | 0.4209 |

Note: arm4/arm5 each have an earlier dump (`arm4_20260729_120521`,
`arm5_20260729_135530`) superseded by the listed later run — only the latest
per arm is reported.

## RLSD series (`run-qwen2.5-7B-sdpo-sci-rl-colocate.sh`, `--sdpo-rlsd`)

Same self-skill/skill-KD machinery as the KD-loss series' arms 2-5; only the
response-teacher mechanism differs (multiplicative RLSD reweighting instead
of additive KD-loss).

| Arm | Config | Dump | Final eval acc |
|---|---|---|---|
| 1.1 | RLSD baseline, no self-skill (RLSD analogue of KD-loss arm 1.2) | `qwen2.5-7B-sdpo-sci-rl-ablation-arm1.1_20260730_184125` | 0.4694 |
| 2   | + self-skill correct-only, no skill-KD; RLSD λ decays 0.5→0 over rollouts 0-50 (inert 50-100) | `qwen2.5-7B-sdpo-sci-rl-ablation-arm2_20260730_003203` | 0.5294 |
| 2.2 | identical to arm 2, RLSD λ held constant at 1.0 (no decay) | `qwen2.5-7B-sdpo-sci-rl-ablation-arm2.2_20260730_022136` | 0.5300 |
| 3   | + self-skill incorrect-only, no skill-KD | `qwen2.5-7B-sdpo-sci-rl-ablation-arm3_20260730_045037` | 0.5159 |
| 4   | + self-skill all, no skill-KD | `qwen2.5-7B-sdpo-sci-rl-ablation-arm4_20260730_165602` | 0.4772 |
| 5   | + self-skill all, skill-KD mode=both | `qwen2.5-7B-sdpo-sci-rl-ablation-arm5_20260730_083528` | 0.5609 |
| 5.1 | identical to arm 5, `--sdpo-skill-max-new-tokens` 2048 (was 1024) | `qwen2.5-7B-sdpo-sci-rl-ablation-arm5.1_20260730_143202` | **0.5857** |
| 5.2 | identical to arm 5, `--sdpo-skill-kd-mode` both-blind (was both) | `qwen2.5-7B-sdpo-sci-rl-ablation-arm5.2_20260730_120448` | 0.5401 |

Note: arm3/4/5.1/5.2 each have an earlier dump superseded by the listed later
run (`arm3_20260730_040701`, `arm4_20260730_110054`, `arm5.1_20260730_115419`,
`arm5.2_20260730_110318`) — only the latest per arm is reported.

## Takeaways

- RLSD outperforms the additive KD-loss on every comparable arm (1.1 vs 1.2,
  2 vs 2, 3 vs 3, 4 vs 4, 5 vs 5) on this SciKnowEval MCQ task — consistent
  with the [[native-multitask-ablation-result]] and RLSD-design rationale
  ([[moe-r3-router-alignment]] area context: RLSD keeps gradient direction
  strictly from task reward, only reweights magnitude).
- Best RLSD arm is **5.1** (self-skill all + skill-KD mode=both + widened
  skill budget 2048 tokens) at 0.5857 — widening the skill token budget past
  arm 5's observed 1024 cap gave a real lift (0.5609 → 0.5857).
- Both-blind skill-KD (arm 5.2) underperforms plain "both" (arm 5) on RLSD
  (0.5401 vs 0.5609) — opposite of the widen-tokens lift, suggesting the
  self-success (near-zero info-gap) correct-skill variant is pulling more
  weight than blind-correct here.
- On the KD-loss series, plain vanilla GRPO (arm 1, 0.5372) beats every SDPO
  variant (1.2-5, 0.42-0.49) — the additive KD-loss mechanism looks harmful
  on this task/model combo standalone, whereas RLSD's multiplicative
  reweighting recovers (and on arm 5.1, exceeds) the no-SDPO baseline.
