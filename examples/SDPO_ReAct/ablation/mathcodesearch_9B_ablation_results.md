# Qwen3.5-9B native multitask (math+code+search) GRPO ablation — completion status

Launcher: `examples/SDPO_ReAct/ablation/run-qwen3.5-9B-sdpo-react-ablation-mathcodesearch.sh`
via `sbatch_run_9B_mathcodesearch.sh` / `sbatch_run_9B_mcs_rlsd_sdpo.sh`. Model =
Qwen3.5-9B, native tool-calling, `--num-rollout 51`. Only the 3 arms GRPO supports
(a/e/f — the prefix-only arms b/c/d need a distillation term to differ from the
baseline, enforced by the script's own arm check) have been run so far; sdpo/rlsd x a-f are still queued (job 1666,
`ip-10-1-37-19`, currently on `rlsd-a`).

"final eval reward" = mean `reward` over `rollout_data/eval_49.jsonl` (last eval,
2720 examples: 480 math + 640 code + 1600 search).

All 3 arms below reached `rollout_data/eval_49.jsonl` with checkpoint saved to
`iter_0000050` and **converted to HF** — fully complete.

| Arm | Config | Dump | Ckpt (torch_dist) | HF ckpt | Final eval reward (math / code / search) |
|---|---|---|---|---|---|
| a | plain GRPO, no skill (baseline) | `...grpo-a-think-minimal_20260815_093017` | `sdpo_ckpts/...grpo-a-think-minimal_ckpt/iter_0000050` | `sdpo_ckpts_hf/grpo-a-think-minimal_iter50_hf` | 0.7485 (0.773 / 0.805 / 0.719) |
| e | + self-skill all + skill-KD mode=both | `...grpo-e-think-minimal_20260814_124250` | `sdpo_ckpts/...grpo-e-think-minimal_ckpt/iter_0000050` | `sdpo_ckpts_hf/grpo-e-think-minimal_iter50_hf` | 0.5827 (0.865 / 0.867 / **0.384**) |
| f | + self-skill all + skill-KD mode=both-blind | `...grpo-f-think-minimal_20260815_030053` | `sdpo_ckpts/...grpo-f-think-minimal_ckpt/iter_0000050` | `sdpo_ckpts_hf/grpo-f-think-minimal_iter50_hf` | **0.7504** (0.790 / 0.789 / 0.723) |

All paths relative to `/fsx/data/haoxiang.zhang/`. `arm a`'s listed run is a
**rerun** of an earlier `arm a` attempt (`...20260814_064707`, which also
succeeded but ran before `SDPO_REACT_SAVE_CKPT=1` was wired in, so it left no
checkpoint) — only the checkpointed rerun is reported/converted.

## Findings

- **Arm f (skill+skill-KD mode=both-blind) is the best overall** (0.7504),
  essentially tied with plain GRPO (arm a, 0.7485) — and unlike the sci/native
  math-only ablations, both-blind does NOT collapse here.
- **Arm e (skill-KD mode=both) has a severe search regression** (0.384 vs
  arm a's 0.719, arm f's 0.723) despite the highest math/code scores of the
  three (0.865/0.867). This is the mirror image of the
  [[native-multitask-ablation-result]] Qwen3.5-4B code-collapse pattern — here
  it's search that collapses under skill-KD mode=both, not code — worth
  flagging before treating skill-KD/both as a safe default across domains/model
  sizes.
- Plain GRPO (arm a) is a strong, balanced baseline across all three domains,
  consistent with the 4B native-multitask result where plain GRPO also won on
  macro-average.
- sdpo/rlsd arms (a/e) for this model+domain are still queued on job 1666 —
  update this table once they land.
