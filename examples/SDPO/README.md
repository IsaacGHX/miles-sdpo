  for p in $(pgrep -f 'train_async|train.py|SGLangEngine|raylet|sglang|ray::|ray job'); do kill -9 $p; done; sleep 8

# SDPO — Prefix-Conditioned Self-Distillation (minimal example)

SDPO ([lasgroup/SDPO](https://github.com/lasgroup/SDPO)) keeps GRPO rollout
unchanged but adds a *self-distillation* signal built from a **correct peer
trace used as a prefix**. "Self" means the teacher is the **current policy**
conditioned on a correct hint — not a separate frozen model. In Miles the
rollout engine already holds the latest policy weights (re-synced every
rollout), so with `--sdpo-self-teacher` (default) we score the teacher against
that engine and **no separate teacher server is needed**.

1. Rollout is exactly GRPO: `n_samples_per_prompt` traces per prompt.
2. Each trace is graded against its label to find the *correct* ones.
3. For each trace, a **random** correct **peer** trace (never itself) is used as
   a prefix. The peer solution is rendered with a prefix template (see below)
   and inserted between the prompt and this trace's response. The teacher scores
   `prompt + prefix + response` and we read its next-token behaviour over the
   response span — the teacher has seen a correct hint that this trace did not.
4. The **student** signal is the original rollout one, conditioned on
   `prompt + response` with **no** prefix (captured during rollout).
5. A per-token divergence between teacher (with prefix) and student (without)
   is computed and written to `sample.opd_reverse_kl`. The framework subtracts
   `opd_kl_coef * opd_reverse_kl` from the GRPO advantages — no training-side
   change needed.

**Correct-trace policy** (this is what actually gets optimised):

| # correct traces in group | behaviour |
|---------------------------|-----------|
| 0 | no KL for anyone (pure GRPO) |
| 1 | no KL for anyone (no valid peer prefix) |
| ≥ 2 | every trace draws a **random** correct peer as prefix, never itself; different traces may draw different peers |

**Divergence** (`--sdpo-divergence`): `reverse_kl`, `forward_kl`, or `jsd` (default).

**Log-prob granularity** (`--sdpo-logprob-mode`):
- `topk` (default): compare the per-position distribution over the student's
  top-k token set **plus one aggregated tail bucket** for all remaining
  vocabulary mass. Needs `--opd-log-prob-top-k > 0` — `128` is a good default:
  it captures the head of the distribution and folds the long tail into a single
  probability `1 - sum(top-k)`, so you approximate the full-vocabulary KL/JSD
  without paying for the whole vocabulary. The top-k log-probs are already
  full-vocab-normalised, so no renormalisation is applied.
- `sampled`: compare only the sampled token's log-prob (a scalar per token,
  like the original OPD / TML signal). No top-k needed.

### Prefix format

The prefix is built from a template at the top of `sdpo.py` (edit to taste):

```python
SOLUTION_TEMPLATE = "\n\nCorrect solution:\n\n{successful_previous_attempt}"
PREFIX_INSTRUCTION = "\n\nCorrectly solve the original question.\n\n"
```

It is rendered from the peer's `response`, tokenized with the rollout
tokenizer, and inserted between the prompt tokens and this trace's response
tokens. Because the response stays at the tail, per-position alignment holds.

### Alignment guarantee

Position `i` of the teacher output is its prediction for response token `i`
(conditioned on `prompt + prefix + response[0:i]`), and the student's position
`i` predicts the same `response[i]` (conditioned on `prompt + response[0:i]`).
The code asserts the teacher's returned token id at each position equals the
student's actual response token, and raises if they ever disagree.

### Truncated (over-length) rollouts

If a trace hits `--rollout-max-response-len`, SGLang stops generation and marks
it `TRUNCATED`, but **keeps all tokens generated so far** — it is not discarded.
Such a trace still flows through SDPO normally (though an over-length trace that
is also wrong simply won't be graded correct, so it won't be used as a prefix).

## Why a group RM

Choosing a prefix from peer traces requires seeing the whole prompt group at
once. Miles delivers that when `--group-rm` is set — the custom RM
(`sdpo_group_reward`) receives the entire group. See
`miles/rollout/sglang_rollout.py::generate_and_rm_group`.

### Teacher: self vs. fixed external

- `--sdpo-self-teacher` (default): score against the rollout engine — the
  current policy, re-synced every rollout. This is the paper's self-distillation
  and needs no extra GPU or server.
- `--no-sdpo-self-teacher --rm-url http://<ip>:<port>/generate`: score against a
  fixed external teacher (e.g. a stronger frozen model). See the commented block
  at the bottom of `run-qwen3-8B-sdpo.sh`.

## Key arguments

```bash
--group-rm
--custom-rm-path examples.SDPO.sdpo.sdpo_group_reward

--use-opd --opd-type sglang        # reuse the OPD advantage hook
--opd-log-prob-top-k 128           # student top-k head; remaining mass -> one tail bucket
--opd-kl-coef 1.0                  # weight of the SDPO penalty
--sdpo-divergence jsd              # reverse_kl | forward_kl | jsd
--sdpo-logprob-mode topk           # topk | sampled
--sdpo-self-teacher                # or --no-sdpo-self-teacher --rm-url <...>
```

## Dataset (SciKnowEval, mixed domains)

Build a mixed-domain science training set (Biology / Chemistry / Physics /
Materials). Every domain's train slice is mixed into one `train.jsonl`; each
domain's val slice is written separately for evaluation:

```bash
python examples/SDPO/build_sci_dataset.py --out-dir /root/sci --val-ratio 0.1
# -> /root/sci/train.jsonl, /root/sci/val_chemistry.jsonl, val_biology.jsonl, ...
```

SciKnowEval ships only a `test` split, so the script slices train/val
deterministically (seeded). Items are mostly multiple-choice; the label is the
answer letter and the model answers inside `\boxed{}`. `_is_correct` in
`sdpo.py` does case-insensitive letter matching for MCQ and falls back to math
grading otherwise. Register more domains for eval by adding more
`--eval-prompt-data <name> <path>` pairs.

## Monitoring

Training exposes two monitoring paths, plus the Ray dashboard:

- **Weights & Biases** — add `--use-wandb --wandb-project <p> --wandb-group <g>`
  (already in the run script). Logs reward, `pg_loss`, `kl_loss`, `entropy_loss`,
  and the SDPO penalty `opd_reverse_kl` per rollout.
- **TensorBoard** — alternatively `--use-tensorboard --tensorboard-dir <dir>`,
  then `tensorboard --logdir <dir>`.
- **Ray dashboard** — `http://<MASTER_ADDR>:8265` (started by `ray start` in the
  script) for per-actor logs, GPU/mem utilisation, and job status.

The SDPO-specific metric to watch is `opd_reverse_kl` (the distillation penalty)
alongside `rewards` / eval pass@1.

## Run

```bash
bash examples/SDPO/run-qwen3-8B-sdpo.sh
```

Tuned for a single 8× H200 (141GB) node: 4 GPUs for training, 4 for rollout,
no teacher server (self-distillation). Fix the hard-coded `/root/...` paths to
match your environment before running.

## Limitations

- **Only `context_parallel_size == 1`.** Context Parallel (CP) shards a single
  sequence's tokens across GPUs (see
  `miles/backends/training_utils/cp_utils.py`). SDPO computes the divergence on
  the full, un-sharded response sequence in the reward stage and stores it in
  `opd_reverse_kl`; with `cp_size > 1` each training rank only holds part of the
  tokens, so the per-token penalty would need to be re-sliced along the CP
  offsets. The example scripts already use `--context-parallel-size 1`, so this
  is only a concern if you raise it.
- The task reward is `1.0` for correct / `0.0` otherwise (MCQ letter match or
  math grading). Adjust `_is_correct` for other task types.
- In `topk` mode the divergence is computed over the **student's** top-k support
  plus one tail bucket, with the teacher queried for exactly those token ids
  (mirrors the `only-student` strategy in the OPD top-k recipe). A token the
  teacher deems negligible is floored to a tiny probability.
