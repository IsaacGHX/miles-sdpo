#!/bin/bash

# SDPO (self-distilled policy optimization) — NVIDIA-Nemotron-3-Nano-4B-BF16 on the
# SciKnowEval (multiple-choice science) task, single 8x H200 (141GB) node.
# Nemotron counterpart of run-olmo3-7B-sdpo-sci.sh; same SDPO/skill config and the
# same SciKnowEval MCQ task (data, grader, eval), only the model changes.
# usage: bash examples/SDPO/run-nemotron3-4b-sdpo-sci.sh
#
# Build the dataset first with:
#   python examples/SDPO/build_sci_dataset.py --out-dir /root/sci --val-ratio 0.1
#
# MODEL NOTE — Nemotron-3-Nano-4B is DENSE (`nemotron_h` = hybrid Mamba + Attention,
# NO MoE: no routed experts, every token flows through the same squared-relu FFN;
# contrast the 30B-A3B sibling which IS MoE). It loads via the AutoBridge path
# (--megatron-to-hf-mode bridge), and the bridge already exists in this repo:
#   - miles_plugins/megatron_bridge/nemotron_h.py  (MilesNemotronHBridge + the
#     hybrid-layer IdentityOp shim and the MambaModel.forward loss_mask shim)
#   - scripts/models/nemotron-3-nano-4b.sh          (the MODEL_ARGS block)
# so no new bridge is needed here. CAVEAT: SDPO's megatron teacher/KD path
# (--sdpo-teacher-backend megatron) was validated on the Olmo3 dense transformer;
# on the Mamba hybrid it is exercised for the first time here — if the teacher
# forward misbehaves, fall back to --sdpo-teacher-backend sglang (see GRPO_ARGS).

set -exf

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "/root/miles/scripts/models/nemotron-3-nano-4b.sh"

SDPO_EXP="${SDPO_EXP:-nemotron3-4b-sdpo-sci-base_$(date +%Y%m%d_%H%M%S)}"
DUMP_DIR="/root/miles/sdpo_dumps/${SDPO_EXP}"
echo "SDPO dump dir: ${DUMP_DIR}"

CKPT_ARGS=(
   # AutoBridge builds the full nemotron_h Megatron provider (incl. all Mamba
   # fields) from HF config.json at load time, so ref-load reads the SAME HF
   # checkpoint directly — there is no separate _torch_dist conversion like Olmo3.
   --hf-checkpoint /root/NVIDIA-Nemotron-3-Nano-4B-BF16
   --ref-load /root/NVIDIA-Nemotron-3-Nano-4B-BF16
   --save /root/NVIDIA-Nemotron-3-Nano-4B-BF16_miles/
   --megatron-to-hf-mode bridge
   --save-interval 50
   --dump-details "${DUMP_DIR}"
   # skip the heavy per-token train_data/*.pt and policy_loss_debug/*.pt dumps;
   # keep rollout_data + sdpo_prompts + skill dumps.
   --no-dump-train-data
   --no-dump-policy-loss-debug
)

# SciKnowEval: all four domains' train splits mixed into one jsonl for training;
# per-domain val splits registered for evaluation.
# Build with: python examples/SDPO/build_sci_dataset.py --out-dir /root/sci
ROLLOUT_ARGS=(
   --prompt-data /root/sci/train.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
   # SDPO dynamic sampling (DAPO-style), OPT-IN and OFF by default. To enable:
   # uncomment all three lines. It oversamples groups and DROPS any with fewer than
   # --sdpo-dynamic-filter-min-correct correct traces, then re-samples, so every kept
   # group can give EVERY trace a correct-peer prefix (no dead teacher==student,
   # zero-KD samples reaching the batch). Set the threshold to 2 for full prefix
   # coverage (1 correct -> that lone trace self-excludes to an empty peer pool);
   # the arg defaults to 0 = keep every group, so the filter is inert until set.
   # over-sampling-batch-size 64 = 2x rollout-batch-size gives headroom to discard.
   # If the model is too weak early (few groups reach 2 correct) the rollout can
   # stall filling the batch — raise oversampling, drop the threshold to 1, or warm
   # up without the filter. Monitor rollout/dynamic_filter/drop_sdpo_* on wandb.
   # --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_sdpo_group_has_prefix
   # --sdpo-dynamic-filter-min-correct 2
   # --over-sampling-batch-size 64
)

RM_ARGS=(
   --group-rm
   --custom-rm-path examples.SDPO.sdpo.sdpo_group_reward
   # SciKnowEval is well-formed L3 multiple-choice: the label is the answer LETTER,
   # graded by a case-insensitive letter match on the extracted <answer> ('mcq').
   # (The math script uses 'dapo' for integer boxed answers instead.)
   --sdpo-grader mcq
   # -------------------------------------------------------------------------
   # SKILL MECHANISM DISABLED -> base SDPO.
   #   * NO skill is generated (--sdpo-self-skill removed; default False), so the
   #     extra per-rollout skill-gen generation is skipped entirely, and
   #     --sdpo-skill-source / --sdpo-skill-max-new-tokens / --sdpo-skill-kd-*
   #     become no-ops (they all require --sdpo-self-skill).
   #   * The skill does NOT replace the trace: --sdpo-response-prefix defaults to
   #     'trace', so the response-SDPO teacher prefix is the correct peer's FULL
   #     solution trace (base SDPO), not a distilled skill.
   # To re-enable the self-skill setup, restore the block below (see
   # examples/SDPO/DESIGN_self_skill.md and run-olmo3-7B-sdpo-sci.sh):
   #   --sdpo-self-skill
   #   --sdpo-skill-max-new-tokens 1024
   #   --sdpo-skill-source correct
   #   --sdpo-skill-kd-coef 1.0
   #   --sdpo-skill-kd-mode self-success
   #   --sdpo-response-prefix skill
   # -------------------------------------------------------------------------
)

EVAL_ARGS=(
   --eval-interval 10
   # --skip-eval-before-train
   # All four SciKnowEval domains as name/path pairs in ONE --eval-prompt-data
   # (nargs='+', parsed as consecutive name path name path ...; repeating the flag
   # would overwrite, not append). eval_rollout runs the datasets concurrently
   # (asyncio.gather), so total time grows sublinearly, not 4x. Graded by the same
   # MCQ letter-match rule as the group RM, via sdpo_eval_reward.
   --eval-prompt-data
      sci_chem /root/sci/val_chemistry.jsonl
      sci_bio  /root/sci/val_biology.jsonl
      sci_phys /root/sci/val_physics.jsonl
      sci_mat  /root/sci/val_material.jsonl
   --n-samples-per-eval-prompt 8
   --log-passrate
   # Full 16k eval length (double the 8K rollout cap). This no longer OOMs: eval
   # skips the OPD top-k logprob request (see generate()'s `evaluation` gate),
   # which was the real cause of the earlier crash — not the sequence length.
   --eval-max-response-len 16384
   --eval-top-p 1
   # SDPO trains with a group RM, which cannot score eval samples (no group step in
   # eval). This per-sample eval RM grades eval pass@1 with the same correctness
   # rule as sdpo_group_reward. Required for eval under --group-rm.
   --eval-custom-rm-path examples.SDPO.sdpo.sdpo_eval_reward
)

PERF_ARGS=(
   # Nemotron-3-Nano-4B is a Mamba+Attention hybrid. TP=1: each of the 4 train GPUs
   # holds the full 4B model + full Adam optimizer state (~80GB static on H200); the
   # Mamba mixer is handled by the bridge provider. Dense -> expert parallelism N/A.
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   # 16384 (was 49152): with TP=1 the model+optimizer eat ~80GB/GPU, leaving ~60GB
   # for activations. At 49152 tokens/microbatch the Mamba SSM training kernel
   # (mamba_split_conv1d_scan_combined) over-allocated and OOMed the forward on a
   # 141GB H200. 16384 keeps a comfortable margin while still ~1.8x the known-good
   # smoke test (scripts/run-nemotron-3-nano-4b.sh uses 9216, but at TP2/PP2 so its
   # per-GPU model share is 1/4 of ours). Raise toward 24576 if memory sits idle.
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 0.0
   # Megatron self-teacher. First use on the Mamba hybrid — if the teacher forward
   # errors or drifts, switch to `--sdpo-teacher-backend sglang` (scores against the
   # rollout engine instead of a megatron training forward).
   --sdpo-teacher-backend megatron
   --sdpo-ema-teacher
   --sdpo-ema-teacher-rate 0.05
   --sdpo-logprob-mode topk
   --opd-log-prob-top-k 100
   --sdpo-divergence jsd
   --sdpo-is-clip 2.0
   --sdpo-kd-loss
   --sdpo-kd-coef 1.0
   --sdpo-kd-max-tokens 8192
   --sdpo-self-teacher
   --sdpo-pure-distill
   # Nemotron-3-Nano-4B is a thinking model: its chat template defaults to
   # enable_thinking=True and ends the prompt with '<|im_start|>assistant\n<think>\n',
   # so every rollout response carries a <think>...</think> chain before the
   # requested <reasoning>/<answer> format. Strip it from the peer solution before
   # it becomes the teacher prefix — otherwise the prefix balloons and the teacher
   # teaches the student to echo the reasoning verbatim. Same reason as the Qwen3
   # script; REQUIRED for thinking models. (Grading strips <think> unconditionally,
   # so this only affects the teacher prefix, not scoring.)
   --sdpo-remove-thinking-from-demonstration
   --sdpo-answer-tag answer
   --entropy-coef 0.00
   --observe-training-entropy
   --calculate-per-token-loss
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --lr-warmup-iters 10
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-sdpo
   --wandb-group nemotron3-4b-sdpo-sci
   --wandb-key "${WANDB_API_KEY}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # 0.9 (was 0.8): at 0.8 the rollout GPUs sat at ~82% (118/143 GB), ~25GB idle.
   # 8k generations have a lower logits/activation peak than 16k, so 0.9 gives the
   # KV cache more room (higher throughput) while keeping ~10% headroom. Drop back
   # to 0.8 if the logits_processor OOMs on a long batch.
   --sglang-mem-fraction-static 0.9
   --sglang-chunked-prefill-size 8192
   --sglang-router-policy round_robin
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# 8x H200, DISAGGREGATED 4+4: 4 GPUs train (TP2 x DP2), 4 run SGLang rollout.
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

ray stop --force 2>/dev/null || true
pkill -9 -f 'ray::' 2>/dev/null || true
sleep 2

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{
     \"env_vars\": {
        \"PYTHONPATH\": \"/root/Megatron-LM/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
        \"OPENAI_API_KEY\": \"${OPENAI_API_KEY}\",
        \"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK\": \"1\"
     }
   }" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   --update-weights-interval 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]}

ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
