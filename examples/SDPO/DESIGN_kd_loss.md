# 设计方案：SDPO 分布级 KD Loss（对齐原版 full_logit_distillation）

## 问题

当前 SDPO 把分布散度 `D(student‖teacher)` 当作 GRPO advantage 走 REINFORCE：
`advantage = -opd_kl_coef × JSD`，再 `pg_loss = -advantage × ∇logπ(采样token)`。

实测症状（bug）：
- `advantage` 恒 ≤ 0（JSD≥0 取负）、量级仅 0.02-0.03、`grad_norm≈0.04`、`pg_clipfrac=0`
- loss(=opd_reverse_kl) 稳定不降 → **模型几乎不更新，蒸馏没起效**

根因：REINFORCE 只对**采样 token**有梯度，且全负小 advantage 无相对方向 → 梯度弱。
原版 lasgroup/SDPO 是**直接的监督蒸馏损失**：`loss = D(student_dist ‖ teacher_dist)`，
对 student 的**整个分布**反向传播 → 梯度强、方向明确（把 student 分布拉向 teacher）。

## 技术可行性（已验证）

- `policy_loss_function(args, batch, logits, ...)` 的 `logits` 是**当前训练前向的 grad-enabled** `[1,T,V]`（pg_loss 就靠它反向）。✅
- `_gather_true_on_policy_full_logits` 基于 `_ReplicatedLossAllGatherLastDim`（`torch.autograd.Function`，
  backward 正确处理 TP 梯度缩放）→ gather 全词表后梯度能回流到 TP 分片 logits。✅
- 因此可在 `policy_loss_function` 内：全词表 log_softmax(grad) → 在 teacher top-k ids 上
  gather student log-prob(grad) → 算分布 KL/JSD → 加进总 loss → 反向到 student logits。✅

## 目标 target

```
loss_kd = sdpo_kd_coef × mean_over_response_tokens( D( P_student ‖ P_teacher ) )
```
- `P_student`：当前策略(无 prefix)在该位置、在 **teacher top-k ids + tail** 上的分布，**带梯度**
- `P_teacher`：当前策略(有 prefix)在该位置、top-k ids 上的分布，**detached target**（no_grad 预计算）
- `D`：`--sdpo-divergence` ∈ {reverse_kl, forward_kl, jsd, jeffrey}
- token 集：**teacher 的 top-k**（KD 惯例：以 teacher 分布为 target 定 support）+ 一个 tail bucket
  - 注：与之前 advantage 版用 student top-k 不同——KD 里 teacher 是 target，用 teacher top-k 更自然
- 纯蒸馏：不再用 GRPO advantage（reward=0 那套），loss 只有 kd（+ 可选 entropy/kl_loss 观测）

## 改动分解（5 处）

### A. teacher target 生产（actor.py `_compute_sdpo_teacher_log_probs`）
- 保留 prefix 序列构造 + teacher forward，但**只算 teacher top-k**（detached），
  存 `rollout_data["sdpo_teacher_topk_logprobs"]`（[R,k]）、`["sdpo_teacher_topk_ids"]`（[R,k]）。
- **不再**算 student top-k、**不再**写 `opd_reverse_kl`/`teacher_log_probs`。
- 无 prefix 的样本：teacher target 置空（该样本 kd loss = 0）。
- CPU 张量存储（[R,k] 小），随 rollout_data 走。

### B. 数据透传
- `train_data_conversion.py`：新增 `sdpo_teacher_topk_logprobs`/`_ids` 已在 rollout_data 里
  （actor 直接写 rollout_data，训练同进程，无需跨 ray 序列化）——确认 actor 写入点在
  `get_batch` 取 batch 之前。实际上 actor 在 train_actor 内先算 teacher target 再 forward_backward，
  同一个 rollout_data，天然可见。**关键**：需让 `get_batch`/`forward_step` 把这两个 key
  按 microbatch 切分后放进 `batch`，供 `policy_loss_function` 读取。
- 若 get_batch 只透传固定 key 列表 → 把两个新 key 加入。

### C. KD loss（losses.py `policy_loss_function`）
- 新增分支：`if getattr(args, "sdpo_kd_loss", False) and "sdpo_teacher_topk_ids" in batch:`
  1. 对每个样本，用 grad-enabled `logits` 经 `get_responses` 取 response 段 logits chunk
  2. `_gather_true_on_policy_full_logits` → `log_softmax`（grad）
  3. 在 teacher top-k ids 上 `gather` → student logp（grad），`tail_s = 1 - sum(exp)`
  4. teacher target：`P_teacher = exp(teacher_topk_logprobs)` + tail（detached）
  5. `kd = D(P_student, P_teacher)` per token，`sum_of_sample_mean`
  6. `loss = sdpo_kd_coef × kd_loss`（+ entropy/kl_loss 观测项，纯蒸馏下 pg_loss 不参与或置 0）
- 复用 losses.py 已有的 `sum_of_sample_mean`、per-sample reduction。

### D. 关掉 advantage-hook SDPO
- run 脚本：去掉走 advantage 的 `--use-opd`（或保留 use_opd 但 opd_kl_coef=0），
  改用新的 `--sdpo-kd-loss --sdpo-kd-coef 1.0`。
- 或：`--use-opd` 仍用于插桩，但 SDPO KD 独立于 advantage 注入 loss。
  倾向：**独立 KD**，最清晰。pure_distill 仍设 reward=0（advantage=0），KD 提供全部梯度。

### E. 新参数（arguments.py）
- `--sdpo-kd-loss`（bool，开启分布 KD loss 模式）
- `--sdpo-kd-coef`（float，默认 1.0，KD loss 权重）

## 关键技术点 / 风险

1. **梯度只应流经 student**：teacher target 必须 `.detach()`（且本就是 no_grad forward 的 CPU 张量）。
2. **TP 全词表 gather 的显存**：`log_softmax` 全词表 [R, 151936] grad 版比 no_grad 重。
   response_len 可能几千 → [几千, 151936] float32 grad → 显存大。
   **缓解**：分 chunk（复用 `log_probs_chunk_size`），或只在 teacher top-k ids 上算（但 log_softmax 需全词表分母）。
   → 用 `fused_vocab_parallel` 或 chunked log_softmax 控制峰值。**这是最大风险点**。
3. **token 集选择**：KD 用 teacher top-k（target support）。student 在这些 id 的概率可能很小 →
   tail 处理要对称。数值上 eps 保护。
4. **microbatch 对齐**：teacher target 按样本存，`get_batch` 切 microbatch 时要同步切分，
   和 logits 的样本顺序一致。
5. **backward 显存**：KD loss 的 grad 经全词表 log_softmax 回流，激活比纯 pg_loss 大 →
   可能需再降 max-tokens 或 topk。

## 验证计划

1. 单测：给定 student logits + teacher target，手算 KD loss + grad 方向，对拍。
2. 数值：一轮跑，看 `grad_norm` 是否明显大于之前的 0.04、`kd_loss` 是否随训练下降。
3. 对比：`sdpo/success_rate` 应开始上升（之前平在 0.46）。
4. 不破坏：普通 GRPO / OPD 不受影响（KD 分支 gated on --sdpo-kd-loss）。

## 工作量

- A（teacher target）：中（改现有方法，去掉 student 侧）
- B（透传）：小
- C（KD loss）：中大（grad 全词表 log_softmax + 分布散度 + chunk 控显存）← 核心+风险
- D/E（参数/配置）：小
- 验证：中

**最大不确定性 = C 的显存**（grad 全词表 log_softmax）。若峰值太高，退路是
top-k 近似的 grad log_softmax（只对 teacher top-k+采样邻域算），但精度略降。
