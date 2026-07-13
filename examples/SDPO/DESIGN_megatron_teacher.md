# 设计方案：SDPO teacher 打分从 SGLang 挪到 Megatron 训练前向

## 目标 / 动机

当前 SDPO（`--opd-type sglang` + self-teacher）把 teacher 打分外包给 SGLang HTTP。
但 SGLang 对「full-sequence logprob」请求强制走 **eager prefill**（`piecewise_cuda_graph_runner.can_run`
在 `return_logprob` 覆盖整条序列时返回 False），8192 token 的 eager prefill 每条几秒。
实测 per-rollout timing：**`teacher_http` 占 84%**，是绝对瓶颈。

原版 SDPO（lasgroup/veRL）把 teacher 打分放在**训练引擎的 RefWorker 批量 forward** 里
（编译过、有 CUDA graph、批量），因此快 ~50x。本方案在 miles 里复刻这个架构：
**用 Megatron 训练前向对「prompt+prefix+response」算 teacher logprob，取代 SGLang HTTP 打分。**

## 现状机制（已确认）

- **rollout→train 数据流**：`convert_samples_to_train_data` 把每个 sample 转成
  `rollout_data`，含 `tokens`（prompt+response）、`response_lengths`、`loss_masks`、
  可选 `teacher_log_probs` / `opd_reverse_kl`（`train_data_conversion.py:32,85-89`）。
- **megatron teacher forward**（`actor.py:355-364`）：`_switch_model("teacher")` →
  `compute_log_prob(data_iterator, store_prefix="teacher_")` → `forward_only(get_log_probs_and_entropy)`
  对 `rollout_data["tokens"]` 批量 forward，logprob 取 response 段。teacher 是
  **独立固定权重**（`load_other_checkpoint("teacher", opd_teacher_load)`）。
- **训练侧消费**（`opd.py:57-88`）：sampled 模式下 `reverse_kl = student_log_probs[i] - teacher_log_probs[i]`，
  逐 token，`advantages -= opd_kl_coef * reverse_kl`。已是现成下游。

## 核心难点

1. **teacher 输入序列不同**：teacher 要看 `prompt + prefix + response`，
   student 看 `prompt + response`。两者 response 段在尾部，逐 token 对齐（SDPO 的
   alignment guarantee）。→ 需要为 teacher 构造**单独的带 prefix token 序列**。
2. **self-teacher = 当前 actor 权重**：miles megatron teacher 是独立固定模型；SDPO 要
   用 actor 自己的当前权重对 prefix 序列 forward（不 switch 到独立 teacher）。
3. **prefix 是动态选的**：每条轨迹随机选一个 correct peer 作 prefix，需在 group RM 阶段
   选好并传到训练侧。

## 方案（分阶段）

### 阶段 A：rollout 侧只选 prefix，不打分（轻量）

改 `examples/SDPO/sdpo.py`：
- `sdpo_group_reward` 保留 `_is_correct` 选 correct peer 的逻辑，但**不再调 `_teacher_score`（HTTP）**。
- 对每条轨迹：选好 prefix peer 后，把 **prefix 的 token ids** 存到 `sample.metadata["sdpo_prefix_tokens"]`
  （render + tokenize prefix，一次性，很轻）。correct=0/1 诊断保留。
- 不再写 `sample.opd_reverse_kl`（改由训练侧算）。返回 reward（纯蒸馏=0）。

### 阶段 B：数据流透传 prefix tokens

改 `train_data_conversion.py`：
- `convert_samples_to_train_data` 增加 `sdpo_prefix_tokens` key（每 sample 的 prefix token 列表）。
- `split_train_data_by_dp` 的透传 key 列表加入 `sdpo_prefix_tokens`（随 partition 切分）。

### 阶段 C：训练侧 teacher forward 走 actor 权重 + prefix 序列（核心）

改 `miles/backends/megatron_utils/actor.py`：
- 新增分支：当 `args.opd_type == "sglang"` 且 SDPO self-teacher 且有 `sdpo_prefix_tokens` 时，
  **不 switch teacher 模型**，而是用**当前 actor 权重**对「prompt+prefix+response」序列
  做一次 `forward_only(get_log_probs_and_entropy)`：
  1. 为 teacher 构造临时 rollout_data：token 序列 = `prompt + prefix + response`，
     response_length 不变（logprob 只取尾部 response 段，与 student 对齐）。
  2. `get_data_iterator` 组 batch → `forward_only` → 取 response 段 logprob，
     存 `rollout_data["teacher_log_probs"]`。
- 关键对齐：teacher 序列尾部的 response 段和 student 的 response 段**逐 token 一一对应**
  （因为 response 在尾部，prefix 只加在中间）。位置数 = response_length，与 student logprob 同形状。

### 阶段 D：训练侧 KL 计算复用现有 opd.py

- `opd.py` 的 sampled 路径已支持 `teacher_log_probs`，直接复用：
  `reverse_kl = student_log_probs - teacher_log_probs`。
- SDPO 的 JSD/top-k 分布散度**在此方案下退化为 sampled-token reverse KL**
  （因为 megatron forward 高效地给出的是采样 token 的 logprob，不是 top-k 分布）。
  → **这是方案的一个取舍**：换到 megatron forward 后，散度粒度从 top-k 分布降为 sampled-token。
  若要保留分布级 KL，需 teacher forward 额外输出 top-k logits（更复杂，阶段 E）。

## 关键取舍（需你确认）

**megatron self-teacher forward 天然给「采样 token 的 logprob」（和 student 同形状），
最自然的散度是 sampled-token reverse KL（= 原 OPD 的默认路径）。**
- 若接受 sampled-token KL：方案止于阶段 D，改动可控，速度追上原版。
- 若坚持 top-k 分布 JSD：需 teacher forward 额外返回每位置 top-k logits，
  并在训练侧做分布散度，改动更大（阶段 E，暂不展开）。

原版 lasgroup/SDPO 用 `full_logit_distillation` + `distillation_topk`，是分布级；
但它在训练引擎里一次 forward 就能拿到 full logits（不额外开销）。miles 的
`get_log_probs_and_entropy` 默认只返回采样 token logprob，要 top-k 需改 forward 输出。

## 风险 / 验证

- **对齐正确性**：必须断言 teacher 序列尾部 response token == student response token（逐位置），
  和 SGLang 版一样的 alignment guarantee。写单测覆盖。
- **显存**：teacher forward 多一遍（prompt+prefix+response 比 prompt+response 长），
  且用 actor 权重（不额外加载模型）。prefix 通常不长，激活增量可控；配合已有 recompute。
- **不破坏现有 OPD**：改动用 `opd_type==sglang && sdpo` 分支隔离，megatron OPD / 普通 GRPO 不受影响。
- **数值对齐**：先跑一轮对比 sampled-token reverse KL 与旧 SGLang 版的 opd_reverse_kl 量级。

## 工作量估计

- 阶段 A/B：小（~50 行，sdpo.py + train_data_conversion.py）
- 阶段 C：中大（~100-150 行，actor.py 新增 teacher-via-actor forward 分支 + 序列构造 + 对齐）
- 阶段 D：极小（复用 opd.py）
- 测试/验证：中（对齐单测 + 一轮数值对比）

**若接受 sampled-token KL 取舍，总体是「中等功能开发」，可实现，速度可追上原版。**
**若要 top-k 分布 KL，还需阶段 E（改 forward 输出 top-k logits），工作量翻倍。**
