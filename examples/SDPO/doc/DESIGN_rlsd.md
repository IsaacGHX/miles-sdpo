# 设计方案：RLSD（RLVR with Self-Distillation，arXiv:2604.03128）

> **状态：已实现。** 本文档是实现前的设计记录，保留供了解决策动机与权衡；
> 下面每个改动点后面标出了对应的实际代码位置。
>
> - `--sdpo-rlsd` / `--sdpo-rlsd-clip-eps` / `--sdpo-rlsd-lambda-init` /
>   `--sdpo-rlsd-lambda-warmup-steps`：
>   `miles/utils/arguments.py`（RLSD 参数块，`add_on_policy_distillation_arguments`
>   内，`--sdpo-response-prefix` 之后）+ 校验逻辑（`validate_args`，与 `--sdpo-kd-loss`/
>   `--use-opd` 互斥的断言）
> - teacher sampled-token logprob 生产：`miles/backends/megatron_utils/actor.py:616`
>   `_compute_sdpo_teacher_log_probs` 的 `sampled` 分支（`distribution_mode=False`），
>   写入 `rollout_data["teacher_log_probs"]`
> - RLSD advantage reweighting 本体：`miles/backends/training_utils/loss_hub/rlsd.py`
>   `apply_rlsd_credit_to_advantages`，在 `miles/backends/training_utils/loss.py`
>   `compute_advantages_and_returns` 里于 `--use-opd` 之后调用
> - skill-KD 解耦（保持 skill 部分不变的约束）：`actor.py` 的 sampled 分支里补种空的
>   `sdpo_teacher_topk_*` 占位再调用 `_append_sdpo_skill_samples`；
>   `losses.py` 的 `run_kd` 改为 `sdpo_kd_loss OR sdpo_skill_kd`

## 一行公式

```
Â_t = A · [ (1-λ) + λ·clip( exp( sign(A)·sg(logP_T(y_t)-logP_S(y_t)) ), 1-ε_w, 1+ε_w ) ]
```

直接替换 GRPO 原本均匀的 `Â_t = A`，走标准 PPO/GRPO clipped surrogate；无额外 loss 项。

`P_T`/`P_S` 都是同一次 rollout 训练开始前的无梯度快照（`P_S` = 训练侧 `log_probs`，
即 `policy_loss_function` 里的 `old_log_probs`；`P_T` = SDPO Megatron self-teacher
sampled-mode forward），完全不涉及"训练侧当前这一步 vs 生成这次 rollout 的推理引擎"
之间的漂移——那条漂移轴已经由代码库既有的 `--use-tis`
（`corrections.py:vanilla_tis_function`，`tis = exp(train_log_probs - rollout_log_probs)`，
乘进 `pg_loss`）覆盖，RLSD 与其正交、可直接叠加，不需要（也不应该）在 `rlsd.py` 里
再算一份同样的漂移修正——早期实现在这里误加了一份重复的 IS 项，已删除。

## 与 SDPO KD-loss 的区别

- **SDPO（`--sdpo-kd-loss`）**：加法蒸馏 loss，`loss += kd_coef·D(P_student‖P_teacher)`，
  teacher 的特权信息（correct-peer prefix）直接进入梯度**方向**——错误 trace 的 token 也会被拉向
  teacher 偏好的分布（OPSD 的信息泄露问题，见论文 §3）。
- **RLSD（`--sdpo-rlsd`）**：乘法重加权 GRPO advantage，方向永远只由 `sign(A)`（真实 task reward）
  决定；teacher 的证据比 `P_T/P_S` 只调节同一条（已定方向的）轨迹内部各 token 的**相对幅度**。

## 关键实现决策

1. **只需 sampled-token teacher logprob，不需要 top-k 分布**：`delta_t` 只用到采样 token 一个标量，
   复用 `_compute_sdpo_teacher_log_probs` 的 `sampled`/`sdpo_logprob_mode=sampled` 分支（原本是
   legacy OPD 路径共用的单次 forward），比 KD-loss 的 top-k 分支更省一次 top-k gather。
2. **不在 `rlsd.py` 里重复做 train-vs-rollout 的 IS 修正**：`delta_t` 全程 `.detach()`，只作为标量
   权重乘在 advantage 上，不引入额外反传路径。曾经在这里额外加过一份
   `exp(student_logp - rollout_logp).clamp(max=is_clip)`——但这正是代码库已有的 `--use-tis`
   （`corrections.py:vanilla_tis_function`）在算的同一个量、同一个漂移轴（训练侧当前 forward
   vs 生成这次 rollout 的推理引擎之间的差异），两者若同时打开会对同一个漂移做两次修正。已删除,
   这条轴交给 `--use-tis` 单独处理,`rlsd.py` 只负责 teacher-vs-student 的证据比重加权。
3. **λ 衰减是可选的、默认关闭**：论文 Algorithm 1 里 `credit_t=(1-λ)+λ·clip(w_t,...)`，λ 衰减到 0 后
   `credit_t≡1`，RLSD 机制完全失效退化成纯 GRPO——这对短消融（`--num-rollout 100` 配 50 步 warmup）
   意味着后一半训练白跑。因此消融脚本默认 `--sdpo-rlsd-lambda-init 1.0 --sdpo-rlsd-lambda-warmup-steps 0`
   （λ 恒为 1，全程生效），只在专门对比"论文原始衰减 schedule"的那一支（sci-rl 脚本的 arm 2）保留
   `0.5→0/50步` 的论文默认值。
4. **与 skill-KD 解耦**：显式约束"只改非 skill 部分"——skill-KD 的散度计算本来只在 `sdpo_kd_loss`
   分支里触发,若不处理,开 `--sdpo-rlsd` 时会静默丢失 skill-KD 信号。改法:
   `_compute_sdpo_teacher_log_probs` 的 sampled 分支也调用 `_append_sdpo_skill_samples`（response
   span 补种空占位 top-k target,kd=0）,`losses.py` 的 `run_kd` 条件从
   `sdpo_kd_loss` 改为 `sdpo_kd_loss OR (sdpo_skill_kd AND skill_tok_mask is not None)`。
5. **互斥关系**：`--sdpo-rlsd` 与 `--sdpo-kd-loss`/`--use-opd` 三者互斥（同一个 teacher-vs-student
   散度只能选一种消费方式：加法 loss / 加法 KL-in-advantage / 乘法 advantage 重加权），并要求
   `--sdpo-teacher-backend megatron --sdpo-logprob-mode sampled`,在 `arguments.py` 里做了断言校验。
6. **Grading 未改动**：`sdpo_group_reward` 的 `_grade_group` 调用完全不受影响,仍用本 session 已修复的
   grader（dapo 路径 strict-box + `grade_answer_verl` fallback）。
