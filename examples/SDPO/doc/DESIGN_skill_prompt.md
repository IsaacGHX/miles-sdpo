# SDPO 正确-trace skill 生成 prompt:两个版本

记录用来把「答对的 trace」蒸馏成 skill 的 prompt。两版都只针对 **correct** 分支
(`_skill_user_prompt` + `_SKILL_SYSTEM_PROMPT`);incorrect 分支产出的是 pitfall
警告,不在此文档范围内。

- **当前版**(`38da9d38e` 起):instance-grounded 长 roadmap(6-10 步)。
- **旧版**(`763b20486`,即 `38da9d38e^`):通用短 bullet(≤3 条)。

两版的核心差异是「实例锚定 vs 通用可迁移」:旧版禁止出现本题的具体量/数值,
当前版反过来要求引用本题的关键量、定理、判别性比较和中间结果——只在最后一步
之前停下,不给出最终答案。

---

## 当前版:instance-grounded roadmap(6-10 步)

### system prompt

```
You distill a worked solution into a SKILL: a concrete solution ROADMAP that a
capable solver could follow to reach the correct answer to THIS problem on their
own. It is grounded in this specific problem — name the key quantities, the
governing relation/theorem to apply, the decisive facts or comparisons that
discriminate the right choice from the wrong ones, and the key intermediate
results along the way — but it stops one step short of the final answer.

Hard constraints:
- Be a clear numbered roadmap: 6-10 short steps, each an imperative instruction.
- Instance-grounded: DO reference this problem's specific quantities, setup, and
the critical intermediate values/comparisons needed to get the answer right.
- Do NOT state the final answer itself (no final letter/number/name, no
'the answer is ...'). Stop at the last step BEFORE committing to the answer, so
the reader must still perform the final selection/computation themselves.
- Output ONLY the numbered steps, nothing else.
```

### user prompt

```
PROBLEM:
{problem}

WORKED SOLUTION (reference, do not echo):
{solution}

Write the solution roadmap (6-10 numbered steps, instance-grounded, stop one
step before the final answer).
```

`{problem}` 经过 `_clean_problem_for_skill` 去掉答案格式脚手架
(如 "put your answer in \boxed{}"),避免答案格式泄漏进 skill。

---

## 旧版:通用短 bullet(≤3 条)

### system prompt

```
You distill a worked solution into a SKILL: the minimal set of transferable,
procedural know-how needed to solve problems of this kind. A skill is NOT the
answer and NOT a step-by-step derivation of this instance. It is generic
procedure: what to recognize in the problem, which concepts/techniques/
theorems/conventions to apply and in what order, common pitfalls to avoid, and
any output-format requirement.

Hard constraints:
- Be EXTREMELY compressed: at most 3 bullets, each a short imperative phrase.
- Procedural and reusable, never instance-specific. No numbers, names, or
intermediate/final values from this particular problem.
- Never reveal or hint at the final answer.
- Output ONLY the bullets (prefixed with '- '), nothing else.
```

### user prompt

```
PROBLEM:
{problem}

WORKED SOLUTION (reference, do not echo):
{solution}

Distill the transferable skill (<=3 terse procedural bullets).
```

---

## 对照

| 维度 | 旧版(短 bullet) | 当前版(roadmap) |
| --- | --- | --- |
| 形式 | ≤3 条 `- ` bullet,短祈使短语 | 6-10 步编号 roadmap,每步一条祈使指令 |
| 实例锚定 | **禁止**出现本题数字/名字/中间值 | **要求**引用本题关键量、设置、判别性中间值 |
| 抽象层级 | 通用可迁移 know-how(this kind of problem) | 针对 THIS problem 的具体解题路线 |
| 答案 | 不透露、不暗示 | 不给最终答案,在最后一步之前停住 |
| 目的 | 泛化技巧,轻量 prefix | 强 teacher hint(privileged info 更接近可解) |

来源:`examples/SDPO/sdpo.py`(`_SKILL_SYSTEM_PROMPT` / `_skill_user_prompt`)。
