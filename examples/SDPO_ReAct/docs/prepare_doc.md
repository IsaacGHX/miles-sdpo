请你先观察这里的代码，
examples/SDPO_ReAct
先确认这里的逻辑是不是如下所述的所有的条件，然后帮我按照我后面的指示来完成进一步的训练环境的搭建和运行测试：
我们的目标是能够将其作为我们的 proposal 的一个有力的验证。

# 一、当前实现情况：

TL; DR
TIR (tool intergrated reasoning) 以及在这个基础上的 SDPO 训练。

## A. SDPO pure distill 的训练流程：

LOOP
1. model 按照数据集 rollout
2. 按照长度限制获得 rollout 的结果；rollout 的时候是要求和环境交互的（#label1: 不良的 multiturn 激发性质）
3. 目前这里的实现是 collocate 的也就是说 rollout 和 forward backward 计算都是同时占满8卡的
4. 对这些潜在有截断的输出来进行 reward model 的判别，用的是这里的 miles/rollout/rm_hub/math_utils.py 的 def grade_answer_verl(solution_str, ground_truth): 也就是说他们的匹配非常的精准
5. reward 得到正确的 trace 可以分流出来等待后面放到 prefix（每个trace 的 prefix 都只会是其他的正确的 trace）
6. 这里计算两个的 
forward: teacher(|rollout_{x_{t+1}} 但是只在自己的生成的部分上面 | rollout_{x_t}, prompt, 正确的完整的trace【包括model 自己的 rollout 和 env 的反馈】) 
student(|rollout_{x_{t+1}} 但是只在自己的生成的部分上面 | rollout_{x_t}, prompt) 
我们用的是 JSD 其中的当中的那个m 用的 0.5(student+teacher), 然后 topk KL 用的是 student 和 teacher 的每个token 的前 k 的 union
7. 对于 thinking model 我们所有的输出都会（think+tool call）都会拿出来计算 JSD（目前我们只实现了没有 thinking model 的）
8. student 和 teacher 的 policy 都是每次计算完之后更新梯度，做下一次 rollout 然后在计算 JSD 的时候这一个 forward 是用的 EMA 加权之后的 student 和 teacher
LOOP

目前会 dump 完整的trace 到文件夹

## B. 目前的工具集: 
python，
- docker 实现，
- 里面安装了非常简单的数学包
- 每次 call 都是独立的；随时按照 rollout 异步 call
- 有执行的上限时间 默认 10s
- 返回值如果是超过2k 会做 mid truncate

# 二、我们的目标：
1. 希望能够训练实现：model 能够自发 multi turn 去解决 agentic 的问题，这个步数应该能够很长，长到他自己的context window max
2. 最终能够解决的问题要多样化，math（aime26 可以，AMC 25）、code、cli、deepsearch（至少要能够 multi hop能力大幅度提升）
3. 我们的 proposal 是能够把上述的 A.6 里面的prefix 给他换成当前这个 model 去自己从自己和环境交互里面得到的 skills
4. 我们希望能够证明的：随着“环境”的多样化和复杂度的提升，model 的上限能够提升（这个可以是训练轨迹也可以是其他的）

# 三、我们之前的实验教会我的以及我的直觉的解决方案和方向：
1. #label1: 我们用 qwen2.5 7b instruct 发现他的 multiturn 能力很差 -- 我希望你能够找到一个合适的 baseline，既能够体现性能大幅度、稳步提升，又能够体现出来multiturn、甚至是并行工具调用 -- 我们就只用 qwen 的 model 来做 debug 我个人觉得 qwen3-4b 是一个好的潜在选项
1.1 因为 qwen2.5 在 <answer><code> 这样子的 user 指令遵循很不错但是在类似 qwen3.5/3.6 这样子的 template tool 注入的条件下效果很差
2. 我们的工具不够完善，python 自己我感觉还是有可以改进的地方，其他的工具也都还没有
3. JSD 也许不是一个很好的 target 可以改进，但是风险高；但是我们一定不能回退开倒车变成 +grpo 之类的东西这样子就没意义；不要有其他 model 参与的 SFT
4. 我个人直觉告诉我自己的每一步生成 skill，可能会 distribution drift；
examples/SDPO/run-qwen3-4B-sdpo-math-colocate.sh 这里的用到的代码就都展示了我们的skill 的已经实现的一些设置，你可以去看他们的定义：
--sdpo-self-skill
--sdpo-skill-source all
--sdpo-skill-max-new-tokens 1024
--sdpo-pitfall-summary-backend self
--sdpo-response-prefix skill 
--sdpo-skill-kd
--sdpo-skill-kd-coef 0.01
--sdpo-skill-kd-mode both

skill 的生成和生成 skill 的policy 也都是两个值得商榷的议题：
到底应该拿什么来作为 skill ，也就是 condense 之后的 prefix ？这个 skill 可以是正确的，也可以是错误的，也可以是错误&正确的 mixed；这里的实验我们没测，目前的examples/SDPO/run-qwen3-4B-sdpo-math-colocate.sh 里面的经验是正确的 skills 可以约等于正确的 trace 作为 prefix 加上总结后的错误的也只会微微掉点

5. 我们认为 sdpo 教会model 的应该是 

1.更好的 CoT，2.更好的格式 3. 因为某些问题他看到了完整的详细的解答所以知道了原来不知道的知识所以更加自信也就更答得对

但是如果我们只去优化model看到了生成的 skill 而去优化后面的解答过程，这个是似乎有些错配
但是如果要优化 skill 的生成怎么去更好地设计一个skill 的 sdpo 的目标，也是值得考虑的（这个似乎又看起来是一个 summary 的任务），这里因为没有 正确的答案可以快速判断一个 skill 是不是一个好的skill，如果要的话只能去快速对比加上 skill 和不加 skill 谁的 acc 高，那这个就太浪费 gpu 了；

---
总而言之，就是要去设计一个优雅、简洁、有效的方法，来提升 model 的性能，多任务解答的泛化性、多任务 skill 生成自我反思的泛化性、以及因而潜在的 bootstrapping 的能力 -- hurestic 地去提升 model 的 TTS（testtime scaling）。
---

# 参考文献：
1. https://github.com/lasgroup/SDPO & https://arxiv.org/abs/2601.20802 这是我们的 sdpo 的实现的参考，不用过多看他们的思路和我们一致，超参数也几乎一致，但是他们的数据集只用了sci 和 livecode bench（不知道为什么在这两个上面的效果出气的好，我们验证了只用 正确的 skill 替代 完整的 正确的 trace 在 sci shuffle 时候也能够达到几乎和完整的 trace 一样的效果）
2. https://arxiv.org/abs/2603.24472 （数学和推理问题上 sdpo 不可抑制地下跌，他们认为是太多的 suppression of epistemic verbalization 但是多 epistemic verbalization 又会 verbose 输出带来不成正比的不够多的 acc 提升）
3. https://huggingface.co/datasets/allenai/Dolci-RL-Zero-Math-7B （这里的数学的 shuffle 也许可以参考）


请遵守：
一定要对于所有的实验的保持可扩展：加入新的 数据集、model、judge、新的测试 target、改动skill 生成、增加工具之类的，都能够最小化而且不侵入地修改其他的代码，所以最好基类做得比较好

我要看到 deliver 一个 model 以及他的训练的 wandb 曲线和对应的 setting， support proposal 中的 claim。

放手去做吧。

tips：这个机器是一个 H200 140G 的，他的 cuda version 是 12.8，他上面有一个定时检测如果一致 idle 时间超过一定时间（这里大概是 45min）就会自动开始一个占用 gpu 的任务防止这个节点被拿走、如果你在跑实验的时候发现它存在请用 `oc-off` 来临时关闭；