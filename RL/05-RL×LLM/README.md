# 05 — RL × LLM：从 RLHF 到 GRPO 的对齐与推理训练

## 一句话开场

> 你费了很大劲训练了一个 GPT，它什么都能生成，但有时候会输出有害内容。你不能直接写一个数学公式作为"友善对话"的奖励——但你可以比较两个回答哪个更好。怎么把这种人类的偏好比较变成模型可以优化的目标？

---

## 正文：渐进式理解

**第一层：问题定义。** 大语言模型（LLM）经过预训练后能合理续写文本，但预训练目标（next token prediction）与"有用、诚实、无害"的人类期望之间存在巨大鸿沟。RL 提供了弥合这一鸿沟的框架：把 LLM 的输出当作"动作"，把人类反馈当作"奖励"，用 RL 算法优化模型。**核心问题：人类的定性偏好如何转化为可量化的奖励信号？传统的 RL 算法（PPO）在大模型场景下有哪些不适应？**

**第二层：核心直觉。** RLHF 的核心直觉非常优雅：① 不要让人写奖励公式——让人做"比较题"（A 比 B 好还是 B 比 A 好），从比较中训练一个 Reward Model；② 用 PPO 最大化 Reward Model 的打分，但同时加一个 KL penalty 约束模型不能偏离原始版本太远（这直接对应 03 的信任域思想）。但 PPO 需要 critic（值函数模型），在 LLM 场景下这意味着再训练一个和 policy 一样大的模型——计算开销翻倍。GRPO 的直觉更简单：对同一个 prompt 生成多个回答，在这些回答的 reward 分布中做归一化——组内得分相对高低就是"优势"，不需要 critic 了。

**第三层：方案细节。** RLHF（Christiano 2017 → InstructGPT 2022）的完整链路：① 收集人类偏好数据（回答 A vs 回答 B）；② 用 Bradley-Terry 模型训练 Reward Model R(s) = log[σ(r_a - r_b)]；③ 用 PPO 最大化 E[R(s)] - β·KL(π||π_ref)，其中 KL penalty 保证模型不偏离初始版本。**GRPO**（Shao 2024）的核心创新是"去掉 critic"：对同一 prompt 生成 G 个回答 → 计算组内 reward 的均值和标准差 → advantage = (r_i - mean) / std → 用 REINFORCE 方式更新策略 + KL penalty。**DPO**（Rafailov 2023）更进一步：不用训练 Reward Model，直接在偏好损失上优化——π* = argmax E[log σ(β·(r(x, y_w) - r(x, y_l)))]，用闭式解把 RL objective 变成简单的分类损失。

**第四层：不同方案的权衡。**

| 维度 | RLHF (PPO) | GRPO | DPO |
|------|-----------|------|-----|
| 依赖 critic | ✅ 需要，参数量翻倍 | ❌ 不需要，组内归一化 | ❌ 不需要 |
| 依赖 reward model | ✅ 需要额外训练 | ✅ 需要（可验证奖励） | ❌ 不需要，直接偏好优化 |
| KL 约束 | ✅ PPO 自带 KL penalty | ✅ 显式 KL penalty | ❌ 隐式（约束在 loss 中） |
| 适用场景 | 通用偏好对齐 | 可验证奖励场景（数学/代码） | 通用偏好对齐 |
| 计算开销 | 高（4 模型：policy + critic + reward + ref） | 中（3 模型：policy + reward + ref） | 低（2 模型：policy + ref） |
| 实际地位 | InstructGPT/ChatGPT 基础 | DeepSeek-R1 训练核心 | 学术界常用简化方法 |

**第五层：总结升华。** 这条演进线的本质是：**在 LLM 场景下不断"脱掉"传统 RL 中不需要的组件。** PPO 需要 critic（游戏场景下奖励噪声大，需要 critic 做价值估计）；但 LLM 的可验证奖励（数学题对错、代码编译正确）信号足够干净——组内归一化就够用。所以 RLOO → GRPO → DAPO 的演进路线是：去掉 critic → 分组归一化 → 工程增强。这与 03 的 TRPO→PPO 的简化方向一致：**去掉复杂，保留必要。**

---

## 学习目标

读完你能：

- 能画出 RLHF 的完整链路图：Human Preference → Reward Model → PPO → Policy Update
- 用一句话说清 KL penalty 在 RLHF 中的作用（约束 policy 不离 reference model 太远，对应 03 的信任域思想）
- 能写出 GRPO 的 advantage 计算式：advantages = (rewards - mean) / std，并说清为什么不需要 critic
- 能用对比表说明 RLHF(PPO)、GRPO、DPO 三者的核心区别（有没有 critic / reward model / 适用场景）
- 面对一个 LLM 对齐任务，能判断用 RLHF 还是 DPO 更适合

---

## 精选论文

**Christiano et al. (2017) "Deep Reinforcement Learning from Human Preferences"**

- **一句话定位**：RLHF 的起源，首次证明人类偏好可通过对比训练转化为可优化的奖励信号
- **阅读重点**：第 3 节——偏好从何而来 + 奖励模型训练
- **时间分配建议**：必读。精读第 3 节（算法核心），第 4 节（Atari 实验）可扫读
- **与本模块的关系**：回答了"人类定性偏好如何变成定量奖励"

**Ouyang et al. (2022) "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT)**

- **一句话定位**：将 RLHF 流程工程化并在 GPT-3 上验证的工程里程碑
- **阅读重点**：第 3 节——三阶段训练流程（SFT → RM → PPO）
- **时间分配建议**：必读。重点读第 3 节的 Fig.2（整体流程图），约 20 分钟
- **与本模块的关系**：回答了"RLHF 在 LLM 上的完整工程实现"

**Rafailov et al. (2023) "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (DPO)**

- **一句话定位**：用分类损失替代 RL 循环的对齐新范式，引发广泛讨论
- **阅读重点**：第 3-4 节——DPO 损失推导 + 与 PPO 的对比
- **时间分配建议**：选读。重点读第 3 节（DPO 目标函数推导），约 30 分钟
- **与本模块的关系**：提供了 RLHF 的替代路线——不用 RL 也能做对齐

**Shao et al. (2024) "DeepSeekMath: Pushing the Limits of LLM Math Reasoning" (GRPO)**

- **一句话定位**：去掉 critic 的分组归一化方法，成为 DeepSeek-R1 的训练核心
- **阅读重点**：第 2.3 节——GRPO 算法伪代码 + 组内 advantage 计算
- **时间分配建议**：必读。精读第 2.3 节（GRPO 算法），约 15 分钟
- **与本模块的关系**：回答了"如何在 LLM 场景下简化 PPO"

---

## 拓展阅读

- **Ayoub et al. (2023) "RLOO: Leave-One-Out Advantage Estimation for LLM Alignment"** — GRPO 的前身，用 Leave-One-Out 基线替代 critic。如果你想理解"去掉 critic"这个思路的演进，建议翻翻。
- **Yu et al. (2025) "DAPO: Dynamic Sampling and Decoupled Policy Optimization"** — 2025 年的进一步解耦方案：policy 和 reward 解耦训练、动态采样策略。代表 GRPO 之后的演进方向。


> 拓展论文不移除，放在各模块的 `拓展/` 文件夹下。核心论文在模块根目录。
---

## 模块间连接

- **前置依赖**：03-策略梯度与信任域（PPO 算法 + KL 约束思想，05 中 PPO 带 KL penalty 就是 03 信任域的直接延续）
- **后续衔接**：06-离线RL与前沿挑战（RL×LLM 中提到的"可验证奖励"思路与离线学习的结合方向）
- **本模块与哪些模块正交**：与 01（理论底座）和 02（DQN）相对独立，可直接阅读；与 03 有强依赖关系
---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model | DPO () | [arXiv](https://arxiv.org/abs/2305.18290) |
| DeepSeekMath: Pushing the Limits of Mathematical Reasoning with Open Language Models | GRPO () | [arXiv](https://arxiv.org/abs/2402.03300) |
| Training language models to follow instructions with human feedback | InstructGPT () | [arXiv](https://arxiv.org/abs/2203.02155) |
| Deep Reinforcement Learning from Human Preferences | RLHF () | [arXiv](https://arxiv.org/abs/1706.03741) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| DAPO: An Open-Source RL System from Scratch | [arXiv](https://arxiv.org/abs/2503.14442) |
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model | [arXiv](https://arxiv.org/abs/2305.18290) |
| DeepSeekMath: Pushing the Limits of Mathematical Reasoning with Open Language Models | [arXiv](https://arxiv.org/abs/2402.03300) |
| Training language models to follow instructions with human feedback | [arXiv](https://arxiv.org/abs/2203.02155) |
| Deep Reinforcement Learning from Human Preferences | [arXiv](https://arxiv.org/abs/1706.03741) |
| REINFORCE Leave One Out | [arXiv](https://arxiv.org/abs/2303.00276) |
