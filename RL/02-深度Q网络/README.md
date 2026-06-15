# 02 — 深度 Q 网络：从 DQN 到 Rainbow

## 一句话开场

> 你下围棋时想"下一步走哪最好"——传统 Q-Learning 用一个巨大的表格记录每个位置的价值。但围棋有 10¹⁷⁰ 种状态，银河系所有原子加起来都不够存这个表——你怎么办？

---

## 正文：渐进式理解

**第一层：问题定义。** 01 的 MDP 框架假设状态空间小到可以用表格存储 Q 值。但真实世界的状态空间无限大（像素、语音、文本），表格 Q-Learning 崩溃。**核心问题：如何用参数化函数（神经网络）来近似 Q 函数，同时保证训练稳定？**

**第二层：核心直觉。** 想象你在玩 Atari 游戏，屏幕像素就是状态。你不能枚举所有像素排列，但你可以训练一个神经网络：输入像素帧，输出每个动作的 Q 值估计。问题在于：RL 的样本是时序相关的（连续几帧高度相似），直接像监督学习那样更新，网络会"学疯掉"——刚学的被下一批几乎一样的样本覆盖。DQN 的直觉是用一个"经验池"存下过去的经历，从中随机抽样打乱相关性——就像错题本随机翻阅而不是死记硬背同一道题。

**第三层：方案细节。** DQN 有三项关键设计：① **Experience Replay**：把 (s, a, r, s') 存进回放缓冲区，随机采样小批量训练，打破时序相关性；② **Target Network**：用固定参数的旧网络计算 Q_target，每隔 N 步才更新一次——减少 Q 网络和 target 之间的耦合振荡；③ **奖励裁剪**：把奖励缩放到 [-1, 1] 之间，提高数值稳定性。但 Q_target = r + γ·max Q(s', a') 中的 max 操作会导致系统性高估（取 max 多个噪声估计 = 高估值）。**Double DQN** 的修复：用当前网络选动作（argmax），用目标网络算 Q 值——解耦选择与评估。

**第四层：不同方案的权衡。** DQN 之后出现了多个改进方向，Rainbow 将它们整合：

| 变体 | 核心思想 | 解决的问题 | 代价 |
|------|---------|-----------|------|
| **Double DQN** | 解耦动作选择和价值评估 | DQN 的高估偏差 | 几乎为 0，开箱即用 |
| **Dueling DQN** | 将 Q 分解为 V(s) + A(s, a) | 某些状态下动作选择不重要 | 网络输出多一倍 |
| **Prioritized Replay** | 按 TD-error 大小加权采样 | 均匀采样效率低 | 额外排序开销 |
| **Rainbow** | 整合以上所有改进 | 单个改进的边际收益递减 | 实现复杂，超参数多 |

**第五层：总结升华。** DQN 在 2015 年首次证明了"深度网络 + RL"能解决复杂问题。它的三大设计（Replay、Target、Clip）至今仍被广泛使用。但它暴露了值函数方法的两大根本局限：离散动作和高估偏差——前者引出策略梯度（03），后者引出 Actor-Critic 架构的持续改进（04）。

---

## 学习目标

读完你能：

- 能画出 DQN 的完整训练循环图：Environment → Replay Buffer → Q-Network → Loss → Gradient Update
- 用一句话说清 Experience Replay 为什么是必要的（打破时序相关性）
- 用一个具体例子（如 CartPole）说清 Target Network 如何稳定训练
- 能推导出 Double DQN 的 Q_target 计算式，说清它为什么能修正高估
- 面对一个新的控制问题，能判断 DQN 是否适合（离散动作 vs 连续动作）

---

## 精选论文

**Mnih et al. (2015) "Human-level Control through Deep Reinforcement Learning" (DQN)**

- **一句话定位**：深度 RL 的开山之作，CNN + Experience Replay + Target Network 在 Atari 上超越人类
- **阅读重点**：第 3-4 节——算法伪代码 + 三大设计原理
- **时间分配建议**：必读。建议精读第 3 节（算法设计），第 4 节可扫读（实验结果）
- **与本模块的关系**：回答了"怎么用神经网络替代 Q 表"

**Van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-Learning"**

- **一句话定位**：诊断并修复 DQN 高估偏差的简短有力论文
- **阅读重点**：第 3-4 节——高估的理论分析 + Double DQN 算法
- **时间分配建议**：必读，约 15 分钟能读完核心公式
- **与本模块的关系**：回答了"DQN 留下什么麻烦，怎么修正"

**Hessel et al. (2018) "Rainbow: Combining Improvements in Deep Reinforcement Learning"**

- **一句话定位**：整合 DQN 六个改进方向的集成论文
- **阅读重点**：第 3 节——各改进的消融对比表
- **时间分配建议**：选读，重点看第 3 节的 Table 1（消融实验）
- **与本模块的关系**：提供 DQN 改进路径的全局视野

---

## 拓展阅读

- **Wang et al. (2016) "Dueling Network Architectures for Deep Reinforcement Learning"** — 将 Q 值分解为状态值 + 动作优势，提高某些场景下的学习效率。如果你对 DQN 架构改进感兴趣可以翻翻。


> 拓展论文不移除，放在各模块的 `拓展/` 文件夹下。核心论文在模块根目录。
---

## 模块间连接

- **前置依赖**：01-基础理论（MDP + Bellman 方程 + Q-Learning）
- **后续衔接**：03-策略梯度与信任域（DQN 的局限→直接优化策略的范式切换）、04-连续控制与最大熵（DQN 只支持离散动作，04 扩展到连续空间）
- **本模块与哪些模块正交**：与 05-RL×LLM（应用场景不同）基本正交，但 Double DQN 的"解耦"思想在 GRPO 中也有体现（优势估计与值函数解耦）
---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Human-level control through deep reinforcement learning | DQN () | [arXiv](https://arxiv.org/abs/1312.5602) |
| Deep Reinforcement Learning with Double Q-learning | DoubleDQN () | [arXiv](https://arxiv.org/abs/1509.06461) |
| Rainbow: Combining Improvements in Deep Reinforcement Learning | Rainbow () | [arXiv](https://arxiv.org/abs/1710.02298) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| Human-level control through deep reinforcement learning | [arXiv](https://arxiv.org/abs/1312.5602) |
| Deep Reinforcement Learning with Double Q-learning | [arXiv](https://arxiv.org/abs/1509.06461) |
| Dueling Network Architectures for Deep Reinforcement Learning | [arXiv](https://arxiv.org/abs/1511.06581) |
| Rainbow: Combining Improvements in Deep Reinforcement Learning | [arXiv](https://arxiv.org/abs/1710.02298) |
