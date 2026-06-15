# 04 — 连续控制与最大熵：从 DDPG 到 SAC

## 一句话开场

> DQN 和 PPO 都能告诉你"该按左键还是右键"(离散动作)，但你开车时方向盘打多少度（连续动作）它们做不到。机器人关节角度、自动驾驶中的油门开度、化学反应的温度调节——这些真实世界的控制问题都需要连续的输出。

---

## 正文：渐进式理解

**第一层：问题定义。** 现实世界的决策问题绝大多数需要输出连续值（扭矩、角度、温度、位移）。值函数方法（DQN）天然只能做离散动作（取 argmax 需要遍历所有动作）；策略梯度方法（PPO）虽理论上可输出连续的均值和方差，但其 Actor-Critic 实现在连续动作空间中的探索效率和稳定性有限。**核心问题：如何设计一个能在连续动作空间中高效探索并稳定收敛的 RL 算法？**

**第二层：核心直觉。** 想象你学开车——方向盘是连续角度，不是左/中/右三档。一开始你不可能知道打 30 度还是 35 度好，只能"大致试试"。这里有两个关键直觉：① 你不需要学所有动作的 Q 值（连续空间也枚举不完），而是学一个 Actor（输出动作）+ 一个 Critic（评价动作）——Actor 负责提方案，Critic 负责打分；② 你不想太早锁定某个固定角度——你要保持一点"随机性"（熵）来持续探索。最大熵原则背后的直觉就是："在所有能达到同一奖励水平的策略中，选那个最分散、最不武断的。"

**第三层：方案细节。** 这条路线有三个里程碑：**DDPG**（Lillicrap 2016）把 DQN 的架构改造成 Actor-Critic——Actor 网络输出连续动作，Critic 网络输出 Q 值，用 DQN 的 Replay + Target 技巧稳定训练。**TD3**（Fujimoto 2018）诊断出 DDPG 的"高估偏差"并做了三项修补：Clipped Double Q-Learning（用两个 Critic 取小值）、延迟 Actor 更新、Target Policy Smoothing（在目标动作上加噪声）。**SAC**（Haarnoja 2018）走了一条更根本的改进路线：优化目标从"最大化回报"变成"最大化回报 + α × 策略熵"——强制策略保持探索，并且自动调节 α 这个温度系数。

**第四层：不同方案的权衡。**

| 维度 | DDPG | TD3 | SAC |
|------|------|-----|-----|
| 核心思想 | DQN 的 Actor-Critic 化 | 修复 DDPG 的高估偏差 | 最大熵框架 |
| 探索机制 | OU 噪声加在动作上 | 高斯噪声 | 熵正则化，天然探索 |
| 稳定性 | 较差（超参数敏感） | 较好 | 最好（自动调温） |
| 实现复杂度 | 低 | 中 | 中高 |
| 实际地位 | 历史里程碑 | 工程改进的典范 | 连续控制事实 SOTA |

**一个贯穿的设计红线：探索-利用的权衡。** DDPG 靠外部噪声探索，TD3 通过修剪 Q 值间接约束，SAC 直接在最优化目标里加入熵——三者代表了"从工程手段到理论框架"的演进。

**第五层：总结升华。** 连续控制是把 RL 从"游戏 AI"推向"物理世界"的必经之路。SAC 的最大熵框架和 03 的 KL 约束（TRPO/PPO）异曲同工——都是"别让策略坍缩太快"。区别在于：KL 约束限制策略更新的步幅（跨时间），最大熵限制策略的瞬时确定性（同一时刻）。两者在 LLM 对齐场景中都有体现——PPO 用 KL penalty，而偏好优化（DPO）从另一个角度实现了类似的约束效果。

---

## 学习目标

读完你能：

- 能画出 Actor-Critic 在连续控制中的更新循环图：Actor 输出动作 → Critic 评估 Q 值 → 两个 Loss 交替更新
- 用一句话说清 DDPG 和 DQN 的根本区别（DQN 输出 Q 值取 argmax vs DDPG 的 Actor 直接输出动作）
- 能解释 TD3 为什么需要两个 Critic 并取小值（双 Q 缓解高估）
- 能写出 SAC 的优化目标并说清熵项的作用（最大化回报 + 维持探索）
- 面对一个连续控制问题，能选择 DDPG / TD3 / SAC 并说明理由

---

## 精选论文

**Lillicrap et al. (2016) "Continuous Control with Deep Reinforcement Learning" (DDPG)**

- **一句话定位**：将 DQN 扩展到连续动作空间的首篇工作，Actor-Critic 连续控制的起点
- **阅读重点**：第 3 节——算法伪代码 + Actor/Critic 网络设计
- **时间分配建议**：必读。精读第 3 节（算法），第 4 节（实验）扫读即可
- **与本模块的关系**：回答了"连续动作空间怎么做 RL"

**Fujimoto et al. (2018) "Addressing Function Approximation Error in Actor-Critic Methods" (TD3)**

- **一句话定位**：系统诊断并修复 DDPG 高估问题的工程典范
- **阅读重点**：第 3-4 节——高估的理论分析 + Clipped Double Q-Learning
- **时间分配建议**：选读。重点读第 3 节（两个 Q 网络的设计直觉），约 15 分钟
- **与本模块的关系**：回答了"DDPG 的高估问题怎么修"

**Haarnoja et al. (2018) "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor" (SAC)**

- **一句话定位**：最大熵框架下的连续控制 SOTA，自动调温让探索不再依赖超参数调优
- **阅读重点**：第 3-4 节——最大熵目标推导 + 自动温度调节
- **时间分配建议**：必读。建议精读第 3 节（最大熵目标 + SAC 算法的设计逻辑），约 30 分钟
- **与本模块的关系**：回答了"如何在连续控制中实现高效且稳定的探索"

---

## 拓展阅读

- **Silver et al. (2014) "Deterministic Policy Gradient Algorithms"** — DPG 的原始论文，从理论层面证明了确定性策略梯度定理，是 DDPG 的理论基础。如果你对"为什么 Actor 可以输出确定性动作还能做策略梯度"感兴趣可以翻翻。


> 拓展论文不移除，放在各模块的 `拓展/` 文件夹下。核心论文在模块根目录。
---

## 模块间连接

- **前置依赖**：01-基础理论（Actor-Critic 框架的基础直觉）、02-深度Q网络（DDPG 本质是 DQN 的 Actor-Critic 化）
- **后续衔接**：05-RL×LLM（SAC 的最大熵思想在探索策略中有启发意义）
- **本模块与哪些模块正交**：与 06-离线RL（聚焦在线交互 vs 离线数据）互为正交，可以独立阅读
---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Continuous control with deep reinforcement learning | DDPG () | [arXiv](https://arxiv.org/abs/1509.02971) |
| Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor | SAC () | [arXiv](https://arxiv.org/abs/1801.01290) |
| Addressing Function Approximation Error in Actor-Critic Methods | TD3 () | [arXiv](https://arxiv.org/abs/1802.09477) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| Continuous control with deep reinforcement learning | [arXiv](https://arxiv.org/abs/1509.02971) |
| Deterministic Policy Gradient Algorithms | — |
| Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor | [arXiv](https://arxiv.org/abs/1801.01290) |
| Addressing Function Approximation Error in Actor-Critic Methods | [arXiv](https://arxiv.org/abs/1802.09477) |
