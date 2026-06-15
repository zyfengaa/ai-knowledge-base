# 03 — 策略梯度与信任域：从 REINFORCE 到 PPO

## 一句话开场

> 值函数方法（DQN）学了 Q 值再从中选动作——这有点像先给每个动作打分再从最高分的选。但能不能**跳过打分**，直接学一个策略："在状态 s 下，我有多大概率做动作 a"？——但你直接沿着梯度更新策略，很容易一步把所有学到的都毁掉。怎么保证更新是稳的？

---

## 正文：渐进式理解

**第一层：问题定义。** 值函数方法有两个固有限制：① 只能处理离散动作（选 Q 值最大的动作）；② Q 值存在高估偏差。策略梯度方法绕过值函数，直接参数化策略 π(a|s) 并用梯度上升最大化累积回报。**新的问题：策略参数的一小步更新可能让策略发生巨大变化（比如从"向左"变成"向右"），导致训练崩溃——你需要在"更新幅度够大"和"更新幅度安全"之间找到平衡。**

**第二层：核心直觉。** 想象你是一个钢琴手在学一首新曲子。每次你改变指法，不能变得太剧烈——如果今天用 1-3-5 明天突然全改成 2-4-5，你的肌肉记忆全废了。你要保证每次练习的指法变化在"可接受的范围"内。TRPO 的思想类似：在策略空间中画一个"信任域"——只有在信任域内才允许策略更新。PPO 更聪明：不显式画边界，而是用 clip 截断——一旦更新超出范围就把它拉回来，简单粗暴但有效。

**第三层：方案细节。** 策略梯度的核心公式是 ∇J(θ) = E[∇log π_θ(a|s) · Q_π(s, a)]。**REINFORCE**（Williams 1992）用蒙特卡洛回报估计 Q，极大方差极大。**TRPO**（Schulman 2015）引入 KL 散度约束：max E[π_θ/π_θ_old · A(s, a)] 满足 KL(π_θ || π_θ_old) ≤ δ——保证更新不离开信任域，但需要用共轭梯度求解二阶优化，实现复杂。**PPO**（Schulman 2017）用 clip 目标函数替代 KL 约束：L_clip = E[min(r_t(θ)·A, clip(r_t, 1-ε, 1+ε)·A)]——超过范围就截断，一阶优化即可，大大简化实现。

**第四层：不同方案的权衡。**

| 维度 | REINFORCE | TRPO | PPO |
|------|-----------|------|-----|
| 稳定性 | 差——方差极大，极易崩溃 | 好——KL 约束理论上界更新量 | 好——Clip 截断足够实用 |
| 实现复杂度 | 极简——十几行代码 | 复杂——共轭梯度 + 二阶近似 | 简单——只需 clip，一阶优化 |
| 计算效率 | 高（ON 步更新一次） | 低（每步需求第二阶） | 高（每步只需一阶） |
| 实际地位 | 教学用的理论起点 | 里程碑但不常用 | 事实标准，RLHF 的基础算法 |

**第五层：总结升华。** 从 DQN 到策略梯度是一次范式切换——从"学值函数"到"直接学策略"。TRPO 用 KL 约束稳定策略更新，PPO 用 clip 简化之，使策略优化成为可落地的技术。**这条约束思想的红线从 TRPO→PPO 直接延续到 RLHF：LLM 对齐中 PPO 带 KL penalty 约束 policy 不离 reference model 太远，本质上就是 TRPO 信任域思想在 LLM 场景的延续。**

---

## 学习目标

读完你能：

- 能写出 Policy Gradient 的核心公式 ∇J(θ) = E[∇log π_θ · Q]，并说清每个符号的含义
- 能画出 TRPO 的 KL 约束目标函数图，说清"信任域"的几何直觉
- 能用一句话说清 PPO 的 Clip 目标函数为什么有效（超过范围就截断）
- 面对一个新问题，能判断用 REINFORCE vs PPO（方差容忍度 vs 实现复杂度）
- 能说清 KL 约束思想从 TRPO 如何延伸到 LLM 的 RLHF 中

---

## 精选论文

**Williams (1992) "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"**

- **一句话定位**：REINFORCE 的原始论文，策略梯度方法的理论起点
- **阅读重点**：第 2-3 节——梯度推导 + 基线（baseline）的引入
- **时间分配建议**：略读，了解核心公式即可（约 10 分钟）
- **与本模块的关系**：回答了"直接优化策略的理论可能性"

**Schulman et al. (2015) "Trust Region Policy Optimization" (TRPO)**

- **一句话定位**：用 KL 散度约束策略更新的里程碑，首次解决策略更新的稳定性问题
- **阅读重点**：第 3-4 节——KL 约束 + 共轭梯度求解思路
- **时间分配建议**：重点读第 3 节（理论直觉），第 5 节（算法）可跳读
- **与本模块的关系**：回答了"如何保证策略更新不崩坏"

**Schulman et al. (2017) "Proximal Policy Optimization Algorithms" (PPO)**

- **一句话定位**：用 clip 简化 TRPO，成为 RLHF 的基础算法
- **阅读重点**：第 3-4 节——Clipped Surrogate Objective + 算法伪代码
- **时间分配建议**：必读。建议精读第 3 节，花约 20 分钟理解 Clip 的直觉
- **与本模块的关系**：回答了"如何让信任域约束变得简单实用"

**Mnih et al. (2016) "Asynchronous Methods for Deep Reinforcement Learning" (A3C)**

- **一句话定位**：引入多线程异步训练的策略梯度方法，Actor-Critic 的代表作
- **阅读重点**：第 2-3 节——A3C 的 n-step 更新 + 多线程架构
- **时间分配建议**：选读。如果时间紧可以跳过，重点了解 Actor-Critic 框架即可
- **与本模块的关系**：展示了策略梯度 + 并行训练的实际工程方案

---

## 拓展阅读

- **Greensmith et al. (2004) "Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning"** — 策略梯度方差缩减的理论分析。如果你想深入理解为什么需要 baseline 以及如何做方差缩减，可以翻翻。


> 拓展论文不移除，放在各模块的 `拓展/` 文件夹下。核心论文在模块根目录。
---

## 模块间连接

- **前置依赖**：01-基础理论（MDP + Bellman + 策略/值函数的基本概念）、02-深度Q网络（了解值函数方法的局限）
- **后续衔接**：04-连续控制与最大熵（PPO 支持离散/连续，但连续场景下 SAC 通常更好）、05-RL×LLM（PPO 是 RLHF 的基础，KL 约束思想贯穿到 GRPO）
- **本模块与哪些模块正交**：与 01（理论底座）有前置依赖；与 06（离线RL）比较独立
---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Asynchronous Methods for Deep Reinforcement Learning | A3C () | [arXiv](https://arxiv.org/abs/1602.01783) |
| Proximal Policy Optimization Algorithms | PPO () | [arXiv](https://arxiv.org/abs/1707.06347) |
| Simple statistical gradient-following algorithms for connectionist reinforcement learning | REINFORCE () | — |
| Trust Region Policy Optimization | TRPO () | [arXiv](https://arxiv.org/abs/1502.05477) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| Asynchronous Methods for Deep Reinforcement Learning | [arXiv](https://arxiv.org/abs/1602.01783) |
| Greensmith VarianceReduction 2004 | — |
| Proximal Policy Optimization Algorithms | [arXiv](https://arxiv.org/abs/1707.06347) |
| Simple statistical gradient-following algorithms for connectionist reinforcement learning | — |
| Trust Region Policy Optimization | [arXiv](https://arxiv.org/abs/1502.05477) |
