# 03 — 多智能体系统

## 一句话开场

> 一个 Agent 写代码、另一个 Agent 审查代码、第三个 Agent 写测试——它们怎么分工、怎么沟通、怎么确保最终产品不是一团糟？这就是多智能体系统的核心问题：把"一个人的战斗"变成"一个团队的协作"。

## 正文：渐进式理解

**第一层：问题定义。** 单 Agent 的瓶颈很明显：一个 LLM 的视角有限，容易陷入局部最优，而且缺乏自我纠错的能力（自己写的 bug 自己很难发现）。多智能体系统的思路是：**多个专业化的 Agent 各司其职，通过结构化对话协作完成复杂任务。**

**第二层：核心直觉。** 想一想工作中怎么和同事协作的：有人负责总体设计（planner），有人负责具体执行（worker），有人负责检查质量（critic）。如果让一个人同时做这三件事，他可能会忽略自己的错误。多 Agent 协作就是把这个分工搬到 AI 系统里。

**第三层：方案细节。** 多智能体系统有四种典型的协作模式：① **对话式**——两个 Agent（Executor + Critic）来回对话迭代改进（AutoGen）；② **角色扮演式**——多个 Agent 各司其职，通过结构化流程协作（MetaGPT）；③ **市场式**——多个 Agent 各自独立工作，结果发布到共享"黑板"由协调 Agent 汇总（AgentVerse）；④ **层级式**——Manager Agent 分配任务给多个 Worker Agent。

**第四层：不同方案的权衡。**

| 模式 | 代表工作 | 优点 | 代价 | 适用场景 |
|------|---------|------|------|---------|
| **对话式** | AutoGen | 实现简单，灵活度高 | 对话可能发散 | 代码审查、迭代改进 |
| **角色扮演式** | MetaGPT | 流程清晰，输出结构化 | 灵活性差 | 软件开发、文档生成 |
| **市场式** | AgentVerse | 并行度高，探索多样 | 结果整合困难 | 创意生成、方案比选 |
| **层级式** | 自定义实现 | 控制力强，安全可控 | Manager 单点瓶颈 | 大型项目 |

**第五层：总结升华。** 多智能体系统是 Agent 领域最有"想象力"的方向——它把 Agent 从"工具"扩展为"虚拟团队"。但实际应用中，不是所有任务都适合多 Agent。一个常见陷阱是"为了多 Agent 而多 Agent"——很多任务一个 Agent 配合好的 Prompt 就能解决。

---

## 学习目标

读完你能：

- 用一句话说清多 Agent 协作的核心优势：专业分工 + 交叉检查 + 并行执行
- 面对一个新任务，能判断该用哪种多 Agent 模式
- 用 AutoGen 搭建一个简单的多 Agent 对话系统
- 理解 Generative Agents 中 Agent 的日常决策循环：感知 → 记忆检索 → 计划 → 行动
- 指出多 Agent 系统的主要失败模式：对话发散、角色冲突、信息冗余、成本失控

---

## 精选论文

**Park et al. (2023) "Generative Agents: Interactive Simulacra of Human Behavior"** — [arXiv](https://arxiv.org/abs/2304.03442)

- **一句话定位**：Stanford 小镇——25 个 Agent 在模拟世界中生活、社交、工作
- **阅读重点**：第 3-4 节（记忆流 → 检索 → 反思 → 计划 → 社交）
- **时间分配建议**：第 3 节精读（架构设计），第 5 节精读（涌现行为）
- **与本模块的关系**：迄今为止最完整的 Agent 框架

**Wu et al. (2023) "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"** — [arXiv](https://arxiv.org/abs/2308.08155)

- **一句话定位**：微软多 Agent 框架，支持灵活对话拓扑
- **阅读重点**：第 3 节（Agent 对话模式和终止条件）
- **时间分配建议**：快速浏览框架设计，配合 demo 运行体验
- **与本模块的关系**：多 Agent 系统最重要的实用框架

**Hong et al. (2023) "MetaGPT: Meta Programming for Multi-Agent Collaborative Framework"** — [arXiv](https://arxiv.org/abs/2308.00352)

- **一句话定位**：角色扮演式多 Agent 的集大成者
- **阅读重点**：第 3 节（角色定义和 SOP 流程）
- **时间分配建议**：关注流程设计而非实现细节
- **与本模块的关系**：展示了"角色扮演 + 结构化流程"的威力

---

## 拓展阅读

- **Li et al. (2023) "AgentVerse: Facilitating Multi-Agent Collaboration"** [arXiv](https://arxiv.org/abs/2308.10848) —— 市场式多 Agent 框架
- **Qian et al. (2023) "ChatDev: Communicative Agents for Software Development"** [arXiv](https://arxiv.org/abs/2307.07924) —— 对话式多 Agent 做软件开发
- **Gudibande et al. (2023) "The False Promise of Imitating Proprietary LLMs"** [arXiv](https://arxiv.org/abs/2305.15717) —— 讨论 Agent 的能力边界
- **Multi-Agent Survey (2024)** [arXiv](https://arxiv.org/abs/2402.01680) —— LLM 多智能体综述

> 拓展论文放在各模块的 拓展/ 文件夹下。

---

## 模块间连接

- **前置依赖**：先理解 01-Agent 基础范式。建议先读 04-Agent 记忆与知识
- **后续衔接**：读完后建议进入 05-工程框架与协议 或 06-安全与前沿
- **本模块与哪些模块正交**：02-推理与规划