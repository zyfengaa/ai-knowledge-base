# 03 — 多智能体系统

## 一句话开场

> 一个 Agent 写代码、另一个 Agent 审查代码、第三个 Agent 写测试——它们怎么分工、怎么沟通、怎么确保最终产品不是一团糟？这就是多智能体系统的核心问题：把"一个人的战斗"变成"一个团队的协作"。

## 正文：渐进式理解

**第一层：问题定义。** 单 Agent 的瓶颈很明显：一个 LLM 的视角有限，容易陷入局部最优，而且缺乏自我纠错的能力（自己写的 bug 自己很难发现）。多智能体系统的思路是：**多个专业化的 Agent 各司其职，通过结构化对话协作完成复杂任务**——就像软件开发不是一个人写所有代码，而是 PM + 开发 + 测试 + 运维各司其职。

**第二层：核心直觉。** 想一想你工作中怎么和同事协作的：① 有人负责总体设计（planner）；② 有人负责具体执行（worker）；③ 有人负责检查质量（critic）。如果让一个人同时做这三件事，他可能会忽略自己的错误（确认偏误）。多 Agent 协作就是把这个分工搬到 AI 系统里：每个 Agent 有自己的角色设定和工具集，通过对话/消息传递来协同。

**第三层：方案细节。** 多智能体系统有四种典型的协作模式：

1. **对话式（Conversational）**：两个 Agent（Executor + Critic）来回对话，迭代改进。AutoGen 的 AssistantAgent + UserProxyAgent 模式
2. **角色扮演式（Role-based）**：多个 Agent 各自扮演专业角色（产品经理、架构师、开发者），通过结构化流程协作。MetaGPT 是典型代表
3. **市场式（Market/Blackboard）**：多个 Agent 各自独立工作，结果发布到共享"黑板"，由协调 Agent 汇总。适合头脑风暴/创意类任务
4. **层级式（Hierarchical）**：一个 Manager Agent 负责分配任务给多个 Worker Agent，Worker 只向 Manager 汇报。适合有明确分工的任务

**第四层：不同方案的权衡。**

| 模式 | 代表工作 | 优点 | 代价 | 适用场景 |
|------|---------|------|------|---------|
| **对话式** | AutoGen | 实现简单，灵活度高 | 对话可能发散，需要终止条件 | 代码审查、迭代改进 |
| **角色扮演式** | MetaGPT | 流程清晰，输出结构化 | 角色设定影响大，灵活性差 | 软件开发、文档生成 |
| **市场式** | AgentVerse | 并行度高，探索多样 | 结果整合困难 | 创意生成、方案比选 |
| **层级式** | 自定义实现 | 控制力强，安全可控 | Manager 单点瓶颈 | 任务明确的大型项目 |

**一个贯穿所有模式的设计轴：Agent 自治度 vs. 人类可控性。** Agent 越自治（自己决定做什么、怎么协作），效率越高，但越可能偏离预期。

**第五层：总结升华。** 多智能体系统是 Agent 领域最有"想象力"的方向——它把 Agent 从"工具"扩展为"虚拟团队"。但实际应用中，多 Agent 的效果高度依赖任务性质和角色设计：不是所有任务都适合多 Agent。一个常见的陷阱是"为了多 Agent 而多 Agent"——实际上很多任务一个 Agent 配合好的 Prompt 就能解决。多 Agent 的真正价值在于：① 引入不同视角（批判性检查）；② 并行执行独立子任务；③ 模拟复杂交互（社交/经济模拟）。

---

## 学习目标

读完你能：

- **用一句话说清多 Agent 协作的核心优势**：专业分工 + 交叉检查 + 并行执行
- **面对一个新任务，能判断该用哪种多 Agent 模式**：需要迭代改进？→ 对话式。需要结构化输出？→ 角色扮演式。需要多样性？→ 市场式
- **用 AutoGen 搭建一个简单的多 Agent 对话系统**：定义角色 → 设定工具 → 启动对话
- **理解 Generative Agents 中 Agent 的日常决策循环**：感知 → 记忆检索 → 计划 → 行动
- **能指出多 Agent 系统的主要失败模式**：对话发散、角色冲突、信息冗余、成本失控

---

## 精选论文

**Park et al. (2023) "Generative Agents: Interactive Simulacra of Human Behavior"**

- **一句话定位**：Stanford 小镇——25 个 Agent 在模拟世界中生活、社交、工作，展示了 Agent 行为的涌现
- **阅读重点**：第 3-4 节（Agent 架构：记忆流 → 检索 → 反思 → 计划 → 社交）。图 2 是整个论文的核心
- **时间分配建议**：第 3 节精读（架构设计），第 5 节精读（涌现行为），实验数据略读。这篇论文很长但可读性高
- **与本模块的关系**：迄今为止最完整的 Agent 框架，影响后续几乎所有工作

**Wu et al. (2023) "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"**

- **一句话定位**：微软开源的实用多 Agent 框架，支持灵活的对话拓扑
- **阅读重点**：第 3 节（Agent 对话模式和终止条件）。第 4 节（coding/问答等案例）
- **时间分配建议**：快速浏览框架设计，重点关注对话终止机制（多 Agent 中最难的问题）。配合 `multiagent_collab.py` 运行体验
- **与本模块的关系**：多 Agent 系统最重要的实用框架，连接学术研究和工程实践

**Hong et al. (2023) "MetaGPT: Meta Programming for Multi-Agent Collaborative Framework"**

- **一句话定位**：角色扮演式多 Agent 的集大成者——让不同角色 Agent 协作输出完整的软件工程文档
- **阅读重点**：第 3 节（角色定义和 SOP 流程）。理解 "产品需求文档 → 架构设计 → 技术方案 → 代码实现" 的自动化流程
- **时间分配建议**：关注流程设计而非具体实现细节。MetaGPT 的价值在于"把软件工程 SOP 翻译给 Agent"
- **与本模块的关系**：展示了"角色扮演 + 结构化流程"在多 Agent 中的威力

---

## 拓展阅读

- **Li et al. (2023) "AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors"** — 市场式多 Agent 框架，强调 Agent 群体的自组织行为。如果你对"Agent 自发协作"感兴趣可以看这篇。
- **Qian et al. (2023) "ChatDev: Communicative Agents for Software Development"** — 对话式多 Agent 做软件开发的早期工作。ChatDev 和 AutoGen 思路接近，但聚焦编程场景。

> 拓展论文不移除，放在 `03-多智能体系统/拓展/` 文件夹下。

---

## 模块间连接

- **前置依赖**：先理解 01-Agent 基础范式（多 Agent 的每个成员都是一个 ReAct Agent）。建议也先读 04-Agent 记忆与知识（Generative Agents 的核心是记忆系统）
- **后续衔接**：读完后建议进入 **05-工程框架与协议**（多 Agent 框架的工程实现在这里有更系统的讨论）或 **06-安全与前沿**（多 Agent 的安全风险更复杂）
- **本模块与哪些模块正交**：02-推理与规划（多 Agent 不依赖具体推理策略，可以各自用不同策略）

---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation | AutoGen (2023) | [arXiv](https://arxiv.org/abs/2308.08155) |
| Generative Agents: Interactive Simulacra of Human Behavior | Generative (2023) | [arXiv](https://arxiv.org/abs/2304.03442) |
| MetaGPT: Meta Programming for Multi-Agent Collaborative Framework | MetaGPT (2023) | [arXiv](https://arxiv.org/abs/2308.00352) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors | [arXiv](https://arxiv.org/abs/2305.04091) |
| AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation | [arXiv](https://arxiv.org/abs/2308.08155) |
| ChatDev: Communicative Agents for Software Development | [arXiv](https://arxiv.org/abs/2307.07924) |
| The False Promise of Imitating Proprietary LLMs | [arXiv](https://arxiv.org/abs/2304.03442) |
| Generative Agents: Interactive Simulacra of Human Behavior | [arXiv](https://arxiv.org/abs/2304.03442) |
| MetaGPT: Meta Programming for Multi-Agent Collaborative Framework | [arXiv](https://arxiv.org/abs/2308.00352) |
| A Survey on Multi-Agent Systems in Large Language Models | [arXiv](https://arxiv.org/abs/2402.01680) |
