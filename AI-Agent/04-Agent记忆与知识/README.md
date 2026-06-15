# 04 — Agent 记忆与知识

## 一句话开场

> Agent 和你聊完天，下次再来怎么还记得你上周说的事？它查完资料，下次能直接引用不用再搜一遍？它犯了个错，下次能不再犯？——这三个问题分别对应 Agent 的短期记忆、外部知识库和长期经验学习。

## 正文：渐进式理解

**第一层：问题定义。** 基础 ReAct Agent 是"无状态"的：每次对话从零开始，没有过去、没有经验。在真实场景中，Agent 需要：① 记住当前对话上下文（短期记忆）；② 能访问大量外部知识（知识库/RAG）；③ 能从过去的经验中学习（长期记忆 + 反思）。记忆与知识模块解决的问题就是：**Agent 怎么"记住"和"学习"？**

**第二层：核心直觉。** 想象一个新同事入职的第一天 vs. 来了半年。第一天的同事：不了解业务、不熟悉系统、不知道谁负责什么——这就是无记忆的 Agent。半年后的同事：知道常见问题、知道找谁问、知道容易踩什么坑——这就是有记忆的 Agent。记忆系统的目标就是加速这个"从生手到老手"的过程，而且是用自动化方式。

**第三层：方案细节。** Agent 的记忆系统可以分成三个层次：

1. **短期记忆（Short-term）**：当前对话的上下文，由 LLM 的 Context Window 天然提供。限制是窗口大小（4K~128K tokens），超出则遗忘最早的内容。
2. **长期记忆（Long-term / Episodic）**：跨会话的知识积累。通常用 Embedding + 向量数据库实现：Agent 把重要信息存储为向量，下次相关场景自动检索。记忆的内容可以是：事实（"用户偏好 X"）、经验（"上次做 Y 踩过坑"）、知识（"系统文档"）。
3. **反思记忆（Reflective）**：比"记住事实"更高层——Agent 从多个经验中抽象出模式。Generative Agents 的"反思"是典型的：Agent 在一天结束时总结"我今天做了三件事 → 我的身份是个教授 → 我明天应该备课"。

**第四层：不同方案的权衡。**

| 记忆层次 | 实现方式 | 优点 | 代价 | 代表工作 |
|---------|---------|------|------|---------|
| **短期记忆（上下文）** | 原生 Context Window | 零成本，天然支持 | 窗口有限，长对话溢出 | — |
| **长期记忆（向量检索）** | Embedding + Vector DB | 跨会话持久化，相似度检索 | 检索质量依赖 Embedding；存储成本 | RAG / MemGPT |
| **反思记忆（抽象总结）** | LLM 二次总结 | 提炼出可迁移经验 | 计算成本高；可能过度抽象 | Generative Agents |
| **结构化知识记忆** | Graph DB / 知识图谱 | 关系查询精确 | 构建和维护成本高 | GraphRAG |

**一个贯穿所有方案的设计轴：记忆的精度 vs. 检索的灵活性。** 精确记忆（知识图谱）检索准确但灵活度低；模糊记忆（向量检索）灵活但可能检索到不相关内容。

**第五层：总结升华。** 记忆系统是 Agent 从"一次性工具"进化为"持续伙伴"的关键。没有记忆的 Agent 每次从头算起——有记忆的 Agent 能积累经验、迭代优化。重要的是理解"记忆不是存储，而是检索"：记忆系统的瓶颈不是存了多少，而是**在正确的时候检索到正确的信息**。Reflexion（02 模块）和 Generative Agents（03 模块）的成功，很大程度上归功于它们巧妙设计的记忆检索机制。

---

## 学习目标

读完你能：

- **用一句话说清 Agent 三层记忆的区别**：短期是"对话框"，长期是"笔记本"，反思是"日记本"
- **面对一个需要外部知识的 Agent 需求，能判断用 RAG 还是微调**：知识动态变化？→ RAG。知识固定且高频使用？→ 微调
- **理解 Generative Agents 的检索机制的三个信号**：时效性、重要性、相关性
- **实现一个简单的向量记忆系统**：Embedding → 存储 → 检索 → 注入上下文
- **能解释 MemGPT 的"虚拟上下文管理"思想**：把 LLM 的 Context Window 当作"主存"，向量库当作"磁盘"，需要时换入换出

---

## 精选论文

**Lewis et al. (2020) "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"**

- **一句话定位**：RAG 的提出——在生成前先检索相关知识，是 Agent 知识注入的基础范式
- **阅读重点**：第 3 节（RAG 框架：Query Encoder → Retriever → Generator）。图 1 就够理解全部
- **时间分配建议**：10 分钟理解框架即可。实验细节不需要深究。核心是"检索+生成"的端到端架构
- **与本模块的关系**：RAG 是 Agent 访问外部知识的标准方法，影响后续所有记忆系统设计

**Packer et al. (2023) "MemGPT: Towards LLMs as Operating Systems"**

- **一句话定位**：把 LLM 的 Context Window 类比为"主存"，向量数据库为"磁盘"，实现虚拟内存管理
- **阅读重点**：第 3 节（MemGPT 架构：主上下文 → 滑动窗口 → 外部存储）。图 1 非常清晰
- **时间分配建议**：框架设计精读，实验数据略读。核心思想是"分层存储 + 换入换出"的操作系统类比
- **与本模块的关系**：连接短期记忆和长期记忆的桥梁——用操作系统概念优雅地解决了"窗口溢出"问题

**Madaan et al. (2023) "Self-Refine: Iterative Refinement with Self-Feedback"**

- **一句话定位**：Agent 自己对输出反思并迭代改进——"自我批判"的记忆循环
- **阅读重点**：第 2 节（Self-Refine 循环：Generate → Feedback → Refine）。理解"反馈如何作为临时记忆"参与下一轮生成
- **时间分配建议**：快速阅读，核心思想简单。重点理解"反馈也是记忆的一种形式"
- **与本模块的关系**：展示了 Agent 如何利用"当前输出的反馈"作为短期记忆来迭代优化

---

## 拓展阅读

- **Zhong et al. (2024) "GraphRAG: Unlocking LLM Discovery on Narrative Private Data"** — 微软的知识图谱版 RAG。如果你遇到"多个实体之间的关系检索"的问题，可以看这篇。
- **Modarressi et al. (2023) "MemPrompt: An Interactive Memory Model for Large Language Models"** — 让记忆成为 Prompt 的一部分，而不是系统组件。如果你对"轻量级记忆方案"感兴趣可以翻翻。

> 拓展论文不移除，放在 `04-Agent记忆与知识/拓展/` 文件夹下。

---

## 模块间连接

- **前置依赖**：先理解 01-Agent 基础范式的 ReAct 循环（记忆系统是在 ReAct 基础上添加"回忆"能力）。02-推理与规划中的 Reflexion 是反思记忆的先行版
- **后续衔接**：读完后建议进入 **03-多智能体系统**（Generative Agents 记忆是核心组件）或 **06-安全与前沿**（记忆系统的隐私和安全问题）
- **本模块与哪些模块正交**：05-工程框架与协议（工程实现与记忆策略独立），02-推理与规划（记忆增强推理但不依赖记忆）

---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| MemGPT: Towards LLMs as Operating Systems | MemGPT (2023) | [arXiv](https://arxiv.org/abs/2310.08560) |
| RAG Lewis 2020 | RAG (2020) | [arXiv](https://arxiv.org/abs/2005.11401) |
| Self-Refine: Iterative Refinement with Self-Feedback | Self (2023) | [arXiv](https://arxiv.org/abs/2303.17651) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| GraphRAG: Unlocking LLM Discovery on Narrative Private Data | [arXiv](https://arxiv.org/abs/2404.16130) |
| MemGPT: Towards LLMs as Operating Systems | [arXiv](https://arxiv.org/abs/2310.08560) |
| MemPrompt: A Memory-Augmented Prompting Framework | [arXiv](https://arxiv.org/abs/2205.08149) |
| RAG Lewis 2020 | [arXiv](https://arxiv.org/abs/2005.11401) |
| Self-Refine: Iterative Refinement with Self-Feedback | [arXiv](https://arxiv.org/abs/2303.17651) |
