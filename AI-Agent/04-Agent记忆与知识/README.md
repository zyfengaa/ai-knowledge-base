# 04 — Agent 记忆与知识

## 一句话开场

> Agent 和你聊完天，下次再来怎么还记得你上周说的事？它查完资料，下次能直接引用不用再搜一遍？它犯了个错，下次能不再犯？——这三个问题分别对应 Agent 的短期记忆、外部知识库和长期经验学习。

## 正文：渐进式理解

**第一层：问题定义。** 基础 ReAct Agent 是"无状态"的：每次对话从零开始，没有过去、没有经验。在真实场景中，Agent 需要：① 记住当前对话上下文（短期记忆）；② 能访问大量外部知识（知识库/RAG）；③ 能从过去的经验中学习（长期记忆 + 反思）。

**第二层：核心直觉。** 想象一个新同事入职的第一天 vs. 来了半年。第一天的同事不了解业务、不熟悉系统——这就是无记忆的 Agent。半年后的同事知道常见问题、知道容易踩什么坑——这就是有记忆的 Agent。记忆系统的目标就是加速这个"从生手到老手"的过程。

**第三层：方案细节。** Agent 记忆系统分三个层次：① **短期记忆**——当前对话上下文，由 LLM 的 Context Window 提供；② **长期记忆**——跨会话知识积累，通常用 Embedding + 向量数据库实现，存储事实、经验、知识；③ **反思记忆**——从多个经验中抽象出模式（Generative Agents 的 Agent 在一天结束时总结"我今天做了三件事 → 我明天应该备课"）。

**第四层：不同方案的权衡。**

| 记忆层次 | 实现方式 | 优点 | 代价 | 代表工作 |
|---------|---------|------|------|---------|
| **短期记忆**（上下文） | 原生 Context Window | 零成本 | 窗口有限 | — |
| **长期记忆**（向量检索） | Embedding + Vector DB | 跨会话持久化 | 检索质量依赖 Embedding | RAG / MemGPT |
| **反思记忆**（抽象总结） | LLM 二次总结 | 可迁移经验 | 计算成本高 | Generative Agents |
| **结构化知识** | Graph DB / 知识图谱 | 关系查询精确 | 构建维护成本高 | GraphRAG |

**第五层：总结升华。** 记忆系统是 Agent 从"一次性工具"进化为"持续伙伴"的关键。重要的不是"存了多少"，而是**在正确的时候检索到正确的信息**。Reflexion 和 Generative Agents 的成功，很大程度上归功于它们巧妙设计的记忆检索机制。

---

## 学习目标

读完你能：

- 用一句话说清 Agent 三层记忆的区别：短期是"对话框"，长期是"笔记本"，反思是"日记本"
- 面对需要外部知识的 Agent 需求，能判断用 RAG 还是微调
- 理解 Generative Agents 检索机制的三个信号：时效性、重要性、相关性
- 实现一个简单的向量记忆系统：Embedding → 存储 → 检索 → 注入上下文
- 解释 MemGPT 的"虚拟上下文管理"思想：Context Window 当作"主存"，向量库当作"磁盘"

---

## 精选论文

**Lewis et al. (2020) "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** — [arXiv](https://arxiv.org/abs/2005.11401)

- **一句话定位**：RAG 的提出——在生成前先检索相关知识
- **阅读重点**：第 3 节（RAG 框架：Query Encoder → Retriever → Generator）
- **时间分配建议**：10 分钟理解框架即可
- **与本模块的关系**：RAG 是 Agent 访问外部知识的标准方法

**Packer et al. (2023) "MemGPT: Towards LLMs as Operating Systems"** — [arXiv](https://arxiv.org/abs/2310.08560)

- **一句话定位**：把 Context Window 类比为"主存"，向量库为"磁盘"，实现虚拟内存管理
- **阅读重点**：第 3 节（MemGPT 架构）
- **时间分配建议**：框架设计精读。核心是"分层存储 + 换入换出"
- **与本模块的关系**：连接短期记忆和长期记忆的桥梁

**Madaan et al. (2023) "Self-Refine: Iterative Refinement with Self-Feedback"** — [arXiv](https://arxiv.org/abs/2303.17651)

- **一句话定位**：Agent 自己对输出反思并迭代改进
- **阅读重点**：第 2 节（Self-Refine 循环：Generate → Feedback → Refine）
- **时间分配建议**：快速阅读。核心思想简单
- **与本模块的关系**：展示了"反馈作为临时记忆"的迭代优化

---

## 拓展阅读

- **Zhong et al. (2024) "GraphRAG: Unlocking LLM Discovery on Narrative Private Data"** [arXiv](https://arxiv.org/abs/2404.16130) —— 知识图谱版 RAG
- **Modarressi et al. (2023) "Memory-Assisted Prompt Editing to Improve GPT-3"** [arXiv](https://arxiv.org/abs/2201.06009) —— 记忆作为 Prompt 的一部分

> 拓展论文放在各模块的 拓展/ 文件夹下。

---

## 模块间连接

- **前置依赖**：先理解 01-Agent 基础范式的 ReAct 循环
- **后续衔接**：读完后建议进入 03-多智能体系统 或 06-安全与前沿
- **本模块与哪些模块正交**：05-工程框架与协议