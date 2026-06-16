﻿# 05 — 应用技术

> 模型的知识停在训练时——怎么让它回答「今天天气怎么样」或者解一道从未见过的数学题？

## 正文：渐进式理解

**第一层：问题定义。** 一个纯 LLM 有三大致命弱点：① **知识陈旧**——训练集截止日期后的事情一概不知；② **缺乏推理链**——复杂问题（数学 / 逻辑 / 多步规划）直接输出容易出错；③ **容易幻觉**——模型不知道「不知道」，对于不知道的事实会自信地编造。这三种问题性质不同，需要三种互补的技术：RAG 解决知识更新，CoT 解决推理能力，基础 Prompt Engineering 解决交互方式。

**第二层：核心直觉。**
- **RAG（Retrieval-Augmented Generation）** = 给 LLM 配一个搜索引擎/知识库。先检索相关文档，再把文档和问题一起发给 LLM——这样模型每次回答问题都是「开卷考试」
- **CoT（Chain-of-Thought）** = 让 LLM 先写草稿再给答案。不是在出题时就逼它给答案，而是说「我们先一步一步想」——把推理过程显式化
- **Prompt Engineering** = 学会跟 LLM 打交道的「沟通技巧」——不同措辞方式会得到完全不同质量的回答

**第三层：方案细节。**

**RAG 标准管线**：
`
User Query → Embedding → Vector DB 检索（Top-K）→ 检索结果 + Query → LLM 生成 → 最终回答
                                                                     ↑
                                                            （知识注入，缓解幻觉）
`

**CoT 的主要变体**：

| 变体 | 做法 | 适用场景 |
|------|------|---------|
| Zero-shot CoT | 加一句"Let's think step by step" | 通用推理 |
| Few-shot CoT | 给 2-3 个推理示例再提问 | 需要格式约束 |
| Self-Consistency | 多次 CoT 采样，投票选最一致的答案 | 高可靠性需求 |
| Tree-of-Thoughts (ToT) | 并行探索多条推理路径 + 剪枝 | 复杂规划问题 |

**第四层：不同方案的权衡。**

| 维度 | RAG | Fine-tuning | Prompt Engineering |
|------|-----|-------------|-------------------|
| 解决的问题 | 知识更新 / 幻觉 | 格式 / 风格 / 领域适配 | 推理 / 任务适配 |
| 是否需要训练 | ❌ 不需要 | ✅ 需要 | ❌ 不需要 |
| 数据需求 | 知识库文档 | 100+ 标注样本 | 几个示例 |
| 成本 | 中（需要向量数据库） | 高（算力 + 标注） | 低（零成本） |
| 适合场景 | 事实性问答 / 知识密集 | 格式规范 / 风格迁移 | 通用推理 / 快速验证 |
| 不能解决的问题 | 复杂推理 / 格式适配 | 知识更新 | 深度的领域适配 |

> **推荐组合**：RAG + Prompt Engineering 做第一层（零成本、快速迭代），Fine-tuning 做第二层（当格式 / 风格需求固化后）。

**第五层：总结升华。** RAG 和 CoT 从不同方向弥补 LLM 的天生缺陷：**RAG 解决「不知道的」，CoT 解决「想不清楚的」**。两者完全互补——RAG 给模型装数据库，CoT 给模型装思考过程。一个有经验的 LLM 应用开发者会同时用这两套工具，再加一层 Fine-tuning 做风格定型。

---

## 学习目标

读完你能：

- 画出 RAG 的完整管线（Query → Embedding → Retrieval → Augment → Generate）并解释每一步做什么
- 用一句话说清 RAG 和 Fine-tuning 的核心分工（RAG 管知识、Fine-tuning 管风格）
- 给一个具体任务场景，判断应该用 Zero-shot CoT、Few-shot CoT 还是 Self-Consistency
- 解释为什么 RAG 只能「缓解」幻觉而不能「根除」幻觉——因为 LLM 仍有概率忽略检索到的信息
- 在实际项目中按照 RAG → Prompt Engineering → Fine-tuning 的顺序构建应用（复杂度递增，收益递减）

---

## 精选论文

**Lewis et al. (2020) "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" [[arXiv](https://arxiv.org/abs/2005.11401)]**

- **一句话定位**：RAG 的提出论文，检索 + 生成的融合范式，至今是 LLM 落地的核心架构
- **阅读重点**：第 2 节（RAG 的两种变体——RAG-Sequence 和 RAG-Token 的区别）和第 3 节（实验验证）
- **时间分配建议**：时间紧只读第 2 节理解 RAG 是什么（Figure 1 的架构图胜过千言万语）；时间充裕精读第 4 节（参数化记忆 vs 非参数化记忆的讨论）

**Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" [[arXiv](https://arxiv.org/abs/2201.11903)]**

- **一句话定位**：CoT 思维链，第一次证明 LLM 可以通过「逐步推理」显著提升复杂问题解决能力
- **阅读重点**：第 2 节（CoT 的定义和示例——Table 1 的标准 Few-shot vs CoT Few-shot 对比非常直观）和第 3 节（实验——CoT 在 GSM8K 和 MAWPS 上的巨大提升）
- **时间分配建议**：实验细节可以跳读，核心是理解 CoT 的「为什么有效」——引导模型生成中间推理步骤

---

## 模块间连接

- **前置依赖**：建议先读 **01-Transformer 起源**（理解模型的基本工作方式）和 **04-推理与部署优化**（理解部署的约束）。本模块的 RAG 和 CoT 是「怎么用」的问题，前序模块是「为什么模型长这样」
- **后续衔接**：读完本模块后推荐进入 **06-前沿方向**——Agent 和 MLLM 是应用技术的自然延伸（RAG + Tool Use → Agent；Text-only → MLLM）
- **本模块与哪些模块正交**：与 03-训练与对齐范式完全正交——「怎么训」和「怎么用」是两个独立维度


---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | CoT () | [arXiv](https://arxiv.org/abs/2201.11903) |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | RAG () | [arXiv](https://arxiv.org/abs/2005.11401) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | [arXiv](https://arxiv.org/abs/2201.11903) |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | [arXiv](https://arxiv.org/abs/2005.11401) |
