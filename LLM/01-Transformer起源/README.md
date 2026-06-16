﻿# 01 — Transformer 起源

> 你拿到一段文本，怎么让模型「同时看到所有词的位置关系」？这就是 Self-Attention 要解决的根本问题。

## 正文：渐进式理解

**第一层：问题定义。** 在 Transformer 之前，序列建模用 RNN/LSTM——每个时间步等上一个输出，像流水线一样串行生产。这有两大致命缺陷：① 无法并行（n 长度的序列必须走 n 步）；② 长距离遗忘（梯度随步数指数衰减）。Attention 在 RNN 时代只是「辅助」——帮模型聚焦重要位置，但序列依赖的瓶颈没解决。

**第二层：核心直觉。** 想象一个会议室里所有人同时说话（所有位置同时交互），而不是一个一个轮流发言（RNN 的串行传递）。Self-Attention 做的就是：让每个词同时「看」所有其他词，直接计算谁和谁最相关。这就是 **「全连接的位置相关性」**——建立所有位置间的「直达通道」，路径长度 O(1)（RNN 需要 O(n)）。

**第三层：方案细节。** Transformer 的核心组件：

`
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
`

1. **Scaled Dot-Product Attention**：Q（查询）× K^T（键）得到注意力分数矩阵，每个位置看每个位置；除以 √d_k 防梯度消失；softmax 归一化；最后乘 V 输出加权信息
2. **Multi-Head**：把 Q/K/V 分成 h 组，每组独立算 Attention，最后拼接——相当于请 h 个专家各自关注不同维度
3. **Encoder-Decoder 结构**：Encoder 做双向理解（每层 Self-Attention），Decoder 做自回归生成（Masked Self-Attention + Cross-Attention 看 Encoder）
4. **Position Encoding**：Sinusoidal 位置编码（固定频率的正余弦波），给 Self-Attention 注入位置信息——因为没有 RNN 的序列结构了

**第四层：不同方案的权衡。** 三大架构路线对比：

| 维度 | Encoder-only | Decoder-only | Encoder-Decoder |
|------|-------------|-------------|-----------------|
| 代表模型 | BERT | GPT 系列 | T5 |
| 理解能力 | ✅ 最强（双向） | ⚠️ 一般（单向） | ✅ 强（双向编码器） |
| 生成能力 | ❌ 不能生成 | ✅ 原生支持 | ✅ 支持 |
| Scaling 友好度 | ⚠️ 一般 | ✅ 最顺畅 | ❌ 参数翻倍 |
| 当前状态 | 退居特征提取 | **事实标准** | 特定任务 |

**第五层：总结升华。** Transformer 的革命性不在于单个组件有多精巧，而在于**完全并行化**使得大规模训练成为可能。没有 Transformer → 没有 GPT-3 → 没有 Scaling Law → 没有今天的 LLM 生态。O(n²) 复杂度是它付出的代价——后面所有工作（FlashAttention / PagedAttention）都在对抗这个复杂度。

---

## 学习目标

读完你能：

- 画出 Encoder-Decoder 的整体结构：Self-Attention → FFN → Add&Norm 的排列
- 写出 Scaled Dot-Product Attention + Multi-Head 的完整公式并解释每个符号
- 用一句话说清 Self-Attention 和 RNN 的本质区别（并行 vs 串行）
- 解释为什么 Attention 的复杂度是 O(n²) 以及为什么这是「值得的代价」
- 面对任务需求，给出选哪种架构（Encoder-only / Decoder-only / Encoder-Decoder）的决策理由

> 每一条学习目标都能被客观检验——你可以说「我做到了」或「我没做到」。避免「理解 XX 原理」这种不可检验的描述。

---

## 精选论文

**Vaswani et al. (2017) "Attention Is All You Need" [[arXiv](https://arxiv.org/abs/1706.03762)]**

- **一句话定位**：Transformer 架构的诞生论文，彻底取代 RNN/CNN 成为序列建模标准，被引 100,000+
- **阅读重点**：第 3 节（Scaled Dot-Product Attention + Multi-Head Attention 的公式推导）和第 5.4 节（位置编码的具体生成方式）。Encoder-Decoder 结构图（Figure 1）值得反复看
- **时间分配建议**：时间紧只读 3.1~3.3 节掌握注意力公式 + 看 Figure 1 和 Figure 2；时间充裕精读整个 Method 部分 + 附录中 Learning Rate Scheduler 的设计
- **与本模块的关系**：它是本模块的全部内容——所有关于 Transformer 的知识都出自这篇论文

---

## 模块间连接

- **前置依赖**：无（这是整个体系的起点）。如果对 Seq2Seq + Attention 不熟悉，建议先翻翻 Bahdanau 2015。
- **后续衔接**：读完本模块后，建议进入 **02-架构演进迭代**——理解原始 Transformer 的 5 个组件各自被怎么改进了
- **本模块与哪些模块正交**：与 05-应用技术（RAG / CoT）、06-前沿方向完全正交，可独立阅读


---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Attention Is All You Need | Vaswani2017 () | [arXiv](https://arxiv.org/abs/1706.03762) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| Attention Is All You Need | [arXiv](https://arxiv.org/abs/1706.03762) |
