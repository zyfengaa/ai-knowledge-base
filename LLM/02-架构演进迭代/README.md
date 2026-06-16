﻿﻿﻿﻿# 02 — 架构演进迭代

> 原始 Transformer 的每个组件都「能用但不够好」——6 年时间，5 个关键改进各自解决一个独立问题。

## 正文：渐进式理解

**第一层：问题定义。** Vaswani 2017 的原始 Transformer 有 5 个「能用但不完美」的组件：位置编码用固定正弦波（不够灵活）、注意力计算 O(n²) 且 IO 效率低、注意力头冗余浪费计算、LayerNorm 做了很多不必要的运算、ReLU 激活函数收敛不够快。这 5 个问题互不依赖，所以各自独立演进。

**第二层：核心直觉。** 可以这么想：
- **RoPE**：给每个位置一个「旋转角度」，距离越远角度差越大——模型从「记位置编号」变成「算相对距离」
- **FlashAttention**：把「从显存取数据 → 算 → 写回显存」这个流程改成「分块算完再写」——避免反复读显存，利用更快的 SRAM
- **GQA**：所有注意力头共享 K/V，只有 Q 头不一样——推理时只需缓存一组 K/V，显存减半
- **RMSNorm**：去掉 LayerNorm 的 mean centering（减均值）——只保留 scaling（除以标准差），省掉一次统计量计算
- **SwiGLU**：ReLU 在负半轴直接砍掉信息 → SwiGLU 用 sigmoid 门控保留一部分负值 → 信息流动更丰富

**第三层：方案细节。** 五个改进速览：

| 改进 | 核心公式 / 操作 | 复杂度变化 | 效果 |
|------|----------------|-----------|------|
| RoPE | 用旋转矩阵给 Q/K 编码，内积自然包含相对位置 | O(n²) 不变 | 外推能力从 1x → 8x+ |
| FlashAttention | 分块 tiling + 在线 softmax 重计算 | O(n²) 但常数降低 | 训练加速 2-4x |
| GQA | 所有头的 K/V 池化为 n 组 | O(n²) 不变，KV Cache /n | 推理显存减半 |
| RMSNorm | 只做 Var(x) 归一化，去掉 Mean(x) | O(d) 一样但省一步 | 训练更稳定 |
| SwiGLU | Swish(x×W) × (x×V)——门控线性单元 | 参数增约 30% | 训练损失更低 |

**第四层：不同方案的权衡。**

位置编码对比：

| 方案 | 外推能力 | 训练友好 | 实现复杂度 | 代表模型 |
|------|---------|---------|-----------|---------|
| Absolute | ❌ 不能外推 | ✅ 简单 | 低 | GPT-2 |
| RoPE | ✅ 强（+NTK 可达 128K） | ✅ | 中 | **LLaMA/Qwen/DeepSeek** |
| ALiBi | ⚠️ 中等 | ✅ | 低 | MPT, Bloom |
| YaRN | ✅ 最优外推 | ⚠️ 需调参数 | 中 | 需在 RoPE 基础上加 |

注意力头设计对比：

| 方案 | KV Cache 大小 | 质量损失 | 代表 |
|------|--------------|---------|------|
| MHA（全部独立） | 最大 | 无 | 原始 Transformer |
| MQA（共享一套 K/V） | 最小 | ⚠️ 一点 | PaLM |
| **GQA（分组共享）** | 一半 | 极小 | **LLaMA 2/3, Qwen 2.5** |
| MLA（低秩压缩） | 最小 | 极小 | DeepSeek V2 |

**第五层：总结升华。** 现代 LLM 的标准配置是 **RoPE + GQA + SwiGLU + RMSNorm + FlashAttention**，这套组合从 2023 年 LLaMA 确立以来至今没有大的变化。后续改进（MLA / Mamba / BitNet）要么是效率优化要么是新范式尝试但尚未成为新一代标准。理解这 5 个改进，你就理解了当今所有主流 LLM 的「骨架」。

---

## 学习目标

读完你能：

- 用一句话说清 RoPE、GQA、FlashAttention、RMSNorm、SwiGLU 各自解决什么问题
- 列出当前主流 LLM 的「标准配置」清单（RoPE + GQA + SwiGLU + RMSNorm + FlashAttention）
- 在新的 LLM 推出时，通过分析它用了哪些组件来判断它的「血统」
- 说出至少 3 个 RoPE 的替代方案（ALiBi / Absolute / YaRN）及其各自优劣
- 理解为什么 FlashAttention 是「IO 感知」而非「计算优化」——它的核心贡献在访存模式而非算法

---

## 精选论文

**Su et al. (2021) "RoFormer: Rotary Position Embedding" [[arXiv](https://arxiv.org/abs/2104.09864)]**

- **一句话定位**：RoPE 是目前最主流的位置编码，LLaMA/Qwen/DeepSeek 全系列在用
- **阅读重点**：第 3 节——RoPE 的旋转矩阵推导和相对位置编码性质证明。核心 insight 是：用旋转矩阵替代加法来做位置编码
- **时间分配建议**：公式推导可跳读（理解思想即可），重点看 Figure 2 的位置编码可视化对比

**Dao et al. (2022) "FlashAttention: Fast and Memory-Efficient Exact Attention" [[arXiv](https://arxiv.org/abs/2205.14135)]**

- **一句话定位**：IO 感知注意力算法，所有现代训练/推理框架标配
- **阅读重点**：第 2 节（IO 复杂度分析 + tiling 策略 + 重计算思想），比具体算法实现更重要
- **时间分配建议**：Section 3 的块级实现细节（Algorithm 1）可跳读，理解「为什么 IO 是瓶颈」比理解具体分块方案更重要

**Ainslie et al. (2023) "GQA: Training Generalized Multi-Query Transformer Models" [[arXiv](https://arxiv.org/abs/2305.13245)]**

- **一句话定位**：MHA → MQA → GQA 的演化路径，LLaMA 2/3 采用的注意力头设计
- **阅读重点**：第 2 节（三种设计的对比）和实验结果——GQA 在质量上与 MHA 相当，速度与 MQA 相当
- **时间分配建议**：如果对 LLaMA 架构熟悉可以直接看 Table 1 的对比总结

**Zhang & Sennrich (2019) "Root Mean Square Layer Normalization" [[arXiv](https://arxiv.org/abs/1910.07467)]**

- **一句话定位**：RMSNorm，去掉 mean centering 的简化 LayerNorm，LLaMA 系列的标准归一化
- **阅读重点**：核心思想很简单——只做除以标准差，不做减均值。看第 2 节公式即可
- **时间分配建议**：全文很短（~4 页），建议通读。实验部分可以跳读

**Shazeer (2020) "GLU Variants Improve Transformer" [[arXiv](https://arxiv.org/abs/2002.05202)]**

- **一句话定位**：SwiGLU 激活函数，LLaMA/Qwen/DeepSeek/PaLM 的标准选择
- **阅读重点**：第 2 节（GLU 变体的形式化定义）和第 3 节（实验对比）
- **时间分配建议**：如果对激活函数熟悉可只读 Table 1 的对比结果；SwiGLU = Swish × Linear 的直觉比公式重要

---

## 模块间连接

- **前置依赖**：建议先读 **01-Transformer 起源**。本模块讨论的 5 个改进都是对原始 Transformer 的组件迭代，如果不理解原始架构的位置编码 / 注意力 / 归一化 / 激活函数分别是什么，无法理解改进的意义
- **后续衔接**：读完本模块后可以进入任意后续模块——03 关注训练流程、04 关注推理部署、05 关注应用、06 关注前沿
- **本模块与哪些模块正交**：与 05-应用技术（RAG / CoT）完全正交。与 03-训练与对齐范式弱正交（改进的架构会影响训练策略的取舍）


---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | FlashAttention () | [arXiv](https://arxiv.org/abs/2205.14135) |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | GQA () | [arXiv](https://arxiv.org/abs/2305.13245) |
| Root Mean Square Layer Normalization | RMSNorm () | [arXiv](https://arxiv.org/abs/1910.07467) |
| RoFormer: Enhanced Transformer with Rotary Position Embedding | RoFormer () | [arXiv](https://arxiv.org/abs/2104.09864) |
| GLU Variants Improve Transformer | SwiGLU () | [arXiv](https://arxiv.org/abs/2002.05202) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | [arXiv](https://arxiv.org/abs/2205.14135) |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | [arXiv](https://arxiv.org/abs/2305.13245) |
| Root Mean Square Layer Normalization | [arXiv](https://arxiv.org/abs/1910.07467) |
| RoFormer: Enhanced Transformer with Rotary Position Embedding | [arXiv](https://arxiv.org/abs/2104.09864) |
| GLU Variants Improve Transformer | [arXiv](https://arxiv.org/abs/2002.05202) |
