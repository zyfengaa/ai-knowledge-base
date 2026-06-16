﻿﻿﻿# 01 — 视觉编码器

## 一句话开场

> 一张 224x224 的彩色图片（150,528 个数字），怎么变成 Transformer 能处理的 token 序列？这不只是"切成小块"那么简单——切割方式决定了模型能看到什么、漏掉什么。

## 正文：渐进式理解（5 层）

**第一层：问题定义。** 视觉编码器解决的根本问题是"图像作为 2D 像素矩阵，怎么转换成 1D 序列输入到 Transformer"。CNN 用卷积滑动窗口提取特征（2D→2D→...→1D），ViT 用 Patchify 一步完成 2D→1D 的转换。核心矛盾：**空间局部性 vs 全局感受野**。

**第二层：核心直觉。** 把图像想象成拼图：CNN 是一块一块局部拼（先看小区域，再汇总成大区域）；ViT 是把所有拼图块一次性摊开，让模型自己学会每块之间的关系。ViT 牺牲了 CNN 的"局部亲和性"归纳偏置（inductive bias），换来了更大的灵活性和可扩展性——代价是需要更多数据来学习"相邻 patch 应该有关系"这件事。

**第三层：方案细节。** ViT 流程：① 将图像分成固定大小 Patch（如 16x16），② 每个 Patch 拉平成 1D 向量，③ 经过线性投影变成 Patch Embedding，④ 加位置编码（Position Embedding），⑤ 送入标准 Transformer Encoder。[CLS] token 输出作为图像整体表征。关键设计选择：Patch Size 决定序列长度和计算成本；位置编码类型（绝对/相对/RoPE）决定空间信息保留方式；Class Token vs GAP 决定输出聚合方式。

**第四层：不同方案的权衡。**

| 方案 | 优点 | 代价 | 适用场景 |
|------|------|------|---------|
| **ViT (2021)** | 简洁、可扩展、大模型效果好 | 需大量预训练数据、缺层级特征 | 通用视觉编码 |
| **Swin (2021)** | 层级特征（类似CNN）、窗口高效 | 实现复杂、窗口限制长程交互 | 检测/分割等需层级特征 |
| **EVA-02 (2023)** | MIM预训练+CLIP蒸馏、收敛快 | 依赖教师模型 | 高性价比视觉编码 |
| **InternViT (2024)** | 大参数(6B)、动态分辨率 | 计算量极大 | 需要最强视觉能力VLM |

**第五层：总结升华。** 视觉编码器演进从 CNN 到 ViT 再到大规模 ViT，本质是"放弃人工归纳偏置 → 利用大规模数据自学偏置 → 用更大数据自学更强偏置"的过程。ViT 是绝对通用的视觉编码器，后续改进围绕效率、层级特征、分辨率优化，没有突破性的 idea 改变。

---

## 学习目标

读完你能：

- 画出 ViT 的完整计算链路：Patchify → Linear Projection → Position Embedding → Transformer Encoder → [CLS] / Pooling
- 用一句话说清 ViT 和 CNN 在归纳偏置上的根本区别：ViT 没有 locality bias，靠数据学习空间关系
- 面对 VLM 项目能判断该用哪种视觉编码器：层级特征→Swin，最强性能→InternViT，通用→EVA/ViT
- 理解 Patch Size 对序列长度和计算量的影响：16x16 在 224² 上产生 196 token
- 解释为什么 ViT 需要大规模预训练（如 JFT-300M）才能超过 ResNet

---

## 精选论文

**Dosovitskiy et al. (2021) "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" [[arXiv](https://arxiv.org/abs/2010.11929)] (ViT)**

- **一句话定位**：打破 CNN 对图像建模的垄断，证明纯 Transformer + Patchify 可在足够数据下超过 ResNet，所有后续 VLM 的视觉编码器都是 ViT 变体
- **阅读重点**：§3.1 Model Architecture（核心设计）、§4 Experiments（数据规模的影响——数据不够 ViT 比不过 ResNet 是关键 insight）
- **时间分配建议**：30 分钟精读架构部分，60 分钟看实验和数据规模分析
- **与本模块的关系**：定义了视觉编码器的 Transformer 范式，是 VLM 视觉端的基石

**Liu et al. (2021) "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" [[arXiv](https://arxiv.org/abs/2103.14030)] (Swin)**

- **一句话定位**：修复 ViT 缺乏层级特征的问题，用窗口自注意力+移位窗口实现高效的多尺度特征提取
- **阅读重点**：§3.1 Shifted Window based Self-Attention（移位窗口设计）、§3.2 Architecture Variants（Swin-T/B/L 配置）
- **时间分配建议**：40 分钟精读核心机制，对比 ViT 全局注意力和 Swin 窗口注意力的计算复杂度差异
- **与本模块的关系**：展示了如何在 Transformer 中引入 CNN 式层级特征

**Fang et al. (2023) "EVA-02: A Visual Representation for Neon Genesis" [[arXiv](https://arxiv.org/abs/2303.11331)] (EVA-02)**

- **一句话定位**：通过 MIM（Masked Image Modeling）+ CLIP 知识蒸馏的高效视觉编码器，同等参数达到 CLIP 同等或更优性能
- **阅读重点**：§3 Method（MIM + 蒸馏的预训练框架）、实验结果对比
- **时间分配建议**：30 分钟精读预训练策略，架构细节可泛读
- **与本模块的关系**：展示了视觉编码器预训练策略的演进：监督学习→CLIP→MIM→MIM+蒸馏

---

## 拓展阅读

- **Touvron et al. (2021) "DeiT: Data-efficient Image Transformers" [[arXiv](https://arxiv.org/abs/2012.12877)]** — 用知识蒸馏在 ImageNet-1K 上训练 ViT，不需 JFT-300M 大规模数据。如果你手头只有 ImageNet-1K 级别数据，建议看看
- **Dosovitskiy et al. (2023) "NaViT" [[arXiv](https://arxiv.org/abs/2307.06304)]** — 让 ViT 支持动态分辨率输入（patchify + 打包训练）。对高分辨率/多分辨率输入感兴趣可翻翻
- **Zhu et al. (2024) "InternViT: Scaling Vision Foundation Models for Multimodal Understanding" [[arXiv](https://arxiv.org/abs/2405.01457)]** — 6B 参数视觉编码器，展示"视觉编码器越大越好"的 scaling 结论

---

## 模块间连接

- **前置依赖**：LLM-01 Transformer 基础知识（Self-Attention / Multi-Head / FFN）
- **后续衔接**：02-跨模态对齐（CLIP 的视觉编码器就是 ViT 的对比学习版本）、SD-02（扩散模型也使用 ViT 编码器）
- **本模块与哪些模块正交**：独立于 03-融合架构、04-数据、05-评估，可以随时阅读

---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| EVA-02: A Visual Representation for Neon Genesis | EVA02 () | [arXiv](https://arxiv.org/abs/2303.11331) |
| Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | Swin () | [arXiv](https://arxiv.org/abs/2103.14030) |
| An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | ViT () | [arXiv](https://arxiv.org/abs/2010.11929) |

---
