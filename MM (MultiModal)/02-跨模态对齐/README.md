# 02 — 跨模态对齐

## 一句话开场

> 你有一张"猫"的图片和一段描述"一只橘猫坐在窗台上"的文字——它们本质上是完全不同的信号（像素 vs 字符串），但模型怎么知道它们指的是同一个事物？跨模态对齐就是建立这座桥梁。

## 正文：渐进式理解（5 层）

**第一层：问题定义。** 跨模态对齐的核心问题是"怎么让图像和文本在同一个向量空间中有意义地对应起来"。不是让模型"看到"图像的同时"读到"文本，而是让两者的表示具有可比性——猫的图片向量和"猫"的文本向量在空间中应该靠近。

**第二层：核心直觉。** 想象你是一个翻译，但翻译的不是两种语言，而是"图像语言"和"文本语言"。没有字典（没有对齐好的逐词映射），你只有大量"图文对"——一张图配一段文字。你的策略是：训练两个编码器（一个看图像、一个读文本），让它们对同一对图文输出相似的向量，对不同对的图文输出不同的向量。这就是对比学习的核心直觉。

**第三层：方案细节。** CLIP 使用双塔结构：Image Encoder（ViT）和 Text Encoder（Transformer），在 4 亿图文对上训练。训练时，batch 内 N 个图文对构成 NxN 的相似度矩阵，对角线是正样本（配对），其他是负样本。用 InfoNCE (Contrastive Loss) 最大化正样本间的 cosine similarity，最小化负样本间的 similarity。SigLIP 的改进是用 Sigmoid Loss 替代 Softmax-based InfoNCE，因为 sigmoid loss 是 pointwise 的（每对独立计算），不再需要 batch 内负样本的全局 softmax 归一化，因此对 batch size 不敏感——batch size 可以小 16 倍而性能不降。

**第四层：不同方案的权衡。**

| 方案 | 对齐方式 | batch size 敏感性 | 细粒度 | 代表工作 |
|------|---------|-----------------|--------|---------|
| 对比学习 (InfoNCE) | 图文级对齐 | 高（需大batch） | 粗粒度 | CLIP (2021) |
| 对比学习 (Sigmoid) | 对级对齐 | 低（batch不敏感） | 粗粒度 | SigLIP (2023) |
| 生成式对齐 (CapFilt) | 生成文本帮助对齐 | 不敏感 | 中粒度 | BLIP (2022) |
| 交叉注意力对齐 | token级跨模态 | 不敏感 | 细粒度 | UNITER (2020) |

**第五层：总结升华。** 跨模态对齐是整个多模态领域的"地基"——没有这个共享空间，后续所有的融合和推理都无从谈起。CLIP 双塔范式的成功，本质上是用"海量弱监督数据 + 对比学习"解决了"没有高质量标注数据怎么办"的问题。后续所有改进（SigLIP / BLIP / DataComp）都在回答同一个问题：怎么让这个对齐更高效、更准确、更可扩展。

---

## 学习目标
- 画出 CLIP 双塔架构 + InfoNCE loss 的计算链路
- 用一句话说清 Sigmoid Loss 为什么比 InfoNCE 更省 batch size
- 能对比 CLIP / SigLIP / BLIP 三个方案在对齐策略上的本质区别
- 理解 CLIP 能做到 zero-shot 分类的原理（图文共享空间 → 文本 prompt class name 作为分类权重）
- 知道 CLIP embedding 在 Stable Diffusion / VLM 中的实际作用

---

## 精选论文

**Radford et al. (2021) 'Learning Transferable Visual Models From Natural Language Supervision' (CLIP)**
- 一句话定位：开创"4亿图文对 + 对比学习"的双塔对齐范式，零样本分类和图文检索的工业标准
- 阅读重点：§2 Approach（对比学习目标 + 双塔架构）、§3 Zero-Shot Transfer（如何用文本 prompt 做分类）
- 时间分配建议：60 分钟精读核心方法，零样本实验可泛读
- 与本模块的关系：跨模态对齐的奠基工作

**Zhai et al. (2023) 'Sigmoid Loss for Language Image Pre-Training' (SigLIP)**
- 一句话定位：用 sigmoid loss 替代 softmax 使对比学习对 batch size 不再敏感，成为 DeepSeek-VL / InternVL2 等最新 VLM 的实际选择
- 阅读重点：§3 Method（Sigmoid Loss 公式推导）、§5 Experiments（和 CLIP 的对比）
- 时间分配建议：30 分钟精读 loss 公式对比，实验可泛读
- 与本模块的关系：CLIP 的高效改进版，当前最实用的对齐方案

**Li et al. (2022) 'BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation'**
- 一句话定位：BLIP 引入 CapFilt（Caption + Filter）机制，用生成+过滤提升图文数据质量，同时支持理解和生成任务
- 阅读重点：§3 Method（CapFilt 数据增强流程）、§3.2 Understanding & Generation 统一架构
- 时间分配建议：45 分钟精读 CapFilt 机制和 MED 架构
- 与本模块的关系：展示了"通过生成式方法提升对齐质量"的思路

---

## 拓展阅读
- **DataComp (2023)** — 系统研究图文数据配比对 VLM 性能的影响。如果你需要实际构建训练数据集，这是必读
- **LiT (2022)** — Locked-image Tuning，冻结 CLIP 视觉编码器只训练 text encoder。适合在已有视觉模型上快速适配

---

## 模块间连接
- **前置依赖**：01-视觉编码器（CLIP 的 Image Encoder 就是 ViT）
- **后续衔接**：03-多模态融合架构（CLIP 对齐后的特征进入 LLM）、04-训练数据与规模化（DataComp / CapFilt 具体数据方法）
- **正交于**：05-评估体系

---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| BLIP: Bootstrapping Language-Image Pre-training for Unified VLU and Generation | BLIP () | [arXiv](https://arxiv.org/abs/2201.12086) |
| Learning Transferable Visual Models From Natural Language Supervision | CLIP () | [arXiv](https://arxiv.org/abs/2103.00020) |
| DataComp: In search of the next generation of multimodal datasets | DataComp () | [arXiv](https://arxiv.org/abs/2304.14108) |
| Sigmoid Loss for Language Image Pre-Training | SigLIP () | [arXiv](https://arxiv.org/abs/2303.15343) |

---
