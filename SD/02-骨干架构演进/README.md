# 02 — 骨干架构演进

## 一句话开场

> 你有一个"从噪声中提纯图像"的神奇算法，但需要一个足够强大的神经网络来执行它——就像一个顶级厨师需要一套趁手的厨具。扩散模型的骨干架构从 U-Net 进化到 Transformer，背后的问题始终是：**什么样的网络结构最擅长"去噪"？**

---

## 正文：渐进式理解

**第一层：问题定义。** 扩散模型的数学框架不限定去噪网络的具体结构，但网络设计直接决定了：① 能否处理高分辨率图像（计算效率）；② 能否有效融合条件信息（文本/图像）；③ 能否随参数量扩展（Scaling Law）。**找到一个好的骨架，是扩散模型从理论走向实用的核心工程问题。**

**第二层：核心直觉。** 扩散模型需要一种"既能看清全局布局（低噪声/大尺度结构），又能关注局部纹理（高噪声/小尺度细节）"的网络结构。U-Net 的 skip connection 天然适合这个需求——编码器压缩全局信息，解码器恢复细节，skip connection 保留浅层纹理。这就像画一幅画：先画轮廓（编码器下采样），再填细节（解码器上采样），时不时回头看看草图（skip connection）。

**第三层：方案细节。** 扩散模型的骨干架构经历了两个关键阶段：

**Phase 1：U-Net（DDPM → LDM / SD）**

核心设计元素：
- **下采样-上采样对称结构**：多层 CNN 块逐步降低空间尺寸再恢复
- **Skip connections**：直接将编码器层的特征拼接到对应解码器层
- **Time embedding（Sinusoidal PE）**：通过 FiLM / AdaGN 将时间步信息注入每层
- **Cross-Attention（仅 LDM）**：在各层插入 Cross-Attention 层融合文本嵌入
- **Self-Attention（低分辨率层）**：在 16×16 及以下分辨率空间引入自注意力

**Phase 2：DiT（Diffusion Transformer）**

核心设计改变：
- **Patchify**：将图像/潜空间特征通过卷积打成 patch 序列，输入 Transformer 块
- **Transformer blocks**：标准的多头自注意力 + MLP，交替使用 modulate 机制注入时间/条件
- **Scalable**：参数量、FLOPs、性能之间呈现清晰的 Scaling Law

**第四层：不同方案的权衡。**

| 维度 | U-Net（LDM 风格） | DiT（Transformer 风格） |
|------|------------------|-----------------------|
| **核心操作** | 卷积 + 下采样/上采样 | 自注意力 + 线性层 |
| **感受野** | 有限（层级堆叠扩展） | 全局（自注意力一步到位） |
| **条件融合** | Cross-Attention 注入 | AdaGN / In-context conditioning |
| **Scaling 行为** | 收益递减（~1B 参数后瓶颈） | 清晰 Scaling Law（3B+ 仍然有效） |
| **推理速度** | 较快（对高分辨率友好） | 较慢（O(N2) 注意力） |
| **训练效率** | 对数据量不敏感 | 需要更多数据 |
| **现在地位** | SD 1.5 / SDXL 的工业标准 | SD3 / Flux / Sora 的下一代标准 |

**第五层：总结升华。** U-Net → DiT 的转变不仅仅是架构替换，它代表了**从手工设计到 Scaling Law** 的理念跃迁。在 2022 年，U-Net 是唯一可行的选择，它让扩散模型实用化；到 2023 年，DiT 证明了"大模型需要大骨架"，推动扩散模型进入了"更大 = 更好"的新阶段。当前 SD3 的 MMDiT 和 Flux 都在 DiT 的基础上进一步优化多模态融合。

---

## 学习目标

读完你能：

- **画出 LDM / Stable Diffusion 的三段式架构**：标注 VAE、U-Net、CLIP 的输入/输出和连接方式
- **说清"在潜空间做扩散"为什么比像素空间高效**：指出 VAE 压缩比例（8× 下采样）和计算节省的来源
- **对比 U-Net 和 DiT 在信息处理方式上的核心差异**：卷积 vs 注意力的本质区别
- **解释为什么 DiT 符合 Scaling Law 而 U-Net 存在瓶颈**：指出两者参数量-性能曲线的差异
- **为一个新任务选择去噪骨干架构并给出理由**：基于计算预算、数据量、分辨率要求做决策

---

## 精选论文

**Rombach et al. (2022) "High-Resolution Image Synthesis with Latent Diffusion Models" [[arXiv](https://arxiv.org/abs/2112.10752)]**

- **一句话定位**：LDM（Stable Diffusion）奠基论文，定义了 U-Net + 潜空间 + Cross-Attention 三合一架构，整个开源 SD 生态的基石
- **阅读重点**：Section 3（潜空间压缩 + 感知压缩的权衡）、Section 3.3（Cross-Attention 条件机制）、Figure 2（架构全景图）
- **时间分配建议**：建议精读 Section 3（约 4 页），理解 perceptual compression 和 semantic compression 的区别。Section 4（实验）可略读
- **与本模块的关系**：回答了"什么样的 U-Net 架构最适合在潜空间做扩散"

**Peebles & Xie (2023) "Scalable Diffusion Models with Transformers" [[arXiv](https://arxiv.org/abs/2212.09748)]**

- **一句话定位**：DiT 论文，证明 Transformer 在扩散模型中存在 Scaling Law，开启了 U-Net → Transformer 的架构迁移
- **阅读重点**：Section 3（DiT block 的四种设计变体）、Section 4（Scaling Law 实验设计）、Figure 3（FID-vs-GFLOPs 曲线）
- **时间分配建议**：Section 2-3（约 5 页）是核心，了解 DiT block 的设计空间。Section 4-5（实验）建议略读但记住其 scaling 结论
- **与本模块的关系**：回答了"Transformer 能否替代 U-Net 作为扩散模型骨架"

---

**Esser et al. (2024) "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" [[arXiv](https://arxiv.org/abs/2403.03206)]**

- **一句话定位**：SD3 的奠基论文，提出 MMDiT（双流多模态 Diffusion Transformer）+ Rectified Flow，是 DiT 之后最重要的架构创新
- **阅读重点**：Section 3（MMDiT 架构：文本和图像各走一路 Transformer 再融合）、Section 4（Rectified Flow 公式与训练）
- **时间分配建议**：Section 3（约 5 页）为必读，理解双流设计为什么比单流 Cross-Attention 更适合多模态。Section 5（实验）看 scaling 曲线即可
- **与本模块的关系**：回答了"DiT 之后，下一代架构长什么样"

**Black Forest Labs (2024) "FLUX: Rectified Flow Transformers" [[Blog](https://blackforestlabs.ai/announcing-black-forest-labs/)]**

- **一句话定位**：Flux 是目前质量最接近 Midjourney 的开源模型，采用 Rectified Flow + Transformer 架构，验证了 Scaling Law 在 12B 参数级别的有效性
- **阅读重点**：架构设计（双流/单流 Transformer 的变体设计）、Rectified Flow 的训练策略
- **时间分配建议**：目前无正式论文，建议阅读官方技术博客和开源代码。重点关注其 Scaling 策略（参数量与质量的关系）
- **与本模块的关系**：回答了"工程化的 Rectified Flow Transformer 应该怎么设计"

**Tian et al. (2024) "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction" [[arXiv](https://arxiv.org/abs/2404.02905)]**

- **一句话定位**：VAR 提出了"下一尺度预测"替代"下一 token 预测"，是一种全新的生成范式，证明了非扩散路线也能达到扩散模型相同质量
- **阅读重点**：Section 3（多尺度 VQVAE + 自回归预测）、Figure 2（VAR 的 Coarse-to-Fine 生成过程）
- **时间分配建议**：Section 3（约 4 页）理解 VAR 的核心设计。重要的是将其与扩散模型对比——它不是扩散模型但值得作为参照系
- **与本模块的关系**：回答了"图像生成是否一定需要扩散？"——作为对比范式放在本模块


## 拓展阅读

- **Ho et al. (2020) "DDPM" Section 3.3** — 原始 U-Net 架构描述。如果你对比 LDM U-Net 和 DDPM U-Net 的差异，可以溯源回这里。
- **Esser et al. (2024) "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" [[arXiv](https://arxiv.org/abs/2403.03206)]** — SD3 提出的 MMDiT（双流多模态 Transformer）。如果你对 Transformer 架构的最新设计感兴趣，建议读 Section 3（MMDiT 设计），这是 2024 年的事实标准设计。

- **Jia et al. (2025) "D2iT: Dynamic Diffusion Transformer for Accurate Image Generation" [[arXiv](https://arxiv.org/abs/2501.04569)]** — 提出动态 DiT 架构。如果你对"DiT 在推理效率上的优化"感兴趣，建议阅读其动态 token 剪枝策略。
- **Du et al. (2026) "ElasticDiT: Efficient Diffusion Transformers via Elastic Architecture and Sparse Attention"** — 弹性架构 + 稀疏注意力优化 DiT 推理。如果你关心"DiT 如何在资源受限设备上部署"，这篇有工程参考价值。
- **Gu et al. (2024) "DART: Denoising Autoregressive Transformer for Scalable Text-to-Image Generation" [[arXiv](https://arxiv.org/abs/2410.12431)]** — 将去噪与自回归结合的尝试。如果你对比 VAR 和 DART 的设计差异，可以发现"下一尺度"vs"下一 token"两条路线的区别。

> 拓展论文不移除，放在 `拓展/` 文件夹下。

---

## 模块间连接

- **前置依赖**：建议先完成 **01-扩散理论基础**，理解扩散模型的基本工作原理。需要知道 VAE 的编码-解码基本概念。
- **后续衔接**：读完本模块后，建议进入 **03-条件注入机制**（理解 Cross-Attention 和 CFG 如何让模型"听指令"）或 **04-可控生成与适配**（理解 ControlNet/LoRA 如何扩展模型能力）。
- **本模块与哪些模块正交**：本模块与 05（采样加速）正交——架构设计影响推理速度，但加速方法本身独立于具体骨架。


