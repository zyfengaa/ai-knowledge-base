# 06 — 多模态扩展

## 一句话开场

> 你学会了用扩散模型生成图像——但为什么不能生成视频？为什么不能用文字生成 3D 模型？为什么不能用一张图生成它的多视角序列？多模态扩展要解决的问题是：**如何把图像扩散的成功经验迁移到更多输出模态（视频/3D/多视图）？**

---

## 正文：渐进式理解

**第一层：问题定义。** 扩散模型在 2D 图像上取得了巨大成功，但现实世界的信息是三维的、动态的、多通道的。视频生成需要在空间维度上额外对齐时序；3D 生成需要保证几何一致性；多视图生成需要跨视角的对应关系。**问题的核心是：图像扩散已经解决了"单帧质量"的问题，但多模态扩展需要额外解决"模态之间的一致性"——无论是时间（视频）还是空间（3D/多视图）的。**

**第二层：核心直觉。** 多模态扩展的核心思路有两种：

- **直接扩展法（Video Diffusion / Multi-view Diffusion）**：把扩散模型从二维（空间）拓展到三维（空间+时间）或四维（空间+视角）。你有一个视频时，它本质上是一叠连续的图像——扩散模型从前只是对单张图做去噪，现在对"一叠图"做去噪，并在时间维度上加注意力机制以保证帧与帧之间的连续性。

- **利用先验的 3D 生成法（SDS / Score Distillation Sampling）**：不直接训练 3D 扩散模型，而是利用一个已经训练好的 2D 扩散模型作为"裁判"，指导优化一个 3D 表示（如 NeRF / 网格）。模型从每个角度渲染一张图，2D 扩散模型评价"这个角度的图真不真实"，反向梯度更新 3D 表示。

**第三层：方案细节。**

**视频扩散——以 Stable Video Diffusion（SVD）为例：**
- 以图像扩散模型（SD 2.1）为初始化，在时序维度加入 temporal attention 层
- 训练数据为视频片段，每个片段由 N 帧组成，在时间维度上做 3D 卷积/注意力
- 训练策略分两阶段：第一阶段在视频数据上 fine-tune（适应运动分布），第二阶段做特定长度/帧率的适配
- 推理时输入一张参考图 + 随机初始化的帧序列，同时去噪所有帧，依赖 temporal attention 保证一致性

**3D 生成——以 DreamFusion / SDS 为例：**
- 初始化一个可微的 3D 表示（NeRF / 网格/ 3D Gaussian）
- 在随机视角下渲染 2D 图像
- 用预训练 2D 扩散模型（Imagen / SD）计算渲染图的 score（评估真实程度）
- 将 score 的梯度反向传播到 3D 表示，优化其几何和纹理
- 整个过程不依赖 3D 训练数据

**第四层：不同方案的权衡。**

**视频扩散：**

| 维度 | 逐帧独立生成 | 时序注意力（SVD） | 全 3D 卷积（Sora） |
|------|------------|-----------------|------------------|
| **时序一致性** | ❌ 差（闪烁） | ✅ 较好 | ✅ 最优 |
| **计算成本** | 低 | 中（N倍图像生成） | 高（3D 卷积 > 2D） |
| **实现难度** | 低（直接用图像模型） | 中（需时序层） | 高（需大量视频数据） |
| **现状** | 工业级不可用 | Stable Video Diffusion | Sora / 闭源 |

**3D 生成：**

| 维度 | SDS（DreamFusion） | 前馈 3D 生成（Point-E） | Zero-1-to-3（多视图） |
|------|-------------------|----------------------|---------------------|
| **速度** | 慢（分钟级优化） | 快（秒级前馈） | 快（秒级） |
| **质量** | 较好（多视角优化） | 一般 | 对已知类别好 |
| **泛化性** | 好（利用 2D 先验） | 受限（需 3D 训练数据） | 较好 |
| **场合适用** | 高质量单物体 | 快速原型 | 多视图生成 |

**第五层：总结升华。** 多模态扩展是扩散模型从"图像生成工具"走向"通用内容生成引擎"的必经之路。当前这个模块处于"范式跃迁的第 6 阶段"——基础框架已经搭建，但每个方向都有大量 open 问题。视频扩散在追赶图像扩散的质量（Sora 展示了可能性但未开源），3D 生成在寻找"等同于 LDM 那样的潜空间突破"，多视图在探索更好的跨视角一致性设计。本模块的核心矛盾是：**输出维度越高，一致性和计算效率之间的 trade-off 越尖锐。**

---

## 学习目标

读完你能：

- **解释视频扩散的时序注意力机制**：它和图像 Cross-Attention 的关系和区别
- **写出 SDS 的优化目标**：为什么 2D 扩散模型的 score 可以指导 3D 模型优化
- **对比"直接训练 3D 扩散"和"SDS 利用 2D 先验"的优缺点**
- **为一个多模态生成任务选择技术路线**：根据输出模态（视频/3D/多视图）和计算预算做决策
- **识别视频生成和图像生成在评价上的关键差异**：为什么 FID 对视频不够用

---

## 精选论文

**Poole et al. (2023) "DreamFusion: Text-to-3D using 2D Diffusion"**

- **一句话定位**：SDS（Score Distillation Sampling）的提出，利用 2D 扩散模型实现文本到 3D 的生成，开启了 2D → 3D 的范式
- **阅读重点**：Section 3（SDS 公式推导）、Figure 2（SDS 优化流程）、Section 4（3D 表示 + SDS 的结合）
- **时间分配建议**：Section 3.1（约 3 页）理解 SDS 的损失函数设计是核心。实验结果看 Figure 5-7 了解质量上限
- **与本模块的关系**：回答了"如何利用已有的 2D 扩散模型生成 3D 内容"

**Blattmann et al. (2023) "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets"**

- **一句话定位**：Stable Video Diffusion，将图像扩散模型扩展到视频生成的开源标准方案
- **阅读重点**：Section 2（时序注意力层设计 + 两阶段训练策略）、Figure 2（架构图：在 U-Net 中加入 1D 时序注意力）
- **时间分配建议**：Section 2（约 4 页）理解视频扩散的核心架构创新。Section 3（实验）看 Figure 5 的帧数扩展曲线
- **与本模块的关系**：回答了"如何高效地将图像扩散模型扩展到视频领域"


**Brooks et al. (2024) "Video Generation Models as World Simulators" (Sora Technical Report)**

- **一句话定位**：Sora 技术报告，展示了大规模视频扩散 DiT 的 Scaling Law，证明视频生成可以通过足够的计算量逼近物理世界模拟
- **阅读重点**：Video DiT + 时空 patch 设计、视频生成的 Scaling Law 曲线
- **时间分配建议**：建议重点理解其设计原则：时空 patch 化、Video DiT、大规模训练的策略借鉴意义
- **与本模块的关系**：回答了"视频扩散模型在足够大规模下能达到什么效果"

**Ma et al. (2024) "Latte: Latent Diffusion Transformer for Video Generation"**

- **一句话定位**：Latte 是将 DiT 成功迁移到视频生成的开源代表工作，验证了 Transformer 架构在视频扩散中的有效性
- **阅读重点**：Section 3（时空 Transformer 块设计）、Figure 2（Latte 架构全景）
- **时间分配建议**：Section 3（约 4 页）为核心，理解时空注意力的设计空间。Section 4（实验）看 FVD 指标
- **与本模块的关系**：回答了"视频 DiT 的具体架构怎么设计"

**Wu et al. (2024) "Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer"**

- **一句话定位**：Direct3D 将 3D 生成带到 DiT 时代——在 3D 潜空间上训练 Diffusion Transformer
- **阅读重点**：Section 3（3D VAE + 3D DiT 设计）、Figure 2（完整生成链路）
- **时间分配建议**：Section 3（约 5 页）理解 3D 潜空间的核心创新。实验看定性结果和生成速度
- **与本模块的关系**：回答了"扩散模型在 3D 领域如何做潜空间扩散"

## 拓展阅读

- **Brooks et al. (2024) "Video Generation Models as World Simulators"** — Sora 技术报告。如果对视频扩散的 Scaling Law 上限感兴趣，这篇展示了大规模视频数据 + DiT 架构能达到的质量。尽管未开源技术细节，但设计原则值得参考。
- **Lin et al. (2023) "Magic3D: High-Resolution Text-to-3D Content Creation"** — SDS 的改进版本，利用两阶段优化（NeRF→Mesh）提升 3D 生成分辨率。如果你对"SDS 后怎么进一步提升质量"感兴趣，值得一读。
- **Watson et al. (2023) "ControlNet for Multi-View Generation"** — 将 ControlNet 扩展到多视图生成的尝试。如果你关心"可控生成的方法在 3D 领域怎么用"，这篇有启发。

- **THUDM (2024) "CogVideoX: Text-to-Video Diffusion Models with An Open-Source License"** — 开源视频生成的重要里程碑。如果你需要"可部署的开源视频生成方案"，CogVideoX 是目前最成熟的选择之一。
- **Wu et al. (2024) "Improved Video VAE for Latent Video Diffusion Model"** + **Chen et al. (2024) "OD-VAE: An Omni-dimensional Video Compressor"** — 视频 VAE 的改进工作。如果你需要理解"视频扩散的基础——时空压缩"，这两篇值得对比阅读。
- **Yang et al. (2024) "Hunyuan3D 1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation"** — 腾讯开源的统一 3D 生成框架。如果你想看"工业级 3D 生成系统怎么设计"，这是很好的参考。
- **Chen et al. (2024) "3D-Adapter: Geometry-Consistent Multi-View Diffusion for High-Quality 3D Generation"** — 多视图一致性的重要进展。如果你关心"多视图扩散的质量瓶颈"，这篇方法值得深入。
- **Yi et al. (2024) "GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion"** — 将 3D Gaussian Splatting 与扩散模型结合的早期工作。如果你关注"3D 表示与扩散的结合"，这篇是起点。
- **Schwarz et al. (2025) "Generative Gaussian Splatting: Generating 3D Scenes with Video Diffusion Priors"** — 利用视频扩散先验做 3D 场景生成。如果你对"如何用视频数据增强 3D 生成"感兴趣，这篇有创新视角。

- **Liu et al. (2023) "Zero-1-to-3: Zero-shot One Image to 3D Object"** — 将扩散模型用于多视角生成。如果你想理解"单图到多视角的零样本转换"这条技术路线，这篇是起点。


> 拓展论文不移除，放在 `拓展/` 文件夹下。

---

## 模块间连接

- **前置依赖**：建议先完成 **01-扩散理论基础**（理解扩散框架）和 **02-骨干架构演进**（理解 DiT/U-Net 架构），因为视频扩散在架构层面修改了骨干网络。
- **后续衔接**：读完本模块后，可以回顾 **04-可控生成与适配**（ControlNet 在视频/3D 上的变体是当前研究方向之一）。
- **本模块与哪些模块正交**：本模块与 03（条件注入机制）正交——多模态扩展中的条件注入方式有自身的特点，不完全依赖于图像条件注入的经验。
