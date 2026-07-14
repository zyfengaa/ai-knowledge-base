# SD / 扩散模型演进史（2014-2026）

> 从 GAN 鼎盛到扩散模型革命，再到 DiT 和视频生成，视觉生成领域七年的进化之路

---

## 写在前面

这份演进史追踪的是**扩散模型（Diffusion Models）**的发展脉络，重点聚焦 **Stable Diffusion 家族**及其生态。和 LLM / ASR 演进史的写法一致——分阶段、每个模型交代"前置条件"和"核心改进"。

整条时间线可以浓缩为六个阶段：

```
2020 ── DDPM ──────────── "扩散模型可以用了"
2021 ── LDM ───────────── "从像素搬到潜空间"
2022 ── SD 1.5 ────────── "开源引爆，AI 画图人人可用"
2023 ── ControlNet ────── "从随机生成到精准控制"
2024 ── DiT + FLUX ────── "从 U-Net 到 Transformer 的范式革命"
2025-26 ── Sora 2 ─────── "从图片到世界模拟器"
```

---

## 第一阶段：扩散模型的奠基（2015-2021）——"扩散从理论到实用"

### 前置条件

在扩散模型之前，图像生成领域的主流是 **GAN（生成对抗网络，2014）**。Goodfellow 提出的 Generator + Discriminator 对抗训练在 2016-2020 年间统治了生成任务。但它的问题：

- **训练不稳定**——G 和 D 需要精心平衡，容易 mode collapse（模式坍塌）
- **多样性差**——GAN 倾向于只生成数据分布中的部分模式
- **缺乏概率基础**——没有显式的似然函数，难以评估生成质量

另一条线是 **VAE（变分自编码器，2013）**，有概率基础但生成质量模糊。**Autoregressive 模型**（PixelCNN / PixelRNN，2016）质量高但串行生成极慢。

行业需要一个**训练稳定、生成多样、有概率解释**的生成范式。

---

### 2015 — Sohl-Dickstein · 扩散模型理论提出

**"Deep Unsupervised Learning Using Nonequilibrium Thermodynamics"（ICML 2015）**

| 维度 | 说明 |
|------|------|
| 核心贡献 | **第一次提出扩散模型的完整框架**——前向过程逐步加噪声破坏数据分布，反向过程学习从噪声恢复。用热力学非平衡过程类比生成建模 |
| 局限 | 仅在 toy dataset 上验证，未在 ImageNet 级别大规模数据上证明可行性。当时的计算资源也不足以支撑 |

---

### 2020.06 — DDPM（UC Berkeley）· ✅ 论文开源

**"Denoising Diffusion Probabilistic Models"（Ho, Jain & Abbeel，NeurIPS 2020）**

| 维度 | 说明 |
|------|------|
| 核心改进 | **将扩散模型简化为实用配方**，引爆了整个领域：(1) 预测添加的噪声 ε 而非预测去噪图像；(2) 简单加权 MSE 损失，训练稳定；(3) 线性噪声调度，T=1000 步。在 CIFAR-10、LSUN 上首次达到媲美 GAN 的生成质量 |
| 参数量 | 随 U-Net 大小而异（典型 35M-114M） |
| 影响 | **~17,600+ 引用**。DDPM 让所有人意识到"扩散模型真的能用了" |
| 局限 | 推理需要 1000 步，生成一张图需要数分钟，无法实用 |

---

### 2020.10 — DDIM · ✅ 论文开源

**"Denoising Diffusion Implicit Models"（Song, Meng & Ermon，ICLR 2021）**

| 维度 | 说明 |
|------|------|
| 核心改进 | 提出**非马尔可夫反向过程**，实现确定性采样。**50-100 步就能达到 1000 步的效果**，无需重新训练模型。速度提升 10-20 倍 |
| 意义 | 让扩散模型从"理论上可行"变成"实际可用"。**至今仍是 SD 生态最常用的采样器之一**（DDIM / DPM++ 等采样器均受其启发） |
| 影响 | ~6,600+ 引用 |

---

### 2021.05 — Diffusion Beats GANs（OpenAI）· ✅ 论文

**"Diffusion Models Beat GANs on Image Synthesis"（Dhariwal & Nichol，2021）**

| 维度 | 说明 |
|------|------|
| 核心改进 | **扩散模型首次在 FID 指标上超越 GAN**。关键技术：U-Net 架构系统消融 + BigGAN 风格上下采样 + 自适应组归一化 + **Classifier Guidance**（用分类器梯度引导生成）。在 ImageNet 类条件生成上达到 SOTA |
| 影响 | **~7,500+ 引用**。宣告扩散模型正式取代 GAN 成为图像生成领域的第一范式 |

---

### 2021.12 — LDM（LMU Munich / Runway）· ✅ 论文开源

**"High-Resolution Image Synthesis with Latent Diffusion Models"（Rombach et al.，CVPR 2022）**

| 维度 | 说明 |
|------|------|
| 架构 | **U-Net + VAE + Cross-Attention** |
| 核心改进 | **(1) 潜空间扩散**——用预训练 VAE 将图像压缩到低维潜空间（约 48× 压缩率），然后在潜空间做扩散。计算量降低数十倍！(2) **Cross-Attention 条件注入**——把文本、分割图、边缘图等各种条件通过 Cross-Attention 注入 U-Net，是后来 ControlNet 等技术的基础 |
| 参数量 | Base 1.45B（扩散 U-Net）+ VAE + 文本编码器 |
| 意义 | **Stable Diffusion 的直接技术底座**。没有 LDM，就不可能有后来的 SD 开源生态。**LDM 是整个视觉生成领域最重要的单一论文** |
| 影响 | ~10,000+ 引用 |

---

## 第二阶段：闭源产品的军备竞赛（2021-2022）——"谁先做出好用的产品"

### 前置条件

DDPM / LDM 从学术上证明了扩散模型可行，但行业还没看到产品形态。2021-2022 年间，几家巨头竞相推出闭源的文生图产品。

### 2021.01 — DALL·E 1（OpenAI）· ❌ 闭源

| 维度 | 说明 |
|------|------|
| 架构 | **GPT-3 解码器（12B）+ dVAE（离散 VAE）**——不是扩散模型，是**自回归 Transformer** |
| 分辨率 | 256×256 |
| 训练数据 | 2.5 亿图文对 |
| 核心贡献 | **第一个从文本直接生成图像的大规模模型**，展现了"零样本"能力——给文字描述直接画图，无需微调。能混合概念（"牛油果形状的椅子"）|
| 局限 | 256×256 分辨率模糊，复杂 prompt 理解有限。自回归架构串行生成慢。后来被 DALL·E 2（扩散模型）快速取代 |

### 2022.04 — DALL·E 2（OpenAI）· ❌ 闭源

| 维度 | 说明 |
|------|------|
| 架构 | **unCLIP（扩散模型 + CLIP）**——第一次在文生图中用扩散模型替代自回归 |
| 分辨率 | **1024×1024**（4 倍提升） |
| 核心改进 | 精确度改善 **71.7%**，写实度改善 **88.8%**。新增 Inpainting（擦除重绘）、Outpainting（扩展边界）、Variations（同一 prompt 多个变体）。2022 年 7 月开放商业使用权，11 月开放 API |
| 影响 | 证明了"扩散模型 + CLIP"是文生图的最佳组合。但闭源且昂贵，为后来的开源爆发埋下伏笔 |

### 2022.05 — Imagen（Google）· ❌ 未公开

| 维度 | 说明 |
|------|------|
| 架构 | **冻结 T5-XXL（4.6B 参数） + 扩散级联**——64×64 → 256×256 → 1024×1024 |
| 核心发现 | **"缩放文本编码器比缩放扩散模型更有效"**——这个发现后来被 SD3 / FLUX 继承（都用 T5-XXL） |
| 核心改进 | 引入 Efficient U-Net 架构 + Dynamic Thresholding 采样。**COCO 零样本 FID 7.27，超越 DALL·E 2**。提出 DrawBench 评测基准 |
| 争议 | Google 因担忧滥用决定**不公开发布**，错失了在图像生成市场的先机。教训：安全顾虑太过保守会丢失技术领导地位 |

### 2022.07 — Midjourney V3 · ❌ 闭源

| 维度 | 说明 |
|------|------|
| 核心特征 | **Discord 优先的产品策略**——不做网页或 App，直接在 Discord server 里用 `/imagine` 生图。2022 年 7 月用户破百万。`--stylize` 和 `--quality` 参数控制生图风格和质量 |
| 意义 | **证明了"产品体验 > 模型参数"**。在技术不如 DALL·E 2 的情况下，通过极致的 UI/UX 获得了大量用户 |

---

## 第三阶段：开源引爆 + 生态成型（2022.08-2023）——"谁能把能力扩散出去"

### 前置条件

DALL·E 2 和 Midjourney 证明"AI 生成图像"有巨大市场需求，但所有能力掌握在几家公司手里。开发者想要一个**自己能控制、能部署、能修改、能微调**的模型。LDM 论文已开源，只差一个组织将其产品化。

### 2022.08 — Stable Diffusion 1.4（Stability AI）· ✅ 开源

| 维度 | 说明 |
|------|------|
| 架构 | **U-Net + CLIP ViT-L/14 + VAE**（基于 LDM 论文） |
| 参数量 | **~860M**（扩散 U-Net） |
| 分辨率 | 512×512 |
| 训练数据 | **LAION-5B**（50 亿图文对，经过筛选） |
| 核心贡献 | **将 LDM 论文产品化并免费开源**。消费级 GPU（4-6GB 显存）即可运行。**2 个月内成为史上增长最快的开源项目** |
| 意义 | **AI 图像生成的"Linux 时刻"**。全球开发者基于 SD 1.4 构建了无数应用——Automatic1111 WebUI、DreamStudio、各种 Colab notebook。如果没有开源，AI 生图可能到今天还是少数公司的特权 |

### 2022.10 — SD 1.5（Stability AI / Runway）· ✅ 开源

| 维度 | 说明 |
|------|------|
| 核心改进 | **最经典的 SD 版本**。在 SD 1.2 基础上继续训练，增加了更多微调步数。生成质量、稳定性都有提升 |
| 影响 | 生态**最成熟**的基座模型——CivitAI 上数十万衍生 Checkpoint + LoRA 几乎都以 1.5 为基础。**所有控制技术（ControlNet / LoRA / Textual Inversion / DreamBooth）都首先在 SD 1.5 上验证**。时至 2026 年，SD 1.5 衍生模型仍是最多的 |

### 2022.11 — SD 2.0 / 2.1（Stability AI）· ✅ 开源

| 维度 | 说明 |
|------|------|
| 分辨率 | **768×768**（首次提升原生分辨率） |
| 文本编码器 | 从 CLIP ViT-L 升级到 **OpenCLIP ViT-H** |
| 核心教训 | 因过度清洗 NSFW 数据导致生成"失去了味道"，且因 VAE 和 latent space 不兼容 **SD 1.5 生态**，社区继续留在 SD 1.5。**开源界的经典教训：生态兼容性有时比技术指标更重要** |

### 2023.02 — ControlNet（Stanford / Lvmin Zhang）· ✅ 开源

**"Adding Conditional Control to Text-to-Image Diffusion Models"（ICCV 2023）**

| 维度 | 说明 |
|------|------|
| 作者 | **Lvmin Zhang（张吕敏 / lllyasviel）**——华人学者，单枪匹马做了这个改变行业的工作 |
| 核心改进 | **Zero Convolutions（零初始化卷积层）**——锁住预训练模型参数不动，通过可训练副本 + 零卷积层逐步激活控制信号。实现 8 种空间控制条件：**Canny Edge**（结构控制）、**Depth**（深度控制）、**OpenPose**（姿态控制）、**HED**（软边缘）、**Segmentation**（语义分割）、**Normal Map**、**Scribble**、**M-LSD**（直线检测）。**单张消费级 GPU（RTX 3090）即可训练** |
| 训练机制 | 冻结原模型 → 复制编码器层 → 用零卷积连接。零卷积确保训练开始时输出为 0，不对原模型产生干扰，然后逐步激活控制信号。这使得 ControlNet 可以用**极小数据（<50k 样本）** 学到有效的控制 |
| 意义 | **扩散模型的"精准控制"时代开启**。没有 ControlNet，AI 生图永远是"抽卡"——你永远不知道模型会画出什么。有了 ControlNet，你可以画好线稿/摆好姿势/确定深度，让 AI 在此基础上发挥创意。**这是 SD 生态中最关键的工程创新** |

### 2023 — LoRA for SD · ✅ 开源

| 维度 | 说明 |
|------|------|
| 来源 | 微软 LoRA（*Low-Rank Adaptation*，2021）原为 LLM 微调设计，2023 年被社区迁移到 SD |
| 核心改进 | 在 U-Net Cross-Attention 层注入低秩矩阵（rank 4-128），仅 **~5MB 额外参数**即可微调出特定风格/角色/概念。可叠加多个 LoRA 同时使用 |
| 意义 | **催生了 CivitAI 上的"LoRA 经济"**——一人训练一个特定角色/风格，社区直接下载即用。降低了微调门槛到极致，任何人都可以在个人 PC 上训练自己的 LoRA |

### 2023.07 — SDXL（Stability AI）· ✅ 开源

| 维度 | 说明 |
|------|------|
| 参数量 | **~3.5B（Base）+ 6.6B（Refiner）** |
| 分辨率 | **1024×1024** 原生 |
| 文本编码器 | **双编码器**：CLIP ViT-L + OpenCLIP ViT-G（bigG） |
| 架构 | **双模型流水线**：Base 负责主生成（U-Net）、Refiner 负责细节增强（另一个 U-Net）。两步走：先用 Base 生成 1024×1024 粗图，再用 Refiner 增强细节 |
| 核心改进 | 卷积 U-Net 架构的**巅峰之作**：参数膨胀到 3.5B，双 CLIP 编码器让文本理解更好，1024 原生分辨率不再需要 upscale。社区接受度很高，至今仍是大规模使用的基座 |
| 意义 | U-Net 路线的"最后辉煌"——2024 年后所有新模型全面转向 DiT |

### 2023.08 — SDXL Turbo（Stability AI）· ✅ 开源

| 维度 | 说明 |
|------|------|
| 核心改进 | **Adversarial Diffusion Distillation（ADD）**——将 SDXL 蒸馏为 **1-4 步生成**。用对抗损失 + 蒸馏损失联合训练，在 1-4 步内达到 SDXL 质量 |
| 意义 | **实时生成成为可能**。这也是"少步生成"路线的代表性工作，同期的 LCM（Latent Consistency Model）也是类似方向 |

### 2023.09 — DALL·E 3（OpenAI）· ❌ 闭源

| 维度 | 说明 |
|------|------|
| 架构 | 扩散模型 + CLIP + VAE（具体未公开） |
| 分辨率 | 最高 **2048×2048** |
| 核心改进 | **(1) 原生集成 ChatGPT**——用户自然描述需求，ChatGPT 自动优化 prompt 再生成，从"写 prompt"变成"说需求"；(2) **语义理解大幅提升**——精确遵循复杂描述，逐词对齐；(3) 更强的安全措施——拒绝生成敏感内容 |
| 意义 | 产品体验的标杆。DALL·E 3 教育了"普通用户如何用 AI 生图"——不需要学 prompt 工程 |

### 2023.10 — PixArt-α（华为诺亚方舟实验室）· ✅ 开源

**"Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis"**

| 维度 | 说明 |
|------|------|
| 架构 | **DiT（Diffusion Transformer）**——0.6B 参数 |
| 文本编码器 | Flan-T5-XXL |
| 核心改进 | **训练成本仅为 SD 1.5 的 10.8%**（675 A100 GPU 天，约 $26,000）。三阶段训练策略：像素依赖 → 文本-图像对齐 → 美学质量提升。效果媲美 Midjourney |
| 后续版本 | **PixArt-Σ（2024.03）**：通过"由弱到强"策略实现 **4K（3840×2160）** 图像生成。引入 KV 压缩自注意力（降 34% 时间）|
| 意义 | **DiT 路线的重要前期探索**。证明了 Transformer 架构可以用极低成本达到 U-Net 的质量。为 SD3 / FLUX 全面转向 DiT 提供了实验依据 |

### 2023.12 — Midjourney V6 · ❌ 闭源

| 维度 | 说明 |
|------|------|
| 训练 | 9 个月从零训练 |
| 核心改进 | 照片级真实感再上新高度。**首次支持在图像中生成文字**。Prompt 理解更加精确，长文本描述也能还原 |

---

## 第四阶段：范式革命——DiT + Flow Matching（2024）——"从 U-Net 到 Transformer"

### 前置条件

2023 年底，行业发现 U-Net 架构的 Scaling Law 到达瓶颈——SDXL 已经把卷积堆到 3.5B 参数但收益递减。同时 **DiT 论文（2022.12，Peebles & 谢赛宁）**证明了 Transformer 在扩散模型中具有更好的缩放特性。

两条新方向交汇：
1. **MM-DiT（多模态扩散 Transformer）**——文本和图像 token 在同一个 Transformer 空间中双向交互
2. **Rectified Flow（整流流）**——用直线路径替代随机漫步，推理效率大幅提升

### 2024.02 — Stable Diffusion 3 预览 + MM-DiT 论文

| 维度 | 说明 |
|------|------|
| 架构 | **MM-DiT（Multimodal Diffusion Transformer）** |
| 推理范式 | **Rectified Flow**——替代 DDPM 的随机漫步去噪，用直线路径一步到位 |
| 文本编码器 | **三编码器**：T5-XXL（11B）+ CLIP-L + OpenCLIP-G |
| VAE | 升级为 **16 通道 VAE**——之前 4 通道在小字/细节上经常模糊 |
| 四大变革 | **(1) U-Net → DiT**：图像切为 patch 送 Transformer；(2) **MM-DiT 圆桌会议**：文本和图像 token 在同一空间双向交互，不再是单向注入；(3) **T5-XXL 替代 CLIP**：真正理解长难句和复杂逻辑；(4) **Rectified Flow**：直线去噪路径 |
| 后续 | **SD3 Medium（2024.06）**开源但广受批评——许可证限制 + 人体结构扭曲。**SD3.5 Large（2024.10）**8.1B 参数救场——QK-Normalization 稳定训练，人体结构大幅改善，许可证更宽松 |

### 2024.05 — HunyuanDiT（腾讯混元）· ✅ 开源

| 维度 | 说明 |
|------|------|
| 架构 | **DiT——1.5B 参数** |
| 训练数据 | 中文原生数据 |
| 核心改进 | **业界首个中文原生 DiT 架构的开源文生图模型**。支持中英双语输入，最长 256 字符。多轮对话式生图。与腾讯现网版本完全一致 |
| 意义 | DiT 路线在中文领域的首次大规模实践。同期开放训练代码 + LoRA 训练。国内社区接受度高 |

### 2024.07 — Kolors 可图（快手）· ✅ 开源

**"Kolors: Effective Training of Diffusion Model for Photorealistic Text-to-Image Synthesis"**

| 维度 | 说明 |
|------|------|
| 架构 | SDXL U-Net（2.6B）+ **GLM 文本编码器** |
| 文本编码器 | **ChatGLM3-6B-Base**——中英双语预训练模型（超 1.4 万亿 token），语义理解远超 CLIP |
| 核心改进 | **(1) 用 GLM 替代 CLIP/T5**——对中文语义理解质的飞跃；(2) **首个原生支持中文文字生成**的模型——构建了 5 万常用汉字数据集；(3) CogVLM 多模态大模型打标增强语义。在人类评估中视觉吸引力**与 Midjourney-v6 持平** |
| 意义 | 证明了"中文场景需要中文文本编码器"这个朴素但重要的道理 |

### 2024.08 — FLUX.1（Black Forest Labs）· ✅ 开源

| 维度 | 说明 |
|------|------|
| 参数量 | **12B——当时最大的开源文生图模型** |
| 架构 | **Rectified Flow Transformer（MM-DiT + Flow Matching + RoPE）** |
| 文本编码器 | T5-XXL + CLIP-L |
| 规格 | **FLUX.1 [schnell]**（Apache 2.0，4-8 步、12GB 显存）、**FLUX.1 [dev]**（非商用，20 步）、**FLUX.1 [pro]**（闭源 API） |
| 背景 | **SD 原班人马**：Robin Rombach、Andreas Blattmann、Dominik Lorenz 离开 Stability AI 后创立 **Black Forest Labs（黑森林实验室）**。a16z 3100 万美元种子轮。4 个月后再获 2 亿美元融资，估值超 10 亿美元 |
| 生成质量 | 人体结构、手部细节、材质光照全面超越 SD3 / 接近 Midjourney。天然支持任意宽高比 |
| 意义 | **开源生图的质量天花板被 FLUX 重新定义**。FLUX 的出现标志着"开源模型质量 > 闭源模型"成为可能。SD 的核心作者出走 → 建立新公司 → 做出更好的模型——这段故事本身也是 SD 领域最戏剧性的一章 |

### 2024.10 — SD3.5 Large / Large Turbo / Medium · ✅ 开源

| 维度 | 说明 |
|------|------|
| 参数量 | **8.1B（Large）** / 2.5B（Medium） |
| 核心改进 | Stability AI 的"救赎之作"——QK-Normalization 稳定训练，人体结构大幅改善。在多个评测中达到接近 FLUX 的质量。许可证更为宽松 |
| 意义 | 证明了 Stability AI 在 DiT 路线上仍然有竞争力。但核心团队已走、品牌影响力下滑 |

---

## 第五阶段：多模态融合 + 视频生成（2025-2026）——"从图片到世界"

### 前置条件

2024 年两条主线并行：
- **DiT 替代 U-Net 成为新标准**——FLUX 和 SD3.5 在质量上全面超越 U-Net 时代
- **视频生成爆发**——Sora（2024.02）证明 DiT 在视频领域的可行性。中国视频模型（可灵、Vidu、Pika 等）也全面跟进

### 2025 — FLUX.2（Black Forest Labs）

| 维度 | 说明 |
|------|------|
| 核心改进 | 支持 **10 张参考图**同时输入，**4MP 输出**分辨率。图像生成质量继续提升 |

### 2025.04 — Midjourney V7 Alpha

"草稿模式"——速度提升 10 倍、成本减半。全新的创意工作流

### 2025.09 — Sora 2（OpenAI）· ❌ 闭源

| 维度 | 说明 |
|------|------|
| 核心改进 | 视频领域的"GPT-3.5 时刻"——原生音视频同步生成、物理模拟精度大幅提升（水流动态提升 70%）、**Cameo**（用户形象植入）。上线独立 iOS / Android 应用 |
| 意义 | 文生视频从"demo 阶段"正式进入"产品阶段" |

---

## 六大核心技术演变总结

| 维度 | U-Net 时代（2022-2023） | DiT 时代（2024-2026） |
|------|----------------------|---------------------|
| **主干网络** | U-Net（卷积归纳偏置） | Transformer（数据驱动缩放） |
| **文本编码器** | CLIP（关键词匹配） | T5-XXL / GLM（真正理解语言） |
| **文本-图像交互** | Cross-Attention 单向注入 | MM-DiT 双向圆桌会议 |
| **推理路径** | DDPM 随机漫步（1000 步） | Rectified Flow 直线路径（1-50 步） |
| **图像表示** | VAE 4 通道潜空间 | VAE 16 通道潜空间 |
| **控制方式** | ControlNet + LoRA | 同上（延续到 DiT） |

---

## 三大模型家族一览

```
Stable Diffusion 家族（Stability AI）：
  2022 SD 1.4 → SD 1.5 → SD 2.0/2.1 → 2023 SDXL → SDXL Turbo → 2024 SD3 Medium → SD3.5 Large
  路线：开源引爆 → 2023 生态巅峰 → 2024 组织动荡 → DiT 转型
  ⚠️ 2024 核心团队出走创立 Black Forest Labs

FLUX 家族（Black Forest Labs）：
  2024.08 FLUX.1 [schnell/dev/pro] → 2024.10 FLUX 1.1 Pro → 2025 FLUX.2
  路线：SD 原班人马，DiT + Flow Matching，开源 + 闭源双轨

DALL·E 家族（OpenAI）：
  2021 DALL·E 1（GPT-3 + dVAE）→ 2022 DALL·E 2（unCLIP）→ 2023 DALL·E 3（ChatGPT 集成）
  路线：闭源，产品体验驱动

Midjourney 家族：
  2022 V1 → V2 → V3 → V4 → 2023 V5 → V5.1 → V5.2 → V6 → 2025 V7 Alpha
  路线：闭源，Discord 原生，美学驱动

Imagen 家族（Google）：
  2022 Imagen 1 → 2023 Imagen 2 → 2024 Imagen 3 → 2025 Imagen 4 / 4 Ultra
  路线：闭源，谨慎发布，逐年迭代

国内模型（中文场景）：
  2023 PixArt-α/Σ（华为诺亚）—— DiT 先驱，$26K 超低成本
  2024 HunyuanDiT（腾讯混元）—— 首个中文 DiT，1.5B
  2024 Kolors 可图（快手）—— GLM 编码器，中文文字生成
```

---

## 六个改变行业的关键节点

| 节点 | 时间 | 为什么关键 |
|------|------|-----------|
| **① DDPM** | 2020.06 | 扩散模型的实用配方，点燃了整个领域。没有 DDPM 就没有后续一切 |
| **② LDM** | 2021.12 | 潜空间扩散让 AI 生图走进消费级 GPU——普通开发者也能参与 |
| **③ SD 1.4 / 1.5** | 2022.08 | 开源引爆全球，AI 画图从"少数公司的特权"变成"人人可用" |
| **④ ControlNet** | 2023.02 | 从"随机抽卡"到"精准控制"的转折点。没有 ControlNet，SD 只是一个"高级抽奖机" |
| **⑤ MM-DiT（SD3 + FLUX）** | 2024 | U-Net → Transformer 的范式革命。2024 年后所有新模型全面转向 DiT |
| **⑥ Sora 2** | 2025 | 视频生成从 demo 变产品。从"生成一张好看的图"到"生成一个可信的世界" |

---

## 总结：九年的范式演变

| 时期 | 核心问题 | 代表 | 驱动力量 |
|------|---------|------|---------|
| 2015-2020 | 扩散能不能用？ | DDPM → LDM | 理论基础 + 算力提升 |
| 2021-2022 | 能不能做出好产品？ | DALL·E 2 / Imagen / MJ | 产品化能力 |
| 2022-2023 | 能不能扩散出去？ | SD 1.5 开源生态 | 社区 + ControlNet / LoRA |
| 2024 | 能不能更高效、更好？ | DiT / FLUX / Flow Matching | Transformer Scaling + 范式革命 |
| 2025-2026 | 能不能理解世界？ | Sora 2 / 视频生成 | 从图片到世界模拟器 |

> **视觉生成领域的竞争已从"谁的模型更大"变成"谁的生态更丰富、控制更精准、成本更低"。** 从 Conv 到 Transformer 的范式革命让所有玩家回到同一起跑线，而开源生态和"世界模型"方向将决定下一个五年的赢家。

---

**Sources:**

- [Deep Unsupervised Learning Using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — Sohl-Dickstein 2015
- [DDPM: Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al. 2020
- [DDIM: Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Song et al. 2020
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — Dhariwal & Nichol 2021
- [LDM: High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Rombach et al. 2021
- [DALL·E 2: Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125) — OpenAI 2022
- [Imagen: Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487) — Google 2022
- [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — Zhang et al. 2023
- [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — Stability AI 2023
- [PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://arxiv.org/abs/2310.00426) — Huawei 2023
- [SD3: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stability AI 2024
- [Kolors: Effective Training of Diffusion Model for Photorealistic Text-to-Image Synthesis](https://github.com/Kwai-Kolors/Kolors) — Kuaishou 2024
- [FLUX.1: Rectified Flow Transformer](https://blackforestlabs.ai/) — Black Forest Labs 2024
- [DiT: Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) — Peebles & Xie 2022
- [Sora: Video Generation as World Simulation](https://openai.com/sora) — OpenAI 2024-2025
- [HunyuanDiT: A Diffusion Transformer for Text-to-Image Generation](https://arxiv.org/abs/2405.08748) — Tencent 2024
- [PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation](https://arxiv.org/abs/2403.04692) — Huawei 2024
- Midjourney 版本历史 — [docs.midjourney.com](https://docs.midjourney.com/)
