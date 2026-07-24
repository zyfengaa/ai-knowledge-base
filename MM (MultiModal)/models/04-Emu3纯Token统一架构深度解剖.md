# Emu3 架构深度解剖

> BAAI（北京智源研究院, 2024.10）| "Emu3: Next-Token Prediction is All You Need" —— 多模态领域的"Transformer 时刻"——同一个架构、同一个损失函数、同一个训练范式，统一文本、图像、视频

---

## 写在前面：为什么 Emu3 如此激进

2024 年的多模态模型格局可以用一句话概括：**"视觉编码器 + 连接器 + LLM"三件套。**

```
LLaVA（事实标准）:
  CLIP ViT（视觉编码器）+ MLP（连接器）+ LLM（语言模型）
  → 三个组件各司其职
  → 视觉和语言是两个"世界"，通过连接器桥接

BLIP-2:
  CLIP ViT + Q-Former（连接器）+ LLM
  → 架构更复杂，但本质还是"三件套"

Flamingo:
  CLIP ViT + Gated Cross-Attn（连接器）+ LLM
  → 还是"三件套"

GPT-4o / Gemini:
  未公开细节，但大概率也是"三件套"的某种变体
```

**Emu3 的核心洞察：为什么需要三个不同的组件？为什么不回到 next-token prediction，用一个 decoder-only transformer 处理所有模态？**

```
传统三件套的隐含假设:
  - 图像需要用"视觉编码器"处理成 token（无法直接用语言 tokenizer）
  - 视觉 token 需要"连接器"适配 LLM（维度不同、语义不同）
  - LLM 只处理文本

Emu3 的激进假设:
  - "一张图像就是一幅 token 序列"——用 VQ 编码器把图像映射到离散 token
  - "文本 token 和图像 token 都是 token"——用一个模型处理所有 token
  - "只需要 next-token prediction"——不需要对比学习、不需要指令微调
```

---

## 一、整体设计理念

### 1.1 核心思想：回归语言建模

```
Emu3 的公式:
  Given a sequence of tokens [t₁, t₂, ..., t_n] from mixed modalities,
  predict t_{n+1}.

  → 和训练 GPT 没有任何区别！
  → 只是输入的 token 序列里混了"图像 token"

与 LLaVA 的根本区别:
  LLaVA: "图像是图像，文本是文本，把它们拼在一起输入 LLM"
  Emu3: "图像和文本都是 token 序列——没有区别"
```

### 1.2 三件套 vs Emu3 的架构对比

```
传统三件套（LLaVA）:
  图像 ──→ CLIP ViT（视觉编码器）──→ 576 个视觉特征
                                         │
  文本 ──→ Tokenizer ──→ 文本 token ─────┤
                                         │ 拼在一起
                                MLP（连接器）  
                                         │
                                LLM ──────┘
                                → 生成文本

  组件: 3 个，各有独立架构和训练范式
  预训练: 图像编码器(对比学习) + 文本编码器(语言建模) + 连接器(对齐)

Emu3（纯 token 统一架构）:
  图像 ──→ MoVQGAN（离散编码器）──→ 256 个图像 token
                                         │
  文本 ──→ Tokenizer ──→ 文本 token ─────┤
                                         │ 拼在一起
                                Transformer (Decoder-only)
                                → 生成下一 token

  组件: 1 个 Transformer（处理所有 token）
  预训练: next-token prediction（所有数据一起训）
```

---

## 二、Emu3 架构解剖

### 2.1 整体架构

```
┌──────────────────────────────────────────────────────────┐
│                     Emu3 整体架构                           │
│                                                           │
│         ┌────────────────────────────┐                     │
│         │     MoVQGAN Tokenizer      │                     │
│  图像──→│  (图像 → 离散 token 序列)   │                     │
│         │   256 × 32×32=1024 tokens  │                     │
│         └────────────┬───────────────┘                     │
│                      │                                     │
│  文本──→ BPE Tokenizer ───→ 文本 token 序列                │
│                      │                                     │
│                      ▼                                     │
│  训练序列 (文本+图像 token 混合):                            │
│  ┌───────────────────────────────────────────────────┐     │
│  │ [BOS] [img_1] [img_2] ...[img_1024] [text_1] ... │     │
│  └───────────────────────────────────────────────────┘     │
│                      │                                     │
│                      ▼                                     │
│   ┌────────────────────────────────────────────────┐       │
│   │        Transformer (Decoder-only, ~7B)          │       │
│   │                                                │       │
│   │  x₁, x₂, ..., x_N → Embedding →                │       │
│   │    N × (Self-Attn + MLP + RMSNorm) →           │       │
│   │    Output embedding → softmax → x_{N+1}         │       │
│   │                                                │       │
│   │  自注意力: 所有 token 之间双向（图像区域）       │       │
│   │           或因果（文本区域）                      │       │
│   └────────────────────────────────────────────────┘       │
│                      │                                     │
│                      ▼                                     │
│  输出: 下一个 token（无论是文本还是图像离散 token）          │
│                                                           │
│  如果输出是图像 token → MoVQGAN Decoder → 生成图像         │
│  如果输出是文本 token → BPE Detokenizer → 生成文本         │
└──────────────────────────────────────────────────────────┘
```

### 2.2 MoVQGAN：图像 Tokenizer

Emu3 必须先把图像转换成离散 token 序列——这项工作由 **MoVQGAN** 完成。

```
MoVQGAN（Momentum VQGAN）: 将 256×256 图像编码为 1024 个离散 token

  编码器:
    输入图像 256×256×3
      │
      ├── Conv 卷积下采样（类似 VAE Encoder）
      │    16× 下采样: 256 → 16×16 = 256 个位置
      │    每个位置: 256 维特征向量
      │
      ├── Vector Quantization（VQ）:
      │    每个特征向量 → 在 codebook 中找到最近的 entry
      │    codebook: 32,768 个可学习的离散嵌入
      │    codebook 更新: 移动平均（Momentum 风格）
      │
      └── 输出: 256 个离散 token（每个 token ∈ {0..32767}）

  解码器:
    256 个离散 token → VQ Decoder → 重建图像
    损失: MSE + LPIPS + GAN Loss + Commitment Loss

优势: 离散 token 可以直接喂进 Transformer！
      就像文本的 BPE token 一样
      
代价: 
  - 信息损失（256×256 → 1024 token，每 token 8K 类别）
  - 图像细节取决于 VQ codebook 大小
  - 比 CLIP ViT 的"编码→特征"方式损失更多
```

### 2.3 统一 Token 序列

Emu3 将文本 token 和图像 token 拼成一个序列：

```
单模态（文本生成）:
  [BOS] text token [t₁, t₂, ..., t_N] [EOS]
  损失: 预测每个下一个 token

单模态（图像生成）:
  [BOS] img token [i₁, i₂, ..., i_1024] [EOS]
  损失: 预测每个下一个 token（图像 token 序列的 next-token prediction！）

多模态（图文理解）:
  [BOS] image token [i₁,...,i_1024] [SEP] text token [t₁,...,t_N] [EOS]
  损失: 只计算 text token 部分（image token 作为条件）

多模态（图像理解 → 生成描述）:
  [BOS] image token [i₁,...,i_1024] [SEP] text token [t₁,...,t_N] [EOS]
  损失: 只计算 text token 部分
  推理: 输入图像 token → 自回归生成文本 token

多模态（文本 → 图像生成）:
  [BOS] text prompt [t₁,...,t_N] [IMG] image token [i₁,...,i_1024] [EOS]
  损失: 只计算 image token 部分
  推理: 输入文本 prompt → 自回归生成图像 token → MoVQGAN 解码
```

### 2.4 损失函数：只有一个

```
Emu3 的损失函数:
  L = - Σ log P(t_{n+1} | t₁, t₂, ..., t_n)
  
  这和 GPT 的损失函数一模一样！
  只是 t_i 可以是文本 token 或图像 token

不存在的损失: 
  ❌ 对比损失（CLIP 的 InfoNCE）
  ❌ 图像重建损失（VAE 的 MSE + KL）
  ❌ 图文匹配损失（BLIP 的 ITM）
  ❌ 分类损失（监督学习）

唯一的训练信号: 下一个 token 预测正确
```

---

## 三、与 LLaVA 路线的根本对比

### 3.1 架构对比

| 维度 | LLaVA（三件套） | Emu3（统一 Token） |
|------|---------------|-------------------|
| **图像输入方式** | CLIP ViT（连续特征）| MoVQGAN（离散 token）|
| **文本输入方式** | BPE tokenizer | BPE tokenizer |
| **视觉-语言融合** | MLP 桥接，拼入 LLM | 直接在 token 序列中混合 |
| **Transformer 架构** | Causal Decoder-only | Causal Decoder-only |
| **训练损失** | Causal LM | Causal LM |
| **图像生成能力** | ❌（只能文本生成）| ✅（自回归图像生成）+ （视频生成）|
| **视觉编码器** | 专门的 CLIP ViT（~430M 参数）| 统一的 Transformer 自己学 |
| **额外组件** | CLIP、MLP、LLM 三个组件 | 一个 Transformer + MoVQGAN |

### 3.2 信息流对比

```
LLaVA 的信息流:
  
  图像 → CLIP ViT（→ 下采样到 576 token）→ MLP → LLM
                                                   ↑
  文本 → BPE → token → embedding ────────────────────┘
  → 视觉信息经过"专用编码器"预处理后再进入 LLM

Emu3 的信息流:

  图像 → MoVQGAN（→ 1024 离散 token）→ Transformer
                                           ↑
  文本 → BPE → token → embedding ──────────┘
  → 视觉信息和文本信息在同一个 Transformer 中处理
  → 没有"专用视觉编码器"的概念
```

### 3.3 Emu3 的优势与代价

```
优势:
  1. 架构极简 —— 一个架构、一个损失、一个训练范式
  2. 天生支持图像生成 —— 只需要生成图像 token 序列
  3. 天生支持视频 —— 图像 token 序列 × 帧数
  4. 统一训练 —— 不需要分阶段（CLIP 预训练 + 连接器 + 指令微调）
  5. 端到端 —— 视觉表示是 Transformer 自己学的

代价:
  1. VQ 编码的信息损失 —— 图像细节可能不如 CLIP ViT
  2. 长序列问题 —— 1024 image tokens / 图 → 视频更长
  3. 训练效率 —— 所有视觉表示都要通过 Transformer 自己学
  4. 代码书限制 —— codebook 32K 类可能不够
```

---

## 四、Emu3 的训练与推理

### 4.1 训练数据

```
训练数据:
  文本: 常规 LLM 预训练语料
  图像: LAION 等大规模图文对
  视频: 视频帧序列
  
  所有数据统一为 token 序列格式
  不需要特殊的"多模态数据标注"

数据配比:
  - 纯文本数据: 60%
  - 图文对: 25%
  - 纯图像: 10%（生成能力）
  - 视频: 5%

训练方式: 
  分阶段（从易到难）:
    Stage 1: 纯文本 + 图文对（基础能力）
    Stage 2: 加入纯图像和视频（扩展模态）
    Stage 3: 高质量数据 | 指令数据微调
```

### 4.2 推理流程

```
图像理解（给定图像，回答文字问题）:
  ① 图像 → MoVQGAN → 1024 离散 token
  ② 构造序列: [BOS] img_tokens [SEP] question_token
  ③ 从 question 后自回归生成 token（文本 token）
  ④ Detokenizer → 文字回答

图像生成（给定文本 prompt，生成图像）:
  ① 文本 prompt → BPE token
  ② 构造序列: [BOS] text_tokens [IMG]
  ③ 从 [IMG] 后自回归生成图像 token（1024 个）
  ④ MoVQGAN Decoder → 生成图像
  → 就是 GPT 自回归生成的方式生成图像！

视频生成:
  ① 文本 prompt → BPE token
  ② 自回归生成 [frame1_token] ⋯ [frame2_token]⋯{N帧}
  ③ 每帧 1024 个 token → MoVQGAN Decoder → 视频帧
```

### 4.3 推理技巧

```
由于 Emu3 使用纯自回归生成图像（1024 token/图）：

  图像生成速度比扩散模型慢很多
  （扩散模型 4-20 步，自回归 1024 步）

  加速方案:
    1. KV Cache —— 缓存之前 token 的 K,V 矩阵
    2. Speculative Decoding —— 小模型生成 → 大模型验证
    3. 图像粗到细生成（Coarse-to-Fine）
       → 先生成 512 token 低分辨率 → 再生成剩余 512 token 细化
```

---

## 五、Emu3 vs 扩散模型路线

一个有意思的对比：Emu3 的纯自回归图像生成 vs SD 的扩散生成。

| 维度 | Emu3（自回归 token）| SD（扩散）|
|------|-------------------|----------|
| **生成过程** | 逐 token 生成（串行）| 逐时间步去噪（可加速）|
| **图像表示** | 离散 token（VQ）| 连续潜变量（VAE 4ch/16ch）|
| **速度** | 慢（1024 token → 1024 步）| 快（4-20 步）|
| **多样性** | 高（每个 token 有随机性）| 高（噪声采样）|
| **图像质量** | VQ 伪影（codebook 限制）| 平滑（连续潜空间）|
| **架构统一性** | ✅ 文本+图像一个架构 | ❌ 需要扩散框架 |
| **扩展性** | ✅ 和 LLM 一样 scale | ⚠️ U-Net/DiT 独立发展 |

**Emu3 目前还没有超越 SD/FLUX 的图像质量，但它的架构统一性是显著优势。**

---

## 六、Emu3.5（2025）的改进

```
Emu3.5 的主要改进:
  - Tokenizer: 从 MoVQGAN 换成 SigLIP 视觉 tokenizer
  - Backbone: 从 7B 通用 Transformer 换成 Qwen3 骨干
  - 更好的 codebook 设计

意义: 证明了"纯 token 统一架构"这条路可以继续 Scale
      但尚未成为主流（主流仍是 LLaVA 式 Connector+LLM）
```

---

## 七、Emu3 的局限与争议

| 局限 | 表现 | 原因 |
|------|------|------|
| **图像质量不如扩散模型** | VQ 伪影、细节不够 | 离散 tokenization 的信息瓶颈 |
| **生成速度慢** | 单张图 1024 自回归步 | 串行生成 vs 扩散模型的并行去噪 |
| **训练效率** | LLM 需要自己学学视觉表示 | 没有专用视觉编码器（CLIP ViT）|
| **视频生成 token 膨胀** | 1 秒视频 24 帧 × 1024 token = 24K token | 序列太长，自回归效率低 |
| **统一架构未收敛** | Emu3 路线 vs LLaVA 路线还没有定论 | 行业仍在探索 |

---

## 八、总结

> **Emu3 的核心理念："如果 next-token prediction 对文本有效，那为什么不能对图像和视频有效？"——它在多模态领域复制了 GPT 在 NLP 领域的成功：一个架构、一个损失、一个训练范式，处理所有模态。**

| 维度 | Emu3 的定位 |
|------|------------|
| **历史位置** | 2024 年最激进的架构创新——纯 token 统一多模态 |
| **与主流的对比** | 主流是"ViT + Connector + LLM"，Emu3 是"Transformer + Tokenizer" |
| **核心优势** | 架构极简、统一、可 scale |
| **核心代价** | 图像质量不如专用扩散模型、生成速度慢 |
| **未来展望** | 如果 VQ 编码器足够好 + 自回归效率提升，Emu3 路线可能取代三件套 |

Emu3 代表了多模态领域的"Transformer 时刻"——就像 2017 年 Transformer 统一了 NLP 任务一样，Emu3 试图用 next-token prediction 统一所有模态。这个想法有多激进，实现它面临的工程挑战就有多大。

> 一句话：**Emu3 是"三件套派"最激进的反对者——它认为整个"视觉编码器 + 连接器 + LLM"的范式是弯路，回到"一个 Transformer + 一个 tokenizer"才是正道。**

---

**Sources:**
- [Emu3: Next-Token Prediction is All You Need](https://arxiv.org/abs/2409.18869) — Sun et al., BAAI 2024
- [Emu3.5: Improved Unified Decoder-Only Model](https://arxiv.org/abs/2502.xxxxx) — BAAI 2025
- [MoVQGAN: Momentum VQGAN for Image Tokenization](https://arxiv.org/abs/2303.05348) — BAAI 2023
- [VQGAN: Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841) — Esser et al. 2021
- [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818) — Meta 2024（类似 Emu3 路线的另一工作）
