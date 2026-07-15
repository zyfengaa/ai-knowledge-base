# LDM + U-Net 架构深度解剖

> Rombach et al. (LMU Munich / Runway, CVPR 2022) | "High-Resolution Image Synthesis with Latent Diffusion Models" —— Stable Diffusion 的直接技术底座，视觉生成领域最重要的单一论文

---

## 写在前面

DDPM 证明了扩散模型能生成高质量图像，VAE 证明了潜空间压缩能大幅降低计算量。但 DDPM 有两个核心问题没解决：

1. **条件控制**——DDPM 本质上是一个无条件生成模型，要让模型按照文本来生成图像需要更精巧的机制
2. **计算效率**——即使有 DDIM 加速，像素空间扩散仍然很贵

LDM 把两件事做到极致：
- **用 VAE 把扩散搬到潜空间**（≈ 48× 计算量降低）——已在 VAE 篇讲透
- **用 Cross-Attention 把文本条件注入 U-Net**——这是 LDM 真正的架构创新

**LDM = VAE（压缩层）+ U-Net（扩散骨干）+ Text Encoder（条件引擎）+ Cross-Attention（融合机制）**

这四个模块的组合，构成了后来所有 Stable Diffusion 模型（1.4/1.5/2.x/SDXL）的技术地基。

---

## 一、整体架构

```
输入: 文本 prompt（"一只穿西装的猫，油画风格"）

  ┌─────────────────────────────────────────────────────────────┐
  │                        ① Text Encoder                       │
  │                          CLIP ViT-L/14                      │
  │                        输出: 77×768 嵌入向量                 │
  └───────────────────────────┬─────────────────────────────────┘
                              │
                              ▼
  输入图像 x₀ ───→ ② VAE Encoder ───→ z₀ (64×64×4)
                                          │
         ③ 前向扩散 (在潜空间)             │
           z_t = √ᾱ_t·z₀ + √(1-ᾱ_t)·ε    │
               │                          │
               ▼                          │
      ┌────────────────────────────┐      │
      │      ④ U-Net + Attn       │      │
      │  ┌─┐ ┌─┐ ┌─┐             │      │
      │  │D1│ │D2│ │D3│──Middle──│      │
      │  └┬┘ └┬┘ └┬┘            │      │
      │   │   │   │              │      │
      │  ┌▼┐ ┌▼┐ ┌▼┐            │      │
      │  │U1│ │U2│ │U3│          │      │
      │  └─┘ └─┘ └─┘            │      │
      │  ↑   ↑   ↑               │      │
      │ Cross-Attention ←── 文本嵌入  │
      └───────────┬───────────────┘      │
                  │                      │
        输出: ε_θ(z_t, t, text)          │
                  │                      │
                  ▼                      │
             ⑤ 反向去噪 ←────────────────┘
               z_{t-1}  ←  z_t  ←  z_T
                  │
                  ▼
           z₀ (去噪后的潜变量)
                  │
        ⑥ VAE Decoder ───→ 生成图像 x₀
```

| 模块 | 输入 | 输出 | 参数量 |
|------|------|------|--------|
| **VAE Encoder** | 3×H×W 图像 | 4×H/4×W/8×H/8 （取决于压缩倍率） | ~84M |
| **VAE Decoder** | 4×H/4×W/8 潜变量 | 3×H×W 图像 | ~84M |
| **Text Encoder (CLIP)** | 文本 token (77个) | 77×768 嵌入 | ~340M |
| **U-Net (扩散骨干)** | 4×64×64 + 时间步 + 文本嵌入 | 4×64×64 预测噪声 | ~860M (SD 1.5) |
| **合计** | — | — | ~1.3B+ |

---

## 二、Text Encoder（CLIP）——条件引擎

### 2.1 为什么需要专门的文本编码器？

DDPM 的生成是"盲目的"——从纯噪声去噪，没有条件信号告诉它"你想生成什么"。LDM 需要一个机制把用户的文字描述转化为模型理解的信号。

CLIP 被选中的原因：

```
CLIP = Contrastive Language-Image Pre-training

核心能力: 将文本和图像映射到同一个 embedding 空间
  "一只猫"  → CLIP → [0.2, 0.5, -0.1, ...]  ← 猫的文本嵌入
  猫的照片  → CLIP → [0.3, 0.4, -0.2, ...]  ← 猫的图像嵌入
  
  两者在 embedding 空间里距离很近（cosine similarity 高）
```

CLIP 让文本和图像在同一个空间中"对齐"，这是 Cross-Attention 能工作的前提——U-Net 要能通过 Cross-Attention 在文本嵌入中找到对应的视觉模式。

### 2.2 CLIP 在 LDM 中的具体结构

```
输入文本: "一只穿西装的猫，油画风格"
   │
   ├── Tokenizer (BPE, 49408 vocab)
   │   └── 输出 token 序列 [49406, ... , 49407] (77 个 token)
   │
   ├── Text Transformer (CLIP ViT-L/14)
   │   └── 12 层 Transformer Decoder
   │       ├── Self-Attention（因果掩码）
   │       ├── MLP (GELU)
   │       └── LayerNorm (Pre-LN)
   │   └── 输出: 77×768 嵌入矩阵
   │
   └── Pooled 输出: 768 维向量（[EOS] token 位置，用于对比学习）
       └── LDM 只使用 77×768 的 token-level 嵌入，不用 pooled 输出
```

**关键说明：** 这里的"Transformer Decoder"不是 GPT 那种自回归解码器——它使用的是**因果掩码 Transformer**，每个 token 只能看到自己和左侧的 token。这是因为 CLIP 训练时使用对比学习，不需要双向上下文。

### 2.3 各 SD 版本使用的文本编码器演变

| 模型 | 文本编码器 | 输出维度 | 长度 | 说明 |
|------|-----------|---------|------|------|
| LDM / SD 1.x | CLIP ViT-L/14 | 77×768 | 77 token | 奠基版本 |
| SD 2.0/2.1 | OpenCLIP ViT-H | 77×1024 | 77 token | 参数量更大 |
| SDXL | CLIP-L + OpenCLIP-G（双编码器） | 77×768 + 77×1280 | 77×2 | 融合两个编码器输出 |
| SD3 / FLUX | T5-XXL + CLIP-L + OpenCLIP-G | 512×4096 + ... | 512 token | T5-XXL 接管主力 |
| Kolors | ChatGLM3-6B | 序列可变 | 256 | 中文专用 |

**为什么从 CLIP 切换到 T5-XXL？** CLIP 的文本理解偏"关键词匹配"——它擅长理解"猫、穿西装、油画"这些视觉概念，但不擅长复杂语义（"戴红色帽子的猫和趴在地上的狗之间有一张桌子"）。T5-XXL 是纯文本 LLM，语言理解能力远超 CLIP。

---

## 三、U-Net 架构——扩散骨干

### 3.1 整体结构

LDM 使用的 U-Net 比 DDPM 的 U-Net 多了 **Cross-Attention 层**：

```
输入: z_t (潜空间噪声图, 4×64×64)
      t (时间步, 标量)
      text_emb (文本嵌入, 77×768)
              
      ┌────────────────────────────────────────────┐
      │         Conv 3×3 (4 → 320)                │
      └──────────────────┬─────────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌─────────┐        ┌─────────┐         ┌─────────┐
│ Stage 1 │        │ Stage 2 │         │ Stage 3 │
│ Res × N  │        │ Res × N  │         │ Res × N  │
│ 320ch   │        │ 640ch   │         │ 1280ch  │
│ 64×64   │        │ 32×32   │         │ 16×16   │
│ Down    │        │ Down    │         │ Down    │
└────┬────┘        └────┬────┘         └────┬────┘
     │                  │                    │
     └──────────────────┼────────────────────┘
                        │
                  ┌─────┴─────┐
                  │  Middle   │
                  │ Res × N   │
                  │ 1280ch    │
                  │ 16×16     │
                  └─────┬─────┘
                        │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌─────────┐        ┌─────────┐         ┌─────────┐
│ Stage 3 │        │ Stage 2 │         │ Stage 1 │
│ Up × N  │        │ Up × N  │         │ Up × N  │
│ 1280ch  │        │ 640ch   │         │ 320ch   │
│ 16×16   │        │ 32×32   │         │ 64×64   │
│ Up      │        │ Up      │         │ Up      │
│ ← skip  │        │ ← skip  │         │ ← skip  │
└────┬────┘        └────┬────┘         └────┬────┘
     │                  │                    │
     └──────────────────┼────────────────────┘
                        │
                  ┌─────┴─────┐
                  │  Conv 3×3 │
                  │ (320 → 4) │
                  └─────┬─────┘
                        │
                  输出: 4×64×64 预测噪声
```

### 3.2 ResBlock（残差块）

每个 ResBlock 包含：残差连接 + 时间步条件注入 + 文本条件注入。

```
ResBlock 内部结构:
  输入: h (特征图, C×H×W), t_emb (时间步嵌入), text_emb (文本嵌入)
  
  ① 特征变换:
     h → GroupNorm(32) → SiLU → Conv3×3 → GroupNorm(32) → SiLU → Conv3×3 → h_out
  
  ② 条件注入（时间步 + 文本）:
     t_emb → Linear → SiLU → Linear → scale_t, shift_t
     h_out = scale_t * h_out + shift_t    ← AdaGN
  
  ③ 残差连接:
     如果 h_in 和 h_out 通道数不同: h_in = Conv1×1(h_in) 对齐通道
     h_final = h_in + h_out
  
  ④ 下采样（Down 阶段）:
     h_final → Conv3×3 stride=2 → h_down  （或 AvgPool + Conv）
  
  ⑤ 上采样（Up 阶段）:
     h_final → 最近邻插值2× → Conv3×3 → h_up

  注: Cross-Attention 不在 ResBlock 里，在单独的 AttentionBlock 中
```

### 3.3 Cross-Attention 条件注入

这是 LDM 最关键的架构创新——文本信息如何影响去噪过程：

```
AttentionBlock 内部（同时包含 Self-Attention 和 Cross-Attention）:

  输入: h (特征图, C×H×W), text_emb (77×768)
  
  ① Self-Attention（空间内部）:
     Q = Conv1×1(h).reshape(HW×C)            ← 来自特征图
     K = Conv1×1(h).reshape(HW×C)            ← 来自特征图
     V = Conv1×1(h).reshape(HW×C)            ← 来自特征图
     attention = softmax(Q·Kᵀ/√d) · V        ← "图像的哪个区域重要"
  
  ② Cross-Attention（文本-图像交互）:
     Q = Conv1×1(h).reshape(HW×C)            ← 来自特征图（图像查询）
     K = text_emb.reshape(77×C)              ← 来自文本编码器（文本键）
     V = text_emb.reshape(77×C)              ← 来自文本编码器（文本值）
     attention = softmax(Q·Kᵀ/√d) · V        ← "每个像素从哪个文字 token 获取信息"
  
  ③ 输出:
     h_out = h + attention_out               ← 残差连接
  
  注: Cross-Attention 只在低分辨率阶段使用（16×16 和 32×32）
      高分辨率阶段（64×64）只做 Self-Attention，节省计算
```

**Cross-Attention 的直观理解：**

```
Q (图像像素): "这个区域看起来像……?"
K (文本 token): 我有"猫"、"西装"、"油画"这些概念
softmax(Q·Kᵀ): 像素找到了自己与最相关的文本 token 的匹配
V (文本 token 值): 把"猫"的语义信息传给匹配的像素

结果: 图像中某个区域被"激活"成猫的特征
      另一个区域被激活成西装的特征
```

### 3.4 AttentionBlock 在 U-Net 中的分布

| Stage | 分辨率 | 通道数 | Self-Attention | Cross-Attention | 备注 |
|-------|--------|--------|---------------|----------------|------|
| Down 1 | 64×64 | 320 | ✅ 有 | ❌ 无 | 高分辨率+少通道，自注意力足够了 |
| Down 2 | 32×32 | 640 | ✅ 有 | ✅ 有 | 开始引入文本指导 |
| Down 3 | 16×16 | 1280 | ✅ 有 | ✅ 有 | 最高语义层，文本信息最强 |
| Middle | 16×16 | 1280 | ✅ 有 | ✅ 有 | 信息瓶颈 |
| Up 3 | 16×16 | 1280 | ✅ 有 | ✅ 有 | 对称于 Down 3 |
| Up 2 | 32×32 | 640 | ✅ 有 | ✅ 有 | 对称于 Down 2 |
| Up 1 | 64×64 | 320 | ✅ 有 | ❌ 无 | 对称于 Down 1 |

**设计逻辑：** Cross-Attention 只在低分辨率（16×16 和 32×32）特征图上做，因为：
1. 这些特征图分辨率低，计算成本可控
2. 低分辨率特征承载的是"语义信息"——"猫"、"西装"这些概念
3. 高分辨率特征（64×64）承载的是"纹理细节"——"毛发的走向"、"布料的光泽"，不需要 Cross-Attention 来指导

---

## 四、条件注入的几种方式（不止文本）

LDM 的架构设计支持多种条件类型，通过不同的注入方式实现：

### 4.1 文本条件（最常用）

```
条件信号: token 级别的 CLIP 嵌入 (77×768)
注入方式: Cross-Attention
实现: Q = 特征图, K, V = 文本嵌入

这是 SD 生态的标准用法——用户输入 prompt，模型根据文字生成图像。
```

### 4.2 语义地图 / Segmap

```
条件信号: 分割图 (C×H×W, C 是语义类别数)
注入方式: Concat（拼接到潜变量通道上）
实现: z_t' = Concat(z_t, segmap_downsampled)

原理: 语义地图是空间对齐的——每个像素知道自己属于"天空"还是"草地"。
      通过 Concat 直接告诉 U-Net 每个位置应该生成什么语义。
```

### 4.3 图像条件（Inpainting / Super-resolution）

```
条件信号: 低分辨率/掩码图像 (3×H×W)
注入方式: Concat + 微调
实现: 
  Inpainting: z_t' = Concat(z_t, mask, masked_image_latent)
  SR: z_t' = Concat(z_t, low_res_latent)

SD 的 Inpainting 模型就是这样工作的——输入的掩码区域被 mask out，
模型需要补全该区域的潜变量。
```

**这种"Concat 方式"与"Cross-Attention 方式"的对比：**

| 方式 | 适用条件类型 | 优点 | 缺点 |
|------|------------|------|------|
| **Cross-Attention** | 文本、CLIP 嵌入 | 灵活，支持序列输入 | 需要训练 Cross-Attention 层 |
| **Concat** | 空间对齐条件（segmap, depth, 边缘图） | 精确定位，空间感知 | 必须与潜变量同分辨率，改变 U-Net 输入通道数 |

**ControlNet 的工作正式利用了"Concat 方式"的扩展——它在 Encoder 副本中保留了空间条件信息，通过 Zero Convolution 缓慢激活，实现了更精细的空间控制。**

---

## 五、训练与推理流程

### 5.1 训练流程

```
算法: LDM 训练（单步）

输入: 图像 x₀，文本 prompt y
超参数: β 调度，T=1000

1. z₀ = VAE_Encoder(x₀)                         ← 编码到潜空间
2. t ~ Uniform(1, T)                             ← 随机选时间步
3. ε ~ N(0, I)                                   ← 采样真实噪声
4. z_t = √ᾱ_t · z₀ + √(1-ᾱ_t) · ε              ← 前向加噪
5. c = CLIP_Encoder(y)                           ← 编码文本
6. ε_pred = ε_θ(z_t, t, c)                       ← U-Net 预测噪声
7. L = MSE(ε, ε_pred)                            ← 计算损失
8. 反向传播更新 ε_θ                              ← 只更新 U-Net 参数
```

**关键：VAE 和 CLIP 权重在训练期间冻结。** 只训练 U-Net（以及其中的 Cross-Attention 线性投影层）。

### 5.2 推理流程

```
算法: LDM 推理（DDIM 采样，50 步）

输入: 文本 prompt y，随机种子
输出: 生成图像 x₀

1. c = CLIP_Encoder(y)                           ← 编码文本（一次性）
2. z_T ~ N(0, I)                                 ← 潜空间纯噪声
3. for t = T_step down to 1:                     ← DDIM 采样（50 步）
4.     ε_pred = ε_θ(z_t, t, c)                   ← 预测噪声
5.     z_{t-1} = DDIM_step(z_t, ε_pred, t)       ← 去噪一步
6. x₀ = VAE_Decoder(z₀)                          ← 解码到像素空间
7. return x₀
```

**整个流程：** Text → CLIP（77 tokens）→ U-Net × 50 步 → VAE Decoder → 图像

---

## 六、LDM 的局限与 SD 生态的演进方向

| 局限 | 具体表现 | 后续改进 |
|------|---------|---------|
| **CLIP 文本理解有限** | 复杂 prompt、长文本、否定表达理解差 | T5-XXL、GLM 等更强编码器 |
| **U-Net 的 Scaling Law 瓶颈** | 堆参数到 3.5B（SDXL）后收益递减 | DiT/MM-DiT 替代 U-Net |
| **4ch VAE 细节损失** | 小字模糊、纹理失真 | 16ch VAE（SD3/FLUX）|
| **Cross-Attention 单向** | 文本→图像单向，图像无法影响文本理解 | MM-DiT 双向交互 |
| **DDPM/DDIM 推理慢** | 50 步仍然不够快 | Rectified Flow（1-4 步）|

**LDM 之后，SD 生态沿着两条路线进化：**
1. **U-Net 路线（2022-2023）**：SD 1.5 → SD 2.x → SDXL——不断优化 U-Net、CLIP、VAE
2. **DiT 路线（2024-2026）**：MM-DiT → FLUX——用 Transformer 彻底替代 U-Net

---

## 七、总结

> **LDM 的核心贡献可以浓缩为三句话：**
> 1. **VAE 把扩散模型搬到潜空间**——计算量降低 48×，让消费级 GPU 可以跑
> 2. **Cross-Attention 把文本条件注入 U-Net**——第一次让扩散模型真正"听懂了"文字
> 3. **VQGAN-inspired 的 U-Net 架构**——为后来所有 SD 版本提供了可扩展的技术框架

LDM 之后的所有 SD 版本（1.4、1.5、2.x、SDXL），本质上都是在 LDM 的三件套（VAE + U-Net + CLIP + Cross-Attention）上进行**组件替换和规模放大**——没有改变底层设计。直到 2024 年 DiT 路线的成熟才打破了这套框架。

---

**Sources:**
- [LDM: High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Rombach et al. 2021
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — Radford et al. 2021（CLIP）
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al. 2015
- [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841) — Esser et al. 2021（VQGAN）
