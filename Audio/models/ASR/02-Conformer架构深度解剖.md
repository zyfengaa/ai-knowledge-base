# Conformer 架构深度解剖

> Google | 卷积增强 Transformer —— ASR 编码器的事实标准

---

## 写在前面

Conformer（Gulati et al., 2020）的贡献可以一句话概括：

> **"发现纯 Self-Attention 做 ASR 编码器不够好，CNN + Self-Attention 并联才是正解。"**

这个判断如此准确，以至于 Conformer 发布后迅速取代了 Speech-Transformer，成为 ASR 编码器的**事实标准**。此后的所有主流 ASR 模型——Whisper、SenseVoice、GLM-ASR、Qwen2-Audio、Qwen3-ASR 的 AuT——编码器都是 Conformer 或其变体。

---

## 第一章 前置条件：纯 Transformer 做 ASR 编码器为什么不够

### 1.1 2018: Speech-Transformer 的突破与不足

2018 年，Google 将 Transformer（Vaswani et al., 2017）引入 ASR，提出了 Speech-Transformer。它的 Encoder 是纯 Self-Attention + FFN 结构：

```
纯 Transformer Encoder Block:
  Input → MHA (Self-Attention) → FFN → Output
```

**优势**：Self-Attention 可以捕捉任意距离的帧间依赖，且可以并行计算（不像 RNN 需要串行）。

**很快发现了一个问题**——ASR 编码器需要同时做好两件事：

| 建模类型 | 需要捕获什么 | 擅长者 | 在 ASR 中的重要性 |
|---------|-------------|-------|-----------------|
| **局部模式** | 相邻帧的音素过渡（如 "b" → "a" 的共振峰过渡）、声道共鸣变化 | **CNN** | 极高——音素识别的核心 |
| **全局上下文** | 整句的句法结构、说话人语速、话题一致性 | **Self-Attention** | 高——但只在消歧义时关键 |

纯 Transformer 的问题：**Self-Attention 在浅层也会看到很远的地方，对相邻帧的局部关系建模效率低**。一个直观的类比——你在辨认"ba"和"pa"的区别时（送气 vs 不送气），关心的只是那个瞬间的几帧，而不是全世界的帧。

### 1.2 已有的局部建模方案

在 Conformer 之前，已经有几种尝试补全局部建模的方案：

| 方案 | 做法 | 问题 |
|------|------|------|
| **Transformer + Conv 前处理** | 用 CNN 做频谱前端提取，再送入 Transformer | 局部信息在注意力中仍会被稀释 |
| **相对位置编码** | Transformer-XL 的 RPE | 改进了位置感知，但没增加局部连接偏置 |
| **限制注意力窗** | 让 Self-Attention 只能看前后 K 帧 | 放弃了全局能力的优势 |

Conformer 的直觉：**不要选边站——让 CNN 和 Self-Attention 各司其职，并联互补。**

---

## 第二章 Conformer Block 深度解剖

### 2.1 宏观结构：Macaron 三明治

Conformer Block 的核心是 **Macaron-Net 风格的三明治结构**：

```
输入 x_i
    │
    ├── ½ × FFN₁ (half-step feed-forward)
    │   └── x̃_i = x_i + ½·FFN(x_i)
    │
    ├── MHSA (Multi-Head Self-Attention)
    │   └── x'_i = x̃_i + MHSA(x̃_i)
    │
    ├── Conv Module (卷积模块)
    │   └── x''_i = x'_i + Conv(x'_i)
    │
    ├── ½ × FFN₂ (half-step feed-forward)
    │   └── x'''_i = x''_i + ½·FFN(x''_i)
    │
    ├── LayerNorm (后归一化)
    │
    └── 输出 y_i = LayerNorm(x'''_i)
```

**为什么叫 Macaron？** 马卡龙饼干是两片外壳夹一片馅料。这里的"两片外壳"是两个半份 FFN，"馅料"是 Self-Attention + Conv 的组合。论文的消融实验证明：**两个半份 FFN + 共享参数总量等价于一个全量 FFN，但效果更好。**

### 2.2 子模块一：Multi-Head Self-Attention

Conformer 沿用 Transformer-XL 的 **相对位置编码（Relative Positional Encoding, RPE）**，而不是绝对位置编码。

**绝对位置编码 vs 相对位置编码的区别**：

```
绝对位置编码:
  Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V
  其中 K = K_content + K_position(position_embedding[pos])
  
相对位置编码:
  Attention = softmax(Q·Kᵀ_content + Q·Rᵀ_relative + 其他偏置项)
  注意力权重 = "内容之间的匹配" + "位置之间的相对距离"
```

RPE 的关键优势：**在推理时可以处理比训练时更长的序列**（位置编码不会被固定在某个绝对位置上）。对于 ASR 任务，这意味着模型可以处理任意长的音频。

### 2.3 子模块二：Convolution Module（核心创新）

Convolution Module 是 Conformer 与纯 Transformer 最大的区别：

```
输入
  │
  ├── LayerNorm
  │
  ├── Pointwise Conv1d (1×1, 扩张因子=2)
  │   └── 将通道数翻倍，为 GLU 做准备
  │
  ├── GLU (Gated Linear Unit)
  │   └── 将通道分裂为 A 和 B → g = A ⊙ σ(B)
  │   └── 门控机制控制信息流通过量
  │
  ├── 1D Depthwise Conv1d (kernel=32, causal)
  │   └── 轻量级卷积：每个通道独立卷积核
  │   └── 感受野 = 32 帧 ≈ 320ms 音频
  │
  ├── BatchNorm
  │
  ├── Swish 激活
  │
  ├── Pointwise Conv1d (1×1)
  │   └── 投影回原始通道数
  │
  └── Dropout
```

#### 各步的详细意图：

**Step 1: LayerNorm**——与常见的 Pre-Norm 结构一致，稳定训练。

**Step 2: Pointwise Conv + 通道扩张**——把通道数从 d 扩大到 2d。这些额外的通道将在 GLU 中被用作为"门"。

**Step 3: GLU**——门控线性单元：

```
输入分为两半: A = X[:, :d], B = X[:, d:2d]
输出: Y = A ⊙ σ(B)
```

σ(B) ∈ (0, 1) 起到"阀门"作用——A 中的哪些信息可以通过由 σ(B) 决定。这比 ReLU 的"硬截断"更灵活**，模型可以学习到哪些局部模式要保留、哪些要抑制。

**Step 4: Depthwise Conv1d**——这是"让模型看到局部"的关键操作。**Depthwise 卷积每个通道只有一个卷积核**，不像标准卷积每个核同时看所有通道。参数量为 `kernel_size × channels`（标准卷积是 `kernel_size × in_channels × out_channels`）。kernel_size=32 在大规模消融中表现最佳。

**Step 5: BatchNorm**——注意这是 BN 不是 LN。BN 在小 batch 时不稳定，但对于 1D 卷积的时序任务效果不错。Swish 激活（`x · σ(x)`）因其非单调性在此处优于 GELU。

**Step 6: Pointwise Conv 回投影**——把通道数恢复为 d。

### 2.4 参数配置

| 规格 | 参数总量 | Encoder Blocks | 隐藏维度 | Attention Heads | FFN 维度 |
|------|---------|---------------|---------|----------------|---------|
| **Conformer S** | ~10.3M | 16 | 144 | 4 | 576 |
| **Conformer M** | ~30.7M | 16 | 256 | 4 | 1024 |
| **Conformer L** | ~118.8M | 17 | 512 | 8 | 2048 |
| **Whisper Large 级** | ~327M | 32 | 1280 | 20 | 5120 |

### 2.5 消融实验的关键发现

论文作者做了一系列消融实验，结果揭示了 Macaron 设计的有效性：

| 配置 | 参数量 | WER (dev/test) | 结论 |
|------|-------|---------------|------|
| 全 Conformer | 118.8M | **2.1/4.3** | 基线 |
| - 去掉卷积模块 | 96.7M | 2.6/5.0 | **+0.5/+0.7** ← 卷积贡献显著 |
| - 去掉 Macaron（单 FFN） | 96.7M | 2.3/4.6 | +0.2/+0.3 |
| - 替换 ReLU FFN | 118.8M | 2.3/4.5 | +0.2/+0.2 |
| - 替换绝对位置编码 | 118.8M | 2.2/4.5 | +0.1/+0.2 |
| Depthwise Conv kernel=32 vs 7 | 118.8M | **2.1/4.3** vs 2.2/4.5 | 更大感受野更好 |

**最关键的结论**：去掉卷积模块的 Transformer → WER 提升 0.7。这直接量化和验证了"纯 Self-Attention 做 ASR 编码器不够好"这一核心判断。

---

## 第三章 Conformer 的后继影响

### 3.1 被哪些模型采用

| 模型 | 年份 | 编码器 | 与 Conformer 的关系 |
|------|------|-------|-------------------|
| **Whisper** | 2022 | Conformer 32 层 | 直接使用，仅改动了前端 Conv 和位置编码 |
| **SenseVoice** | 2024 | SANM（70 层） | 类似思路：Self-Attn + FSMN conv，比 Conformer 更轻量 |
| **GLM-ASR** | 2024 | Conformer 32/12 层 | 直接使用，增加了部分 RoPE |
| **Qwen2-Audio** | 2024 | Whisper-large-v3 Encoder | 间接使用了 Conformer 架构 |
| **Qwen3-ASR AuT** | 2026 | Audio Transformer | 保留了 Conv 下采样 + Self-Attn 结构 |
| **WeNet** | 2021 | Conformer | 直接使用 U2 框架支持 Conformer 和 Transformer |

### 3.2 Conformer 后的一些变体

| 变体 | 年份 | 改进 |
|------|------|------|
| **Squeezeformer** | 2022 | 降低计算量：简化结构、替换时间下采样位置、更合理的激活函数选择 |
| **Branchformer** | 2022 | 用两个分支（全局 Attention + 局部 MLP）代替卷积模块 |
| **E-Branchformer** | 2023 | 改进 Branchformer 的双分支合并方式 |
| **Conformer XL** | 2023 | 更大更深，参数量达到 600M+ |

所有这些变体都没有取代 Conformer 的地位——Conformer 仍然是最广泛使用的编码器架构。

---

## 第四章 一张图总结

```
纯 Transformer (2018)
  └── 全局能力强，局部建模弱
      │
需要解决的问题: Self-Attention 看太远了，
               相邻帧的局部关系建模效率低
      │
Conformer 的解决方案:
  ┌─────────────────────────────────────┐
  │         FFN₁ (½ step)               │
  │              ↓                       │
  │     Multi-Head Self-Attention        │  ← 全局上下文
  │         (相对位置编码)               │
  │              ↓                       │
  │     ┌────────────────────┐           │
  │     │ Convolution Module │           │  ← 局部模式
  │     │ LN → PWConv → GLU  │           │
  │     │ → DWConv → BN →    │           │
  │     │ Swish → PWConv     │           │
  │     └────────────────────┘           │
  │              ↓                       │
  │         FFN₂ (½ step)               │
  │              ↓                       │
  │         LayerNorm                    │
  └─────────────────────────────────────┘
         ↑  "各司其职，并联互补"
```

**一句话总结 Conformer：在 Transformer 的 Self-Attention 旁边并联一个深度可分离卷积，让全局和局部各司其职，然后用 Macaron 结构粘在一起。**

---

**Sources:**
- [Conformer: Convolution-augmented Transformer for Speech Recognition (arXiv:2005.08100)](https://arxiv.org/abs/2005.08100)
- [Speech-Transformer (arXiv:1804.00067)](https://arxiv.org/abs/1804.00067)
- [Squeezeformer (arXiv:2206.00888)](https://arxiv.org/abs/2206.00888)
- [Branchformer (arXiv:2207.02971)](https://arxiv.org/abs/2207.02971)
- [Espnet Conformer Implementation](https://github.com/espnet/espnet/blob/master/espnet2/asr/encoder/conformer_encoder.py)
- [Transformer-XL: Relative Positional Encoding (arXiv:1901.02860)](https://arxiv.org/abs/1901.02860)
