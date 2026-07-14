# Qwen3-ASR 架构深度解剖

> 阿里巴巴通义千问团队出品 | 开源最强多语种语音识别模型（Apache 2.0）

---

## 写在前面：两种不同的产品

Qwen3-ASR 系列目前存在**两个不同的产品线**，本文分析覆盖后者：

| 产品 | 发布时间 | 开源状态 | 定位 |
|------|---------|---------|------|
| **Qwen3-ASR-Flash** | 2025.09 | 闭源，仅 API | DashScope 云端实时 API，参数量未公开 |
| **Qwen3-ASR** (开源版) | 2026.01 | **开源 (Apache 2.0)**，1.7B / 0.6B | 可本地部署，SOTA 精度 |

> **注意**：Flash 版本是仅 API 的闭源产品，与开源的 Qwen3-ASR 是两个独立模型。本文所有分析均针对 **2026 年 1 月开源的 Qwen3-ASR**。

开源版本分为三个子模型：

| 模型 | 总参数量 | 核心用途 |
|------|---------|---------|
| **Qwen3-ASR-1.7B** | ~2B | 高精度 ASR + 语言识别，SOTA |
| **Qwen3-ASR-0.6B** | ~0.9B | 高效 ASR，RTF=0.064，2000× 实时比 |
| **Qwen3-ForcedAligner-0.6B** | ~0.9B | NAR 词级时间戳对齐，平均误差 42.9ms |

---

## 前提：遗留三大问题

在 Qwen3-ASR 之前，ASR 领域存在三个长期未解决的痛点：

### 1. 流式 vs 离线 —— 仍需两套部署

```
传统方案:
  WeNet U2: 统一训练，但推理仍需选择 mode (static/streaming)
  Whisper: 天然不支持流式
  SenseVoice: 流式精度下降明显

本质矛盾:
  流式 → 小窗口 + 低延迟 → 精度下降
  离线 → 全上下文 → 高精度 + 高延迟
  → 需要两套模型 / 两套部署
```

### 2. 方言覆盖 —— Whisper 覆盖 99 种语言但方言很差

| 模型 | 粤语 CER | 方言覆盖 |
|------|---------|---------|
| Whisper large-v3 | ~10.9% | 仅覆盖标准语言，无细分方言 |
| SenseVoice | ~9.8% | 50+ 语言，方言有限 |
| 目标 (Qwen3-ASR) | **<8%** (实际 7.32%) | **22 种中国方言 + 英语口音** |

### 3. 歌唱识别 —— 没有 ASR 模型能处理音乐/歌声

- 传统 ASR 在 BGM 下 WER > 50%
- 人类说话和歌唱的声学特征差异极大（音高范围、持续时长、共振峰分布）
- 无开源模型将歌唱识别作为一等功能

Qwen3-ASR 在这三个问题上都取得了突破性进展。

---

## 一、整体架构设计哲学

Qwen3-ASR 采用 **Encoder-Projector-Decoder 三段式架构**，整体设计理念可概括为：

> **"重编码、精投影、强解码"**

这套思想落地为非对称的 Audio Transformer + LLM 联合架构：

```
原始音频 (16kHz 单声道)
    │
    ├─ ① Mel 频谱前端 ─── 128-bin log-mel，25ms 窗 / 10ms 步长，100Hz
    │
    ├─ ② Conv2D 下采样 ── 3 层 stride=2 → 8× 时间下采样 (100Hz → 12.5Hz)
    │
    ├─ ③ AuT Encoder ──── 24/18 层双向 Transformer (FlashAttention + 动态窗口)
    │
    ├─ ④ Projector ────── LayerNorm → GELU → Linear → Linear (对齐到 LLM 空间)
    │
    ├─ ⑤ Qwen3 Decoder ── 28 层自回归 Transformer (GQA + SwiGLU + RMSNorm)
    │
    └─ 输出文本 Token (151,936 vocab)
```

### 为什么是 Encoder-Projector-Decoder？

Qwen3-ASR 的架构选择与 GLM-ASR 有相似的哲学——**编码器专注于声学理解，解码器专注于语言生成**，但与 GLM-ASR 不同的是：

| 维度 | Qwen3-ASR | GLM-ASR-Nano |
|------|-----------|-------------|
| **编码器结构** | Conv2D + 双向 Transformer (AuT) | Conv1D + 双向 Transformer (Whisper 风格) |
| **下采样方式** | 3× Conv2D stride=2 (8×) | 1× Conv1D stride=2 + Projector Pooling(4×) |
| **下采样率** | **8×** (100Hz → 12.5Hz) | **8×** (100Hz → ~12.5Hz) |
| **投影器** | 简单 2 层 Linear (无额外压缩) | 4× Pooling + 3 层 MLP |
| **解码器** | Qwen3 LLM 28 层 (含 Cross-Attention?) | 6 层自研 / 28 层 LLaMA |
| **流式支持** | ✅ **原生统一** (动态窗口 1-8s) | ❌ 不支流式 (编码器非因果) |

> **关键区别**：Qwen3-ASR 通过**动态 FlashAttention 窗口**实现了单一模型同时支持流式和离线，这是 GLM-ASR 不具备的能力。

---

## 二、各模块深度解剖

### 2.1 音频前端（Mel Spectrogram）

**定位**：将 16kHz 原始波形转为 128 维 log-mel 频谱特征，帧率 100Hz。

```
16kHz 原始波形
    │
    ├── 分帧: 25ms 窗口 (400 采样点), 10ms 步长 (160 采样点)
    ├── FFT: 512 点 → 功率谱
    ├── Mel 滤波: 128 个 Mel 滤波器 (Slaney scale, 0-8000Hz)
    ├── 对数压缩: log10(clamp(mel, min=1e-10))
    ├── 动态范围裁剪: max(log_spec, log_spec.max() - 8.0)
    └── 归一化: (x + 4.0) / 4.0
    
输出: [batch, 128, T]    T = 帧数 ≈ 100 × 音频秒数
```

| 参数 | 值 |
|------|-----|
| 采样率 | 16000 Hz |
| Mel 滤波器数 | 128 |
| FFT 窗口大小 (n_fft) | 400 采样点 (25ms) |
| 步长 (hop_length) | 160 采样点 (10ms) |
| Mel 尺度 | Slaney scale |
| 频率范围 | 0 - 8000 Hz |
| 帧率 | **100 Hz** |

**与 Whisper 的异同**：整体遵循 Whisper 风格的前端处理（包括 Slaney Mel 尺度和动态范围裁剪），但追加了 `(x + 4.0) / 4.0` 的归一化步骤，有助于训练稳定性。

---

### 2.2 Conv2D 下采样（3 层，8×）

**定位**：将频谱从声学特征空间投影到 Transformer 隐藏空间，同时完成 **8× 时间下采样**。

这是 Qwen3-ASR 与 GLM-ASR 在结构上的**第一个重要区别**——Qwen3-ASR 使用 **Conv2D**（同时压缩频率维度和时间维度），而 GLM-ASR 使用 Conv1D（仅压缩时间维度）。

```
输入: [batch, 1, 128, T]   (channel=1, mel_bins=128, time_frames=T)
    │
    ├── Conv2D(1 → 480, kernel=3×3, stride=2, padding=1)
    │   └── GELU 激活
    │   [batch, 480, 64, T/2]
    │
    ├── Conv2D(480 → 480, kernel=3×3, stride=2, padding=1)
    │   └── GELU 激活
    │   [batch, 480, 32, T/4]
    │
    ├── Conv2D(480 → 480, kernel=3×3, stride=2, padding=1)
    │   └── GELU 激活
    │   [batch, 480, 16, T/8]
    │
    ├── Reshape: [batch, T/8, 480 × 16] = [batch, T/8, 7680]
    │   └── 频率维 16 个 bin 全部拼接到通道维
    │
    └── Linear(7680 → d_model, no bias)  →  [batch, T/8, d_model]
```

| 参数 | 值 | 含义 |
|------|-----|------|
| Conv2D 层数 | 3 | 每层 stride=2 |
| 频率压缩 | **128 → 64 → 32 → 16** | 最终只剩 16 个 frequency bin |
| 时间压缩 | T → T/2 → T/4 → **T/8** | **8× 时间下采样** |
| 输出序列维度 | 7680 (480×16) | 将频率扁平化为特征维 |
| 最终 Linear | 7680 → d_model | 无 bias |
| 激活函数 | GELU | 贯穿所有 Conv2D 层 |

**为什么用 Conv2D 而非 Conv1D？**

```
Conv1D (GLM-ASR):
  只压缩时间轴 → 保留全部 128 维频率信息
  输出维度: 1280 (通过 Conv1D 通道扩展)
  → 编码器需要自己学习频率结构

Conv2D (Qwen3-ASR):
  同时压缩时间和频率 → 逐层抽象频率模式
  频率维: 128 → 64 → 32 → 16
  → 形成频率层次化表示，编码器负担更轻
  
本质: Conv2D = 内置了"频率局部性"先验
```

#### 分块处理（Per-Chunk Convolution）

这是一个**关键的实现细节**：Conv2D 不是在整段频谱上一次性计算的，而是按 **chunk 分块处理**。

```
输入频谱 T 帧
    │
    ├── 按 100 帧分块: chunk_1 [0:100], chunk_2 [100:200], ...
    │
    └── 每个 chunk 独立通过 3× Conv2D:
        chunk (100 帧) → Conv2D → 产出 13 个 token
        └── 100/8 ≈ 12.5 → 实际输出 13 tokens/chunk
```

| 属性 | 值 |
|------|-----|
| Chunk 大小 | 100 帧 (≈ 1 秒音频) |
| 输出 token 数 / chunk | 13 (100/8 ≈ 12.5 → 13) |
| Token 帧率 | **12.5 Hz** (≈ 80ms / token) |

---

### 2.3 AuT Encoder（Audio Transformer 编码器）

这是 Qwen3-ASR 最核心的模块，也是命名中 **"AuT"** 的由来。

#### 模型规格对比

| 参数 | 1.7B 版本 | 0.6B 版本 |
|------|----------|-----------|
| `d_model` | **1024** | **896** |
| `num_layers` | **24** | **18** |
| `num_attention_heads` | **16** | **14** |
| `head_dim` | **64** | **64** |
| `intermediate_size` (FFN) | **4096** (4×) | **3584** (4×) |
| 编码器参数量 | **~300M** | **~180M** |

#### 每个 Encoder Layer 的内部分解

```
输入: x [batch, seq_len, d_model]
    │
    ├── LayerNorm (含 bias，区别于解码器的 RMSNorm)
    │
    ├── Multi-Head Self-Attention (双向 / 非因果)
    │   ├── Q, K, V 投影均带 bias（与解码器不同！）
    │   ├── sinusoidal 位置编码（非 RoPE，按 chunk 重置）
    │   ├── FlashAttention + 动态窗口 (1-8s)
    │   │   └── 块对角注意力掩码，跨 chunk 不可见
    │   └── Output 投影 (含 bias)
    │
    ├── + 残差连接
    │
    ├── LayerNorm (含 bias)
    │
    ├── FFN (MLP)
    │   ├── fc1: d_model → intermediate_size (含 bias)
    │   ├── GELU 激活
    │   └── fc2: intermediate_size → d_model (含 bias)
    │
    └── + 残差连接
```

#### 关键设计细节

**1. 位置编码：Sinusoidal（非 RoPE）**

与 Qwen3 解码器的 MRoPE 不同，AuT 编码器使用**传统的正弦位置编码**：

```python
log_timescale_increment = log(10000) / (d_model/2 - 1)
inv_timescales = exp(-arange(d_model/2) * log_timescale_increment)
pe = concat(sin(pos * inv_timescales), cos(pos * inv_timescales))
# 形状: [seq_len, d_model]
```

**每 chunk 位置重置**：Encoder chunk 中的位置从 0 开始，而非全局递增。这意味着：
- 每个 chunk (100 帧 → 13 tokens) 独立编码位置
- chunk 之间的位置互相独立
- 窗口化注意力确保了跨 chunk 不会产生位置混淆

**2. 动态 FlashAttention 窗口（核心创新）**

这是 Qwen3-ASR **最重要的架构创新**——一个模型同时支持流式和离线。

```
训练阶段:
  动态窗口 1s ~ 8s 随机采样
  → 模型学会适应不同上下文的注意力模式

推理阶段:
  离线模式: 窗口 = 8s (104 tokens) → 全上下文，最高精度
  流式模式: 窗口 = 2s (26 tokens) → 低延迟，约 92ms TTFT
```

```
注意力掩码示意 (8s 窗口, 4 chunks):

Chunk 0 [tokens 0:13]    ←→ [tokens 0:104]  ← 可看到前 8s
Chunk 1 [tokens 13:26]   ←→ [tokens 0:104]
Chunk 2 [tokens 26:39]   ←→ [tokens 0:104]
  ...    ...                     ...
Chunk 8 [tokens 104:117] ←→ [tokens 104:208] ← 滑到下一窗口

注意: 超出窗口的 token 不可见 → 块对角注意力掩码
```

| 窗口 | Token 数 | 特点 |
|------|---------|------|
| 1s 窗口 | 13 tokens | 极限低延迟，适用于极实时场景 |
| 2s 窗口 | 26 tokens | 推荐流式 (92ms TTFT) |
| 4s 窗口 | 52 tokens | 平衡模式 |
| **8s 窗口** | **104 tokens** | 离线模式 (默认推理配置) |

**3. 编码器投影（proj1 + proj2）**

编码器输出最后的投影层，将 AuT 隐藏空间映射到解码器 LLM 的嵌入空间：

```
h = LayerNorm(h, ln_post)        # 含 bias
h = GELU(h @ proj1 + b_proj1)    # d_model → d_model
h = h @ proj2 + b_proj2          # d_model → output_dim

1.7B: 1024 → 1024 → 2048        (匹配 Qwen3-1.7B hidden_size)
0.6B: 896 → 896 → 1024           (匹配 Qwen3-0.6B hidden_size)
```

该投影器仅 2 层 Linear + GELU，比 GLM-ASR 的 3 层 MLP + Pooling 更简洁，因为：
- Conv2D 已完成了 8× 下采样，无需额外压缩
- LLM 解码器足够强大，可以处理更长的音频 token 序列

---

### 2.4 Qwen3 LLM 解码器

解码器是基于 **Qwen3 系列语言模型**的自回归文本生成器。

#### 模型规格

| 参数 | 1.7B 版本 | 0.6B 版本 |
|------|----------|-----------|
| `hidden_size` | **2048** | **1024** |
| `num_hidden_layers` | **28** | **28** |
| `num_attention_heads` (Q) | **16** | **16** |
| `num_key_value_heads` (KV) | **8** | **8** |
| GQA 压缩比 | **2:1** (16Q / 8KV) | **2:1** (16Q / 8KV) |
| `head_dim` | **128** | **128** |
| `intermediate_size` (FFN) | **6144** (3×) | **3072** (3×) |
| `vocab_size` | **151,936** | **151,936** |
| 位置编码 | **MRoPE** (标准 RoPE) | MRoPE |
| 归一化 | **RMSNorm** (eps=1e-6) | RMSNorm |
| 激活函数 | **SwiGLU** | SwiGLU |
| Embedding 绑定 | **是** (tie_word_embeddings) | 是 |
| 是否含 bias | **否** (所有层无 bias) | 否 |

#### 每个 Decoder Layer 的内部分解

```
输入: [batch, seq_len, hidden_size]
    │
    ├── 逐头 Q/K RMSNorm (pre-RoPE 归一化——独特设计)
    │
    ├── Masked Self-Attention (因果/单向)
    │   ├── Q 投影 (16 头, 每头 128 维)
    │   ├── K 投影 (8 头)        ← GQA 2:1
    │   ├── V 投影 (8 头)        ← KV Cache 减半
    │   ├── 逐头 RMSNorm (Q/K 归一化)
    │   ├── MRoPE (标准 RoPE，theta=1,000,000)
    │   │   └── mrope_section=[24,20,20] (音频时全部相同)
    │   ├── FlashAttention + 因果掩码
    │   └── Output 投影
    │
    ├── + 残差连接
    │
    ├── RMSNorm
    │
    ├── SwiGLU MLP
    │   ├── gate_proj: hidden → intermediate (SiLU 激活)
    │   ├── up_proj:   hidden → intermediate
    │   ├── 逐元素乘法: gate_output × up_output
    │   └── down_proj: intermediate → hidden
    │
    └── + 残差连接
```

#### GQA（Grouped Query Attention）分析

Qwen3 解码器在 1.7B 和 0.6B 版本中统一使用 **GQA 2:1** (16Q / 8KV)：

```
标准多头注意力 (MHA):
  Q: 16 组, K: 16 组, V: 16 组
  → KV Cache = 16 × head_dim × seq_len × layers

GQA 2:1 (Qwen3-ASR):
  Q: 16 组, K: 8 组, V: 8 组
  → 每 2 组 Q 共享 1 组 K, V
  → KV Cache = 8 × 128 × seq_len × 28

收益: KV Cache 降到 MHA 的 1/2
```

与 GLM-ASR 的 GQA 4:1 (16Q/4KV) 相比，Qwen3 使用更保守的 2:1 比率，这意味着：
- KV Cache 减半（而非 75%）
- 注意力多样性损失更小 → 精度更高
- 这是 LLM 精度与推理效率之间的权衡选择

#### MRoPE（多维 RoPE）

Qwen3 系列使用 **MRoPE**（Multi-dimensional RoPE），原始设计用于多模态输入。在 Qwen3-ASR 中：

```
mrope_section = [24, 20, 20]   # 三个维度分别编码
head_dim = 128
  ├── 维度 1: 前 24 维 → 文本位置编码
  ├── 维度 2: 中 20 维 → 音频时间编码 (与维度 3 相同)
  └── 维度 3: 后 20 维 → 音频时间编码 (与维度 2 相同)

对于纯音频 ASR 场景:
  维度 1 = 维度 2 = 维度 3 → 退化为标准 RoPE
```

> **实际效果**：虽然 MRoPE 设计为多模态位置编码，但在纯语音场景下三个维度使用相同的值，等价于标准 RoPE (head_dim=64，与 GLM-ASR 的 50% partial RoPE 不同)。

**逐头 Q/K RMSNorm**：在 RoPE 旋转**之前**对 Q 和 K 进行归一化，这是 Qwen3 系列的一个独特设计，目的是稳定注意力分数的数值范围。

---

### 2.5 Tokenizer 与 Prompt 格式

#### 词汇表

| 参数 | 值 |
|------|-----|
| `vocab_size` | **151,936** |
| Tokenizer 类型 | Qwen3 tokenizer (基于 tiktoken) |
| Embedding 绑定 | 是 (embed_tokens = lm_head) |

#### 特殊 Token

| Token | ID | 用途 |
|-------|-----|------|
| `<\|endoftext\|>` | 151643 | 序列结束 |
| `<\|im_start\|>` | 151644 | 消息开始 |
| `<\|im_end\|>` | 151645 | 消息结束 |
| `<\|audio_start\|>` | 151669 | 音频段开始 |
| `<\|audio_end\|>` | 151670 | 音频段结束 |
| `<\|audio_pad\|>` | 151676 | 音频填充符（由编码器输出替换） |
| `<asr_text>` | 151704 | ASR 文本标记 |

#### Prompt 结构

```
<|im_start|>system\n<|im_end|>\n         ← system prompt（系统消息）
<|im_start|>user\n                       ← user 消息开始
<|audio_start|><|audio_pad|>×N<|audio_end|>  ← 音频 Token（N 个 pad 占位）
<|im_end|>\n                              ← user 消息结束
<|im_start|>assistant\n                   ← assistant 消息开始
```

**输出格式**：
```
language English<asr_text>The transcription.<|im_end|>
```

注意输出以 `language XX<asr_text>...` 格式开头，其中 `XX` 是检测到的语言，后面是 ASR 转录文本。这意味着模型在**识别语言的同时**也完成了语音转录。

---

## 三、训练管线：四阶段递进

Qwen3-ASR 的训练是迄今为止**最复杂的 ASR 训练管线**，包含四个阶段，从无监督预训练到强化学习对齐。

```
Stage 1: AuT Pre-training
  ├── 数据: ~4000 万小时伪标注 ASR 数据 (中 + 英)
  ├── 目标: 学习稳定的音频表示
  ├── 模型: AuT Encoder + 简单的 ASR head
  └── 输出: 预训练音频编码器权重

Stage 2: Omni Multi-task Pre-training
  ├── 数据: 3 万亿 tokens (音频 + 视觉 + 文本)
  ├── 目标: 训练 Qwen3-Omni 基座模型
  ├── 模型: 完整的 Qwen3-Omni (含 AuT Encoder + LLM)
  └── 输出: 多模态对齐的基座模型

Stage 3: ASR SFT (Supervised Fine-Tuning)
  ├── 数据: 多语种 + 流式增强 + context biasing 数据
  ├── 格式: 指令微调格式 (ChatML 模板)
  ├── 增强:
  │   ├── 多语言 ASR: 52 种语言/方言
  │   ├── 流式增强: 随机窗口截断训练
  │   └── Context Biasing: 参考文本引导
  └── 输出: ASR 专业化的指令跟随模型

Stage 4: RL (GSPO Reinforcement Learning)
  ├── 数据: ~5 万条精选话语
  ├── 算法: Group Sequence Policy Optimization
  ├── 目标: 噪声鲁棒性 + 转录稳定性
  └── 输出: 最终 Qwen3-ASR 模型
```

### Stage 1: AuT 预训练

- **规模**：约 40M 小时的伪标注 ASR 数据
- **数据组成**：主要来自中文和英文的弱标注语音数据
- **训练目标**：标准的 ASR 损失（CTC / 交叉熵）
- **意义**：在 PaLM/Qwen 式的"预训练然后微调"范式中，这是 AuT Encoder 的"预训练阶段"——在注入海量 ASR 数据后，编码器学会了稳定的声学表征，为后续多模态训练打下基础。

### Stage 2: Omni 多任务预训练

这是 Qwen3-ASR **最独特的训练阶段**——它并不是像传统 ASR 那样独立训练，而是作为 **Qwen3-Omni 多模态模型的一部分**进行联合训练。

- **规模**：3 万亿 tokens（涵盖音频、视觉、文本）
- **数据组成**：语音识别、图像理解、视频理解、多模态对话等
- **意义**：这让 ASR 能力与 LLM 语言能力深度融合——解码器不仅学会了"将音频映射到文本"，还学会了"理解上下文、遵循指令"，为后续的 Context Biasing 和多语言指令控制奠定基础。

### Stage 3: ASR SFT

在 Omni 基座之上进行 ASR 任务的专业化微调：

- **多语言 ASR 数据**：覆盖 52 种语言/方言
- **流式增强**：训练时随机使用 1-8s 的窗口截断，让模型学会在不同上下文下工作
- **Context Biasing 数据**：构造包含参考文本（如关键词列表）的样本，让模型学会利用外部上下文引导识别
- **指令格式**：使用 ChatML 模板，统一 ASR 输入输出格式

### Stage 4: GSPO —— RL 微调（核心创新）

这是 **ASR 领域首次公开使用强化学习进行模型微调**，GSPO 最早由 Qwen 团队提出（arXiv: 2507.18071）。

#### GSPO vs GRPO 对比

| 维度 | GRPO | **GSPO** |
|------|------|----------|
| 重要性比率 (IS Ratio) | Token 级别 | **序列级别**（含长度归一化） |
| 优势/奖励计算 | 每个 token 共享同一优势 | **序列级别组内相对优势** |
| 裁剪粒度 | 逐 Token 裁剪 | **整条序列裁剪** |
| 训练稳定性 | 长序列易崩溃 | ✅ **稳定** |
| MoE 训练 | 需 Routing Replay | ✅ **无需额外策略** |

#### GSPO 核心公式

$$s_i(\theta) = \left( \frac{\pi_{\theta}(y_i|x)}{\pi_{\theta_\text{old}}(y_i|x)} \right)^{\frac{1}{|y_i|}}$$

通过长度归一化的序列似然比，消除了 GRPO 中逐 token 重要性采样引起的高方差噪声。

#### 在 ASR 中的应用

**奖励设计**：
- 主要奖励：WER/CER 的负对数（越低越好）
- 辅助奖励：转录稳定性（同一音频多次推理的一致性）
- 语言 ID 准确率（52 种语言分类正确性）

**训练数据**：约 5 万条精心挑选的语音片段，重点关注：
- 高噪声场景
- 方言口音极重的话语
- 歌唱/音乐场景
- 多语言混杂话语

---

## 四、核心创新点详细解析

### 4.1 动态 FlashAttention 窗口（流式 + 离线统一）

这是 Qwen3-ASR **最重要的工程创新**。

```
传统方案:
  ┌──────────────────┐
  │ 流式模型  │  小窗口   │  低延迟，中度精度
  └──────────────────┘
  ┌──────────────────┐
  │ 离线模型  │  全音频   │  高延迟，最高精度
  └──────────────────┘
  → 维护两套模型，两套部署 → 成本翻倍

Qwen3-ASR 方案:
  ┌─────────────────────────────────────┐
  │ 同一模型 │ 动态窗口 1s~8s           │
  │         │ 流式 → 2s窗口 → 92ms TTFT│
  │         │ 离线 → 8s窗口 → 最高精度  │
  └─────────────────────────────────────┘
```

**技术实现**：
1. 训练时随机采样窗口大小（1-8s）
2. FlashAttention 使用块对角掩码——每个 token 只能看到窗口内的 token
3. 推理时通过 `n_window_infer=800`（8s）的固定窗口直接运行

**流式推理细节**：

```
流式推理步骤:
  1. 接收 2s 音频 → 200 帧 Mel → 26 tokens
  2. 编码器处理（窗口内自注意力）
  3. 解码器生成 → 5-token rollback
     └── 新模式: 生成最后 5 个 token 被丢弃并重新解码
  4. KV Cache 保留最近 4 个 chunk (≈8s)
  5. 输出文本 → 接收下一个 2s 音频 → 重复
```

| 模式 | 窗口 | TTFT | WER (LibriSpeech other) | 场景 |
|------|------|------|------------------------|------|
| 流式 | 2s | ~92ms | 4.51% | 实时对话 |
| 离线 | 8s | ~500ms | 3.38% | 离线转录 |

> 流式模式下 WER 仅增加 1.13pp (3.38% → 4.51%)，性能损失极小，这是统一架构的显著优势。

### 4.2 52 种语言/方言 —— 开源 ASR 最广方言覆盖

**语言覆盖矩阵**：

```
Qwen3-ASR 语言覆盖 (52 种):
  ├── 标准语言 (30 种):
  │   ├── 英语 (含美式/英式/澳式/印度/...口音)
  │   ├── 中文普通话
  │   ├── 日语、韩语、法语、德语、西班牙语...
  │   └── 共 30 种标准语言
  │
  └── 中国方言 (22 种):
      ├── 北方: 东北话、河北话、河南话、山东话、陕西话、山西话、天津话、宁夏话、甘肃话
      ├── 西南: 四川话、云南话、贵州话
      ├── 中部: 湖北话、湖南话、安徽话、江西方言
      ├── 东南: 福建话、浙江话
      └── 南方: 粤语(香港)、粤语(广东)、吴语、闽南语
```

| 维度 | Qwen3-ASR-1.7B | Whisper-large-v3 | 提升 |
|------|---------------|-----------------|------|
| 语言数 | 52 (30标准+22方言) | 99 (标准语言) | — |
| 方言覆盖 | **22 种中国方言** | 无细分方言 | **独家** |
| 粤语 CER | **~7.32%** | ~10.9% | **-3.6pp** |
| 中文方言平均 | **15.94%** | ~20%+ | **~4pp** |
| 语言 ID 准确率 | **97.9%** | 94.1% | +3.8pp |

### 4.3 歌唱识别 —— 业界首创

Qwen3-ASR 是**首个将歌唱识别作为一等功能的开源 ASR 模型**。

| 数据集 | Qwen3-ASR-1.7B | GPT-4o-Transcribe | Doubao-ASR | 说明 |
|--------|---------------|------------------|-----------|------|
| **M4Singer** (独唱) | **5.98%** | 16.77% | 7.88% | 纯人声歌唱 |
| **EntireSongs-zh** (中文歌曲) | **13.91%** | 34.86% | 23.99% | 含 BGM |
| **EntireSongs-en** (英文歌曲) | **14.60%** | 30.71% | 33.51% | 含 BGM |

```
M4Singer 独唱: Qwen3-ASR < 6% WER
  歌唱识别首次进入"实用级"精度

含 BGM 歌曲: Qwen3-ASR ~14% WER
  超越 GPT-4o 2 倍 (34.86% → 13.91%)

Singing + BGM < 15% WER ← 这一级别首次实现
```

**技术原因分析**：
1. AuT 编码器的 Conv2D 在频率维度逐层抽象，保留了声调/旋律特征
2. Omni 多模态训练让模型接触了大量音乐数据
3. 40M 小时预训练数据中包含了大量唱歌/音乐片段
4. GSPO 强化学习中对歌唱场景进行了专项优化

### 4.4 Context Biasing（上下文偏置）

**功能**：用户可以提供任意长度的背景文本（关键词列表、整篇文档），模型会自动利用这些信息引导识别。

```
传统 ASR:
  热词列表 → 独立的热词模块（WFST / 独立语言模型）
  局限性: 仅支持少量热词，无法理解完整上下文

Qwen3-ASR Context Biasing:
  输入: 音频 + "会议主题是量子计算，关键词包括：
        薛定谔方程、量子纠缠、退相干、量子比特..."
  输出: 音频中出现 "薛定谔方程" 自动偏好识别

  甚至可以是整篇文档:
  输入: 音频 + 患者的完整病历文档
  输出: 医学术语识别准确率大幅提升
```

| 属性 | 值 |
|------|-----|
| 支持的最大上下文长度 | **~10,000 tokens** (API 版本) |
| 训练方式 | SFT 阶段注入 biasing 数据 |
| 实现方式 | LLM 解码器的指令跟随能力——无需额外模块 |
| 可添加内容 | 关键词列表、完整文档、对话历史 |

**技术秘密**：Context Biasing 实际上利用了 **Qwen3 LLM 解码器**的指令跟随能力——将背景文本作为 System Prompt 或对话上下文注入，LLM 在进行 ASR 解码时自动"注意"到这些上下文，从而引导解码方向。这完全不需要额外的热词 WFSG 模块，是纯语言模型能力的体现。

### 4.5 Qwen3-ForcedAligner（非自回归时间戳对齐）

这是 Qwen3-ASR 套件中的一个**独立创新**——基于非自回归 LLM 的词级时间戳对齐器。

```
传统对齐:
  WhisperX: 自回归 + 动态规划 → 133ms 平均误差
  MFA: 高斯混合模型 → 130ms 平均误差

Qwen3-ForcedAligner:
  输入: 音频 + 文本 → 输出: 每个词的 [开始, 结束] 时间戳
  方式: NAR slot-filling (非自回归槽填充)
  架构: 基于 Qwen3-0.6B 改造
  精度: 42.9ms 平均误差 (比 WhisperX 低 67%)
  速度: RTF ≈ 0.001 (每秒处理 ~1000s 音频)
```

| 语言 | Qwen3-ForcedAligner | WhisperX | NFA (MFA) |
|------|---------------------|----------|-----------|
| 中文 | **33.1ms** | — | 109.8ms |
| 英语 | **37.5ms** | 92.1ms | 107.5ms |
| 法语 | **41.7ms** | 145.3ms | 100.7ms |
| 德语 | **46.5ms** | 165.1ms | 122.7ms |
| 日语 | **42.2ms** | — | — |
| **平均** | **42.9ms** | 133.2ms | 129.8ms |

---

## 五、模型矩阵与参数量估算

### 5.1 模型规格总览

| 参数 | Qwen3-ASR-1.7B | Qwen3-ASR-0.6B | Qwen3-ForcedAligner-0.6B |
|------|---------------|---------------|--------------------------|
| **总参数量** | ~2B | ~0.9B | ~0.9B |
| **编码器层数** | 24 | 18 | 18 |
| **编码器维度** | 1024 | 896 | 896 |
| **编码器注意力头** | 16 | 14 | 14 |
| **编码器 FFN** | 4096 | 3584 | 3584 |
| **编码器参数量** | ~300M | ~180M | ~180M |
| **LLM 解码器** | Qwen3-1.7B (28层) | Qwen3-0.6B (28层) | Qwen3-0.6B (NAR) |
| **解码器维度** | 2048 | 1024 | 1024 |
| **解码器注意力头** | 16Q / 8KV (GQA) | 16Q / 8KV (GQA) | — (NAR) |
| **解码器 FFN** | 6144 | 3072 | — |
| **词汇表大小** | 151,936 | 151,936 | 151,936 |
| **输出帧率** | 12.5 Hz | 12.5 Hz | — |
| **流式支持** | ✅ 统一 | ✅ 统一 | ❌ |
| **语言数** | 52 | 52 | 11 |
| **时间戳精度** | — | — | **42.9ms** |

### 5.2 参数量估算

以 1.7B 版本为例：

```
AuT Encoder:
  - Conv2D Stem:
    - Conv2D(1→480, 3×3): 1×480×3×3 + 480 = 4,320 + 480 = 4.8K
    - Conv2D(480→480, 3×3): 480×480×3×3 + 480 = 2,073,600
    - Conv2D(480→480, 3×3): 2,073,600 (同上)
    - Linear(7680→1024): 7680×1024 = 7,864,320
    - Conv 合计: ~12M

  - 24 × Transformer Layer:
    - LayerNorm (bias): 1024 × 2 = 2K
    - QKV 投影: 1024×1024×3 + 1024×3 = 3,147,264
    - Output 投影: 1024×1024 + 1024 = 1,049,600
    - LayerNorm (bias): 1024 × 2 = 2K
    - FFN fc1: 1024×4096 + 4096 = 4,198,400
    - FFN fc2: 4096×1024 + 1024 = 4,198,400
    - 每层: ~12.6M × 24 = ~302M

  - Post LN + Proj:
    - LayerNorm: 1K
    - proj1: 1024×1024 + 1024 = 1,049,600
    - proj2: 1024×2048 + 2048 = 2,099,200

  - Encoder 合计: ~317M

Qwen3-1.7B Decoder (28 层):
  - Token Embedding: 151936×2048 = 311,164,928
  - 28 × (Self-Attention + FFN):
    - Q 投影: 2048×2048 = 4,194,304
    - K 投影: 2048×512 = 1,048,576
    - V 投影: 2048×512 = 1,048,576
    - Output 投影: 2048×2048 = 4,194,304
    - SwiGLU gate: 2048×6144 = 12,582,912
    - SwiGLU up: 2048×6144 = 12,582,912
    - SwiGLU down: 6144×2048 = 12,582,912
    - RMSNorm: 2048×2 = 4K
    - 逐头 Q/K RMSNorm: 2048×2 = 4K
    - 每层: ~48.2M × 28 = ~1,350M
  - LM Head: 绑定了 Token Embedding (不计)
  - Decoder 合计: ~1.66B

总计: 317M + 1,660M ≈ 1.98B (~2B 参数)
```

> 注意：实际参数量可能因权重共享策略、bias 设置等微调而略有差异。0.6B 版本按相同估算约为 0.9B 参数。

---

## 六、推理流程演练

以 10 秒中文语音"今天天气真的很冷"为例：

### Stage 1: 前端特征提取

```
10s 音频 @16kHz → 160,000 采样点
    → 25ms 窗口 / 10ms 步长 → 1000 帧 Mel
    → 动态范围裁剪 + 归一化
    → [1, 128, 1000]
```

### Stage 2: Conv2D 下采样

```
[1, 1, 128, 1000]     ← 添加通道维
    → 按 100 帧分块: 10 个 chunk
    → 每 chunk 独立通过 Conv2D × 3
    → [1, 480, 16, 13] per chunk
    → Reshape: [1, 13, 7680] per chunk
    → Linear: [1, 13, 1024]
    → 拼接: [1, 130, 1024]
    └── 1000 → 130 (8× 下采样), 帧率 = 12.5Hz
```

### Stage 3: AuT Encoder 编码

```
[1, 130, 1024] → 24 层双向 Transformer
    → 动态窗口: 8s = 104 tokens
    → 每 token 可看到窗口内约 104 tokens
    → 窗口滑动覆盖全部 130 tokens
    → 输出: [1, 130, 1024]
```

### Stage 4: 编码器投影

```
[1, 130, 1024]
    → LayerNorm + GELU + Linear(1024→1024)
    → Linear(1024→2048)
    → [1, 130, 2048]
```

### Stage 5: Prompt 构建与 Prefill

```
系统消息: <|im_start|>system\n<|im_end|>\n
用户消息: <|im_start|>user\n
音频段:   <|audio_start|><|audio_pad|>×130<|audio_end|>
用户结束: <|im_end|>\n
回答开始: <|im_start|>assistant\n

完整 prompt ≈ 135 tokens → 送入解码器
解码器 Prefill 生成完整 KV Cache
```

### Stage 6: 自回归解码

```
Step 1:  输入 <s> → 输出 "language" + "Chinese"  ← 语言识别
Step 2:  输出 "<asr_text>"                                 ← ASR 文本开始标记
Step 3:  输出 "今"
Step 4:  输出 "天"
Step 5:  输出 "天"
Step 6:  输出 "气"
  ...
Step 10: 输出 "冷"
Step 11: 输出 "<|im_end|>"                                ← 结束标记
```

### 推理配置

```python
# 推荐推理超参数
max_new_tokens = 256          # ASR 最长输出
use_cache = True              # 启用 KV Cache
do_sample = False             # 贪婪解码（ASR 场景）
temperature = None            # 关闭温度参数
num_beams = 1                 # 流式用 greedy；离线可用 5

# 流式配置
chunk_length_s = 2.0          # 每块 2s
stride_length_s = 1.0         # 50% 重叠
n_window_infer = 800          # 8s 注意力窗口

# 离线配置
max_segment_length = 20 * 60  # 最大 20 分钟
split_by_energy = True        # 能量检测断句
```

---

## 七、推理性能优化

### 7.1 推理瓶颈分析

```
总推理时间分布 (以 10s 音频为例，离线模式):
    ├── 音频特征提取:      ~1%
    ├── Conv2D 下采样:     ~2%
    ├── AuT Encoder 编码:  ~25%  ★ 编码瓶颈
    ├── 投影映射:          ~1%
    ├── KV Cache Prefill:  ~2%
    └── 自回归解码 (10步):  ~69%  ★ 解码瓶颈
```

### 7.2 优化策略矩阵

| 优化手段 | 解决的问题 | 原理 | 效果 |
|----------|-----------|------|------|
| **GQA (2:1)** | KV Cache 过大 | 2 组 Q 共享 1 组 KV | KV Cache 减半 |
| **FlashAttention** | 注意力计算慢 | 分块 + 在线 Softmax | 2-4× 注意力加速 |
| **动态窗口** | 流式/离线分离 | 1-8s 窗口动态切换 | 统一部署 |
| **5-token rollback** | 流式边界处理 | 丢弃最后 5 token 重解码 | 平滑过渡 |
| **非因果编码** | 编码器延迟 | 全 chunk 并行 | 编码器 O(1) 延迟 |
| **INT8 量化** | 模型过大 | 权重压缩 | 显存减半 |
| **TensorRT/vLLM** | 推理框架 | 算子融合 + 内存优化 | 2-5× 加速 |

### 7.3 吞吐量数据

| 模型 | 并发数 | RTF | 等效处理速度 |
|------|--------|-----|-------------|
| Qwen3-ASR-0.6B | 128 | **0.064** | ~2000× 实时 |
| Qwen3-ASR-1.7B | 128 | ~0.15 | ~660× 实时 |
| Qwen3-ForcedAligner-0.6B | — | **~0.001** | ~1000s/s |

---

## 八、QA：架构决策背后的思考

### 8.1 为什么 AuT 编码器用 Sinusoidal 而非 RoPE？

| 编码方案 | 优点 | 缺点 |
|---------|------|------|
| **Sinusoidal (Qwen3-ASR)** | 绝对位置信息明确，每 chunk 位置重置方便 | 长度泛化能力有限 |
| **RoPE (GLM-ASR)** | 相对位置编码，长度泛化好 | 需要处理 chunk 边界 |
| **可学习的 (Whisper)** | 灵活 | 最大长度固定，泛化差 |

Qwen3-ASR 选择 Sinusoidal 的原因是：

1. **每个 chunk 位置从 0 开始**：Sinusoidal 的绝对位置编码天然适合独立分块
2. **动态窗口**：窗口内位置是局部的，不需要全局位置信息
3. **简单高效**：Sinusoidal 无需额外参数，且 FlashAttention 的实现兼容性最好

### 8.2 Conv2D vs Conv1D：策略选择

```
Conv2D (Qwen3-ASR):
  128 mel → 64 → 32 → 16 (频率渐进压缩)
  + 频率维度的层次化抽象
  + 减少编码器自注意力的序列长度
  - 损失了频率分辨率

Conv1D (GLM-ASR / Whisper):
  128 mel 全保留
  + 保留全部频率细节
  - 编码器序列长度长
  - 频率结构需要自注意力自学
```

Qwen3-ASR 的选择反映了"Conv2D 做粗粒度的频率压缩 + 编码器做细粒度的时序建模"的分工哲学。

### 8.3 为什么解码器用 28 层而非更少？

| 模型 | 解码器深度 | 参数量比例 (编码:解码) |
|------|-----------|---------------------|
| **Qwen3-ASR-1.7B** | 28 层 | 317M : 1.66B = **1:5.2** |
| **GLM-ASR-Nano** | 6 层自研 | 约 1:1 (更轻量) |
| **Whisper large-v3** | 32 层 | 32:32 (对称) |

Qwen3-ASR 的解码器非常重（占 80%+ 的参数），这是因为：

1. **复用 Qwen3 基座**：直接使用 Qwen3-1.7B 或 Qwen3-0.6B 作为解码器，无需重新设计
2. **多任务能力**：保留了 LLM 的指令跟随和对话能力
3. **Context Biasing**：深层解码器才能有效利用长上下文引导 ASR

**代价**：推理时大部分计算在解码器，但 RTF=0.064 (0.6B) 证明这在实践中是可行的。

---

## 九、与竞品对比

### 9.1 综合对比

| 维度 | **Qwen3-ASR-1.7B** | **GLM-ASR-Nano** | **Whisper-large-v3** |
|------|-------------------|-----------------|---------------------|
| **开源协议** | Apache 2.0 | MIT | MIT |
| **总参数量** | ~2B | ~1.5B | ~1.5B |
| **平均 WER** | **5.76%** | 7.03% | 7.44% |
| **语言覆盖** | 52 (30+22方言) | 17 种 | 99 种 |
| **中国方言** | **22 种** ✅ | 主要(粤语/吴语/闽语) | 无细分 |
| **歌唱识别** | **<6% WER** ✅ | ❌ | ❌ |
| **流式+离线统一** | ✅ **动态窗口** | ❌ 仅离线 | ❌ 仅离线 |
| **Context Biasing** | ✅ **10K tokens** | ❌ | ❌ |
| **时间戳对齐** | ✅ 42.9ms | ❌ 需外挂 | via WhisperX |
| **RL 微调** | ✅ GSPO | ❌ | ❌ |

### 9.2 标准基准对比

| 数据集 | Qwen3-ASR-1.7B | Whisper-large-v3 | 说明 |
|--------|---------------|-----------------|------|
| LibriSpeech clean | 1.63 | 1.51 | 英文标准 |
| LibriSpeech other | **3.38** | 3.97 | 英文噪声 |
| GigaSpeech | **8.45** | 9.76 | 英文多场景 |
| CommonVoice-en | **7.39** | 9.90 | 众包英语 |
| Tedlium | **4.50** | 6.84 | TED 演讲 |
| WenetSpeech net | **4.97** | 9.86 | 中文网络 |
| WenetSpeech meeting | **5.88** | 19.11 | 中文会议 |
| AISHELL-2 | **2.71** | — | 中文标准 |
| Fleurs-en | **3.35** | — | 多语英语 |

### 9.3 架构对比总结

| 设计维度 | Qwen3-ASR | GLM-ASR-Nano | Whisper-large-v3 |
|---------|-----------|-------------|-----------------|
| 音频前端 | Mel + 动态归一化 | Mel (标准) | Mel (标准) |
| 下采样 | **3× Conv2D stride=2** | 2× Conv1D stride=2 | 2× Conv1D stride=2 |
| 下采样率 | 8× (100→12.5Hz) | 2× (100→50Hz) + 额外投影压缩 | 2× (100→50Hz) |
| 位置编码 | **Sinusoidal (per-chunk)** | 部分 RoPE (50%) | 可学习的绝对编码 |
| 编码器类型 | **AuT** (24层/18层) | 改进 Whisper (32层/12层) | 标准 Encoder (32层) |
| 解码器类型 | Qwen3 LLM (28层, GQA) | LLaMA (28层/6层, GQA) | 标准 Decoder (32层, MHA) |
| 归一化 | LayerNorm (编码) + RMSNorm (解码) | LayerNorm (编码) + RMSNorm (解码) | LayerNorm |
| 注意力 | FlashAttention + 动态窗口 | FlashAttention | 标准 SDPA |
| 激活函数 | GELU (编码) + SwiGLU (解码) | GELU (编码) + SwiGLU (解码) | GELU |
| 流式 | ✅ 统一架构 | ❌ 不支持 | ❌ 不支持 |
| RL 微调 | ✅ GSPO | ❌ | ❌ |

---

## 十、总结：一张图看穿 Qwen3-ASR

```
┌───────────────────────────────────────────────────────────────────────────┐
│                       Qwen3-ASR 架构全景                                   │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  输入: 16kHz 音频                                                          │
│       │                                                                   │
│  ┌────┴──────────┐                                                        │
│  │  Mel 频谱前端   │  128 维, 25ms 窗 / 10ms 步长, 100 Hz                   │
│  │  动态范围裁剪   │  log10(clamp()), (x+4)/4 归一化                        │
│  └────┬──────────┘                                                        │
│       │                                                                   │
│  ┌────┴──────────┐                                                        │
│  │ Conv2D 下采样  │  "重编码"                                              │
│  │  3× Conv2D    │  128→64→32→16 频, T→T/8 时                              │
│  │  stride=2     │  GELU 激活, per-chunk 独立                              │
│  │  Linear 投影  │  7680 → d_model                                        │
│  └────┬──────────┘                                                        │
│       │                                                                   │
│  ┌────┴─────────────────────────────────────┐                             │
│  │         AuT Encoder                       │  "深度感知"                  │
│  │  ┌─────────────────────────────────────┐  │                             │
│  │  │ LayerNorm (bias)                    │  │  24层 (1.7B) / 18层 (0.6B)  │
│  │  │ 双向 Self-Attention (Full)          │  │  d_model=1024/896            │
│  │  │ + Residual                          │  │  16头 / 14头                 │
│  │  │ LayerNorm (bias)                    │  │  Sinusoidal 位置编码          │
│  │  │ FFN: GELU, d_model→4×→d_model       │  │  FlashAttention + 动态窗口   │
│  │  │ + Residual                          │  │  1s ~ 8s 自适应切换          │
│  │  └─────────────────────────────────────┘  │                             │
│  │                        × 24/18 层          │                             │
│  └────┬─────────────────────────────────────┘                             │
│       │                                                                   │
│  ┌────┴──────────┐                                                        │
│  │  Projector    │  "精投影"                                              │
│  │  LayerNorm →  │  1024→1024→2048 (1.7B)                                │
│  │  GELU→Linear  │  896→896→1024 (0.6B)                                  │
│  └────┬──────────┘                                                        │
│       │                                                                   │
│  ┌────┴─────────────────────────────────────┐                             │
│  │    Qwen3 LLM Decoder                      │  "强解码"                   │
│  │  ┌─────────────────────────────────────┐  │  28层, GQA 2:1 (16Q/8KV)   │
│  │  │ 逐头 Q/K RMSNorm                    │  │  head_dim=128               │
│  │  │ Masked Self-Attention (因果)        │  │  MRoPE (退化成标准RoPE)      │
│  │  │ + Residual                          │  │  SwiGLU FFN                  │
│  │  │ RMSNorm                              │  │  RMSNorm                     │
│  │  │ SwiGLU MLP                          │  │  vocab=151,936               │
│  │  │ + Residual                          │  │  无 bias                     │
│  │  └─────────────────────────────────────┘  │                             │
│  │                        × 28 层             │                             │
│  └────┬─────────────────────────────────────┘                             │
│       │                                                                   │
│  输出: 文本 Token (语言 + <asr_text> + 转录内容)                           │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────┘

             一句话总结 Qwen3-ASR：
     "用 3 层 Conv2D 把 100 帧压成 13 token/秒，
      用 AuT 编码器（24层）+ 动态窗口统一流式/离线，
      用 Qwen3 LLM（28层，GQA 2:1）做最强大的 ASR 解码。"
```

---

## 附录：关键配置一览

### AuT 编码器配置

| 配置项 | 1.7B | 0.6B |
|--------|------|------|
| `d_model` | 1024 | 896 |
| `num_layers` | 24 | 18 |
| `num_heads` | 16 | 14 |
| `head_dim` | 64 | 64 |
| `intermediate_size` | 4096 | 3584 |
| `hidden_act` | gelu | gelu |
| `position_encoding` | sinusoidal | sinusoidal |
| `attention_type` | full + windowed | full + windowed |
| `norm_type` | LayerNorm (bias=True) | LayerNorm (bias=True) |
| `conv_channels` | [1, 480, 480, 480] | [1, 480, 480, 480] |
| `conv_kernel` | 3×3 | 3×3 |
| `conv_stride` | 2 | 2 |

### Qwen3 解码器配置

| 配置项 | 1.7B | 0.6B |
|--------|------|------|
| `hidden_size` | 2048 | 1024 |
| `num_layers` | 28 | 28 |
| `num_attention_heads` | 16 | 16 |
| `num_key_value_heads` | 8 | 8 |
| `head_dim` | 128 | 128 |
| `intermediate_size` | 6144 | 3072 |
| `vocab_size` | 151,936 | 151,936 |
| `hidden_act` | silu (SwiGLU) | silu (SwiGLU) |
| `norm_type` | RMSNorm (eps=1e-6) | RMSNorm (eps=1e-6) |
| `rope_theta` | 1,000,000 | 1,000,000 |
| `rope_scaling` | mrope [24,20,20] | mrope [24,20,20] |
| `bias` | False (all layers) | False (all layers) |
| `tie_word_embeddings` | True | True |

### 训练配置

| 阶段 | 数据规模 | 训练目标 |
|------|---------|---------|
| AuT Pretraining | ~40M hours | ASR 损失 (CTC/CE) |
| Omni Pretraining | 3T tokens | 多模态下一词预测 |
| ASR SFT | — | 指令跟随 ASR |
| GSPO RL | ~50K utterances | 噪声鲁棒性 |

---

*本文基于 arXiv 技术报告 (2601.21337)、HuggingFace 模型配置、Qwen 官方技术博客及社区分析整理。*

---

## Sources

- [Qwen3-ASR Technical Report (arXiv:2601.21337)](https://arxiv.org/abs/2601.21337)
- [Qwen3-ASR-1.7B on HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Qwen3-ASR-0.6B on HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ASR-Toolkit GitHub](https://github.com/QwenLM/Qwen3-ASR-Toolkit)
- [Qwen3-ASR Model Blog (qwen.ai)](https://qwen.ai/blog?id=qwen3asr)
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
- [GSPO: Group Sequence Policy Optimization (arXiv:2507.18071)](https://arxiv.org/abs/2507.18071)
- [GSPO 官方博客 (qwenlm.github.io)](https://qwenlm.github.io/blog/gspo/)
- [Qwen3-ASR-0.6B streaming variant on HuggingFace](https://huggingface.co/qfuxa/qwen3-asr-0.6b-streaming)
