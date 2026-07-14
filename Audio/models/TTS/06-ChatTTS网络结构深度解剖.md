# ChatTTS 网络结构深度解剖

> 2noise 社区出品 | 专为对话场景设计的 TTS | 中文社区现象级热度
> 2024 年发布 | Apache 2.0 / CC BY-NC 4.0

---

## 写在前面：ChatTTS 的特殊定位

ChatTTS 是第一个**以对话场景为第一目标**的开源 TTS。传统 TTS（Tacotron 2、FastSpeech 2）训练数据以有声书和朗读语音为主，生成的语音音质好但"读感"重。ChatTTS 使用大量真实对话数据进行训练，第一次让开源 TTS 听起来像在"聊天"而不是"朗读"。

| 维度 | 传统 TTS | ChatTTS |
|------|---------|---------|
| 训练数据 | 有声书/朗读 (LJSpeech, LibriTTS) | **对话/播客/访谈 (10 万+ 小时)** |
| 语音风格 | 播音员朗读 | **真人聊天** |
| 韵律控制 | 难 | ✅ 精细 token 级控制 |
| 情感丰富度 | 平缓 | ✅ 自然起伏 |

---

## 一、整体架构设计哲学

### 核心思想

> **"用 LLM 理解文本，用 DVAE 编码语音，用说话人嵌入控制音色。"**

ChatTTS 的架构可以分解为三个阶段：
1. **理解**：GPT（LLaMA 架构）将文本转换为音频 token 序列
2. **编码**：DVAE 将音频 token 解码为 Mel 频谱
3. **还原**：Vocos 声码器将 Mel 频谱还原为波形

### 架构总览

```
文本: "你今天还好吗?"
    │
    ├── ① Tokenizer (文本 token: 21178 vocab)
    │
    ├── ② GPT (LLaMA, 20 层, 768hd)
    │   ├── Text Refinement: 优化韵律/发音
    │   └── 音频码生成: 输出 4 流离散音频 token (626 vocab)
    │
    ├── ③ DVAE Decoder (ConvNeXt, 12 层)
    │   └── 离散 token → Mel 频谱 (100 bands)
    │
    ├── ④ Vocos 声码器
    │   └── Mel 频谱 → 24kHz 波形
    │
    └── 波形: "你今天还好吗?" (带有自然对话韵律)
```

---

## 二、各模块深度解剖

### 2.1 Tokenizer 与词汇空间

**定位**：将文本和音频分别编码为离散 token，送入 GPT 统一处理。

| 词汇表 | 大小 | 内容 |
|--------|------|------|
| `num_text_tokens` | 21178 | 中文/英文字词 + 标点 + 控制符 |
| `num_audio_tokens` | 626 | 音频码本（每流） |
| `num_vq` | 4 | VQ 流数 |

**四流音频码（num_vq=4）**：与 EnCodec 的 8 层 RVQ 不同，ChatTTS 将音频压缩为 4 个独立流。每个流捕获不同粒度的声学特征——从"说了什么"到"说话习惯"。

---

### 2.2 GPT（LLaMA 架构 Transformer）

**定位**：核心推理引擎。同时处理文本和音频 token 序列，输出预测的音频 token。

#### GPT 配置

| 参数 | 值 | 含义 |
|------|-----|------|
| `hidden_size` | 768 | 隐藏维度 |
| `intermediate_size` | 3072 | FFN 中间维度 (4×) |
| `num_attention_heads` | 12 | 注意力头数 (每头 64 维) |
| `num_hidden_layers` | **20** | Transformer 层数 |
| `max_position_embeddings` | 4096 | 最大序列长度 |
| `spk_emb_dim` | 192 | 说话人嵌入维度 |
| `num_vq` | 4 | 音频流数 |
| 激活函数 | SiLU (SwiGLU) | LLaMA 标准门控激活 |

#### 每个 Layer 的内部分解

```
输入: [batch, seq_len, 768]
    │
    ├── RMSNorm (LLaMA 标准)
    │
    ├── Masked Self-Attention (因果)
    │   ├── QKV 投影: 768 → 3 × 768
    │   ├── RoPE (旋转位置编码)
    │   └── Output 投影
    │
    ├── + 残差连接
    │
    ├── RMSNorm
    │
    ├── SwiGLU FFN
    │   ├── gate_proj: 768 → 3072 (SiLU 激活)
    │   ├── up_proj: 768 → 3072
    │   ├── 逐元素乘法: gate × up
    │   └── down_proj: 3072 → 768
    │
    └── + 残差连接
```

**参数量估算**：

```
每层:
  Self-Attn: 4 × (768 × 768) = ~2.36M  (QKV + Output)
  SwiGLU: 3 × (768 × 3072) = ~7.08M     (gate/up/down)
  RMSNorm: 768 × 2 = ~1.5K
 每层合计: ~9.44M
20 层合计: ~188.8M

Embedding: 21178 × 768 = ~16.3M
LM Head: (共享权重, 不计)
总共: ~205M (不含音频 token embed)
```

#### 两模式工作流程

```
模式 1: Text Refinement
  输入: "[oral_2][laugh_1]你今天还好吗[break_4]"
        → GPT 精炼 → "[oral_5][laugh_0]你今天还好吗[break_6]"
  作用: 用韵律标签重新校准发音和语气

模式 2: 音频码生成
  输入: 精炼文本 + 说话人嵌入 (192-D)
        → GPT 预测 → 4 流音频 token 序列
```

---

### 2.3 DVAE（离散变分自编码器）

**定位**：将 GPT 输出的离散音频 token 解码为连续的 Mel 频谱。

#### DVAE Decoder 结构——ConvNeXt Block × 12

```
输入: 4 流离散音频 token [batch, 4, T_frame]
    │
    ├── Embedding: 流 ID → 384-D 连续向量
    │
    ├── ConvNeXt Block × 12:
    │   └── 每层:
    │       ├── Depthwise Conv1d(384→384, kernel=7, dilation=2, groups=384)
    │       │   └── 深度可分离卷积——每个通道独立卷积，降低参数量
    │       ├── LayerNorm
    │       ├── Linear(384→2048) — 逐点卷积 1 (升维)
    │       ├── GELU 激活
    │       ├── Linear(2048→384) — 逐点卷积 2 (降维)
    │       ├── LayerScale (可学习残差缩放 γ=1e-6)
    │       └── + 残差连接
    │
    ├── Linear: 384 → 100 (Mel 频带数)
    │
    └── 输出: Mel 频谱 [batch, 100, T_frame]
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `decoder.n_layer` | 12 | ConvNeXt Block 层数 |
| `decoder.idim` | 384 | 输入维度 |
| `decoder.odim` | 384 | 输出维度 |
| `decoder.hidden` | 512 | 隐藏维度 |
| `decoder.bn_dim` | 128 | 瓶颈维度 |
| `decoder.kernel` | 7 | 深度可分离卷积核宽度 |
| `decoder.dilation` | 2 | 膨胀率 |

**为什么用 ConvNeXt 而非 Transformer？** DVAE 的输入是 GPT 已经编码好的音频 token 序列——语义信息已经被充分提取了。DVAE 的职责是"还原"而非"理解"，卷积网络在这类确定性映射任务上比 Transformer 更高效（参数量小、速度快、不丢局部细节）。

---

### 2.4 说话人嵌入（Speaker Embedding）

**定位**：控制生成的音色。支持条件生成和随机采样。

```
说话人嵌入: 192-D 向量，从高斯分布 N(μ, σ) 采样
    │
    ├── 条件生成: 从参考音频提取嵌入 (零样本克隆)
    │   └── 编码器: 参考音频 → embedding (192-D)
    │
    ├── 随机生成: 从预设分布采样
    │   └── 采样: z ~ N(0, I) → 线性投影 → 192-D
    │
    └── 注入方式: 与文本 token 拼接后送入 GPT
```

**韵律控制标签**：ChatTTS 提供了一组细粒度的韵律控制 token：

| 标签 | 范围 | 作用 |
|------|------|------|
| `[oral_0]` ~ `[oral_9]` | 0-9 | 口语化程度（越高越随意） |
| `[laugh_0]` ~ `[laugh_2]` | 0-2 | 笑声强度 |
| `[break_0]` ~ `[break_7]` | 0-7 | 停顿长度 |
| `[uv_break]` | — | 发生停顿 |
| `[lbreak]` | — | 长停顿 |

---

### 2.5 Vocos 声码器

**定位**：Mel 频谱 → 波形。相比 HiFi-GAN 更轻量。

| 参数 | 值 |
|------|-----|
| 输入 | Mel 频谱 (100 bands) |
| 输出 | 24kHz 波形 |
| 推理 | ✅ 实时 (RTF < 0.1 on GPU) |

Vocos 使用卷积 + 转置卷积结构将 Mel 频谱上采样至波形级别，参数量约 10M。

---

## 三、推理流程演练

以生成 "你今天还好吗？[laugh_1]"（约 1.5 秒音频）为例：

### Stage 1: Refinement

```
输入: "你今天还好吗？[laugh_1]"
    │
    ├── GPT Text Refinement
    │   └── 自动调整韵律标签位置
    │
    └── 输出: "你今天还好吗？[laugh_0]"（调整笑声强度）
```

### Stage 2: 音频码生成

```
精炼文本: "你今天还好吗？[laugh_0]"
   + Speaker Embedding (192-D)
    │
    ├── Tokenize → 文本 token 序列 (约 10 个)
    ├── 拼接说话人嵌入
    │
    ├── GPT (20 层 LLaMA) 自回归生成
    │   ├── Step 1: → 4 流音频 token [t_0, t_1, t_2, t_3]
    │   ├── Step 2: → 4 流音频 token [t'_0, t'_1, t'_2, t'_3]
    │   └── ... 直到 EOS
    │
    └── 输出: [4, T_frame] 音频 token 矩阵
```

### Stage 3: DVAE 解码

```
音频 token [4, T_frame]
    │
    ├── 流嵌入 → [4, T_frame, 384]
    ├── ConvNeXt Block × 12
    └── Linear → Mel 频谱 [100, T_frame]
```

### Stage 4: Vocos 声码器

```
Mel [100, T_frame] → Vocos → 24kHz 波形
```

---

## 四、架构设计的深层思考

### 4.1 DVAE + Vocos 分离声码器的设计

VITS 直接将 HiFi-GAN 作为解码器嵌入到模型中（联合训练），ChatTTS 选择了分离设计——DVAE 做离散→连续频谱，Vocos 做频谱→波形。

| 方案 | 优点 | 缺点 |
|------|------|------|
| **联合 (VITS)** | 端到端最优，频谱→波形误差可反向传播 | 训练复杂，更换声码器需重新训练 |
| **分离 (ChatTTS)** | 模块独立，声码器可换可升级 | 两阶段误差累积 |

ChatTTS 选择分离设计的最实际原因：**训练数据版权问题复杂**——VITS 式的端到端训练需要文本-音频配对数据。ChatTTS 使用大量无标注对话音频（DVAE 自监督训练），文本-音频配对需求只在 GPT 训练阶段需要。分离让模型可以用更多无标注数据。

### 4.2 对话韵律来源

ChatTTS 的对话感主要来自两个设计：

1. **训练数据**：27,000+ 小时真实对话音频（播客、访谈、电话录音）→ 模型直接学习到对话的韵律模式
2. **Refinement 机制**：GPT 在生成音频 token 前会自动"优化"韵律标签——即使输入文本没有显式标签，也会自动添加自然的停顿和语气变化

这与传统 TTS 核心差异：传统模型学的是"如何正确发音"，ChatTTS 学的是"如何对话"。

---

## 五、ChatTTS 的问题与局限

### 已知问题

| 问题 | 表现 | 原因 |
|------|------|------|
| **正式场景违和** | 新闻/有声书场景换气频繁、语气词多 | 训练数据以对话为主 |
| **英文弱于中文** | 英文 MOS 比中文低 0.3-0.5 | 训练数据中文占比大 |
| **推理速度** | 30 秒音频 ~4GB VRAM | GPT 20 层自回归 |
| **版权争议** | 训练数据来源不透明 | 未公开详细数据来源 |

---

## 六、总结：一张图看穿 ChatTTS

```
┌──────────────────────────────────────────────────────────────────────┐
│                     ChatTTS 架构全景                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  文本: "你今天还好吗？[laugh_0]"                                      │
│       │                                                              │
│  ┌────┴───────────────┐                                              │
│  │  GPT Refinement    │  LLaMA 架构 (20L, 768d)                     │
│  │  (韵律标签优化)     │  优化停顿/笑声/口语强度                       │
│  └────┬───────────────┘                                              │
│       │  精炼文本                                                     │
│       │                                                              │
│  ┌────┴────────────────────────────────────────┐                     │
│  │  GPT 音频码生成 (LLaMA)                       │                     │
│  │  ├── RMSNorm + Masked Self-Attn (RoPE)      │                     │
│  │  ├── RMSNorm + SwiGLU FFN (768→3072→768)    │                     │
│  │  └── × 20 层                                │                     │
│  └────┬────────────────────────────────────────┘                     │
│       │  4 流离散音频 token (626 vocab)                              │
│       │                                                              │
│  ┌────┴──────────────────────────────────────────────┐               │
│  │  DVAE Decoder (ConvNeXt Block × 12)               │               │
│  │  ├── Depthwise Conv1d(k=7, dilation=2)           │               │
│  │  ├── LayerNorm + GELU + Pointwise Conv            │               │
│  │  ├── LayerScale + Residual                        │               │
│  │  └── Linear(384→100)                             │               │
│  └────┬──────────────────────────────────────────────┘               │
│       │  Mel 频谱 (100 bands)                                        │
│       │                                                              │
│  ┌────┴─────────┐                                                    │
│  │  Vocos       │  频谱→波形                                         │
│  └────┬─────────┘                                                    │
│       │                                                              │
│  波形: 24kHz (带自然的对话韵律和笑声)                                 │
│                                                                      │
│  一句话总结 ChatTTS：                                                │
│  "LLaMA 做文本理解 + 音频码生成，ConvNeXt 做频谱还原，               │
│   对话数据训练——第一次让开源 TTS 听起来像在聊天。"                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 附录：关键配置速查

### GPT 配置

```python
class ChatTTSConfig:
    # 音频
    sample_rate = 24000
    
    # GPT
    gpt = {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 20,
        "max_position_embeddings": 4096,
        "spk_emb_dim": 192,
        "num_text_tokens": 21178,
        "num_audio_tokens": 626,
        "num_vq": 4,
    }
    
    # DVAE
    dvae = {
        "decoder": {
            "idim": 384,
            "odim": 384,
            "hidden": 512,
            "n_layer": 12,
            "bn_dim": 128,
            "kernel": 7,
            "dilation": 2,
        },
        "dim": 512,
    }
    
    # 特征参数
    n_mels = 100
    n_fft = 1024
    hop_length = 256
    win_length = 1024
```

---

*本文基于 2noise/ChatTTS GitHub 仓库、HuggingFace ChatTTS 模型文档及 DeepWiki ChatTTS 源码分析整理。*

**Sources:**
- [2noise/ChatTTS - GitHub](https://github.com/2noise/ChatTTS)
- [ChatTTS DeepWiki - 源码架构分析](https://deepwiki.com/2noise/ChatTTS)
- [ChatTTS-Forge - HuggingFace](https://huggingface.co/spaces/lenML/ChatTTS-Forge)
