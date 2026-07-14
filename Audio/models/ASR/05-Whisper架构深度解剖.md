# Whisper 架构深度解剖

> OpenAI 出品 | 大规模弱监督多语言语音识别

---

## 写在前面：2022 年之前的 ASR 困境

在 Whisper 出现之前，多语言语音识别面临一个根本性困境：

```
传统 ASR 路线图 (Pre-2022):

  语言 A 的标注数据 → 训练模型 A → 只在语言 A 上可用
  语言 B 的标注数据 → 训练模型 B → 只在语言 B 上可用
  语言 C 的标注数据 → 训练模型 C → 只在语言 C 上可用
  ...
  
  结论: 每添加一门新语言 = 重新采集数据 + 重新训练模型
```

**具体困境体现在三个方面：**

| 困境 | 表现 | 代表模型 |
|------|------|---------|
| **语言专用性** | 每个语言需要独立的标注数据、独立的模型训练 | DeepSpeech, Kaldi |
| **微调依赖** | 自监督预训练（wav2vec2 / HuBERT）减少了数据需求，但下游仍需微调 | wav2vec2, HuBERT |
| **零样本缺失** | 没有任何模型能在不微调的情况下直接识别未见过的语言 | 所有前 Whisper 模型 |

wav2vec 2.0（2020）和 HuBERT（2021）通过自监督学习降低了标注数据需求，但其范式仍然是 **"预训练 + 微调"** —— 预训练阶段学习通用的声学表示，微调阶段针对特定语言/任务进行适配。零样本跨语言泛化能力极为有限。

> **Whisper 的核心突破：** 把一个完整的 ASR pipeline（声学特征提取 → 编码 → 解码 → 文本生成）作为一个**单一的 seq2seq Transformer** 来训练，用 **680K 小时** 的弱监督数据覆盖 **100+ 语言**，实现**零样本多语言语音识别**。

---

## 一、整体架构设计哲学

Whisper 的设计理念可以概括为一句话：

> **"用大规模弱监督数据训练一个通用的 Seq2Seq Transformer，把所有 ASR 相关任务统一到一个模型中。"**

这套思想落地为一个**经典的 Encoder-Decoder Transformer 架构**：

```
原始音频 (任意采样率)
    │
    ├─ ① 重采样至 16kHz
    │
    ├─ ② 80-bin (v2) / 128-bin (v3) log-Mel 频谱
    │
    ├─ ③ Conv1d 特征提取 (2层)
    │    第一层: n_mels → n_state, kernel=3, stride=1
    │    第二层: n_state → n_state, kernel=3, stride=2 ← 时间维减半
    │
    ├─ ④ Sinusoidal Positional Encoding (固定, 非学习的)
    │
    ├─ ⑤ Audio Encoder ── N 层双向 Transformer (Pre-LN, GELU)
    │
    ├─ ⑥ Cross-Attention ─── 解码器每层通过交叉注意力访问编码器输出
    │
    ├─ ⑦ Text Decoder ─── N 层自回归 Transformer (Pre-LN, GELU)
    │    输入: learned token embedding + learned positional embedding
    │
    └─ 输出: 多任务 token 序列
```

### 设计特点

1. **对称 Encoder-Decoder**：与 GLM-ASR 的"非对称"设计不同，Whisper 的编码器和解码器深度相同（large 版本都是 32 层）。这是因为弱监督训练需要解码器具备足够的语言建模能力来纠正转录错误。

2. **绝对位置编码分工明确**：编码器使用**固定的** Sinusoidal 位置编码，解码器使用**可学习的**位置编码。这一区别在架构设计中非常关键。

3. **多任务统一解码格式**：ASR、翻译、语言识别、时间戳预测——四个任务共享同一个解码器，通过特殊的 prompt token 切换任务模式。

---

## 二、各模块深度解剖

### 2.1 Conv1d 特征提取器（Conv1d Stem）

**定位**：将 Mel 频谱从声学特征空间投影到 Transformer 隐藏空间，同时完成第一级时间下采样。

```
输入: [batch, n_mels, T]   (T = 帧数 ≈ 100 × 音频秒数, n_mels = 80/128)
    │
    ├─ Conv1d(n_mels → n_state, kernel=3, stride=1, padding=1)
    │   └─ GELU 激活
    │
    ├─ Conv1d(n_state → n_state, kernel=3, stride=2, padding=1)
    │   └─ GELU 激活
    │
    └─ 输出: [batch, n_state, T/2]   (n_state 对应模型的 d_model)
```

| 参数 | tiny | base | small | medium | large |
|------|------|------|-------|--------|-------|
| `n_mels` (v2) | 80 | 80 | 80 | 80 | 80 |
| `n_mels` (v3) | — | — | — | — | 128 |
| `n_state` | 384 | 512 | 768 | 1024 | 1280 |
| 卷积层数 | 2 | 2 | 2 | 2 | 2 |
| 时间缩减率 | **2×** | **2×** | **2×** | **2×** | **2×** |
| 激活函数 | GELU | GELU | GELU | GELU | GELU |

**设计要点**：

1. **第一层 stride=1**：只做维度变换（从 Mel 维度投影到模型维度），不压缩时间
2. **第二层 stride=2**：时间维度减半，相当于以 20ms 的间隔将帧信息聚合到 Transformer 的 token 粒度
3. **kernel=3 的小卷积核**：局部感受野，保持时间分辨率的同时引入邻域上下文

以 30 秒音频为例：
```
30s @16kHz → 3000 mel 帧 (10ms/帧)
  → Conv1d stride=1 → 3000 帧 (维度变为 n_state)
  → Conv1d stride=2 → 1500 帧 (时间减半)
  → 送入 Encoder: 1500 个 token
```

### 2.2 Audio Encoder（音频编码器）

这是 Whisper 的核心模块，结构上是一个**标准 Transformer Encoder**，但有几个关键设计选择。

#### 模型配置表

| 参数 | tiny | base | small | medium | large-v1/v2 | large-v3 |
|------|------|------|-------|--------|-------------|----------|
| `n_audio_layer` | **4** | **6** | **12** | **24** | **32** | **32** |
| `n_audio_state` | **384** | **512** | **768** | **1024** | **1280** | **1280** |
| `n_audio_head` | **6** | **8** | **12** | **16** | **20** | **20** |
| `n_audio_ff` | 1536 | 2048 | 3072 | 4096 | 5120 | 5120 |
| Transformer 类型 | Pre-LN | Pre-LN | Pre-LN | Pre-LN | Pre-LN | Pre-LN |
| 激活函数 | GELU | GELU | GELU | GELU | GELU | GELU |
| 位置编码 | Sinusoidal | Sinusoidal | Sinusoidal | Sinusoidal | Sinusoidal | Sinusoidal |

#### 每个 Encoder Layer 的内部分解

```
输入: x [batch, seq_len, n_state]
    │
    ├── LayerNorm (Pre-LN)
    │
    ├── Multi-Head Self-Attention (双向 / 非因果)
    │   ├── QKV 投影 (线性层, 无 bias)
    │   ├── Scaled Dot-Product Attention (n_audio_head 个头)
    │   ├── Attention 权重 [seq_len, seq_len] — 完整矩阵
    │   └── Output 投影: n_state → n_state
    │
    ├── + 残差连接
    │
    ├── LayerNorm (Pre-LN)
    │
    ├── MLP (FFN)
    │   ├── Linear: n_state → n_audio_ff (维度扩张 4×)
    │   ├── GELU 激活
    │   └── Linear: n_audio_ff → n_state (维度压缩回 n_state)
    │
    └── + 残差连接
```

#### 关键设计细节

**1. Sinusoidal 位置编码（固定、非学习）**

这是 Whisper 编码器最容易被忽视的设计决策。与主流做法（可学习位置编码、RoPE、ALiBi）不同，Whisper 选择了最原始的 Sinusoidal 编码：

```python
# 来自 Whisper 源码 model.py（简化）
position = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                     -(math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

**为什么选择 Sinusoidal？**

| 角度 | 说明 |
|------|------|
| **长度外推** | Sinusoidal 允许模型处理比训练时更长的序列（虽然 Whisper 限制在 30s 以内） |
| **归纳偏置** | 语音中"时间的相对位置"比"绝对位置"更重要——Sinusoidal 的线性变换性使得模型可以依赖相对位置 |
| **与 Conv1d 协同** | 输入已经是经过下采样的帧表示，Sinusoidal 提供了足够的位置线索 |
| **TTS 中已验证** | Transformer TTS（如 Tacotron2）广泛使用 Sinusoidal，Whisper 继承了这个成熟设计 |

**实际效果验证**：
```
Sinusoidal 编码相邻位置之间的编码向量差异平滑且连续：
  pos=100 和 pos=101 的编码差异 ≈ pos=200 和 pos=201 的差异
  这种一致性使得编码器可以"学会"利用位置差而不是绝对位置
```

**2. Pre-Layer Normalization**

Whisper 使用的是 **Pre-LN** 而非 Post-LN：

```
Pre-LN (Whisper 采用):
  x → LayerNorm → SubLayer → + x  →  LayerNorm → SubLayer → + x
              残差路径 I                   残差路径 II

Post-LN (原始 Transformer):
  x → SubLayer → + x → LayerNorm  →  SubLayer → + x → LayerNorm
              残差路径                   残差路径
```

Pre-LN 的优势：
- **训练稳定性更高**：梯度在残差路径中更直接地流动
- **无需 warmup**：Pre-LN 可以使用较大的初始学习率
- **深层训练时更稳定**：对 32 层编码器尤为关键

**3. GELU 激活函数**

所有 FFN 中使用 **GELU**（Gaussian Error Linear Unit）而非 ReLU：

```
ReLU:  f(x) = max(0, x)
GELU:  f(x) = x * Φ(x)    (Φ 为标准正态分布的 CDF)

区别：
  ReLU 在 x<0 时硬性裁剪 → 不可微，神经元死亡风险
  GELU 在 x<0 时概率性保留 → 平滑近似，更优的梯度流
```

GELU 对于 ASR 的特殊意义：语音信号的数值分布是连续的（不像 NLP 中 token embedding 是离散的），平滑激活函数有利于保持声学特征的连续性。

**4. 双向（非因果）注意力**

```
编码器的 Attention Mask:
  位置 | 1  2  3  4  5 ...
  ─────┼─────────────────
   1   | ✓  ✓  ✓  ✓  ✓
   2   | ✓  ✓  ✓  ✓  ✓
   3   | ✓  ✓  ✓  ✓  ✓
  ...

所有位置互相可见 → 每个 token 看到完整的全局上下文
```

这与语言模型不同。语音编码必须看到**完整上下文**才能消除歧义：

```
示例："科学" 和 "柯学" 在发音上极为相似
  只看前 200ms → 无法区分
  看完整 500ms → 通过后续语境可以区分
  
双向注意力就是让模型拥有"听完再判断"的能力
```

### 2.3 Text Decoder（文本解码器）

解码器结构与编码器对称，但有三个关键区别：

#### 模型配置表

| 参数 | tiny | base | small | medium | large-v1/v2 | large-v3 |
|------|------|------|-------|--------|-------------|----------|
| `n_text_layer` | **4** | **6** | **12** | **24** | **32** | **32** |
| `n_text_state` | **384** | **512** | **768** | **1024** | **1280** | **1280** |
| `n_text_head` | **6** | **8** | **12** | **16** | **20** | **20** |
| `n_text_ff` | 1536 | 2048 | 3072 | 4096 | 5120 | 5120 |
| Token Embedding | 可学习 | 可学习 | 可学习 | 可学习 | 可学习 | 可学习 |
| 位置编码 | **可学习** | **可学习** | **可学习** | **可学习** | **可学习** | **可学习** |
| 词汇表大小 | 51865 | 51865 | 51865 | 51865 | 51865 | 51865 |

#### 每个 Decoder Layer 的内部分解

```
输入: token_ids [batch, seq_len]
    │
    ├── Token Embedding (可学习) → [batch, seq_len, n_state]
    ├── Positional Embedding (可学习) → + [batch, seq_len, n_state]
    │
    ├── LayerNorm (Pre-LN)
    │
    ├── Masked Self-Attention (因果/单向)
    │   ├── QKV 投影
    │   ├── Scaled Dot-Product Attention + 因果掩码
    │   └── Output 投影
    │
    ├── + 残差连接
    │
    ├── LayerNorm (Pre-LN)
    │
    ├── Cross-Attention (从编码器读取信息, 仅解码器特有)
    │   ├── Q: 来自解码器当前层
    │   ├── K, V: 来自编码器最后一层输出
    │   ├── 标准注意力 (无掩码, 编码器输出全部可见)
    │   └── Output 投影
    │
    ├── + 残差连接
    │
    ├── LayerNorm (Pre-LN)
    │
    ├── MLP (FFN)
    │   ├── Linear: n_state → n_text_ff
    │   ├── GELU 激活
    │   └── Linear: n_text_ff → n_state
    │
    └── + 残差连接
```

#### 关键设计细节

**1. 可学习的位置编码（与编码器 Sinusoidal 不同）**

| 模块 | 位置编码类型 | 原因 |
|------|-------------|------|
| **编码器** | **Sinusoidal（固定）** | 输入序列长度可变，固定编码提供稳定的位置线索 |
| **解码器** | **可学习** | 输出序列（文本 token）有固定的语义结构，可学习编码能自适应调整 |

可学习位置编码在解码器中的优势：
- 不同 token 可能有不同的"位置重要性"——例如语言指示 token `<|en|>` 和文本 token "hello" 需要不同的位置表示
- 解码器的序列长度通常很短（几十个 token），可学习编码不会面临外推问题
- 与 token embedding 协同优化——位置和语义可以联合学习

**2. Masked Self-Attention（因果注意力）**

```
解码器的 Attention Mask:
  位置 | 1  2  3  4  5 ...
  ─────┼─────────────────
   1   | ✓  .  .  .  .
   2   | ✓  ✓  .  .  .
   3   | ✓  ✓  ✓  .  .
   4   | ✓  ✓  ✓  ✓  .
   5   | ✓  ✓  ✓  ✓  ✓
  
当前 token 只能看到自身和之前的 token → 自回归生成
```

**3. Cross-Attention（交叉注意力）**

Cross-Attention 是连接编码器和解码器的**唯一通道**：

```
Cross-Attention 信息流:
  Encoder 输出: [batch, T/2, n_state]  ← 声学特征
                        ↑
        K, V 来自编码器最后一层（所有位置）
                        ↑
  Q 来自解码器当前层    → Attention 权重决定"听"音频的哪一部分
                        ↑
  解码器各层的 Cross-Attention 权重可以是不同的:
    - 浅层 → 关注音素级别的声学特征
    - 深层 → 关注语义级别的声学特征
```

**为什么 Cross-Attention 不是只在第一层？**

Whisper 在解码器的**每一层**都包含了 Cross-Attention。这意味着：

- 生成每个 token 时，解码器可以从编码器中**反复读取**声学信息
- 每一层都可能关注音频的不同部分——浅层关注局部声学特征，深层关注全局语义
- 这对于长序列生成至关重要——解码器不会"忘记"音频内容

### 2.4 多任务输出格式（Multi-task Training Format）

Whisper 最创新的设计之一：**用一个解码器同时支持 ASR、翻译、语言识别、时间戳预测四个任务**。

#### Token 序列结构

```
基础 ASR（不带时间戳）:
  <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|> Hello world <|endoftranscript|>

翻译（Translate）:
  <|startoftranscript|> <|zh|> <|translate|> <|notimestamps|> 你好世界 <|endoftranscript|>

带时间戳的 ASR:
  <|startoftranscript|> <|en|> <|transcribe|> <|timestamps|>
  <|0.00|> Hello world <|2.50|> <|2.52|> This is a test <|5.80|> <|endoftranscript|>
```

#### 特殊 Token 分类

| 类别 | Token 示例 | 数量 | 作用 |
|------|-----------|------|------|
| **控制 token** | `<|startoftranscript|>`, `<|endoftranscript|>` | 2 | 标记转录开始/结束 |
| **语言 token** | `<|en|>`, `<|zh|>`, `<|fr|>`, ... | 99 | 标记语言 ID |
| **任务 token** | `<|transcribe|>`, `<|translate|>` | 2 | 选择 ASR 或翻译 |
| **时间戳 token** | `<|0.00|>` ~ `<|30.00|>` | 3000 | 标记时间边界（每步 10ms） |
| **时间戳模式** | `<|timestamps|>`, `<|notimestamps|>` | 2 | 切换是否输出时间戳 |
| **空白 token** | `<|blank|>` | 1 | 用于填充（v2 无特殊用处） |

#### 训练时 Prompt 构建

```
训练样本的 token 序列:

  Step 1:  <|startoftranscript|>  ← 第一个 token 始终是这个
  Step 2:  <|en|> 或 <|zh|> ...   ← 语言 ID（训练时从标签获取）
  Step 3:  <|transcribe|> 或 <|translate|>  ← 任务类型
  Step 4:  <|timestamps|> 或 <|notimestamps|>  ← 时间戳模式
  
  中间部分: 实际文本 token（可能穿插时间戳 token）
  
  最后:    <|endoftranscript|>  ← 终止符
```

**这种设计的关键优势**：

1. **任务统一**：四个任务共享同一组参数，互相对齐学习
2. **零样本任务切换**：推理时只需改变 prompt token，无需重新加载模型
3. **语言感知**：语言 token 显式告诉模型"用哪种语言的解码路径"，防止多语言混淆
4. **时间戳作为一级信号**：时间戳 token 与文本 token 在同一输出空间，模型同时学习"说什么"和"何时说"

#### 推理时 Prompt 构建（零样本语言识别）

有趣的是，Whisper 甚至可以不指定语言 token：

```
<|startoftranscript|> <|transcribe|> ...
```

此时模型会自动检测前 30 帧音频中的语言，并内部决定使用哪种语言解码路径。这是因为语言 token 在训练时是从训练数据中自动提取的，模型学会了**从音频中推断语言**的能力。

---

## 三、模型配置对比表（完整版）

| 参数 | tiny | base | small | medium | large-v1/v2 | large-v3 | turbo |
|------|------|------|-------|--------|-------------|----------|-------|
| **编码器** | | | | | | | |
| `n_audio_layer` | 4 | 6 | 12 | 24 | 32 | 32 | 32 |
| `n_audio_state` | 384 | 512 | 768 | 1024 | 1280 | 1280 | 1280 |
| `n_audio_head` | 6 | 8 | 12 | 16 | 20 | 20 | 20 |
| `n_audio_ff` | 1536 | 2048 | 3072 | 4096 | 5120 | 5120 | 5120 |
| **解码器** | | | | | | | |
| `n_text_layer` | 4 | 6 | 12 | 24 | 32 | 32 | **4** |
| `n_text_state` | 384 | 512 | 768 | 1024 | 1280 | 1280 | 1280 |
| `n_text_head` | 6 | 8 | 12 | 16 | 20 | 20 | 20 |
| `n_text_ff` | 1536 | 2048 | 3072 | 4096 | 5120 | 5120 | 5120 |
| **总参数量** | **~39M** | **~74M** | **~244M** | **~769M** | **~1.55B** | **~1.55B** | **~809M** |
| **编码器占比** | ~53% | ~57% | ~62% | ~65% | ~66% | ~66% | ~87% |
| **解码器占比** | ~47% | ~43% | ~38% | ~35% | ~34% | ~34% | ~13% |
| **Mel 维度** | 80 | 80 | 80 | 80 | 80 | **128** | **128** |

### large-v3-turbo：解码器从 32 层压缩到 4 层

turbo 版本的核心改动只有一个：**将 32 层解码器减少到 4 层**，编码器保持 32 层不变。

```
large-v3:     编码器 32 层 + 解码器 32 层 = 1.55B 参数
large-v3-turbo:编码器 32 层 + 解码器  4 层 = 0.81B 参数

参数量减少: ~47%
推理速度:   ~6× 更快（自回归步数中每步的计算量大减）
精度损失:   ~0.5% WER 上升（可接受范围）
```

这个设计的核心理念是：

> **"语音识别的瓶颈在编码器（声学理解），不在解码器（文本生成）"**

与 GLM-ASR 的"非对称"思想不谋而合——只是 Whisper 通过蒸馏将 32→4，而 GLM-ASR 直接从设计时就选择了浅解码器。

### 参数分布估算（以 large-v2 为例）

```
Audio Encoder (66% ≈ 1.02B):
  - Conv1d stem: 80×1280 + 1280×1280 ≈ 1.7M
  - 32 × Transformer Layer:
    - QKV 投影: 3 × 1280 × 1280 = 4.9M
    - Output 投影: 1280 × 1280 = 1.6M
    - FFN up: 1280 × 5120 = 6.6M
    - FFN down: 5120 × 1280 = 6.6M
    - LayerNorms: 4 × 1280 ≈ 5K
    - 每层 ≈ 19.7M × 32 ≈ 630M
  - 编码器总 ≈ 1.02B

Text Decoder (34% ≈ 0.53B):
  - Token Embedding: 51865 × 1280 ≈ 66.4M
  - Positional Embedding: 448 × 1280 ≈ 0.6M
  - 32 × Transformer Layer:
    - QKV 投影: 3 × 1280 × 1280 = 4.9M
    - Self Output: 1280 × 1280 = 1.6M
    - Cross Q 投影: 1280 × 1280 = 1.6M
    - Cross KV 投影: 2 × 1280 × 1280 = 3.3M  (来自编码器投影)
    - Cross Output: 1280 × 1280 = 1.6M
    - FFN up: 1280 × 5120 = 6.6M
    - FFN down: 5120 × 1280 = 6.6M
    - LayerNorms: 6 × 1280 ≈ 8K
    - 每层 ≈ 26.3M × 32 ≈ 842M
  - 解码器总 ≈ 0.53B (含 embedding)

总计 ≈ 1.55B
```

---

## 四、训练数据：弱监督的力量与代价

### v2 版本（2022）：680K 小时弱监督数据

```
680K 小时训练数据的构成:
  ├── 96 种语言
  ├── 117K 小时 英语 (17%)
  ├── 216K 小时 其他语言 (32%)
  │     └── 前 10: 法语、德语、西班牙语、日语、中文、...
  ├── 125K 小时 多语言混合 (18%)
  └── 222K 小时 非英语音频 + 英语翻译标签 (33%)
```

**数据来源**：互联网上的公开音频（YouTube、播客、会议录音等），配合自动生成的转录文本或已有的字幕文件。

**弱监督的含义**：

```
传统监督学习:
  人工标注 → 每条数据都经过人工审核 → 高质量但昂贵

弱监督学习 (Whisper 的路线):
  互联网数据 → 已有转录可能包含噪声 → 大量但质量参差不齐
    - 自动字幕可能有错误
    - 背景音乐可能干扰转录
    - 数据分布偏向热门语言
```

### v3 版本（2023）：5M 小时（大规模伪标签）

```
5M 小时训练数据的构成:
  ├── 680K 小时 原始 v2 弱监督数据 (13.6%)
  └── 4.32M 小时 伪标签数据 (86.4%)
        └── 使用 large-v2 对未标注音频进行推断
              → 将 large-v2 的转录作为 "伪标签" 训练 large-v3
```

**增加的技术变化**：

| 变化 | v2 | v3 | 影响 |
|------|-----|-----|------|
| Mel 维度 | **80** | **128** | 更高的频率分辨率 |
| 数据量 | 680K 小时 | 5M 小时 | 更大的数据覆盖 |
| 伪标签 | 无 | large-v2 伪标签 | "自举"式提升 |
| 语言数 | 96 | 100+ | 更多语言支持 |

**128-bin Mel 带来的影响**：
```
80-bin Mel (v2):   0-8kHz 范围 → 80 个滤波器 → 适用于 16kHz 采样率
128-bin Mel (v3):  0-8kHz 范围 → 128 个滤波器 → 更高的频率分辨率

实际效果: v3 对高频辅音（如 /s/, /ʃ/ 等 voiceless fricatives）的辨识能力更强
        因为这些音素的能量集中在 >4kHz 的高频区域
```

### 数据集的多任务标签策略

Whisper 训练数据中的标签结构利用了互联网数据的天然多样性：

```
每种数据样本同时包含:
  ├── 音频轨道 (必选)
  ├── 语言标签 (多数样本有)
  ├── 原始语言转录 (ASR 任务的标签)
  ├── 英文翻译 (翻译任务的标签)
  └── 时间戳 (若有字幕文件则附带)
```

这种天然的"多任务标签"使得 Whisper 可以在不手动标注的情况下同时学习四个任务。

---

## 五、推理流程深度演练

以 10 秒英语音频 "Hello, this is a test of the Whisper speech recognition system." 为例。

### Stage 1: 音频重采样与预处理

```
输入音频 (任意采样率)
    → 重采样到 16kHz (Whisper 内部始终保持 16kHz)
    → 16kHz × 10s = 160,000 采样点
    → 如果超过 30s (480,000 采样点)，需要分窗处理
```

### Stage 2: 特征提取（Feature Extraction）

```
160,000 采样点
    → 25ms Hamming 窗口 / 10ms 步长
    → STFT → Mel 滤波器组 (80 或 128 个)
    → log(幅值) 变换
    → 输出: [1, n_mels, ~1000]  其中 ~1000 ≈ 160,000 / 160 ≈ 1000 帧
```

### Stage 3: Conv1d 子采样

```
[1, 80, 1000]  (large-v2 的输入)
    │
    ├─ Conv1d(80→1280, k=3, s=1) + GELU → [1, 1280, 1000]
    ├─ Conv1d(1280→1280, k=3, s=2) + GELU → [1, 1280, 500]
    │
    └─ 时间维度从 1000 压缩到 500 (2×)
```

### Stage 4: 编码器编码

```
[1, 1280, 500] → 转置为 [1, 500, 1280] (seq_len × feature_dim)
    → + Sinusoidal Positional Embedding
    → 32 层双向 Transformer
    → 输出: [1, 500, 1280]  (声学特征序列)
```

### Stage 5: Prompt 构建与解码初始化

```
解码器初始输入:
  [<|startoftranscript|>] → Token Embedding → [1, 1, 1280]
    → + Positional Embedding (可学习, pos=0)
    → 进入解码器

编码器输出 [1, 500, 1280] → Cross-Attention 的 K, V
```

### Stage 6: 自回归解码（Autoregressive Decoding）

```
Step 1:  输入 [<|sot|>]
         输出: <|en|> (语言检测)
         模型检测到音频为英语

Step 2:  输入 [<|sot|>, <|en|>]
         输出: <|transcribe|>
         选择 ASR 任务

Step 3:  输入 [<|sot|>, <|en|>, <|transcribe|>]
         输出: <|notimestamps|> (或 <|timestamps|>)
         选择时间戳模式

Step 4:  输入 [..., <|notimestamps|>]
         输出: "Hello" (第一个文本 token)

Step 5:  输入 [..., "Hello"]
         输出: ","

Step 6:  输入 [..., ","]
         输出: "this"

...

Step N:  输入 [..., "system", "."]
         输出: <|endoftranscript|>
         解码结束
```

### 分窗处理（长音频超过 30s）

```
长音频处理流程:
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Window 1  │    │ Window 2  │    │ Window 3  │
  │ 0-30s     │──→│ 20-50s    │──→│ 40-70s    │
  └──────────┘    └──────────┘    └──────────┘
       │               │               │
       ▼               ▼               ▼
  转录文本 1       转录文本 2       转录文本 3
       │               │               │
       └───────────────┼───────────────┘
                       ▼
             合并文本（去重 + 缝合）
```

- 窗口大小 = 30s（固定）
- 窗口步长 = 20s（10s 重叠）
- 重叠区域用于平滑合并——通过 `logprob` 或时间戳对齐来决定最佳拼接点

### 推理计算量分析

以 large-v2 处理 30 秒音频为例：

```
编码器（一次前向）:
  - 输入序列长度: ~1500 token (30s × 100 帧/s ÷ 2 stride)
  - Self-Attention: O(1500² × 1280 × 32 层) ≈ 138G FLOPs
  - FFN: O(1500 × 1280 × 5120 × 2 × 32) ≈ 628G FLOPs
  - 编码器总计算量 ≈ ~766G FLOPs

解码器（自回归, 假设生成 50 个 token）:
  - Prefill (第一个 token): O(1500 × 1280 × 32 层) ≈ 60G FLOPs
  - 每步解码 (49 步): O(1 × 1280 × 32 层) ≈ 1280 × 32 × 49 ≈ 2G FLOPs
  - 解码器总计算量 ≈ ~62G FLOPs

总计算量 ≈ 828G FLOPs
```

```
计算量分布:
  ┌────────────────────────────────────────────┐
  │  编码器 (92.5%)    ████████████████████████░ │
  └────────────────────────────────────────────┘
  ┌────────────────┐
  │  解码器 (7.5%)   │ ███░░░░░░░░░░░░░░░░░░░░░ │
  └────────────────┘
```

> 注意：虽然编码器占总计算的 92.5%，但解码器延迟更高，因为解码器是串行的（每步都需要前向传播），而编码器是并行的（一次前向完成）。

---

## 六、幻觉问题（Hallucination Problem）

### 弱监督的"原罪"

Whisper 最受诟病的问题之一：**在静音或噪声环境下产生幻觉（hallucination）**。

```
无音频输入（静音）→ Whisper 输出类似:
  "Thank you for watching this video. I hope you enjoyed it. Please subscribe..."
  "The speaker began by thanking the audience for their attention..."

这显然不是转录，而是模型"编造"的内容
```

### 幻觉产生的原因

**根本原因：弱监督数据中存在大量"填充式"文本模式。**

```
训练数据中的常见模式:
  YouTube 视频的结尾:
    "Thanks for watching! Don't forget to like and subscribe!"
    → 这类文本在数据集中大量出现
    
  会议录音的开场:
    "Good morning everyone, thank you for joining today's call..."
    → 同样频繁出现
```

当弱监督数据中有大量这类"套路化"文本时，模型学习到的是：

> **"在某些声学特征微弱的情况下，输出高频文本模式的期望收益 > 输出空白"**

这是因为：
1. 训练损失函数（Cross-Entropy）对"输出文本"和"输出空白"的处理不对称
2. 弱监督数据中很少出现"真正的静音+空白转录"样本
3. 模型学会了在没有足够声学证据时，**推测最可能的文本**

### 为什么 large-v3 比 large-v2 更容易产生幻觉？

这是一个反直觉的现象——**数据更多、模型更强的 v3 反而更易产生幻觉**。

```
v2 训练: 680K 小时弱监督数据
  - 数据中的噪声分布更真实
  - 转录错误的多样性更高

v3 训练: 5M 小时 (680K 弱监督 + 4.32M 伪标签)
  - 伪标签由 large-v2 生成
  - 伪标签继承了 large-v2 的偏见 (bias)
  - 模型在"自己的输出"上训练 → 确认偏差 (confirmation bias)
```

**伪标签的放大器效应**：

```
伪标签训练循环:
  Step 1: large-v2 在未标注数据上推断
          对于噪声片段: 可能输出幻觉文本
  Step 2: 将幻觉文本作为伪标签训练 large-v3
          模型学习到: "这个噪声对应这段文本"
  Step 3: large-v3 在噪声片段上 → 更强烈的幻觉
```

这是一个**自我强化的反馈环**——伪标签中的幻觉被放大了。

### 声学特征完整的片段为什么不会幻觉？

```
正常语音片段:
  音频 → Mel → Conv → Encoder → 丰富的声学特征
    → Cross-Attention 权重集中在音频的语音段
    → 解码器主要依赖声学信息生成文本
    → ✅ 准确转录

静音/噪声片段:
  音频 → Mel → Conv → Encoder → 微弱的声学特征
    → Cross-Attention 权重分散或混乱
    → 解码器缺乏声学引导 → 依赖语言模型的先验
    → ❌ 产生幻觉
```

### 缓解方法

| 方法 | 原理 | 效果 |
|------|------|------|
| **temperature 阈值** | 当解码器不确定性高时，增加随机采样 | 部分缓解 |
| **logprob 阈值** | 如果整体 logprob 过低，标记为"不确定性高" | 检测而非修复 |
| **温度回退策略** | 贪婪解码 → 若生成结果不置信 → 退火采样 | 官方推荐策略 |
| **VAD 预处理** | 先做语音活动检测，只处理有语音的片段 | 最实用方案 |
| **压缩比惩罚** | 如果转录文本显著长于常见长度，降低其分数 | 减少重复 |
| **no_speech_threshold** | 检测静音段的 logprob 阈值 | 官方实现 |

### 一个具体的"温度回退"策略示例

```python
# Whisper 官方解码策略（简化）
options = dict(
    beam_size=5,
    best_of=5,
    patience=2.0,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6,
    condition_on_previous_text=True,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

# 温度回退逻辑:
# 1. 先用 temperature=0.0 (贪婪解码)
# 2. 如果 logprob < -1.0 或压缩比 > 2.4 → 使用 temperature=0.2
# 3. 如果不置信则继续升高温度
# 4. 最高到 temperature=1.0 (完全随机采样)
```

---

## 七、Whisper 架构全景总结

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          Whisper 架构全景                                  │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  输入: 任意采样率音频 → 重采样 → 16kHz                                   │
│       │                                                                  │
│  ┌────┴─────────────────────┐                                            │
│  │     Log-Mel 频谱提取       │  80/128 维，25ms 窗 / 10ms 步长            │
│  └────┬─────────────────────┘                                            │
│       │                                                                  │
│  ┌────┴─────────────────────┐                                            │
│  │   Conv1d Stem             │  2 层：stride=1 → stride=2                │
│  │   n_mels → n_state       │  时间减半，维度提升                        │
│  └────┬─────────────────────┘                                            │
│       │                                                                  │
│  ┌────┴─────────────────────┐                                            │
│  │   Sinusoidal Position    │  固定编码，非学习                          │
│  └────┬─────────────────────┘                                            │
│       │                                                                  │
│  ┌────┴──────────────────────────────────────────────────────┐           │
│  │                   Audio Encoder                            │           │
│  │  ┌─────────────────────────────────────────────────────┐   │           │
│  │  │ LayerNorm                                          │   │  4~32 层   │
│  │  │ Multi-Head Self-Attention (双向/非因果, n_head 头)   │   │           │
│  │  │ + Residual                                         │   │           │
│  │  │ LayerNorm                                          │   │           │
│  │  │ FFN: n_state → n_audio_ff (GELU) → n_state         │   │           │
│  │  │ + Residual                                         │   │           │
│  │  └─────────────────────────────────────────────────────┘   │           │
│  └────┬──────────────────────────────────────────────────────┘           │
│       │                                                                  │
│       │  Encoder Output: [batch, T/2, n_state]                           │
│       │                                                                  │
│  ┌────┴──────────────────────────────────────────────────────┐           │
│  │                   Text Decoder                             │           │
│  │                                                             │           │
│  │  Token Embedding (可学习, vocab_size=51865)                 │           │
│  │  + Positional Embedding (可学习, max=448)                  │           │
│  │                                                             │           │
│  │  ┌─────────────────────────────────────────────────────┐   │  4~32 层 │
│  │  │ LayerNorm                                          │   │           │
│  │  │ Masked Self-Attention (因果, n_head 头)              │   │           │
│  │  │ + Residual                                         │   │           │
│  │  │ LayerNorm                                          │   │           │
│  │  │ Cross-Attention (Q:解码器, K,V:编码器输出)           │   │           │
│  │  │ + Residual                                         │   │           │
│  │  │ LayerNorm                                          │   │           │
│  │  │ FFN: n_state → n_text_ff (GELU) → n_state          │   │           │
│  │  │ + Residual                                         │   │           │
│  │  └─────────────────────────────────────────────────────┘   │           │
│  │                                                             │           │
│  │  LM Head: n_state → vocab_size (权重与 token embedding 共享) │           │
│  └────┬──────────────────────────────────────────────────────┘           │
│       │                                                                  │
│  输出: 多任务 Token 序列                                                  │
│        <|sot|> <|lang|> <|task|> <|ts_mode|> text <|eot|>                │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

                  一句话总结 Whisper：
      "用 32 层编码器 + 32 层解码器的对称 Transformer，
       在 680K 小时弱监督数据上训练，
       实现 100+ 语言的零样本语音识别。"
```

---

## 八、从架构角度看 Whisper 的继承与影响

### Whisper 继承了什么

| 架构要素 | 来源 | 对比 |
|----------|------|------|
| Seq2Seq Transformer | Vaswani et al., 2017 | 标准 Encoder-Decoder |
| Sinusoidal PE | Transformer (原始) | 与 TTS 模型一致 |
| Pre-LN | GPT-2 / BERT 后常见 | 非创新的设计选择 |
| GELU | BERT (2018) | 比 ReLU 更平滑 |
| Log-Mel 频谱 | 经典 ASR 前端 | 与 Kaldi 时代一致 |
| Time-stamp token | RNN-T 类模型 | 但用 token 表示而非概率 |

### Whisper 开创了什么

| 创新点 | 影响 |
|--------|------|
| **零样本多语言 ASR** | 改变了"每个语言一个模型"的范式 |
| **多任务统一解码格式** | 开启了"解码 prompt 工程"的先河 |
| **极其简单的架构** | 证明了"数据 > 架构设计"——标准 Transformer + 海量数据即可超越精心设计的复杂系统 |
| **弱监督的有效性** | 推动了伪标签 (pseudo-labeling) 技术路线在语音领域的广泛采用 |

### 后续模型对 Whisper 的继承

| 模型 | 继承 | 改进 |
|------|------|------|
| **GLM-ASR** | Conv1d stem、时间戳 token、多语言 token | RoPE、GQA、4× Pooling、非对称架构 |
| **Whisper v3-turbo** | 完整架构 | 32→4 层解码器蒸馏 |
| **SenseVoice** | 多任务输出格式 | 引入情感、事件识别 |
| **Qwen2-Audio** | 语音理解作为 LLM 输入 | Decoder-only 架构 |

---

## 九、关键代码结构（model.py 源码导读）

Whisper 的模型实现极其简洁（约 400 行 Python），没有任何花哨的定制算子。

### 核心模块

```python
# model.py 的核心类结构

class LayerNorm(nn.Module):           # 标准 LayerNorm
class AudioEncoder(nn.Module):         # 音频编码器
class TextDecoder(nn.Module):         # 文本解码器
class Whisper(nn.Module):             # 完整模型
```

### AudioEncoder 的核心逻辑

```python
class AudioEncoder(nn.Module):
    def __init__(self, n_mels, n_audio_state, n_audio_head, n_audio_layer):
        self.conv1 = nn.Conv1d(n_mels, n_audio_state, 3, 1, 1)   # stride=1
        self.conv2 = nn.Conv1d(n_audio_state, n_audio_state, 3, 2, 1)  # stride=2 → 时间减半
        self.positional_embedding = SinusoidalPositionalEmbedding(...)  # 固定的！
        self.layers = nn.ModuleList([ResidualAttentionBlock(...) for _ in range(n_audio_layer)])
    
    def forward(self, x: torch.Tensor):
        # x: [batch, n_mels, T]
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # [batch, T/2, n_state]
        x = x + self.positional_embedding(x)  # 固定位置编码，无学习参数
        for layer in self.layers:
            x = layer(x)
        return x  # [batch, T/2, n_state]
```

### TextDecoder 的核心逻辑

```python
class TextDecoder(nn.Module):
    def __init__(self, n_vocab, n_text_state, n_text_head, n_text_layer):
        self.token_embedding = nn.Embedding(n_vocab, n_text_state)         # 可学习
        self.positional_embedding = nn.Embedding(n_positions, n_text_state)  # 可学习！
        self.layers = nn.ModuleList([ResidualAttentionBlock(...) for _ in range(n_text_layer)])
    
    def forward(self, x, xa, kv_cache=None):
        # x: [batch, seq_len] token IDs
        # xa: [batch, seq_len_enc, n_state] encoder output
        x = self.token_embedding(x) + self.positional_embedding(x)  # 可学习位置编码
        for layer in self.layers:
            x = layer(x, xa, kv_cache=kv_cache)
        x = self.ln(x)
        logits = x @ self.token_embedding.weight.t()  # LM Head 与 Embedding 共享权重
        return logits
```

### ResidualAttentionBlock 的核心逻辑

```python
class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, cross_attention=False):
        self.attn = MultiHeadAttention(n_state, n_head)
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_state * 4),   # 扩张 4×
            nn.GELU(),
            nn.Linear(n_state * 4, n_state),   # 压缩回 n_state
        )
        self.ln1 = LayerNorm(n_state)
        self.ln2 = LayerNorm(n_state) if cross_attention else None
        self.ln3 = LayerNorm(n_state)
    
    def forward(self, x, xa=None, kv_cache=None):
        x = x + self.attn(self.ln1(x), kv_cache=kv_cache)       # Self-Attention
        if self.cross_attn:                                      # Cross-Attention (仅解码器)
            x = x + self.cross_attn(self.ln2(x), xa)             
        x = x + self.mlp(self.ln3(x))                            # FFN
        return x
```

---

## 附录：关键配置原文速查

### Whisper 模型维度配置

```python
# 来自 whisper/model.py
# whisper/model.py 中的模型维度定义

MODEL_DIMENSIONS = {
    "tiny": {
        "n_mels": 80,
        "n_audio_layer": 4,
        "n_audio_state": 384,
        "n_audio_head": 6,
        "n_text_layer": 4,
        "n_text_state": 384,
        "n_text_head": 6,
    },
    "base": {
        "n_mels": 80,
        "n_audio_layer": 6,
        "n_audio_state": 512,
        "n_audio_head": 8,
        "n_text_layer": 6,
        "n_text_state": 512,
        "n_text_head": 8,
    },
    "small": {
        "n_mels": 80,
        "n_audio_layer": 12,
        "n_audio_state": 768,
        "n_audio_head": 12,
        "n_text_layer": 12,
        "n_text_state": 768,
        "n_text_head": 12,
    },
    "medium": {
        "n_mels": 80,
        "n_audio_layer": 24,
        "n_audio_state": 1024,
        "n_audio_head": 16,
        "n_text_layer": 24,
        "n_text_state": 1024,
        "n_text_head": 16,
    },
    "large": {
        "n_mels": 80,           # v1/v2; v3 改为 128
        "n_audio_layer": 32,
        "n_audio_state": 1280,
        "n_audio_head": 20,
        "n_text_layer": 32,
        "n_text_state": 1280,
        "n_text_head": 20,
    },
}
```

### 特殊 Token ID 定义

```python
# 来自 whisper/tokenizer.py

class Tokenizer:
    # 控制 token
    SOT = "<|startoftranscript|>"       # 转录开始
    EOT = "<|endoftranscript|>"         # 转录结束
    BLANK = "<|blank|>"                 # 空白填充
    
    # 语言 token (99 个)
    LANGUAGES = {
        "en": 50259,   "zh": 50260,   "fr": 50261,
        "de": 50262,   "ja": 50263,   "ko": 50264,
        ...
    }
    
    # 任务 token
    TRANSCRIBE = "<|transcribe|>"       # ASR 任务
    TRANSLATE = "<|translate|>"         # 翻译任务
    
    # 时间戳模式
    TIMESTAMPS = "<|timestamps|>"       # 启用时间戳
    NOTIMESTAMPS = "<|notimestamps|>"   # 禁用时间戳
    
    # 时间戳 token (0.00 ~ 30.00, 每步 10ms)
    # token_id = timestamp_begin + int(time * 100)
    timestamp_begin = 50365  # <|0.00|> = 50365
    # <|30.00|> = 50365 + 3000 = 53365
```

### 解码超参数默认配置

```python
# 来自 whisper/decoding.py

class DecodingOptions:
    # 任务选择
    task: Literal["transcribe", "translate"] = "transcribe"
    language: Optional[str] = None              # None = 自动检测
    
    # 时间戳
    without_timestamps: bool = False
    
    # 解码策略
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    
    # 采样策略
    temperature: Union[float, List[float]] = 0.0  # 默认贪婪解码
    sample_len: Optional[int] = None
    
    # 幻觉缓解
    compression_ratio_threshold: float = 2.4      # 压缩比阈值
    logprob_threshold: float = -1.0                # logprob 阈值
    no_speech_threshold: float = 0.6               # 静音检测阈值
    condition_on_previous_text: bool = True        # 条件前缀
```

---

*本文基于 Whisper 论文 (arXiv:2212.04356)、GitHub 开源代码、HuggingFace 模型卡及社区讨论整理。*

## Sources

- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Whisper GitHub repository - model.py](https://github.com/openai/whisper/blob/main/whisper/model.py)
- [Whisper GitHub repository - decoding.py](https://github.com/openai/whisper/blob/main/whisper/decoding.py)
- [Whisper GitHub repository - tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)
- [OpenAI Whisper large-v3 model card (HuggingFace)](https://huggingface.co/openai/whisper-large-v3)
- [Whisper large-v3-turbo discussion (#2363)](https://github.com/openai/whisper/discussions/2363)
- [OpenAI Whisper large-v3-turbo model card (HuggingFace)](https://huggingface.co/openai/whisper-large-v3-turbo)
