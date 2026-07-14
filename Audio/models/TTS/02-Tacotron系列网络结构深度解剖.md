# Tacotron 系列网络结构深度解剖（Tacotron 1 & 2）

> Google 出品 | 端到端语音合成（文本→频谱）的奠基之作
> Tacotron 1 (2017, INTERSPEECH) → Tacotron 2 (2018, ICASSP)

---

## 写在前面：理解两个版本

Tacotron 系列由 Google 提出，定义了"端到端 TTS"的基本范式——从字符序列直接生成频谱，不再需要音素标注、时长模型、语言学特征工程。

| 版本 | 架构核心 | 声码器 | MOS | 状态 |
|------|---------|--------|-----|------|
| **Tacotron 1** | CBHG + Content-based Attention + GRU Decoder | Griffin-Lim（无训练） | 3.82 | 验证路线可行 |
| **Tacotron 2** | Conv + BiLSTM + Location-Sensitive Attention + LSTM Decoder + Post-Net | WaveNet（条件式） | **4.53** | 接近人类水平（4.58） |

> 本文同时分析两个版本的关键差异，理解"为什么从 1→2 要改这些地方"比单独看任何一个版本都更有价值。

---

## 一、整体架构设计哲学

### 核心设计理念

> **"字符到频谱，一步到位"**

Tacotron 的核心理念是用一个 Seq2Seq 模型替代传统 TTS 管线中的所有组件（文本前端、时长模型、声学模型、声码器）。但它在架构上的关键选择是：

- **频谱作为中间目标**：不直接生成波形（WaveNet 负责这一步），而是先生成信息密度更低的频谱——**把难的问题拆成两个子问题**
- **自回归逐帧生成**：解码器一帧等一帧，保留时序依赖性——**精度优先于速度**
- **Attention 做对齐**：不依赖外部 aligner 或音素标注——**真正端到端的关键**

### 架构总览（Tacotron 2）

```
文本字符 (Character Sequence)
    │
    ├── ① Character Embedding (256-D)
    │
    ├── ② Encoder: 3× Conv1D (512, k=5) + BiLSTM (512)
    │
    ├── ③ Location-Sensitive Attention (128-D, 32 filters, k=31)
    │   └── 对齐编码器输出和解码器时间步
    │
    ├── ④ Decoder: Pre-Net (256+256) → LSTM (1024×2) → Linear → Mel (80-D)
    │   └── 自回归逐帧生成 Mel 频谱
    │
    ├── ⑤ Post-Net: 5× Conv1D (512, k=5, tanh, 残差连接)
    │   └── 精细化 Mel 频谱残差
    │
    ├── ⑥ Stop Token: Sigmoid 输出——模型自己决定何时停止生成
    │
    └── 输出: 80-band Mel Spectrogram → WaveNet Vocoder → 波形
```

---

## 二、各模块深度解剖

### 2.1 Character Embedding（字符嵌入层）

**定位**：将离散的文本字符映射到连续向量空间，使模型能捕获字符间的语义关系。

```
输入: one-hot 字符序列 [batch, T_char]
    │
    └── Embedding(vocab_size=40, embed_dim=256)
        └── 输出: [batch, T_char, 256]
```

| 参数 | Tacotron 1 | Tacotron 2 |
|------|-----------|-----------|
| 词表大小 | ~40（英文字母+标点+空格） | ~40（同） |
| 嵌入维度 | 256 | 256 |
| 是否需要音素 | ❌ 纯字符 | ❌ 纯字符 |

**设计要点**：纯字符输入意味着模型必须自己学习"拼写→发音"的映射。在英语中，这个映射有很多不规则性（"caught"和"cot"的"gh"不发音、"t"不送气等）。Tacotron 能学习到这些不规则性，但需要足够的数据。

---

### 2.2 Encoder（编码器）

**定位**：从字符嵌入中提取富含上下文信息的序列表征，供 Attention 机制参考。

#### Tacotron 1 Encoder（CBHG）

```
输入: [batch, T_char, 256] (字符嵌入)
    │
    ├── Pre-Net: FC(256→128, ReLU, Dropout 0.5)
    │
    ├── CBHG 模块:
    │   ├── Conv1D Bank (K=16):
    │   │   ├── conv-1-128-ReLU (1-gram)
    │   │   ├── conv-2-128-ReLU (2-gram)
    │   │   ├── ...
    │   │   └── conv-16-128-ReLU (16-gram)
    │   │   → 拼接: [batch, T_char, 16×128=2048]
    │   │
    │   ├── MaxPool(stride=1, width=2) — 时间维度不变
    │   │
    │   ├── Conv1D Projection:
    │   │   ├── conv-3-128-ReLU
    │   │   └── conv-3-128-Linear + 残差连接 (+ 原始 Pre-Net 输出)
    │   │   → BatchNorm
    │   │
    │   ├── Highway Network: 4 层 FC(128→128, ReLU) + 门控
    │   │
    │   └── BiGRU(128 cells)
    │
    └── 输出: [batch, T_char, 256] (BiGRU 输出，双向各 128 → 拼接 256)
```

**CBHK 的核心思想**：用多个不同宽度的卷积核并行提取 n-gram 级别的特征（1-gram 到 16-gram），覆盖字符序列中的短程和长程模式。

| 参数 | 值 | 含义 |
|------|-----|------|
| Conv1D Bank K | 16 | 卷积核宽度从 1 到 16 |
| 每种卷积核数量 | 128 | 每个宽度提取 128 维特征 |
| 拼接后维度 | 16×128=2048 | 所有 n-gram 特征拼接 |
| Highway Net 层数 | 4 | 门控残差网络，控制信息流通 |
| BiGRU cells | 128 | 双向 RNN 节点数 |

#### Tacotron 2 Encoder（Conv + BiLSTM）——对比 1 的简化

```
输入: [batch, T_char, 256] (字符嵌入)
    │
    ├── Conv1D × 3:
    │   ├── conv-5-512-BN-ReLU-Dropout(0.5)
    │   ├── conv-5-512-BN-ReLU-Dropout(0.5)
    │   └── conv-5-512-BN-ReLU-Dropout(0.5)
    │   输出: [batch, T_char, 512]
    │
    └── BiLSTM(512 units: 256 per direction)
        └── 输出: [batch, T_char, 512]
```

**为什么从 CBHG 简化为 Conv + BiLSTM？**

| 对比 | CBHG (Tacotron 1) | Conv+BiLSTM (Tacotron 2) |
|------|------------------|------------------------|
| 参数量 | 较大（16 组卷积 + Highway + BiGRU） | **更小**（3 组相同卷积 + BiLSTM） |
| 训练速度 | 慢 | **快** |
| 效果 | MOS 3.82 | MOS 4.53（但改进来自 Decoder 和声码器，不是 Encoder） |
| 设计动机 | 需要多尺度 n-gram 特征 | 3 层 Conv 已足够提取局部上下文，BiLSTM 负责全局上下文 |

Tacotron 2 的编码器简化的启示：**复杂架构设计的价值往往不在于它看起来多精巧，而在于它解决的具体问题是否真的存在。**

| 参数 | Tacotron 1 | Tacotron 2 |
|------|-----------|-----------|
| 卷积层数 | 16 组 (Bank) + 2 投影 | 3 层相同的 Conv1D |
| 卷积核大小 | 1~16（多尺度） | 全 5（固定） |
| 每层滤波器 | 128 | 512 |
| 激活函数 | ReLU | ReLU |
| 正则化 | Dropout(0.5) | BatchNorm + Dropout(0.5) |
| 序列模型 | BiGRU(128) | BiLSTM(512) |
| 总参数量 | ~500K | ~4M（大部分来自卷积） |

---

### 2.3 Attention（注意力机制）

**定位**：Seq2Seq 模型的核心——决定生成当前频谱帧时，应该"看"文本的哪个部分。

#### Tacotron 1: Content-based Tanh Attention（内容加性注意力）

```
e(i,j) = v^T · tanh(W_1 · h_j + W_2 · s_i)
α_i,j = softmax(e(i,j))
c_i = Σ_j α_i,j · h_j

其中:
  h_j (512-D): 编码器第 j 时间步的输出
  s_i (256-D): 解码器 Attention RNN 第 i 步的隐藏状态
  W_1, W_2 (128 × 512, 128 × 256): 投影矩阵
  v (128-D): 注意力向量
  c_i (512-D): 上下文向量（送入解码器）
```

| 参数 | 值 | 含义 |
|------|-----|------|
| 注意力维度 | 128 | 投影后的隐藏维度 |
| 注意力类型 | **Content-based** | 只根据编码器/解码器的内容决定对齐 |
| 位置信息 | **无** | 不做任何位置偏置——长文本不稳定 |

**Tacotron 1 的 Attention 问题**：纯 content-based 意味着每次对齐都是"从零开始搜索"。在长文本中，模型可能"忘了"已经读过哪些位置，导致：
- **字重复**：对同一段文本生成两次
- **跳词**：跳过某段文本
- **提前终止**：模型误以为文本已读完

#### Tacotron 2: Location-Sensitive Attention（位置敏感注意力）

Tacotron 2 的核心改进之一。在 content-based 的基础上引入**累积注意力权重的位置特征**。

```
# 步骤 1: 位置特征
cumulative_weights = Σ(α_{1..i-1})  # 前 i-1 步的累积注意力权重
f_i = Conv1D(cumulative_weights)     # 32 filters, kernel=31
# f_i 的形状: [batch, T_char, 32]

# 步骤 2: 注意力得分
e(i,j) = w^T · tanh(W·s_i + V·h_j + U·f_i,j + b)

# 步骤 3: 归一化
α_i = softmax(e_i)

# 步骤 4: 上下文向量
c_i = α_i · h   →  [512-D]
```

| 参数 | 值 | 含义 |
|------|-----|------|
| 位置卷积滤波器数 | 32 | 从累积权重中提取位置特征 |
| 位置卷积核大小 | 31 | 覆盖 ~31 个字符的位置上下文 |
| 注意力维度 | 128 | 投影后维度 |
| 上下文向量维度 | 512 | 与编码器隐藏状态维度一致 |

**位置敏感的核心意义**：

```
传统 Attention (Tacotron 1):
  每步独立搜索——"这帧应该对齐文本的哪里？"
  没有历史信息——容易跳回已读过的位置

位置敏感 Attention (Tacotron 2):
  累积权重告诉模型"我已经读到这了"
  位置特征编码"附近的对齐趋势"
  → 模型自然地向前移动，很少回头
```

用一个类比理解：**传统 Attention 是每步重新找方向，Location-Sensitive Attention 是沿着已经走过的路径继续前进**。

#### 可视化对比

```
Tacotron 1 Attention (不稳定):
  文本: "the cat sat on the mat"
  帧 1-10:  ████████░░░░░░░░░░░░  (正确)
  帧 11-20: ████████████████░░░░  (重复了"the cat")
  帧 21-25: ░░░░░░░░░░░░██████  (跳过"on the"跳到"mat")

Tacotron 2 Attention (稳定):
  文本: "the cat sat on the mat"
  帧 1-10:  ████████░░░░░░░░░░░░
  帧 11-20: ░░░░████████████░░░░
  帧 21-30: ░░░░░░░░░░░░████████
  每步平滑向右移动——无重复无跳词
```

---

### 2.4 Decoder（解码器）

**定位**：自回归地生成频谱帧——每一帧的生成依赖于之前生成的所有帧。

#### Tacotron 1 Decoder

```
注意力上下文 c_i (512-D) + 上一帧预测 (mel/r 帧)
    │
    ├── Pre-Net: FC(256→128→128, ReLU, Dropout 0.5)
    │
    ├── Attention RNN: 1 层 GRU(256)
    │   └── 产生注意力 query
    │
    ├── Decoder RNN: 2 层残差 GRU(256)
    │   └── 输入 = [Attention RNN 输出, 上下文向量] 拼接
    │
    └── 输出投影: Linear(mel_dim × r) → 预测 r 帧
```

#### Tacotron 2 Decoder

```
注意力上下文 c_i (512-D) + 上一帧 Mel (-1) (80-D)
    │
    ├── Pre-Net: FC(256→256, ReLU, Dropout 0.5) × 2 层
    │   └── Dropout 在推理时作为多样性噪声源
    │
    ├── LSTM × 2 层 (1024 units 每层, Zoneout 0.1)
    │   └── 输入 = [Pre-Net 输出, 上下文向量] 拼接
    │
    ├── 输出投影 1: Linear(1024→80) → 当前帧 Mel 频谱 (80-D)
    │
    └── 输出投影 2: Linear(1024→1) → Sigmoid → Stop Token
```

#### Decoder 对比

| 维度 | Tacotron 1 | Tacotron 2 |
|------|-----------|-----------|
| Pre-Net | 256→128→128 | **256→256→ReLU (2 层，相同维度)** |
| RNN 类型 | GRU | **LSTM** |
| RNN 单元数 | 256 | **1024** |
| RNN 层数 | 3 (1 Attn + 2 Dec) | 2 (纯 Decoder) |
| 正则化 | Dropout(0.5) | **Zoneout(0.1)** |
| 停止机制 | 无（固定帧数） | **Stop Token（模型自停）** |
| 每步输出 | r 帧（r=2） | **1 帧** |
| 残差连接 | Decoder RNN 层间 | **无（LSTM 自带门控）** |

**Pre-Net 的 Dropout 为什么重要？**

Tacotron 2 的 Pre-Net 用了 Dropout(0.5)——推理时也在用。这听起来反直觉：推理时 Dropout 不是应该关闭吗？

原因：**Mel 频谱的单帧预测是一个多模态问题**——给定相同的过去帧和文本，下一个 Mel 帧可以有多种合理的取值（不同的语调、轻重音）。Pre-Net 的 Dropout 在推理时引入了随机性，让解码器在每次前向时看到略有不同的输入，从而产生多样化的输出。

没有 Pre-Net Dropout，模型会退化为"均值预测器"——所有可能的输出都被平均成了一个模糊的频谱。

#### Stop Token 的设计

```
Stop Token 是一个线性投影 + Sigmoid:
  p_stop = σ(W_s · decoder_state + b_s)

训练时: 二分类交叉熵损失（目标: 如果这是最后一帧，p_stop=1，否则=0）
推理时: 当 p_stop > 0.5 时停止生成
```

**为什么需要 Stop Token？** Tacotron 1 固定生成预设帧数——要么太长（尾部噪声），要么太短（句子没说完）。Stop Token 让模型自主判断"这句话说完了"，生成的时长更加自然。

---

### 2.5 Post-Net（后处理网络）

**定位**：对解码器生成的 Mel 频谱做残差精修，提升高频细节。

#### Tacotron 1 Post-Net: CBHG（与 Encoder 相同结构）

```
K=8 的 Conv1D Bank → MaxPool → Conv1D Projection → Highway × 4 → BiGRU
输出: Linear-spectrogram (1025-D FFT bins)
```

#### Tacotron 2 Post-Net: 5 层 Conv1D（更轻量）

```
输入: 解码器输出 Mel [batch, T_mel, 80]
    │
    ├── Conv1D(80→512, k=5, BN, tanh)
    ├── Conv1D(512→512, k=5, BN, tanh)
    ├── Conv1D(512→512, k=5, BN, tanh)
    ├── Conv1D(512→512, k=5, BN, tanh)
    └── Conv1D(512→80, k=5, BN, Linear)  ← 最后一层无激活
    │
    └── 输出: 残差 [batch, T_mel, 80]
        最终 Mel = 解码器输出 + 残差
```

| 参数 | Tacotron 1 Post-Net | Tacotron 2 Post-Net |
|------|--------------------|--------------------|
| 结构 | CBHG (复杂) | **5 × Conv1D (简洁)** |
| 参数量 | 大 | **小** |
| 输出目标 | 线性谱 (1025-D) | **Mel 残差 (80-D)** |
| 激活函数 | ReLU + 门控 | **tanh** |
| 残差连接 | ❌ | ✅ **残差加到解码器输出** |

Tacotron 2 的 Post-Net 输出残差而非完整频谱——这是"残差学习"的思想。解码器生成"大概的" Mel 频谱，Post-Net 补充高频细节和修正误差。训练时两个输出的损失都计算（辅助收敛）。

```
总损失 = L1(解码器输出, 目标) + L1(最终输出, 目标)
```

---

### 2.6 损失函数与训练配置

| 损失项 | Tacotron 1 | Tacotron 2 |
|--------|-----------|-----------|
| Mel 前 Loss | — | ✅ L1 (解码器输出 vs 目标) |
| Mel 后 Loss | ✅ L1 (线性谱 vs 目标) | ✅ L1 (Post-Net 输出 vs 目标) |
| Stop Token Loss | — | ✅ BCE |
| 总损失 | L1(线性谱) | L1(Mel前) + L1(Mel后) + BCE(Stop) |

**训练超参数：**

| 参数 | Tacotron 1 | Tacotron 2 |
|------|-----------|-----------|
| 采样率 | 24kHz | 24kHz (原论文) / 22.05kHz (NVIDIA) |
| 帧长 | 50ms | 50ms |
| 帧移 | 12.5ms | 12.5ms / 11.6ms |
| Mel 带数 | 80 | 80 |
| FFT 点数 | 2048 | 2048 / 1024 |
| 优化器 | Adam | Adam |
| 初始学习率 | 0.001 | 0.001 |
| 学习率衰减 | 多次衰减 | 指数衰减 |
| Batch Size | 32 | 64+ |
| 教师强制 | ✅ | ✅ |
| Reduction Factor r | 2 | 1 |

---

## 三、推理流程演练

以生成文本 "hello world"（约 1.5 秒音频）为例：

### Stage 1: 文本编码

```
输入: ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
      → 11 个字符，每个 256-D 嵌入
      → 3 层 Conv1D(512, k=5) → [11, 512]
      → BiLSTM(512) → [11, 512] (编码器输出)
```

### Stage 2: Prefill + 自回归解码

```
Step 0:  查询编码器最后帧 → Attention → 上下文向量
         Pre-Net(零向量) → LSTM → 输出第 1 帧 Mel

Step 1:  输入第 0 帧 Mel → Pre-Net(Dropout) → LSTM → Attention → 第 1 帧 Mel

Step 2:  输入第 1 帧 Mel → ... → 第 2 帧 Mel

...

Step T-1: Stop Token > 0.5 → 停止解码
```

### Stage 3: Post-Net 精细化

```
解码器输出: [T_mel, 80]
    → 5 层 Conv1D Post-Net → 残差 [T_mel, 80]
    → 最终 Mel = 解码输出 + 残差
```

### Stage 4: WaveNet 声码器（Tacotron 2）

```
Mel 频谱 [T_mel, 80]
    → 转置卷积上采样至 16kHz 对齐
    → WaveNet 逐点自回归生成 16000 × 1.5s = 24000 个采样点
    → 波形
```

### 各阶段数据维度（Tacotron 2）

```
| 阶段 | 形状 | 说明 |
|------|------|------|
| 输入文本 | [11] (字符) | "hello world" |
| Embedding | [11, 256] | 字符嵌入 |
| Encoder | [11, 512] | 3×Conv + BiLSTM |
| Attention 上下文 | [512] | 每步的动态加权编码器输出 |
| Decoder 输出 | [T_mel, 80] | ~120 帧 (1.5s @ 12.5ms = 120帧) |
| Post-Net 输出 | [T_mel, 80] | 残差精修后的最终 Mel |
| WaveNet 输出 | [1, 24000] | 16kHz 采样率 × 1.5s |
```

---

## 四、Tacotron 1 vs Tacotron 2 全景对比

| 模块 | Tacotron 1 | Tacotron 2 | 改进方向 |
|------|-----------|-----------|---------|
| **目标** | 验证端到端可行 | **接近人类音质** | — |
| **Encoder** | CBHG (16 conv bank + BiGRU) | **3×Conv + BiLSTM** | 更简洁高效 |
| **Attention** | Content-based (无位置信息) | **Location-Sensitive** | 消除跳词/重复 |
| **Decoder** | 3 层 GRU (256) | **2 层 LSTM (1024)** | 更大容量，更稳定 |
| **输出** | 线性频谱 (1025-D) | **Mel 频谱 (80-D)** | 相位交给声码器 |
| **Post-Net** | CBHG (K=8) | **5 × Conv1D 残差** | 轻量高效 |
| **停止机制** | 固定帧数 | **Stop Token** | 动态时长 |
| **声码器** | Griffin-Lim (无参数) | **WaveNet (条件式)** | 音质飞跃 |
| **MOS** | 3.82 | **4.53** | 接近人类 (4.58) |

---

## 五、性能分析与优化

### 5.1 推理瓶颈

```
Tacotron 2 推理时间分布 (1.5 秒音频):
    ├── Encoder (Pre-Net + Conv + BiLSTM): ~2%
    ├── Decoder + Attention (逐帧):         ~35%
    │   └── 每帧: Pre-Net → LSTM → Attn → Linear → Post-Net(1层)
    ├── Post-Net (全序列 5 层 Conv):        ~3%
    └── WaveNet 声码器:                    ~60%  ← 核心瓶颈
```

核心瓶颈在 WaveNet 自回归逐点生成。实际产品中通常用 WaveGlow（并行流模型）替换 WaveNet，推理速度从 0.1x 实时率提升到 25-70x 实时率。

### 5.2 Tacotron 2 参数量估算

```
Encoder:
  Character Embedding: 40 × 256 = 10K
  Conv1D × 3: 3 × (5 × 256 × 512 + 512) = 3 × 655K = ~1.97M
  BiLSTM: 4 × (512 × 512 + 512 × 512) = ~2.1M  (输入门、遗忘门、输出门、候选)
  Encoder 合计: ~4.1M

Attention:
  Location Conv: 1 × 31 × 32 = 1K
  Attention Projection: ~128K
  Attention 合计: ~130K

Decoder:
  Pre-Net: 2 × (80 × 256 + 256 × 256) = ~172K
  LSTM × 2: 2 × 4 × ((256+512) × 1024 + 1024 × 1024) = ~14.7M
  Output Projection: 1024 × 80 = 82K
  Stop Token: 1024 × 1 = 1K
  Decoder 合计: ~15M

Post-Net:
  5 × 1D Conv: 5 × (5 × 80 × 512 + 5 × 512 × 512 + ...) ≈ ~8M

WaveNet Vocoder: ~3M (详见 WaveNet 深度解剖)

总参数量: ~30M (Tacotron 2 本身) + ~3M (WaveNet) ≈ 33M
```

> 相比后来的 TTS 模型（VITS ~40M, CosyVoice ~300M, Qwen3-TTS 1.7B），Tacotron 2 的参数量现在看来非常小。

---

## 六、架构设计的深层思考

### 6.1 为什么 Tacotron 2 是两阶段而非端到端？

"端到端"这个词在 TTS 语境下的歧义：
- **严格的端到端**：文本 → 波形，一个模型一步到位——VITS 做到了
- **Tacotron 的端到端**：文本 → 频谱端到端——频谱→波形仍然需要声码器

Tacotron 2 选择两阶段的根本原因：**波形生成的难度和频谱生成的难度不是一个量级。**
- 频谱：80 维 / 帧，12.5ms 帧移，1 秒 = 80 帧
- 波形：1 维 / 采样点，16kHz，1 秒 = 16000 点
- 计算量差 200 倍

把两个不同量级的问题分开做，在 2018 年是最合理的选择。

### 6.2 Tacotron 2 的影响——为什么它是一代标杆

```
Tacotron 2 (2018)
    │
    ├── 启发了非自回归 TTS
    │   └── FastSpeech (2019): "既然对齐可以显式预测，为什么还要自回归？"
    │
    ├── 启发了单阶段端到端
    │   └── VITS (2021): "声学模型和声码器可以是一个模型"
    │
    └── 启发了多说话人 TTS
        └── Jia et al. (2018): "加一个说话人编码器就能零样本克隆"
```

从 2018 到 2022 年，**Tacotron 2 + HiFi-GAN** 组合是工业界部署最广泛的 TTS 方案。即使今天被 Flow Matching 和扩散模型超越，Tacotron 2 作为"端到端 TTS 的参考答案"的地位不可替代。

---

## 七、实际部署效果

### 7.1 主观评测

| 系统 | MOS | 来源 |
|------|-----|------|
| Tacotron 1 + Griffin-Lim | 3.82 | 论文 (LJSpeech) |
| Tacotron 2 + WaveNet | **4.53** | 论文 |
| Tacotron 2 + WaveGlow | 4.35-4.50 | NVIDIA |
| Tacotron 2 + HiFi-GAN | 4.40-4.52 | 社区 |
| 人类录音 | 4.58 | — |

### 7.2 实际场景表现

| 场景 | Tacotron 1 | Tacotron 2 |
|------|-----------|-----------|
| 短句（<5 秒） | ✅ 可用 | ✅ 自然 |
| 长句（>10 秒） | ⚠️ 偶有跳词/重复 | ✅ 稳定 |
| 英语朗读 | ✅ 好 | ✅ 接近真人 |
| 其他语言 | ⚠️ 需要大量调参 | ⚠️ 仍然需要语言适配 |
| 快速合成 | ❌ Griffin-Lim 音质差 | ✅ WaveGlow 替换后实时 |

### 7.3 生产部署

- **Google Assistant** (2019)：Tacotron 2 + WaveNet 用于部分流量
- **NVIDIA NGC**：提供 Tacotron 2 + WaveGlow 的容器化部署方案
- **多个开源项目**：ESPnet、Coqui TTS 的 Tacotron 2 实现

---

## 八、总结：一张图看穿 Tacotron

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Tacotron 2 架构全景                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  文本: "hello world" (11 个字符)                                      │
│       │                                                              │
│  ┌────┴───────────────┐                                              │
│  │  Character Embed    │  40-word vocab → 256-D embedding            │
│  └────┬───────────────┘                                              │
│       │                                                              │
│  ┌────┴───────────────┐                                              │
│  │  Conv1D × 3        │  k=5, 512 filters, BN+ReLU+Dropout          │
│  └────┬───────────────┘                                              │
│       │                                                              │
│  ┌────┴───────────────┐                                              │
│  │  BiLSTM (512)      │  256 前向 + 256 反向                         │
│  └────┬───────────────┘        ──── Encoder ────                     │
│       │                                                              │
│  ┌────┴───────────────────────────────────────────────────────┐      │
│  │  Location-Sensitive Attention                               │      │
│  │  ├── 累积权重 → Conv1D(32 filters, k=31) → 位置特征         │      │
│  │  └── tanh(W·s + V·h + U·f) → softmax → 上下文向量           │      │
│  └────┬───────────────────────────────────────────────────────┘      │
│       │                                                              │
│  ┌────┴───────────────────────────────────────────────────────┐      │
│  │  Decoder （自回归逐帧）                                     │      │
│  │                                                              │      │
│  │  ┌──────────────┐                                            │      │
│  │  │  Pre-Net     │  FC(256)→FC(256)→ReLU, Dropout(0.5)        │      │
│  │  └──────┬───────┘                                            │      │
│  │         ↓                                                    │      │
│  │  ┌──────────────┐                                            │      │
│  │  │  LSTM × 2    │  1024 cells, Zoneout(0.1)                  │      │
│  │  └──────┬───────┘                                            │      │
│  │         ↓                                                    │      │
│  │  ┌──────────────┐  ┌──────────────┐                          │      │
│  │  │  Linear→Mel  │  │  Linear→Stop │  80-D Mel + Stop Token   │      │
│  │  └──────────────┘  └──────────────┘                          │      │
│  └────┬───────────────────────────────────────────────────────┘      │
│       │                                                              │
│  ┌────┴───────────────┐                                              │
│  │  Post-Net          │  5×Conv1D(512, k=5, tanh) + 残差            │
│  └────┬───────────────┘                                              │
│       │                                                              │
│  ┌────┴───────────────┐  ┌────────────────┐                         │
│  │  Mel Spectrogram   │→│  WaveNet/WaveGlow │→ 波形                  │
│  │  (80 bands, T帧)    │  └────────────────┘                         │
│  └────────────────────┘                                              │
│                                                                      │
│  一句话总结 Tacotron 2：                                             │
│  "Conv+BiLSTM 编码文本，Location-Sensitive Attention 对齐，          │
│   LSTM 逐帧生成 Mel，Post-Net 精细化残差，WaveNet 还原波形。"         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 附录：关键配置速查

### Tacotron 2 完整超参数表

```python
# 音频参数
sample_rate = 22050          # 16kHz (原论文) / 22.05kHz (NVIDIA)
hop_length = 256             # 11.6ms (NVIDIA) / 12.5ms (原论文)
filter_length = 1024         # FFT 窗口大小
win_length = 1024            # 窗口长度
n_mels = 80                  # Mel 带数
mel_fmin = 0.0               # Mel 最小频率
mel_fmax = 8000.0            # Mel 最大频率

# Encoder
encoder_embedding_dim = 512
encoder_conv_channels = 512
encoder_conv_kernel_size = 5
encoder_conv_layers = 3
encoder_lstm_units = 256     # 每方向

# Attention
attention_dim = 128
attention_filters = 32
attention_kernel_size = 31

# Decoder
decoder_rnn_dim = 1024
decoder_rnn_layers = 2
decoder_prenet_dims = [256, 256]
max_decoder_steps = 1000     # 最大步数限制
stop_threshold = 0.5         # Stop Token 阈值

# Post-Net
postnet_conv_channels = 512
postnet_conv_kernel_size = 5
postnet_conv_layers = 5
```

### 训练配置

```python
batch_size = 64              # 可随 GPU 内存调整
learning_rate = 1e-3
weight_decay = 1e-6
grad_clip_thresh = 1.0       # 梯度裁剪
mask_padding = True          # 填充部分不参与 loss 计算
```

---

*本文基于 Tacotron 1 (Wang et al., INTERSPEECH 2017)、Tacotron 2 (Shen et al., ICASSP 2018) 论文、NVIDIA/DeepLearningExamples 官方实现及 r9y9/tacotron_pytorch 社区实现整理分析。*

**Sources:**
- [Tacotron: Towards End-to-End Speech Synthesis - Google (INTERSPEECH 2017)](https://arxiv.org/abs/1703.10135)
- [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions - Google (ICASSP 2018)](https://arxiv.org/abs/1712.05884)
- [NVIDIA/tacotron2 - GitHub (BSD-3)](https://github.com/NVIDIA/tacotron2)
- [NVIDIA/DeepLearningExamples - Tacotron2 + WaveGlow](https://github.com/NVIDIA/DeepLearningExamples)
- [r9y9/tacotron_pytorch - GitHub (MIT)](https://github.com/r9y9/tacotron_pytorch)
- [Transfer Learning from Speaker Verification to Multispeaker TTS - Jia et al. (NeurIPS 2018)](https://arxiv.org/abs/1806.04558)
