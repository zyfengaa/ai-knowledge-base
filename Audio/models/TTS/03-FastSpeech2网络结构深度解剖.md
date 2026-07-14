# FastSpeech 2 网络结构深度解剖

> 微软（Microsoft）& 浙江大学出品 | 非自回归 TTS 的里程碑
> 论文发表于 ICLR 2021

---

## 写在前面：FastSpeech 1 → 2 → 2s

FastSpeech 系列的目标明确——**让 TTS 从逐帧自回归变成全序列并行生成**。三个版本逐级推进：

| 版本 | 核心改进 | 训练方式 | 声码器 | MOS |
|------|---------|---------|--------|-----|
| **FastSpeech 1** (2019) | 首次非自回归 TTS | 知识蒸馏（Tacotron 2 教） | WaveGlow | ~3.8 |
| **FastSpeech 2** (2020) | Variance Adaptor + 真实数据训练 | MFA 时长 + 真实 Mel | HiFi-GAN | **~4.45** |
| **FastSpeech 2s** (2020) | 文本→波形一步到位 | 对抗训练 + 多分辨率 STFT | 内置波形解码器 | ~4.0 |

> 本文以 **FastSpeech 2** 为主，因为它是系列中影响最大的版本。FastSpeech 1 已被 2 完全取代，2s 作为纯文本→波形的尝试影响力有限。

---

## 一、整体架构设计哲学

### 核心思想

> **"TTS 不需要自回归——显式对齐 + 并行生成 + 条件信息注入就够了"**

Tacotron 2 用 Attention 隐式对齐、用自回归逐帧生成。FastSpeech 2 的核心理念是：
1. **显式对齐**：用 MFA 预计算每个音素持续多少帧（不再靠 Attention 自学）
2. **并行生成**：一次性算出所有 Mel 帧（不再逐帧等）
3. **条件注入**：对"该多高多快多重"给出明确的 Pitch/Energy/Duration 信号

### 架构总览

```
音素序列 (Phoneme Sequence)
    │
    ├── ① Embedding + Positional Encoding
    │
    ├── ② Encoder: FFT Block × N (N=4~6)
    │   └── Multi-Head Self-Attention + Conv1D + LayerNorm + Res
    │
    ├── ③ Variance Adaptor（核心创新）
    │   ├── Duration Predictor → Length Regulator（扩展帧数）
    │   ├── Pitch Predictor → Pitch Embedding + Add
    │   └── Energy Predictor → Energy Embedding + Add
    │
    ├── ④ Decoder: FFT Block × N（同 Encoder）
    │
    ├── ⑤ Linear Projection → Mel 频谱 (80-D)
    │
    └── ⑥ Post-Net（可选）→ 波形声码器（HiFi-GAN）
```

### 非自回归为什么能行？

```
Tacotron 2（自回归）:
  生成第 t 帧 → 等第 t 帧完成 → 生成第 t+1 帧 → ...
  时序 = O(T)，T 是帧数

FastSpeech 2（非自回归）:
  把所有音素一次性通过 Encoder → Duration Predictor 展开 → Decoder → 所有帧一次性输出
  时序 = O(1)，与帧数无关
```

这种"一次性生成"之所以可行，是因为 **Duration Predictor 提供了帧数的显式信息**——Decoder 不需要像自回归那样"边生成边猜测还剩多少帧"。

---

## 二、各模块深度解剖

### 2.1 音素嵌入与位置编码

**定位**：将离散的音素序列映射到连续向量空间，并注入位置信息。

```
输入: 音素 ID 序列 [batch, T_phone]
    │
    ├── Embedding(vocab_size, d_model=256)
    │   └── [batch, T_phone, 256]
    │
    ├── + Positional Encoding (sin/cos 固定, 或可学习)
    │
    └── 输出: [batch, T_phone, 256]
```

| 参数 | 值 | 含义 |
|------|-----|------|
| 输入层 | 音素而非字符 | 使用音素（phoneme）而非字符（character）——减少拼写不规则带来的模糊性 |
| `d_model` | 256 | 隐藏维度（论文）/ 384（PaddleSpeech） |
| 位置编码 | sin/cos 固定 | Transformer 标准做法 |

**为什么 FastSpeech 2 用音素（phoneme）而非字符（character）？** 

Tacotron 用字符可以成功，因为它有 Attention 可以自学拼写→发音映射。FastSpeech 2 是非自回归——如果 Encoder 对字符"gh"和"f"的输出差不多，Duration Predictor 也无从判断哪个该持续几帧。**音素消除了拼写的不规则性**，让 Encoder 的输入和 Duration Predictor 的目标直接对应。

---

### 2.2 FFT Block（Feed-Forward Transformer Block）

**定位**：编解码器的基本构建单元。每个 FFT Block = Self-Attention + 1D Conv + FFN。

#### 每个 FFT Block 内部分解

```
输入: x [batch, seq_len, d_model=256]
    │
    ├── Multi-Head Self-Attention (2 heads)
    │   ├── QKV 投影: d_model → 3 × d_model
    │   ├── Scaled Dot-Product Attention
    │   ├── 残差连接: x + Attn(x)
    │   └── LayerNorm
    │
    ├── Conv1D 模块（非传统 FFN）
    │   ├── Conv1D(d_model → d_ffn, k=9, ReLU)
    │   ├── Conv1D(d_ffn → d_model, k=1)
    │   ├── 残差连接: x + Conv(x)
    │   └── LayerNorm
    │
    └── 输出: [batch, seq_len, d_model]
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `d_model` | 256 | 隐藏维度 |
| `d_ffn` | 1024 | FFN 中间层维度（4× d_model） |
| `num_heads` | 2 | 注意力头数 |
| Encoder 层数 | 4 | FFT Block 数量 |
| Decoder 层数 | 4 | FFT Block 数量 |
| Conv1D kernel | 9 | 在序列维度上的卷积核宽度 |

**为什么用 Conv1D 而非标准 Transformer 的 FFN？**

```
标准 Transformer FFN:
  FFN(x) = Linear_2(ReLU(Linear_1(x)))
  → 每位置独立，没有跨位置的交互

FastSpeech 2 的 Conv 模块:
  Conv1D(d_model → d_ffn, k=9) + Conv1D(d_ffn → d_model, k=1)
  → 每个位置的输出受前后 4 个位置影响（k=9 → 感受野 = 9）
```

对于 TTS 来说，相邻 Mel 帧之间有强烈的局部相关性——一个音素的起始、稳态、结束过渡需要平滑连续。Conv1D 提供了这种局部上下文建模，而标准 FFN 无法做到。

---

### 2.3 Variance Adaptor（方差适配器）

**定位**：FastSpeech 2 的核心创新。用三个显式预测器替代了 Tacotron 2 的隐式 Attention，给模型提供"该说多快、多高、多重"的明确信号。

```
输入: Encoder 输出 [batch, T_phone, d_model=256]
    │
    ├── (1) Duration Predictor
    │   └── 预测每个音素的持续帧数 d_i
    │
    ├── (2) Length Regulator
    │   └── 将音素序列 [T_phone] 展开为帧序列 [T_mel]
    │       h_frame_j = h_phone_i  对每个属于音素 i 的帧 j
    │
    ├── (3) Pitch Predictor + Pitch Embedding
    │   └── 预测每帧基频 → quantize → Embedding → + 到 h_frame
    │
    └── (4) Energy Predictor + Energy Embedding
        └── 预测每帧能量 → quantize → Embedding → + 到 h_frame
```

---

#### 2.3.1 Duration Predictor（时长预测器）& Length Regulator（长度调节器）

**定位**：取代 Attention 做对齐——显式告诉模型每个音素持续多少帧。

```
网络结构（三个 Predictor 共享）:
  输入: [batch, seq_len, d_model]
    │
    ├── Conv1D(d_model → d_model, k=3, ReLU)
    │   ├── LayerNorm
    │   └── Dropout(0.5)
    │
    ├── Conv1D(d_model → d_model, k=3, ReLU)
    │   ├── LayerNorm
    │   └── Dropout(0.5)
    │
    └── Linear(d_model → 1)
        └── 输出: 对数域时长 log(d_i)
```

**Duration 标签的获取——MFA 流程**：

```
音频 + 文本
    │
    ├── Montreal Forced Aligner (MFA)
    │   └── 基于 HMM 的自动语音对齐工具
    │
    ├── 输出 TextGrid 格式:
    │   "hello" → h(0.12s) + ə(0.08s) + l(0.15s) + oʊ(0.25s)
    │
    ├── 转换为帧数:
    │   d_h = 0.12s / 0.0125s = 9.6 ≈ 10 帧
    │
    └── Ground Truth Duration: d = [10, 6, 12, 20]
```

**Length Regulator——展开操作**：

```
输入: 音素隐藏序列 [batch, T_phone=4, d_model=256]
     + 预测时长 d = [10, 6, 12, 20]

操作: 对每个音素 i，将其隐藏向量 h_i 复制 d_i 份:
  h_phone[0]: [h_0]  →  复制 10 次 → [h_0, h_0, ..., h_0]  (10个)
  h_phone[1]: [h_1]  →  复制 6 次  → [h_1, ..., h_1]        (6个)
  h_phone[2]: [h_2]  →  复制 12 次 → [h_2, ..., h_2]        (12个)
  h_phone[3]: [h_3]  →  复制 20 次 → [h_3, ..., h_3]        (20个)

输出: [batch, T_mel=48, d_model=256]  ← 帧数与目标 Mel 一致
```

| 参数 | 值 | 含义 |
|------|-----|------|
| 时长来源 | MFA 自动对齐 | 不再依赖教师模型（FastSpeech 1 弱点） |
| 预测目标 | 对数域时长 | log(duration) 更符合高斯分布假设 |
| 损失函数 | MSE | 预测时长 vs MFA 时长 |

---

#### 2.3.2 Pitch Predictor（基频预测器）& Pitch Embedding

**定位**：给模型提供"每个音该发多高"的信号——语气语调的核心控制维度。

**Pitch 的提取与量化**：

```
提取:
  音频 → WORLD 声码器 → F0 (基频) 序列 [batch, T_mel]
    → log 变换: log(F0)
    → 归一化: (logF0 - μ) / σ

量化（256 bins）:
  ┌─────────────────────────────────────────┐
  │  归一化后的 pitch 值范围 ~ [-5, 5]       │
  │  → 均匀分为 256 个区间 (bin)            │
  │  → 每个帧分配到最近的 bin               │
  │  → bin index → Embedding(256 → 256)     │
  └──────────────────────────────────────────┘

CWT（连续小波变换）分解——论文的核心设计:
  F0 轮廓 → CWT → F0 频谱图 (多个时间尺度)
    ├── 粗尺度: 句子的整体语调走向（上升/下降）
    ├── 中尺度: 短语的重音模式
    └── 细尺度: 音节的基频微调（四声）
```

**为什么用 CWT 处理 F0？** 原始 F0 值在音素边界处有剧烈跳变（比如清音到浊音时 F0 从 0 跳升到 200Hz）。直接预测这些跳变值，MSE loss 会驱使模型输出"平滑的平均值"——结果是语调平淡。CWT 将 F0 分解为不同时间尺度的分量，每个分量变化更平滑、更容易预测。

**推理流程**：
```
预测 → iCWT (逆变换) → F0 轮廓 → 量化 → Embedding → + 到隐藏状态
```

---

#### 2.3.3 Energy Predictor（能量预测器）& Energy Embedding

**定位**：给模型提供"这帧该多大声"的信号——重音和节奏控制。

```
能量计算:
  每帧音频采样点的 L2 范数:
    Energy_j = sqrt( Σ_t s_t² )
    其中 s_t 是第 j 帧内的原始采样点

量化（同 Pitch，256 bins）:
  Energy → log 变换 → 归一化 → 256 bins → Embedding(256 → 256)
  → + 到隐藏状态
```

| 维度 | Pitch Predictor | Energy Predictor |
|------|----------------|-----------------|
| 输入特征 | F0 (基频) | 帧级 L2 能量 |
| 变换 | **CWT 分解**（多尺度） | 直接归一化（能量变化平缓，CWT 无增益） |
| 量化 bins | 256 | 256 |
| 嵌入维度 | 256 | 256 |
| 损失权重 | 0.1 | 0.1 |
| 预测器结构 | 2×Conv1D + Linear | 同左 |

---

### 2.4 Encoder & Decoder（编解码器对称结构）

**定位**：Encoder 提取音素的上下文化表征，Decoder 从展开后的帧序列生成 Mel 频谱。

```
Encoder:           Decoder:
  音素序列            帧序列 [T_mel]
    │                   │
  (Embedding)       (展开隐藏状态) ← Duration+Lengt
    │                   │
  FFT Block × 4      FFT Block × 4
    │                   │
  Linear + PosEnc    Linear → [T_mel, 80]
    │                   │
  音素隐藏状态         Mel 频谱
```

**Encoder 和 Decoder 结构完全相同**（都是 FFT Block 堆叠），但参数不共享。这种对称设计在实际中被大量复用：

| 复用场景 | 做了什么 |
|---------|---------|
| **FastSpeech 2s** | 替换 Decoder + 输出层（波形解码器） |
| **多说话人版本** | Encoder/Decoder 不变，加 Speaker Embedding |
| **多语言版本** | 替换 Embedding 层，Variance Adaptor 重训 |

---

### 2.5 损失函数

FastSpeech 2 的损失函数由多个子损失加权组合：

```
总损失 = 1.0 × L1(Mel_before) + 1.0 × L1(Mel_after)
       + 1.0 × MSE(duration_pred, duration_gt)
       + 0.1 × MSE(pitch_pred, pitch_gt)
       + 0.1 × MSE(energy_pred, energy_gt)
```

| 损失分量 | 权重 | 含义 |
|---------|------|------|
| Mel L1 (before) | 1.0 | Decoder 直接输出 vs 目标 Mel |
| Mel L1 (after) | 1.0 | Post-Net 精修后 vs 目标 Mel |
| Duration MSE | 1.0 | 音素时长预测精度 |
| Pitch MSE | 0.1 | 基频预测精度 |
| Energy MSE | 0.1 | 能量预测精度 |

**为什么 Pitch/Energy 权重要取 0.1？** 这两个维度的数值范围远小于 Mel（Mel 是 0~∞ 的连续值，Pitch 被归一化到 ±5 区间）。直接用 1.0 权重会主导梯度。

---

## 三、推理流程演练

以合成 "hello world" 为例：

### Stage 1: 文本→音素

```
"hello world"
    → G2P: "hh ax l ow w er l d"
    → 音素 ID 序列 [8]: [hh, ax, l, ow, w, er, l, d]
```

### Stage 2: Encoder 编码

```
ID 序列 [1, 8] → Embedding(40→256) → [8, 256]
    → + Positional Encoding
    → 4× FFT Block (Self-Attn + Conv1D)
    → 音素隐藏状态 [8, 256]
```

### Stage 3: Variance Adaptor

```
Duration Predictor: [8, 256] → [8, 1] → d = [12, 6, 18, 20, 10, 14, 12, 8]

Length Regulator: 按时长展开
  hh(12帧) + ax(6帧) + l(18帧) + ow(20帧) + w(10帧) + er(14帧) + l(12帧) + d(8帧)
  = 100帧

Pitch Predictor: 100帧 → 预测 F0 → 256-bin量化 + Embedding → + 到隐藏状态
Energy Predictor: 100帧 → 预测能量 → 256-bin量化 + Embedding → + 到隐藏状态

输出: [batch, 100, 256]
```

### Stage 4: Decoder + Mel 输出

```
[100, 256] → 4× FFT Block → Linear(256→80) → Mel [100, 80]
    → Post-Net (如使用)
    → 最终 Mel → HiFi-GAN → 波形
```

---

## 四、FastSpeech 2 vs Tacotron 2 全景对比

| 维度 | Tacotron 2 | FastSpeech 2 | 胜负 |
|------|-----------|-------------|------|
| **生成方式** | 逐帧自回归 | **全序列并行** | ✅ FS2 |
| **对齐方式** | Attention（隐式，不稳定） | **Duration Predictor（显式，可控）** | ✅ FS2 |
| **对齐来源** | Attention 自学 | **MFA 外部工具** | ⚠️ FS2 需预处理 |
| **推理速度** | ~1× 实时率 | **~30-270× 实时率** | ✅ FS2 |
| **MOS** | 4.53 | ~4.45 | ✅ T2（微弱） |
| **语速/语调控制** | ❌ 无法直接控制 | ✅ Pitch/Energy/Duration 全可控 | ✅ FS2 |
| **训练速度** | 1×（基准） | **~3×** | ✅ FS2 |
| **训练数据需求** | 文本+音频 | 文本+音频+**MFA 对齐** | ⚠️ T2 更简单 |
| **长文本稳定性** | ⚠️ Attention 偶发漂移 | **稳定（无 Attention 对齐）** | ✅ FS2 |
| **韵律自然度** | 好 | ✅ **更好（有明确的 Pitch/Energy 信号）** | ✅ FS2 |

---

## 五、性能分析

### 5.1 FastSpeech 2 参数量估算

```
Embedding: 40 × 256 = 10K

Encoder (4× FFT Block):
  每层: 
    Multi-Head Attn: 4 × (256 × 256) + 256 × 256 = ~393K
    Conv1D: 256 × 1024 × 9 + 1024 × 256 = ~2.6M
  总计: 4 × 3M ≈ 12M

Variance Adaptor:
  Duration Predictor: 2 × (256 × 256 × 3 + 256 × 256) + 256 ≈ ~1M
  Pitch Predictor: 同结构 ≈ ~1M
  Energy Predictor: 同结构 ≈ ~1M
  Pitch Embedding: 256 × 256 = 65K
  Energy Embedding: 256 × 256 = 65K
  总计: ~3.1M

Decoder (4× FFT Block): 同 Encoder ≈ 12M

Output: 256 × 80 = 20K

总计: ~27M 参数
```

### 5.2 推理瓶颈

```
推理时间分布 (1.5 秒音频):
  Encoder: ~15%
  Duration + Length Regulator: <1%
  Pitch/Energy Predictor: ~3%
  Decoder: ~15%
  HiFi-GAN 声码器: ~67%
  
  总延迟: 50-100ms （主要取决于声码器，而非 FastSpeech 本身）
```

因为非自回归生成不涉及逐帧迭代，FastSpeech 2 本身的推理极快（<30ms 在 GPU 上）。主要的计算瓶颈在 HiFi-GAN 声码器上。

---

## 六、架构设计的深层思考

### 6.1 显式对齐 vs 隐式 Attention——谁是正解？

```
显式对齐（FastSpeech）:
  Pros: 稳定、可控、不跳词、不重复
  Cons: 需要外部对齐工具（MFA），增加了预处理成本
  
隐式 Attention（Tacotron 2）:
  Pros: 端到端培训，无需外部工具
  Cons: 长文本不稳定、无法精确控制语速
```

从历史来看，**显式对齐在工业部署中更受青睐**（WeNet、FastSpeech、一些生产 TTS 系统都用显式对齐），因为稳定性对于产品来说优先级高于"端到端"的完美性。

### 6.2 FastSpeech 2 的影响

```
FastSpeech 2 (2020)
    │
    ├── 非自回归 TTS 的事实标准
    │   └── Glow-TTS (2020): Flow + 显式对齐
    │   └── VITS (2021): MAS 也是显式对齐（但可微分）
    │
    ├── Variance Adaptor 被广泛借鉴
    │   └── 几乎所有工业 TTS 系统都包含 Duration/Pitch 预测
    │
    └── 启发了"条件信息注入"范式
        └── 后续模型（CosyVoice、F5-TTS）用 Flow 替代 Explicit Predictor
```

---

## 七、实际部署效果

### 7.1 主观评测

| 系统 | MOS | 数据集 |
|------|-----|--------|
| FastSpeech 2 + HiFi-GAN | **4.45** | LJSpeech |
| Tacotron 2 + HiFi-GAN | 4.47 | LJSpeech |
| FastSpeech 1 + WaveGlow | 3.82 | LJSpeech |
| Ground Truth | 4.58 | — |

### 7.2 实际场景表现

| 场景 | 表现 |
|------|------|
| **长文本朗读** | ✅ 稳定无跳词 |
| **语速控制** | ✅ 直接修改 Duration 预测值（0.5× ~ 2.0×） |
| **语调控制** | ✅ 修改 Pitch 值（提高/降低基线，改变语调范围） |
| **多说话人** | ✅ 加 Speaker Embedding 即可 |
| **冷启动** | ✅ 受限于 MFA 覆盖的语言（MFA 不支持的语言需要先训 aligner） |

### 7.3 生产部署

- **微软 Azure TTS**：FastSpeech 2 架构被用于部分语音合成管线
- **PaddleSpeech**（百度）：提供 FastSpeech 2 的官方实现
- **ESPnet**：集成 FastSpeech 2 作为 TTS baselines
- **Coqui TTS**：提供 FastSpeech 2 训练脚本

---

## 八、总结：一张图看穿 FastSpeech 2

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FastSpeech 2 架构全景                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  音素: "hh ax l ow w er l d" (8 个音素)                            │
│       │                                                             │
│  ┌────┴───────────┐                                                │
│  │  Embedding     │  40 vocab → 256-D                               │
│  └────┬───────────┘                                                │
│       │                                                             │
│  ┌────┴────────────────────────────────┐                            │
│  │  FFT Block × 4 (Encoder)            │                            │
│  │  ├── Multi-Head Self-Attention (2H) │                            │
│  │  └── Conv1D(k=9) + Residual + LN    │                            │
│  └────┬────────────────────────────────┘                            │
│       │  音素隐藏状态 [8, 256]                                      │
│       │                                                             │
│  ┌────┴────────────────────────────────────┐                        │
│  │  Variance Adaptor                        │                        │
│  │                                           │                        │
│  │  ┌────────────────┐                      │                        │
│  │  │ Duration Pred  │ → d = [12,6,18,20,...]→ Length Regulator    │
│  │  └────────────────┘                      │                        │
│  │  ┌────────────────┐  ┌──────────────┐    │                        │
│  │  │ Pitch Predictor │→│ CWT + Quant  │→ + │                        │
│  │  └────────────────┘  └──────────────┘    │                        │
│  │  ┌──────────────────┐  ┌──────────────┐  │                        │
│  │  │ Energy Predictor │→│ Quant + Emb  │→ + │                        │
│  │  └──────────────────┘  └──────────────┘  │                        │
│  │                                           │                        │
│  │  输出: [100, 256] (帧隐藏状态)            │                        │
│  └────┬────────────────────────────────────┘                        │
│       │                                                             │
│  ┌────┴────────────────────────────────┐                            │
│  │  FFT Block × 4 (Decoder)            │                            │
│  │  (同 Encoder 结构)                   │                            │
│  └────┬────────────────────────────────┘                            │
│       │                                                             │
│  ┌────┴──────┐                                                     │
│  │  Linear   │  256 → 80-D Mel                                      │
│  └────┬──────┘                                                     │
│       │                                                             │
│  ┌────┴──────┐  ┌─────────────┐                                    │
│  │  Mel 谱   │→│ HiFi-GAN    │→ 波形                               │
│  └───────────┘  └─────────────┘                                    │
│                                                                     │
│  一句话总结 FastSpeech 2：                                          │
│  "MFA 显式对齐替代 Attention，Variance Adaptor 注入时长/音调/能量，  │
│   FFT Block 并行生成，TTS 从此跑在 GPU 利用率 90%+ 上。"            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 附录：关键配置速查

### FastSpeech 2 完整超参数表

```python
# 音素
vocab_size = 40             # 英语音素词表大小
phoneme_language = "en"     # 语言

# 模型
model_type = "FastSpeech2"
d_model = 256               # 隐藏层维度
d_ffn = 1024                # FFN 中间维度
num_heads = 2               # Attention 头数
encoder_layers = 4          # Encoder FFT Block 数
decoder_layers = 4          # Decoder FFT Block 数
conv_kernel_size = 9        # Conv 模块卷积核宽度

# Variance Adaptor
predictor_kernel = 3        # 3 个 Predictor 的卷积核宽度
predictor_layers = 2        # 每个 Predictor 的卷积层数
predictor_dropout = 0.5     # Predictor Dropout 率
pitch_quantization = "linear"  # 量化方式
energy_quantization = "linear"
n_bins = 256                # 量化 bin 数

# 训练
batch_size = 48
learning_rate = 0.001
warmup_steps = 4000
betas = [0.9, 0.98]
eps = 1e-9
grad_clip_thresh = 1.0

# 损失权重
mel_loss_weight = 1.0
duration_loss_weight = 1.0
pitch_loss_weight = 0.1
energy_loss_weight = 0.1
```

---

*本文基于 FastSpeech 2 论文 (Ren et al., ICLR 2021)、ming024/FastSpeech2 PyTorch 实现 (MIT License, GitHub)、PaddleSpeech 官方实现 (Apache 2.0) 及 Montreal Forced Aligner (MFA) 文档整理分析。*

**Sources:**
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech - Microsoft (ICLR 2021)](https://arxiv.org/abs/2006.04558)
- [ming024/FastSpeech2 - GitHub (MIT)](https://github.com/ming024/FastSpeech2)
- [PaddleSpeech FastSpeech2 - GitHub (Apache 2.0)](https://github.com/PaddlePaddle/PaddleSpeech)
- [Montreal Forced Aligner - MFA Docs](https://montreal-forced-aligner.readthedocs.io/)
- [FastSpeech: Fast, Robust and Controllable Text to Speech - NeurIPS 2019](https://arxiv.org/abs/1905.09263)
