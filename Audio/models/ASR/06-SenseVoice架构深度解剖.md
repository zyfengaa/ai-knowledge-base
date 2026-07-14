# SenseVoice 架构深度解剖

> 阿里巴巴 FunAudioLLM 出品 | 非自回归、多任务统一语音理解模型

---

## 写在前面：为何我们需要"听懂的"而非"听写的"模型

### 一个核心矛盾

在 2024 年之前，几乎所有 ASR（自动语音识别）模型只做一件事：**把语音转成文字**。情感、背景声、音频事件——这些信号被统统当作"噪声"滤掉。但人类在听一段语音时，接收到的远不止文字信息：

| 语音中的信息维度 | 传统 ASR 的处理 | 真实理解的意义 |
|-----------------|---------------|--------------|
| 说了什么词 | 重点保留 | 核心语义 |
| **说话人的情绪**（开心/悲伤/愤怒） | 丢弃 | 决定回复的语气 |
| **背景音**（掌声/音乐/笑声） | 丢弃 | 决定场景上下文 |
| **音频事件**（哭声/咳嗽/喷嚏） | 丢弃 | 涉及安全或紧急响应 |
| 语种 | 单独用 LID 模型 | 路由到不同的下游 |

传统方案需要串联多个独立模型（ASR + SER + AED + LID），带来**延迟累积、工程复杂、特征割裂**的问题。

### 另一个矛盾：自回归解码的瓶颈

自回归（Autoregressive）解码——每生成一个 token 都需要一次完整的前向计算——是实时 ASR 场景中最主要的延迟瓶颈。以 Whisper 为例：

```
自回归解码过程 (10 秒音频, ~30 个输出 token):
  Step 1: 编码器编码整段音频 (1× 前向)
  Step 2: 解码器生成 "今" (1× 前向)
  Step 3: 解码器生成 "天" (1× 前向)
  Step 4: 解码器生成 "天" (1× 前向)
  ...
  Step 30+: 逐 token 生成结束  ← 30+ 次串行前向

总解码延迟 ≈ 30 × 单步解码延迟 → 对于长音频呈线性增长
```

**非自回归（NAR）架构**的出现正是为了解决这个瓶颈——如果能一次性输出全部 token，延迟将从 O(L) 降为 O(1)。

**SenseVoice 的答案是：Encoder-only + CTC，用一次前向完成全部输出。**

---

## 一、整体架构：Encoder-only + 单 CTC 头

SenseVoice 的架构设计极其简洁——**没有解码器，没有交叉注意力，没有自回归模块**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SenseVoice 整体架构                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  16kHz 音频波形                                                      │
│       │                                                              │
│  ┌────┴────┐                                                         │
│  │ 前端特征  │  80-dim FBank + LFR(m=7, n=6) → 560-dim/帧              │
│  └────┬────┘                                                         │
│       │                                                              │
│  ┌────┴────────────────────────────────────────────────────────────┐ │
│  │  SANM Encoder (50 blocks)                                       │ │
│  │                                                                  │ │
│  │  ┌─────────┐  ┌─────────┐          ┌─────────┐                 │ │
│  │  │SANM B.1 │→│SANM B.2 │→···→│SANM B.50│                 │ │
│  │  └─────────┘  └─────────┘          └─────────┘                 │ │
│  └────┬────────────────────────────────────────────────────────────┘ │
│       │                                                              │
│  ┌────┴────┐                                                         │
│  │ 4 个查询  │  LID + SER + AED + ITN 可学习 embedding                  │
│  │ 可学习   │  (prepended to encoder output)                          │
│  │ Embedding│                                                         │
│  └────┬────┘                                                         │
│       │                                                              │
│  ┌────┴────┐                                                         │
│  │ CTC Head│  单线性层 → 输出概率分布                                  │
│  └────┬────┘                                                         │
│       │                                                              │
│  输出序列: <|lang|><|emo|><|event|><|itn|> transcription              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

             "一次前向，全部输出——没有自回归，没有解码器"
```

### 架构的关键设计决定

| 设计选择 | SenseVoice | 传统 Encoder-Decoder (如 Whisper) | 优势 |
|---------|-----------|---------------------------------|------|
| **解码方式** | CTC (非自回归) | 自回归 Transformer | 延迟从 O(L) 降为 O(1) |
| **解码器** | ❌ 无 | Transformer Decoder | 节省 30-50% 参数 |
| **输出策略** | 贪婪解码，无 beam search | Beam search + 长度惩罚 | 解码零开销 |
| **多任务实现** | 4 个 query embedding 拼接 | 独立模型或 prompt 控制 | 参数极小代价 |

### 为什么可以没有解码器？

CTC（Connectionist Temporal Classification）可以"单帧对齐"——每一帧独立输出一个字符（或空白），然后用去重和合并规则得到最终转录。这意味着：

1. **不需要解码器来建模文本依赖**——CTC 假设帧间条件独立
2. **不需要交叉注意力**——CTC 直接在编码器输出上做分类
3. **不需要因果掩码**——编码器可以双向看到全部上下文

当然，CTC 的"条件独立假设"是近似——它假设每帧的分类决策独立于其他帧。这在实践中通过足够深的编码器来补偿：编码器的双向自注意力已经建模了帧间依赖，CTC 只需做最终的"投票"。

---

## 二、SANM 编码器深度解剖

### 2.1 核心参数

| 参数 | SenseVoiceSmall 值 | 含义 |
|------|-------------------|------|
| `hidden_size` / `output_size` | **512** | 隐藏层维度 |
| `num_blocks` | **50** | SANM 编码器层数 |
| `num_attention_heads` | **4** | 自注意力头数 |
| `linear_units` (FFN 维度) | **2048** | FFN 中间层 (4× hidden) |
| `kernel_size` | **11** | FSMN depthwise 卷积核大小 |
| `dropout_rate` | 0.1 | Dropout 率 |
| `selfattention_layer_type` | **sanm** | 自注意力类型 |
| `normalize_before` | true | Pre-Norm 结构 |
| `pos_enc_class` | SinusoidalPositionEncoder | 正弦位置编码 |

> 注意：SenseVoice-Large（未开源）使用更大配置，层数可能达 70+ 层。

### 2.2 SANM Block 的内部结构

SANM (Self-Attention Network with Memory) 的核心思想与 Conformer 一脉相承：**自注意力负责全局上下文，卷积负责局部模式**。但 SANM 在具体实现上做了显著减法：

```
SANM Block 结构:
  输入 x [batch, seq_len, 512]
      │
      ├── LayerNorm
      │
      ├── Multi-Head Self-Attention (4 头, 相对位置编码)
      │   └── + 残差连接
      │
      ├── LayerNorm
      │
      ├── FSMN Depthwise Conv1d (kernel=11)
      │   └── 轻量局部建模：无 GLU，无 Pointwise Conv，无 BN
      │
      ├── + 残差连接
      │
      ├── LayerNorm
      │
      ├── FFN (Position-wise Feed-Forward)
      │   ├── Linear: 512 → 2048
      │   ├── Swish / ReLU 激活
      │   └── Linear: 2048 → 512
      │
      └── + 残差连接
```

#### 与 Conformer Block 的逐行对比

| 组件 | Conformer Block | SANM Block | 差异内涵 |
|------|----------------|-----------|---------|
| **FFN 分布** | 两个半份 FFN（½ + ½）包裹两端 | 一个全量 FFN 在尾部 | 去掉 Macaron 结构，简化 |
| **Self-Attention** | 相对位置编码 (RPE) | 相对位置编码或正弦编码 | 能力相当 |
| **卷积模块** | Conv Module 含 GLU + DWConv + BN + Swish | 仅 FSMN Depthwise Conv | 极度简化：去掉了 GLU 门控、额外激活和 BN |
| **卷积核大小** | 32（~320ms 感受野） | 11（~110ms 感受野） | 更小的局部感受野 |
| **LayerNorm 位置** | 普通 Pre-Norm | 每个子层前各一个 | 更精细的归一化 |
| **计算量** | 高 | 显著更低 | 适合实时场景 |

### 2.3 FSMN Depthwise Conv — SANM 的核心差异化设计

FSMN（Feedforward Sequential Memory Networks）最初由科大讯飞提出，用于解决 RNN 的串行计算问题，后来被 SenseVoice 团队改造为轻量级深度可分离卷积模块。

```
标准 Depthwise Conv1d:
  - 每个通道独立卷积核
  - 参数量 = channels × kernel_size
  
标准 Conv1d:
  - 所有通道共享一个卷积核的投影版本
  - 参数量 = in_channels × out_channels × kernel_size

FSMN Depthwise Conv (SenseVoice 实现):
  - Depthwise Conv1d kernel=11
  - 参数量 = 512 × 11 = 5,632
  - 无 BN、无 GLU、无 Pointwise Conv 前置
  - 直接接在 Self-Attention 输出上
```

**设计动机**：

1. **极致轻量**：SANM 的卷积部分只占 ~5.6K 参数（相比之下 Conformer 的 Conv Module 占 ~10M 参数），这就是为什么 SenseVoice 能用 50 层编码器但总参数量控制在 234M。
2. **局部先验**：kernel=11 覆盖约 11 帧 ≈ 110ms 音频，恰好覆盖一个音素（phone）的典型时长（50-150ms），这是一个精心选择的局部感受野。
3. **避免过拟合**：相对位置编码已在 Self-Attention 中提供了时序建模能力，卷积只需提供"相邻帧的平滑偏置"，不需要复杂的门控。

**一句话理解 SANM 的设计哲学**：

> Conformer 把卷积模块做得又大又重（GLU + DWConv + BN + Swish），SANM 则把卷积做到极致简单（只留一个 Depthwise Conv），剩余的建模能力全部由 Self-Attention 和深度的堆叠来提供。

### 2.4 关于 tp_blocks 假设

配置文件中有 `tp_blocks: 20` 参数。在 FunASR 的实现惯例中，`tp` 代表 **Time-Pooling**（时间池化）。每 N 层做一次时间维度的下采样。推测 50 个 SANM block 中，有 20 个 block 在其自注意力或卷积后附带 **2× 时间池化**，从而在深层逐步压缩时间分辨率：

```
TP 下采样过程（推测）:
  输入帧数 ≈ T
      │
  ┌───┤ 前几个 SANM block（无 TP，保持 T）
  │   │
  ├───┤ SANM block with TP → 帧数减半: T/2
  │   │
  ├───┤ 几个 SANM block（无 TP）
  │   │
  ├───┤ SANM block with TP → 帧数减半: T/4
  │   │
  └───┘ ...（继续）

最终帧数 ≈ T / 2^(tp_blocks 数量?)
```

如果 50 层中有 20 层做 2× TP，总下采样率可能为 2^3 ~ 2^4 ≈ 8-16×，这对 CTC 对齐有利——CTC 输出的帧率应与文本 token 率接近（1 秒音频 ≈ 3-5 个字符）。

---

## 三、输入特征提取流水线

### 3.1 完整处理流程

```
原始音频 (16kHz 单声道)
    │
    ├── 采样点: 160,000 / 秒
    │
    ├── 特征提取: 80 维 FBank (Filter Bank)
    │   ├── 窗口: 25ms Hamming 窗
    │   ├── 步长: 10ms
    │   └── 输出帧率: 100 帧/秒
    │
    ├── LFR (Low Frame Rate) 拼接
    │   ├── m=7: 每帧拼接前后共 7 帧
    │   ├── n=6: 每隔 n 帧取一个输出帧
    │   └── 输出: 每帧 80×7 = 560 维
    │   └── 输出帧率: 100/6 ≈ 16.7 帧/秒
    │
    ├── Sinusoidal 位置编码 + Dropout
    │
    └── 进入 SANM Encoder: [batch, T', 512]
        (T' = 原始帧数 / 6)
```

### 3.2 LFR 设计的意图

LFR（Low Frame Rate）由 Povey 等人在 Kaldi 时期提出，用于降低 CTC 模型的帧率：

| 参数 | 值 | 含义 |
|------|-----|------|
| `lfr_m` | 7 | 拼接窗口大小（当前帧 ±3 帧） |
| `lfr_n` | 6 | 下采样步长（每 6 帧取一个） |
| 输入维度 | 80 | FBank 维度 |
| 输出维度 | 80 × 7 = **560** | 拼接后的维度 |

没有 LFR 的情况下，10 秒音频产生 ~1000 帧。CTC 需要在每一帧输出一个标签，但"每一帧都是冗余的"——相邻帧的声学信息高度重叠。用 LFR 将帧率从 100 帧/秒降到 ~16.7 帧/秒：

- **计算量降低 6 倍**：编码器处理的序列长度变为 1/6
- **对齐更精准**：CTC 的目标帧数与文本 token 数更匹配
- **信息不损失**：每帧拼接了 7 帧的信息，上下文信息反而更丰富

### 3.3 前端实现细节

```yaml
# config.yaml 中的前端配置
frontend: WavFrontend
fs: 16000           # 采样率
n_mels: 80           # Mel 滤波器数量
frame_length: 25     # 帧长 (ms)
frame_shift: 10      # 帧移 (ms)
window: hamming      # 窗函数类型
lfr_m: 7
lfr_n: 6
```

> 注意：SenseVoice 使用 80 维 FBank（而非 Whisper 的 128 维 Mel），配合 LFR 扩展到 560 维后输入编码器。

---

## 四、多任务联合输出机制

### 4.1 四类 Query Embedding

这是 SenseVoice 最具创新性的设计之一：**在编码器输出序列的首部 prepend 可学习的 query embedding**，每个 embedding 对应一个推理任务。这些 embedding 随模型一同训练，在推理时作为"任务指示器"：

```
编码器输出序列:
  ┌──────────┬──────────┬──────────┬──────────┬──────────────────────────────┐
  │ LID      │ SER      │ AED      │ ITN      │ CTC 帧序列                    │
  │ query    │ query    │ query    │ query    │ (逐帧输出字符概率)              │
  │ [1, 512] │ [1, 512] │ [1, 512] │ [1, 512] │ [T', 512]                    │
  └──────────┴──────────┴──────────┴──────────┴──────────────────────────────┘
         ↓           ↓           ↓           ↓                ↓
         └──────────────┬───────────────────────────────────────┘
                       ↓
               CTC Head (Linear: 512 → vocab_size)
                       ↓
        输出序列: <|lang|><|emo|><|event|><|itn|> transcription...
```

#### 为什么是 4 个 embedding 而非 4 个独立分类头？

这是 **"统一表征学习"** 的核心理念——所有任务共享同一个编码器输出，4 个 query embedding 仅仅是在序列首部加了不同的偏置，引导 CTC 头在隐空间中产生针对性的输出分布。

- **参数增量几乎为零**：4 × 512 = 2,048 个额外参数，与 234M 总参数相比可忽略不计
- **无需单独训练分类头**：一切都在 CTC 损失下端到端优化
- **推理零成本**：4 个 embedding 在前向计算中和普通 token 一模一样

### 4.2 输出格式详解

CTC 输出是一个拼接序列：

```
输出格式（以中文普通话、中性情绪、说话场景、需要 ITN 为例）:

  <|zh|><|NEUTRAL|><|Speech|><|withitn|>今天天气真的很冷

各段含义:
  <|zh|>         ← LID: 识别为中文
  <|NEUTRAL|>    ← SER: 中性情绪
  <|Speech|>     ← AED: 说话/语音场景
  <|withitn|>    ← ITN: 需要文本规整（数字、日期等格式化）
  今天天气真的很冷  ← ASR 转录文本
```

**为什么不需要 `<sos>` 和 `<eos>`？** CTC 模型天然不需要序列起始/终止标记——CTC 的输出长度由输入帧数决定，空白帧（blank token）表示"无输出"。只需从输出帧中解码出非空白 token 序列即可。

### 4.3 LID（语种识别）

| 支持语种 | 标签 |
|---------|------|
| 中文普通话 | `<|zh|>` |
| 英语 | `<|en|>` |
| 粤语 | `<|yue|>` |
| 日语 | `<|ja|>` |
| 韩语 | `<|ko|>` |
| 无语音 | `<|nospeech|>` |

> 模型训练覆盖 50+ 语种，在 auto 模式下自动识别语种。也可以手动指定语种标签跳过自动 LID。

### 4.4 SER（语音情感识别）

| 情感类别 | 标签 | 典型语调特征 |
|---------|------|------------|
| 高兴 | `<|HAPPY|>` | 高基频、快语速、大调域变化 |
| 悲伤 | `<|SAD|>` | 低基频、慢语速、弱能量 |
| 愤怒 | `<|ANGRY|>` | 高能量、快语速、紧喉音质 |
| 中性 | `<|NEUTRAL|>` | 基频平稳、无显著情绪色彩 |
| 恐惧 | `<|FEARFUL|>` | 高频颤抖、呼吸声 |
| 厌恶 | `<|DISGUSTED|>` | 特殊音质、停顿增多 |
| 惊讶 | `<|SURPRISED|>` | 瞬时基频跳升、音强增大 |

> 官方声称 SenseVoice 在 SER 任务上"无需在目标数据集上进行微调即可达到或超过当时最佳的专用情感识别模型"。

### 4.5 AED（音频事件检测）

| 音频事件 | 标签 | 应用场景 |
|---------|------|---------|
| 说话 | `<|Speech|>` | 区分纯语音 vs 非语音 |
| 音乐 | `<|BGM|>` | 背景音乐 |
| 掌声 | `<|Applause|>` | 会议/演讲场景标记 |
| 笑声 | `<|Laughter|>` | 对话情感色彩标记 |
| 哭声 | `<|Cry|>` | 紧急/情绪感知 |
| 喷嚏 | `<|Sneeze|>` | 健康监测 |
| 呼吸 | `<|Breath|>` | 通话质量分析 |
| 咳嗽 | `<|Cough|>` | 健康监测 |

> 注意：SenseVoice 仅在语音数据上训练，AED 作为辅助任务。对于纯事件分类，精确度不如专用 AED 模型（如 BEATS、PANN），但实际场景中足够使用。

### 4.6 ITN（反向文本规整）

| 标签 | 含义 |
|------|------|
| `<|withitn|>` | 输出文本应用 ITN（如 "二零二四" → "2024"） |
| `<|woitn|>` | 输出保持原始口语形式（如 "二零二四"） |

---
## 五、训练配置与损失函数

### 5.1 关键训练配置

```yaml
# 损失函数
criterion: CTC
length_normalized_loss: true    # 按序列长度归一化 CTC 损失

# 优化器
optimizer: adamw
lr: 2e-5                        # 学习率

# 学习率调度
scheduler: warmuplr
warmup_steps: 25000             # 25K 步预热

# 训练轮次
max_epoch: 20
avg_nbest_model: 10             # 平均最后 10 个检查点作为最终模型

# 梯度裁剪
grad_clip: 5

# SpecAugment
specaug: SpecAugLFR
freq_mask_width_range: [0, 30]  # 频率掩码最大宽度 30
num_freq_mask: 1                # 1 个频率掩码
time_mask_width_range: [0, 12]  # 时间掩码最大宽度 12
num_time_mask: 1                # 1 个时间掩码
lfr_rate: 6                     # LFR 步长
```

### 5.2 关于 CTC Loss 的几点说明

**为什么选择 CTC？**

CTC（Connectionist Temporal Classification）的损失函数定义为所有可能对齐路径的负对数似然之和：

```
CTC Loss = -ln( Σ_{π∈B⁻¹(y)} P(π | x) )

其中:
  x = 编码器输出的帧序列
  y = 目标转录文本
  π = 一条对齐路径（包含 blank token）
  B⁻¹(y) = 所有能通过去重合并得到 y 的路径集合
```

CTC 的核心优势在于它**自动处理对齐**——不需要事先知道每一帧对应哪个音素/字符，模型在训练中自己学会对齐。

**`length_normalized_loss = true` 的含义**：

默认的 CTC 损失会对长句子产生更大的损失值（因为更多帧 × 更多对齐路径）。开启长度归一化后，损失按序列长度平均，防止模型偏向短句子：

```
常规 CTC Loss:     L = -ln(P(y|x))
归一化 CTC Loss:   L = -ln(P(y|x)) / len(y)    ← 按输出长度平均
```

**为什么 CTC 适合 SenseVoice 的多任务输出？**

CTC 输出是一个"序列序列"，而 SenseVoice 的输出正是拼接序列：4 个标签 + 转录文本。CTC 的"帧级独立分类"机制天然适配这种结构——每个标签和每个字符都可以由不同帧独立预测出来。

### 5.3 数据相关配置

```yaml
# 数据集配置
dataset: SenseVoiceCTCDataset
batch_size: 14000         # token 级别 batch size
max_token_length: 2000
min_token_length: 60
max_source_length: 2000
min_source_length: 60
max_target_length: 200
min_target_length: 0
shuffle: true
num_workers: 4

# 训练数据规模
# 训练数据超过 400,000 小时，覆盖 50+ 语种
```

### 5.4 推理超参数

```python
# SenseVoice 推理配置
use_ctc = True              # 启用 CTC 解码
beam_size = 1                # greedy 解码（beam_size=1）
vocabulary = [...]           # 词汇表（含 4 类标签 token）
language = "auto"            # 自动语种识别
# 无需: num_beams, max_new_tokens, temperature, repetition_penalty
# 因为这些是自回归解码的参数
```

**SenseVoice 的推理极其简洁**：

1. 一次编码器前向 → 得到帧级隐状态
2. CTC 头线性投影 → 得到帧级概率分布
3. Greedy argmax 解码 → 取每帧概率最大的 token
4. 去重合并（CTC collapse）→ 移除重复 token 和 blank
5. 后处理分离 → 识别出 LID/SER/AED/ITN 标签和转录文本

**整个过程没有循环、没有分支、没有束搜索。**

---

## 六、延迟对比：非自回归 vs 自回归

### 6.1 正式基准对比

| 模型 | 架构 | 参数量 | **10 秒音频推理延迟** | 加速比 (vs SenseVoice) |
|------|------|--------|---------------------|----------------------|
| **SenseVoice-Small** | Encoder-only + CTC | **234M** | **~70ms** | 1× (基线) |
| Whisper-Small | Encoder-Decoder | 244M | 518ms | **7.4× 慢** |
| Whisper-Large-V3 | Encoder-Decoder | 1.55B | 1.28s | **18.3× 慢** |

### 6.2 延迟来源分解

```
10 秒音频推理延迟分解:

SenseVoice-Small (70ms 总延迟):
  ├── 特征提取 + LFR:     ~5ms    (7%)
  ├── SANM 编码器 (50 层): ~63ms   (90%)
  ├── CTC 头投影:         ~1ms    (1.5%)
  └── Greedy 解码:        ~1ms    (1.5%)

Whisper-Small (518ms 总延迟):
  ├── 特征提取:           ~5ms    (1%)
  ├── 编码器 (12 层):     ~20ms   (4%)
  ├── Prefill (KV 初始化): ~30ms  (6%)
  └── 自回归解码 (30+ 步): ~463ms (89%)  ← 核心瓶颈
```

两个关键差异：

1. **单次前向 vs 多次前向**：SenseVoice 一次前向搞定解码；Whisper 需要 30+ 次串行解码步骤
2. **参数规模化影响**：SenseVoice 增加编码器深度不影响解码延迟（仍然是一次前向）；Whisper 增加解码器深度会使自回归阶段线性变慢

### 6.3 为什么 SenseVoice-Small 和 Whisper-Small 参数量接近，延迟差 7 倍？

| 因素 | SenseVoice-Small (234M) | Whisper-Small (244M) |
|------|------------------------|---------------------|
| 参数分布 | 全部在编码器 | ~70% 编码器 + ~30% 解码器 |
| 解码方式 | 非自回归（CTC, 1 步） | 自回归（30+ 步） |
| 解码器参数 | 0 | ~73M（12 层解码器 Transformer） |
| 解码计算量 | O(1) 相对于输出长度 | O(L) 相对于输出长度 |

**本质差异**：SenseVoice 把"解码"的计算压力前置到了编码器（50 层 + 深度可分离卷积），而 Whisper 把计算分散在编码器和解码器之间。当输入音频变长时：

- SenseVoice 的延迟随**输入长度**线性增长（编码器 O(N)）
- Whisper 的延迟随**输出长度**线性增长（解码器 O(L) × N），且输出长度 ≈ 输入长度 × ~0.03（每帧 16.7 帧/秒，每 3 帧输出一个字符）

对于**长音频**场景（如会议记录），SenseVoice 的 O(N) 方式显著优于 Whisper 的 O(L) × O(N) 方式。

---

## 七、Conformer vs SANM：同为卷积增强，设计哲学不同

### 7.1 关键差异总结

| 维度 | Conformer | SANM (SenseVoice) | 内涵 |
|------|----------|-------------------|------|
| **卷积组件复杂度** | 高（PWConv → GLU → DWConv → BN → Swish → PWConv） | 极低（仅 DWConv） | SANM 把卷积简化为"位置偏置" |
| **FFN 分布** | 两个 ½ FFN 包裹（Macaron） | 一个全量 FFN 在尾部 | Macaron 被证明不是必须的 |
| **卷积核大小** | 32（~320ms） | 11（~110ms） | Conformer 需要更大感受野补偿弱注意力？ |
| **位置编码** | 相对位置编码 (RPE) | 正弦编码 + SANM 自注意力中的隐式相对位置 | SANM 结合了绝对和相对方法 |
| **归一化位置** | 每个子模块内 LayerNorm/BN | 每个子模块前单独 LayerNorm | 更精细的归一化 |
| **残差结构** | 标准 Pre-Norm | 每个子模块独立残差 | 50 层深度的稳定性保障 |
| **GLU 门控** | 是 | 否 | 卷积模块中的门控被证明可省略 |
| **BN (BatchNorm)** | 是（卷积内部） | 否 | 减少对 batch size 的依赖 |

### 7.2 设计哲学对比

```
Conformer 设计哲学:
  "卷积模块是编码器中最重要的部分，应该做得充分"
  
  表现:
  - GLU 门控 → 控制信息流
  - 大 kernel (32) → 大感受野
  - BatchNorm → 卷积专用归一化
  - Macaron 结构 → 强化 FFN 的两端包裹

SANM 设计哲学:
  "自注意力才是主力，卷积只补充局部偏置"
  
  表现:
  - 无 GLU → 卷积只做线性变换
  - 小 kernel (11) → 精确覆盖音素级感受野
  - 无 BN → 减少归一化层的复杂度
  - 一个 FFN → 去掉 Macaron 结构
```

### 7.3 为什么 SANM 如此"轻量"却能工作？

关键发现：**当编码器深度足够大时（50 层），卷积模块可以极为轻量**。

Conformer 时期的假设是"每个 block 都要能独立完成局部+全局建模"。SANM 的假设是"50 层堆叠时，整网的表达能力来自深度，单个 block 只需完成自己的小任务"。

打个比方：
- **Conformer**：每个士兵装备精良（全副武装），但只有 12-17 个士兵
- **SANM**：每个士兵只带一把匕首（极度轻装），但有 50 个士兵

深度的堆叠补偿了单个 block 能力的不足。CTC 头也不需要像自回归解码器那样精细的中间表示——它只需要在帧级做独立分类。

---

## 八、技术演进脉络：SenseVoice 在 ASR 发展史中的位置

### 8.1 从 CTC 到 Transformer 再到融合

```
CTC (Graves, 2006)                      ← 非自回归的鼻祖
  │
  ├── DeepSpeech (Baidu, 2014)          ← RNN + CTC 的大规模实践
  │
  ├── Speech-Transformer (Google, 2018)  ← 自注意力引入 ASR，但自回归
  │
  ├── Conformer (Google, 2020)          ← 卷积+注意力融合，事实标准
  │   │
  │   ├── Whisper (OpenAI, 2022)        ← Enc-Dec + 大规模弱监督
  │   │
  │   ├── SenseVoice (Alibaba, 2024)    ── Encoder-only + CTC ← 回归非自回归
  │   │                                   │   但用 SANM 代替纯 Transformer/RNN
  │   │                                   │   + 多任务 query embedding 创新
  │   │
  │   └── GLM-ASR (Zhipu AI, 2025)     ← Encoder (重) + Decoder (轻) + 投影器
  │
  └── Decoder-Only 路线
      ├── Qwen2-Audio (Alibaba, 2024)  ← 音频 token 化 + LLM
      └── Qwen3-ASR AuT (Alibaba, 2026)← Audio Transformer + LLM
```

### 8.2 SenseVoice 的历史贡献

| 贡献 | 意义 |
|------|------|
| **非自回归架构的重大回归** | 证明在足够深的编码器下，CTC 可以达到与自回归模型竞争力相当的精度，但延迟低一个数量级 |
| **多任务统一输出格式** | 4 个 query embedding 的设计极简优雅，为后续多任务语音模型提供了范式参考 |
| **轻量卷积模块的有效性验证** | 证明在深度编码器 + 强自注意力的组合下，卷积模块可以被大幅简化 |
| **超大训练规模验证** | 400,000 小时训练数据的规模验证了非自回归模型在大数据下的可扩展性 |

---

## 九、一张图总结 SenseVoice

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SenseVoice 架构全景                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  16kHz 音频 → 80-dim FBank → LFR(m=7, n=6) → 560-dim                        │
│       │                                                                       │
│  ┌────┴──────────────────────────────────────────────────────────────────┐   │
│  │                   SANM Encoder (50 blocks)                             │   │
│  │                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────┐      │   │
│  │  │  SANM Block (×50)                                           │      │   │
│  │  │  ┌─────────────────────────────────────┐                    │      │   │
│  │  │  │ LayerNorm → Self-Attn(4h, 512)      │  ← 全局上下文      │      │   │
│  │  │  │ LayerNorm → FSMN DWConv(k=11)       │  ← 局部音素级      │      │   │
│  │  │  │ LayerNorm → FFN(512→2048→512)       │  ← 非线性变换      │      │   │
│  │  │  └─────────────────────────────────────┘                    │      │   │
│  │  │  每个 block 约 3.15M 参数，50 层共约 157.5M                 │      │   │
│  │  └─────────────────────────────────────────────────────────────┘      │   │
│  └────┬──────────────────────────────────────────────────────────────────┘   │
│       │                                                                       │
│  ┌────┴──────────────────────────────────────────────────────────────────┐   │
│  │  4 个 trainable query embeddings + CTC Head (Linear 512→vocab)        │   │
│  │  ┌──────────┬──────────┬──────────┬──────────┐                       │   │
│  │  │  LID Q   │  SER Q   │  AED Q   │  ITN Q   │  ← 统一表征学习       │   │
│  │  └──────────┴──────────┴──────────┴──────────┘                       │   │
│  └────┬──────────────────────────────────────────────────────────────────┘   │
│       │                                                                       │
│  ┌────┴──────────────────────────────────────────────────────────────────┐   │
│  │  CTC Greedy Output: <|zh|><|NEUTRAL|><|Speech|><|withitn|>今天...     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│  一句话总结 SenseVoice：                                                       │
│  "用 50 层 SANM 编码器一次前向完成全部分析，                                  │
│   用 4 个 query embedding 统一管理多任务输出，                                 │
│   用 CTC 非自回归解码实现 10 秒音频只需 70ms。"                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## ==Sources==

- [SenseVoice GitHub - FunAudioLLM](https://github.com/FunAudioLLM/SenseVoice)
- [SenseVoiceSmall HuggingFace config.yaml](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
- [FunAudioLLM technical overview (Baidu Developer)](https://developer.baidu.com/article/detail.html?id=6908664)
- [Conformer: Convolution-augmented Transformer for Speech Recognition (arXiv:2005.08100)](https://arxiv.org/abs/2005.08100)
- [FSMN: Feedforward Sequential Memory Networks (arXiv:2203.04743)](https://arxiv.org/abs/2203.04743)
- [CTC: Connectionist Temporal Classification (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [FunASR: A Fundamental End-to-End Speech Recognition Toolkit](https://github.com/modelscope/FunASR)
