﻿# 04 — 编码器进化

## 从 GRU 到 Conformer，ASR 编码器经历了什么？

回到你熟悉的 CNN+GRU 时代。GRU/LSTM 做序列建模有一个根本性的问题：它是串行的。处理一段 30 秒的语音（960 帧 @ 25ms 帧移），GRU 需要老老实实跑 960 步——每步依赖上一步的状态。这就意味着你不能用 GPU 的并行能力来加速计算，而且反向传播梯度经过 960 步传到前面，要么爆炸要么消失。

2017 年 Transformer 在机器翻译上横空出世，ASR 研究者们很快想到一件事：Self-Attention 是高度并行的——输入 T 帧，一次性算出 T×T 的 Attention 矩阵，而且任意两帧之间的距离都是 1，不存在长程依赖问题。

### Speech-Transformer：第一步尝试

Dong 等人在 2018 年直接把 Transformer Encoder 搬来做了 ASR 的声学编码器。论文里做的事情其实很简单：把 Transformer 的 Positional Encoding 从 NLP 改成语音型（因为语音帧之间不像 NLP 词之间有明显的边界），然后接 CTC 训练。结果证明并行编码确实能工作，这是 Transformer 在 ASR 的"投石问路"。

### Conformer：CNN + Attention 的融合

纯 Self-Attention 有一个被忽视的问题：它没有"局部归纳偏置"。CNN 天生知道相邻的像素/频率是相关的——3×3 卷积核的视野就是一个局部窗口。但 Attention 是对序列里所有位置一视同仁的，你需要在 Positional Encoding 上做很多文章才能让它"理解"局部关系。对于语音信号，这一点特别重要——语音的底层特征（共振峰、能量跳变、过零率）都是高度局部的，你丢了局部性就丢了大量信息。

Conformer 的解决方案是：**不选了，两个都要。**

它的结构是这样的——一个 Macaron 风格的 Sandwich：FF → (Conv Module + Self-Attention) → FF。中间的 Conv Module 做了一个精巧的复合：Pointwise Conv → GLU 激活 → Depthwise Conv → BatchNorm。Depthwise Conv 负责在时间和频率维度上做局部特征提取（类似 CNN 的角色），Self-Attention 负责建模长程依赖。两个并行，最后串接，一次性拿到"局部细节 + 全局上下文"。

这个设计的巧妙之处在于：它不是简单地把"CNN 和 Attention 两个模块"串在一起，而是在特征维度上让他们做了互补——卷积核感受野有限但平移不变，Attention 覆盖全序列但对位置信息敏感有限。这两者互为补充，不是谁替代谁。

## 学习目标

读完你能：

- 用一句话说清楚为什么 Transformer 比 RNN 更适合做声学编码
- 对比 Conv Module 和 Self-Attention 在"做同一件事"时的不同方式——一个抓局部，一个抓全局
- 理解 Conformer 的 Macaron 结构长什么样——不需要默写每一层的维度，但能画出大致块图
- 知道 Speech-Transformer 和 Conformer 的关系是"引路人和解决方案"的关系

## 编码器设计的三个关键维度

### 1. 降采样（Subsampling）—— 为什么不做 O(T²) 的计算？

16kHz 语音按 25ms 帧移，30 秒对话就有 960 帧。Transformer 的 Self-Attention 是 O(T²) 的——T=960 还能跑，但 T=1920 直接翻四倍。所以实际系统都在进入 Transformer 层之前对序列做降采样。

**Speech-Transformer 的做法**：2 层 Conv2d（每层 stride=2），960 帧 → 240 帧，Attention 计算量降到 1/16。Conformer 延续了同样的设计：先用一个 2 层 CNN 以 stride=2 做 subsampling，再接入 Conformer blocks。目前主流实现的输出帧长约为输入帧长的 1/4。

注意降采样率不是越大越好——降太多会让帧对齐精度下降。1/4 是主流折中，流式场景下部分实作会用 1/2。

### 2. 位置编码 —— 为什么语音更需要"相对位置"？

Transformer 的 Self-Attention 本身是集合运算——打乱输入顺序 Attention 矩阵不变，所以必须靠位置编码注入顺序信息。

语音里真正重要的是**帧与帧之间的相对距离**，不是"这是第 47 帧"这个绝对位置：

- 快速语速下第 47-50 帧对应一个音素 /e/，慢速下第 47-58 帧对应同一个 /e/
- 如果用绝对位置编码，同一个音素在不同语速下的位置表示完全不同
- 重复帧（声门闭合、长元音）在绝对编码下看起来"位置很远"，但在语音上它们是连续的

Conformer 用的是 **Relative Positional Encoding（来自 Transformer-XL）**，Attention 计算的是"这两帧相隔多远"而不是"第一帧的绝对位置是多少"。语速变化时相对位置保持稳定，这正是语音建模需要的。

### 3. 因果（Causal）vs 非因果 —— 离线与流式的设计分岔

Self-Attention 默认是双向的——第 t 帧可以看到第 t+1 帧（"未来"）。离线场景没问题，但流式 ASR 要求每帧只能看过去和有限的未来。

两种常见方案：

- **Causal Self-Attention + Causal Depthwise Conv**：每帧只能 attend 到当前及之前的帧。Conformer 的 Conv Module 通过调整 padding 方式即可改成 causal 版本
- **Chunk-wise Attention**：把语音切成小块（chunk_size ≈ 256ms），每帧 attend 到当前 chunk + 左侧历史帧（context_size）。既提供了左上下文参考，又控制了对未来的依赖延迟。WeNet 和 RNNT 系统常用这种方案

在实际工程中，"能不能流式"往往不是编码器的问题，而是训练范式（CTC/AED/RNNT）的问题——编码器本身可以训练一个因果版本同时支持两种模式。

## 精选论文

**Dong et al. (2018) "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" [[arXiv](https://arxiv.org/abs/1804.06993)]**

这篇的工作量不大，但它问了一个很重要的问题：Transformer 到底能不能用在 ASR 上？答案是能。它就是那个"投石问路"的工作。如果你对 Transformer 已经很熟悉了，这篇可以快读——重点是它的实验设置和结果分析。

**Gulati et al. (2020) "Conformer: Convolution-augmented Transformer for Speech Recognition" [[arXiv](https://arxiv.org/abs/2005.08100)]**

Conformer 是 2020 年至今 ASR 编码器的事实标准——LibriSpeech 上的 WER 刷新、ESPnet 里的默认配置、WeNet / WenetSpeech 等框架都在用它。你在工作中应该已经接触到了，这一篇是帮你理解"为什么 Conformer 能成为标准"的。

重点读它的 Attention + Conv Module 的融合设计。如果你能理解"为什么把卷积嵌入 Transformer 里这么管用"，那这篇就算读透了。

## 拓展阅读

- **Peng et al. (2022) "Branchformer: Parallel MLP-CNN-Attention Hybrid Architecture" [[arXiv](https://arxiv.org/abs/2207.07682)]** — Conformer 的改进版本，把串行的 Macaron 结构改成双分支并行的"全局分支 + 局部分支"。效果有一点提升，但思路比 Conformer 更清晰——如果你想了解"Conformer 之后还有什么"，可以翻翻。

> 注意：Squeezeformer、Zipformer、Emformer 这些主要是效率优化，不在"突破性 idea"的范畴内。
---

## 本章思考题（附解答）

### 基础层

#### Q1：用一句话说清楚为什么 Transformer 比 RNN 更适合 ASR 声学编码。

**A：**

> Transformer 的 Self-Attention 是**高度并行的**——处理 960 帧语音时不需要像 RNN 那样串行跑 960 步，而且任意两帧之间的 Attention 距离恒为 1，不存在 RNN 的梯度消失/爆炸问题。

展开几个关键点：

| 维度 | RNN (GRU/LSTM) | Transformer |
|---|---|---|
| 计算方式 | 串行，每步依赖上一步状态 | 并行，一次性计算 T×T Attention |
| 长程依赖 | 难以捕捉（梯度衰减） | 任意两帧距离为 1 |
| 训练速度 | 慢（不能利用 GPU 并行） | 快（可并行） |
| 帧数增加时 | 线性增加计算时间 | O(T²) 计算量（需降采样缓解） |

注意 Transformer 的 O(T²) 复杂度在帧数太大时也是问题——所以需要降采样。

---

#### Q2：Conformer 用"不选了，两个都要"的方式解决什么问题？

**A：**

解决 **Self-Attention 缺少局部归纳偏置**的问题：

- **Self-Attention**：对序列所有位置一视同仁，可以建模长程依赖，但对局部细节（共振峰、能量跳变）不敏感
- **CNN（Depthwise Conv）**：天生有局部感受野，对相邻帧的变化敏感，但覆盖范围有限

Conformer 让两者**在特征维度上互补**——Conv Module 提取局部细节，Self-Attention 捕捉全局上下文，两者输出拼接后一起传递到下一层。

> 一句话：**CNN 给了 Attention 一双"近视眼"来关注细节，Attention 给了 CNN 一双"广角镜"来把握全局。**

---

#### Q3：画出 Conformer 的 Macaron 结构图，并标注每个组件的作用。

**A：**

```
输入
  │
  ▼
[Feed Forward 模块 ①]  ← Macaron 前半
  │
  ▼
[Conv Module] ──── 并行 ──── [Self-Attention Module]
  │  局部特征（Depthwise Conv）    │  长程依赖（Relative PE）
  └────────── 拼接输出 ────────────┘
  │
  ▼
[Feed Forward 模块 ②]  ← Macaron 后半
  │
  ▼
[LayerNorm + 残差连接]  ← Pre-Norm 设计
  │
  ▼
  输出
```

关键点：
- **两个 FF 组成"Macaron 三明治"**——只是命名习惯，功能就是标准 FFN（线性层 + Swish 激活 + Dropout）
- **Conv Module**：Pointwise Conv → GLU → Depthwise Conv → BatchNorm → Swish → Pointwise Conv
- **Self-Attention**：Multi-Head Self-Attention + Relative Positional Encoding
- 每个模块外围都有残差连接和 Pre-Norm

---

#### Q4：Conformer 的 Conv Module 包含哪些子层？为什么这样设计？

**A：**

完整子层顺序：

```
Pointwise Conv (1×1) → GLU 激活 → Depthwise Conv → BatchNorm → Swish → Pointwise Conv (1×1)
```

每层的作用：

| 子层 | 作用 |
|---|---|
| **Pointwise Conv (1×1)** | 升维/降维，沿通道维度做线性变换 |
| **GLU (Gated Linear Unit)** | 门控机制——一半通道做门控信号，另一半做特征，可以理解为选择"哪些局部信息值得保留" |
| **Depthwise Conv** | 核心步骤——在时间和频率维度上做逐通道的局部卷积，提取局部声学特征 |
| **BatchNorm** | 稳定训练（Conv Module 里用 BN 不是 LN，因为卷积的通道统计有规律） |
| **Swish** | 激活函数，f(x) = x · sigmoid(x)，比 ReLU 更平滑 |
| **Pointwise Conv (1×1)** | 恢复通道数，匹配输出维度 |

> Depthwise Conv 是 Conv Module 里真正做"局部特征提取"的组件——它的计算量远小于普通卷积，但能在时间和频率两个维度上捕捉局部模式。

---

#### Q5：如果要让 Conformer 支持流式 ASR，需要做哪些改动？

**A：**

主要有三处需要改动：

**① Self-Attention → Causal/Chunk Attention**
- 默认的 Self-Attention 是双向的（第 t 帧能看到第 t+1 帧）
- 流式场景改为 **Chunk-wise Self-Attention**：每帧只能 attend 到当前 chunk + 左侧 context 区域
- 或者直接用 **Causal Self-Attention**（Mask 掉未来帧）

**② Depthwise Conv → Causal Depthwise Conv**
- 标准卷积的 padding 会让"未来帧"信息泄露到当前位置
- 改为只在左侧 padding（或者使用 CausalConv 的实现），保证帧 t 只看 t 及之前的帧

**③ 配合支持流式的训练范式**
- 编码器改了因果还不够——训练范式也必须是流式兼容的（CTC 或 RNNT）
- AED 即使 Causal Encoder 也无法流式（Decoder 需要整句编码结果）
- 可以训练一个**共享权重的非因果版 + 因果版**，做 dual-mode 推理

常见工程配置（WeNet 风格）：
```
因果 Conformer:
  - Chunk_size: 16 frames (160ms)
  - Left_context: 32 frames (320ms)
  - Right_context: 0 frames (strictly causal)
  - 训练时随机选 chunk_size，推理时固定
```

---

#### Q6：为什么语音编码器中相对位置编码比绝对位置编码更有效？

**A：**

核心原因：**语音的语义信息不依赖于帧的绝对位置。**

| 场景 | 绝对位置编码的问题 | 相对位置编码的优势 |
|---|---|---|
| **语速变化** | "cat" 慢读 40 帧 VS 快读 15 帧，/æ/ 的绝对位置完全不一样 | 相对位置只看"和前一帧隔多近"，语速变化时稳定 |
| **元音拉伸** | 长元音 /a:/ 的重复帧在绝对位置上不断递增，模型被强制"记住距离远" | 相对距离保持稳定，同一元音的相邻帧关系一致 |
| **噪音帧干扰** | 中间插入噪音帧让后续所有帧的绝对位置偏移 | 相对距离偏移只有噪音插入点附近受影响 |
| **跨说话人** | 不同说话人语速习惯不同，绝对位置统计量不同 | 相对关系跨说话人更一致 |

Transformer-XL / Conformer 采用的 Relative Positional Encoding 将位置编码**融入 Attention 的偏置项（bias）** 中，而不是加到输入向量上：

```
Attention(Q, K) = QK^T + relative_position_bias
```

这样模型学到的是"偏移量 i−j 用什么偏置"，而不是"位置 i 的向量是什么"。

---

### 应用层

#### Q7：你有一个离线 ASR 系统（Conformer + AED），现在要求改成低延迟流式。哪些可以复用？哪些需要重训？

**A：**

**可以复用的：**
- **训练数据**（语音-文本对），无需重新标注
- **特征提取管道**（FBank + CMVN 参数）
- **词典和 Tokenizer**（BPE / 字表）
- **Conformer 权重可以部分初始化流式模型**——非因果版训练好的权重可以作为因果版训练的起始点

**需要重训的：**
| 组件 | 原因 |
|---|---|
| **Causal Conformer 编码器** | Attention 和 Conv 的因果掩码变了，需要重新训练 |
| **训练范式** | AED 不能流式，需要换 CTC 或 RNNT |
| **解码策略** | beam search 的约束条件变了 |
| **推理框架** | 可能需要换支持流式的推理引擎（如 WeNet、NCNN） |

**推荐迁移路径**：
```
Conformer + AED (离线)
  ↓ 1. 保留 Conformer 权重（非因果→因果初始化）
  ↓ 2. 换成 CTC 头（简单场景）或 RNNT 头（高精度场景）
  ↓ 3. 用原有数据 + 流式模拟重训
Conformer + RNNT (流式)
```

---

#### Q8：你的 Conformer 模型在测试集上 WER 比预期高了 15%。你能从编码器设计的角度列出三条排查方向吗？

**A：**

**排查方向一：降采样率是否合适？**
- 如果你的语速极快（每秒 5-6 个音节），降采样 1/4 后每帧要编码更多语音内容
- 尝试降采样改为 1/2（两层 CNN，其中一层 stride=1），或者前接一个更密集的 CNN front-end

**排查方向二：Conformer 的层数和 Attention head 数是否匹配数据量？**
| 数据量 | 推荐配置 |
|---|---|
| < 100h | 12 层 Conformer 可能过拟合 → 降到 8 层 + 增加 Dropout (0.2) |
| 100-1000h | 16-18 层为标准配置 |
| > 10000h | 20+ 层可以发挥优势 |

**排查方向三：位置编码的上下文长度是否覆盖了你的语音？**
- Relative Positional Encoding 有一个最大长度限制（如 max_len=512）
- 如果你的语音超过这个长度（如 1 分钟以上的极长语音），Attention 的偏移偏置会被截断
- 检查 max_positions 是否匹配你的训练数据中最长语音的帧数（降采样后）

> 补充：WER 高不一定是编码器的问题——也检查声学特征（FBank 参数是否匹配）、训练范式（CTC 独立假设是否成为瓶颈）、语言模型（如果用了 external LM 的话）。

---

### 评价层

#### Q9：Conformer "两个都要"的策略——CNN + Attention 并行融合有什么潜在代价？

**A：**

代价主要在三个维度：

**① 计算开销**
- 标准 Transformer 已经需要 O(T²) 的计算量
- 额外增加 Depthwise Conv + GLU + Pointwise Conv，每层多了 ~30-50% 的 FLOPs
- 对于移动端和低功耗设备，这个代价不是总能接受

**② 结构复杂度**
- Macaron Sandwich（FF → Conv+Attn → FF）让模型层数翻倍（两个 FF）
- 超参数增多（CNN 通道数、卷积核大小、GLU 维度），调参难度增大
- 对比：Branchformer 把结构改成**双分支并行的全局/局部分支**，思路更清晰，后续改进更容易

**③ 设计冗余？**
- 如果数据量小（< 100h），CNN + Attention 的融合优势不明显——标准 Transformer + 充分正则化可能效果相当
- 在超大语种（> 10000h）上 Conformer 的优势才充分体现

**一句话评价**：Conformer 是一个优秀的工程折中，但不是理论上"最优雅"的设计。它牺牲了简洁性和计算效率来换取精度上限。如果你的场景对延迟和算力敏感（手机、嵌入式），需要仔细权衡是否值得多这 50% 的计算量换 5-10% 的相对 WER 下降。

---

#### Q10：从 RNN → Transformer → Conformer → Branchformer，ASR 编码器的设计趋势是什么？

**A：**

这条演进路线背后有一条清晰的逻辑：

```
串行 → 并行              (RNN → Transformer)
全局并行 → 全局+局部融合   (Transformer → Conformer)
串行融合 → 并行融合       (Conformer → Branchformer)
```

**核心趋势：从"单一范式"到"互补范式融合"**

| 编码器 | 核心思想 | 局限性 | 下一代的解决方向 |
|---|---|---|---|
| **RNN/LSTM** | 串行时序建模 | 不能并行，梯度问题 | → 用 Attention 替代串行 |
| **Transformer** | 全局并行 Attention | 没有局部归纳偏置 | → 加入 CNN 补充局部特征 |
| **Conformer** | 串行融合（CNN + Attn 在同一个 block 内） | 结构复杂，调参麻烦 | → 简化融合方式 |
| **Branchformer** | 并行双分支（全局/局部各走各的） | 分支合并可能丢失信息 | → 改进分支交互方式 |

**另一个趋势：效率优化成为标配**
- Squeezeformer、Zipformer、Emformer 不是突破性 idea，但解决了"Conformer 太贵"的问题
- 未来的编码器大概率是在"精度-效率"的 Pareto 前沿上找更好的点，而不是再做一个"全新型号"

> 对工程选型的启示：2024-2025 年的新项目建议以 Conformer 或 Branchformer 为基线，然后根据场景需求考虑是否切到效率优化版。**开新"类 Transformer 架构"能带来的边际收益正在快速缩小，更大的收益在数据、训练策略和推理优化上。**

---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Conformer: Convolution-augmented Transformer for Speech Recognition | Gulati et al. (2020) | [arXiv](https://arxiv.org/abs/2005.08100) |
| Speech-Transformer: A No-Recurrence Seq2Seq Model for Speech Recognition | Dong et al. (2018) | [arXiv](https://arxiv.org/abs/1804.06993) |

---
