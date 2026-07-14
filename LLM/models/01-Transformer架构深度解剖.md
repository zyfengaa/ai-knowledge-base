# Transformer 架构深度解剖

> Google | "Attention Is All You Need" —— 所有现代 LLM 的技术地基

---

## 写在前面

Transformer（Vaswani et al., 2017）的贡献不需要复杂描述，一句话就够了：

> **"用纯自注意力替代 RNN 做序列建模，实现并行计算 + 长程依赖一步到位。"**

这一刀有多狠——2017 年之前，NLP 的通用架构是 RNN/LSTM（串行，梯度问题，无法并行），Transformer 之后，所有主流模型都基于它。GPT、BERT、T5、LLaMA、DeepSeek、Qwen、GLM……全是 Transformer 的变体或子集。

---

## 第一章 前置条件：RNN 时代的困境

### 1.1 序列建模的老办法

在 Transformer 之前，处理序列数据的标准工具是**循环神经网络（RNN）**及其变体 LSTM、GRU。

```
RNN 的递推公式:
  h_t = tanh(W_h · x_t + U_h · h_{t-1} + b_h)

每一步 t 的计算依赖上一步的隐状态 h_{t-1}。
这意味着: 无法并行。第 100 步必须等第 99 步算完。
```

**RNN 的三个核心问题：**

1. **串行计算** → 训练慢。一个 512 词的句子要串行 512 步。GPU 的并行能力被浪费
2. **梯度消失/爆炸** → 长序列处理不了。即使 LSTM 引入了门控机制缓解了梯度问题，但实际上 100 步以上的依赖仍然很难捕捉
3. **长程依赖弱** → "文档开头的实体"和"文档结尾的指代"之间的关系，RNN 几乎无法建模

### 1.2 注意力机制：最初的辅助角色

Bahdanau et al.（2015）在机器翻译中提出了注意力机制——让解码器在每步生成时"聚焦"到源端的不同位置：

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V
```

但这个 Attention 只是 RNN 的附属模块——编码器仍然是 RNN，解码器也仍然是 RNN。Attention 只是用来"辅助对齐"的。

**Transformer 的突破性想法**：既然 Attention 能让每一步看到所有位置，那为什么还需要 RNN？**只用 Attention 做所有的事情。**

---

## 第二章 整体架构

Transformer 是一个 **Encoder-Decoder** 架构，最初为机器翻译设计：

```
输出序列
    ↑
┌─── Decoder ───┐
│  Self-Attn →  │
│  Cross-Attn → │  ← Encoder 的最后输出作为 K, V
│  FFN          │
└───────┬───────┘
        │
┌─── Encoder ───┐
│  Self-Attn →  │
│  FFN          │
└───────┬───────┘
        │
   输入序列 (词嵌入 + 位置编码)
```

**Encoder**：将输入序列编码为一组上下文相关的表征（双向，每步都能看到所有位置）
**Decoder**：自回归地生成输出序列（因果，每步只能看到已生成的位置 + 通过 Cross-Attention 看编码器的输出）

---

## 第三章 核心模块深度解剖

### 3.1 Scaled Dot-Product Attention（缩放点积注意力）

这是 Transformer 最基础的"原子操作"：

```
输入: Q (Query), K (Key), V (Value)  形状: [batch, seq_len, d_k]

公式: Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
                                          ↑
                                    缩放因子 √d_k
```

**为什么需要除以 √d_k？**

当 d_k 很大时，Q 和 K 点积的方差很大，softmax 的结果会趋向于"一个位置接近 1、其余接近 0"——梯度极小，训练困难。除以 √d_k 让方差稳定在 1 附近：

```
d_k = 64 时, Q·K 的方差 ≈ 64, softmax 对数值差异极度敏感
除以 √64 = 8 后, 方差 ≈ 1, softmax 回归正常范围
```

### 3.2 Multi-Head Attention（多头注意力）

不做一个 Attention，而是**把 Q/K/V 投影到 h 个不同的子空间，各做一次 Attention，再拼回去**：

```
输入: Q, K, V  [batch, seq_len, d_model]
    │
    ├─ 线性投影到 h 个头:
    │   head_i = Attention(Q·W_Q_i, K·W_K_i, V·W_V_i)
    │   每个 head 在 d_k = d_model / h 的子空间中计算
    │
    ├─ Concat: [head_1, ..., head_h]  [batch, seq_len, d_model]
    │
    └─ 线性投影: Concat · W_O → 输出  [batch, seq_len, d_model]
```

**为什么需要多头？**

单个 Attention 只能捕捉"一种"关系模式。多头允许模型同时在多个不同的表征子空间中关注输入：
- 一个头可能关注语法依赖（主语-谓语）
- 一个头可能关注位置距离（相邻词）
- 一个头可能关注语义相似性

论文的配置（Base 模型）：h=8, d_k=d_v=64, d_model=512

### 3.3 Position-wise Feed-Forward Network（逐位置前馈网络）

每个位置独立通过同一个两层 MLP：

```
FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2
        = ReLU(x·W_1 + b_1)·W_2 + b_2
```

- 第一层：d_model → d_ff（通常是 4× 扩张：512→2048）
- 激活函数：ReLU
- 第二层：d_ff → d_model（2048→512）

**Attention 负责"在哪里看"，FFN 负责"看到了什么"**。每个位置的 FFN 是独立计算的，没有跨位置交互。

### 3.4 位置编码（Positional Encoding）

**问题**：Self-Attention 本身是**置换不变的**——把序列顺序打乱，Attention 的结果一样。模型感知不到"第一个词"和"第二个词"的区别。

**解决方案**：在输入嵌入上叠加一个位置编码信号。

Transformer 用的是**固定正弦/余弦位置编码**（不是学习的）：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**设计意图**：
- 每个位置有唯一的编码（确定性，不需要学习）
- 不同维度的周期不同（从 2π 到 10000·2π），低维编码位置细节、高维编码全局位置
- 相邻位置在编码空间中接近
- 编码的线性变换可以表达相对位置（通过三角恒等式：sin(α+β) = sinα·cosβ + cosα·sinβ）

**局限性**：固定位置编码不能 extrapolate 到比训练更长的序列。后来的 RoPE（旋转位置编码，LLaMA 使用）和 ALiBi 解决这个问题。

### 3.5 残差连接 + LayerNorm

每个子层（Attention/FFN）外面都套一层"残差连接 + LayerNorm"：

```
output = LayerNorm(x + Sublayer(x))
```

**残差连接**让梯度可以直接流过深层网络（不受梯度消失影响），是 Transformer 能堆到 96 层+（GPT-3）的关键设计。

**LayerNorm** 对每个样本的**所有特征维度**做标准化（对比 BatchNorm 是对整个 batch 的一个特征维度做标准化）：

```
LN(x) = γ · (x - μ) / √(σ² + ε) + β
```

LayerNorm 在 Transformer 中的使用后来被 RMSNorm 简化替代（少了均值偏移的计算，LLaMA 等模型使用）。

### 3.6 掩码机制

**Encoder**：没有掩码，每个位置可以看到所有位置 → 双向
**Decoder**：**因果掩码**（causal mask）——每个位置只能看到它自己和它左边的位置，不能看到右边（因为它自回归地生成未来的词）：

```
    我  爱  你
我   1   0   0
爱   1   1   0
你   1   1   1

上三角区域掩码为 -∞ → softmax 后注意力为 0
```

**Cross-Attention 的掩码**：Decoder 的 Cross-Attention 中，Query 来自 Decoder，Key/Value 来自 Encoder（双向），没有位置限制。

---

## 第四章 模型配置

### 4.1 论文中的两个规格

| 参数 | Base | Big |
|------|------|-----|
| d_model | 512 | 1024 |
| d_ff | 2048 | 4096 |
| h (heads) | 8 | 16 |
| d_k, d_v | 64 | 64 |
| Encoder 层数 | 6 | 6 |
| Decoder 层数 | 6 | 6 |
| 总参数量 | ~65M | ~213M |
| P_drop (dropout) | 0.1 | 0.3 |
| label_smoothing | 0.1 | 0.1 |

### 4.2 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| 学习率调度 | **Warmup + 衰减**（先线性增加到 7e-4，再按步数的倒数平方根衰减）|
| Warmup 步数 | 4,000 |
| Batch 大小 | ~25,000 source + 25,000 target tokens |
| 训练数据 | WMT 2014 英德（4.5M 句子对）/ 英法（36M 句子对） |
| 硬件 | 8× P100 GPU (Base) / 8× V100 (Big) |
| 训练时间 | 英德 Big: 3.5 天 / 英法 Big: 3.5 天 |

### 4.3 与 RNN 的计算复杂度对比

| 维度 | RNN | Transformer |
|------|-----|-------------|
| 每层计算复杂度 | O(n·d²) | **O(n²·d)**（n 为序列长度） |
| 顺序计算步数 | **O(n)**——串行 | **O(1)**——并行 |
| 最大路径长度 | O(n)——穿过所有隐状态 | **O(1)**——一步 Attention |
| 瓶颈 | 训练慢、长程弱 | n² 复杂度——超长序列显存爆炸 |

> **Transformer 对短序列比 RNN 快，对超长序列比 RNN 慢**（n² 的劣势）。但对绝大多数 NLP 任务（n < 1024），Transformer 完胜。

---

## 第五章 推理流程演练

以机器翻译为例：输入 "I love you" → 输出 "我爱你"

### Stage 1: 嵌入 + 位置编码

```
输入: ["I", "love", "you"] (3 个 token)
    → 词嵌入: 每个 token → 512 维向量
    → + 位置编码: 每个位置加入位置信号
    → Encoder 输入: [3, 512]
```

### Stage 2: Encoder 编码

```
6 层 Encoder，每层:
  Step 1: Multi-Head Self-Attention（双向）
          — "love" 能看到 "I" 和 "you" 和 "love" 自己
  Step 2: Residual + LayerNorm
  Step 3: FFN（ReLU → 扩张到 2048 → 降回 512）
  Step 4: Residual + LayerNorm

输出: Encoder 的最后层输出 → [3, 512]，作为 Decoder 的 K, V
```

### Stage 3: Decoder 生成

```
Decoder 输入: <sos> (start of sequence token)

Step 1（生成"我"）:
  - Masked Self-Attn: 只看 <sos>（因果掩码）
  - Cross-Attn: <sos> 的 Query 看 Encoder 的 3 个输出位置
  - FFN → 预测 "我"

Step 2（生成"爱"）:
  - Masked Self-Attn: 看 <sos> + "我"（因果掩码）
  - Cross-Attn: "我" 的位置看 Encoder 输出
  - FFN → 预测 "爱"

Step 3（生成"你"）:
  同上

Step 4（生成 <eos>）:
  输出结束符 → 终止
```

### Stage 4: 自回归的串行性

注意：虽然 Encoder 是完全并行的（一次前向处理所有位置），**Decoder 的生成是串行的**——每步只能生成一个 token。这是 LLM 推理延迟的核心瓶颈，后来的 KV Cache 和 speculative decoding 都是为了缓解这个问题。

---

## 第六章 后续影响——Transformer 的遗产

### 6.1 三条架构路线

Transformer 的设计衍生出了三条路线：

```
Transformer (2017, Google)
    │
    ├── Encoder-Only (BERT 路线)
    │    └── 双向 → 理解类任务（分类、标注、搜索）
    │
    ├── Encoder-Decoder (T5 路线)
    │    └── 完整的 Encoder + Decoder → 翻译、摘要、生成+理解
    │
    └── Decoder-Only (GPT 路线)  ← 最终胜出
         └── 因果 → 生成类任务（如今 "everything is generation"）
```

Decoder-Only 之所以最终胜出，原因在于：
- 足够简单——不需要维护两个模块
- In-context learning——所有任务可以统一为"输入文本→生成文本"
- Scaling 效果好——更大的 Decoder 直接提升生成质量

### 6.2 Transformer 的后续改动

| 改动 | 解决的问题 | 谁做的 |
|------|-----------|--------|
| **Pre-LN** → 把 LayerNorm 放到子层之前 | 训练更稳定 | 后续几乎所有 LLM |
| **RoPE** → 旋转位置编码 | 更好的位置感知 + 可以外推到更长序列 | LLaMA、Qwen、DeepSeek |
| **GQA** → 分组查询注意力 | KV Cache 节省 50-75% | LLaMA 2 70B、LLaMA 3、Qwen |
| **SwiGLU** → 替换 ReLU | 表达能力更强的门控 FFN | LLaMA、几乎所有现代 LLM |
| **RMSNorm** → 替换 LayerNorm | 省去均值计算，更高效 | LLaMA、几乎所有现代 LLM |
| **MLA** → 潜空间压缩注意力 | KV Cache 节省 75-93% | DeepSeek-V2/V3 |
| **Sliding Window Attention** | 线性复杂度处理长序列 | Mistral、Mixtral |

但所有这些改动都是**在 Transformer 框架内**的优化。**基础架构——Self-Attention + FFN + Residual + LayerNorm——从 2017 年定型至今没有变过。**

---

## 第七章 一句话总结

```
Transformer 前的 NLP:
    RNN: 串行 → 慢
    LSTM: 能记但记不久 → 对长序列仍然弱
    Attention: 只是 RNN 的辅助模块

Transformer 的决定性一刀:
    "RNN 不需要了。Self-Attention 自己做全部的序列建模。
    并行让它快，全局连接让它看得远。"

Transformer 后的所有 LLM:
    GPT、BERT、T5、LLaMA、DeepSeek、Qwen、GLM……
    全是在 Transformer 这条根上长出的不同枝条。
```

---

**Sources:**
- [Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762)
- [Visualizing A Neural Machine Translation Model (Bahdanau Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Scaling Laws for Neural Language Models (arXiv:2001.08361)](https://arxiv.org/abs/2001.08361)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (arXiv:2305.13245)](https://arxiv.org/abs/2305.13245)
- [LLaMA: Open and Efficient Foundation Language Models (arXiv:2302.13971)](https://arxiv.org/abs/2302.13971)
