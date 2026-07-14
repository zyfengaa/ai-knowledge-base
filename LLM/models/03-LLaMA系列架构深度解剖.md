# LLaMA 系列架构深度解剖

> Meta | 开源 LLM 的引擎——"LLM 最佳实践"的定型者

---

## 写在前面

LLaMA 系列的贡献可以浓缩为一句话：

> **"用更多的训练数据补偿模型容量，然后把这一切开源。"**

这条路线在 LLaMA 1（2023.02）中被验证：7B 模型在多数 benchmark 上超过 GPT-3（175B）。2023.07 的 LLaMA 2 允许商用。2024.07 的 LLaMA 3.1 405B 成为首个匹配 GPT-4 的开源模型。整个 LLaMA 系列见证了开源 LLM 从"追赶"到"并跑"的全过程。

更重要的是——LLaMA 系列是"LLM 最佳实践"的**定型者**。RoPE（旋转位置编码）、SwiGLU 激活函数、RMSNorm、GQA（分组查询注意力）——这些东西现在几乎是所有 LLM 的标配，但 LLaMA 论文是第一个把它们组合成参考架构的。

---

## 第一章 前置条件：GPT-3 闭源后的真空

### 1.1 2023 年初的局面

ChatGPT（2022.11）引爆了全球需求，但整个行业面临一个尴尬局面：

- **OpenAI GPT-3/GPT-4** — 能力最强，但闭源 API only，贵、不透明、供应商锁定
- **其他闭源模型** — Google PaLM、Anthropic Claude — 同样是 API，没有权重可下载
- **已有开源模型** — BLOOM（176B，HuggingFace 社区）性能不够好；OPT（Meta 2022）训练不充分

市场需要的不是一个"能在某个 benchmark 上接近 GPT 的模型"，而是一个**开发者能自己部署、微调、修改的开源模型**。

### 1.2 LLaMA 的核心洞察

LLaMA 论文最关键的一句话：

> **"最近的研究表明，对于给定的计算预算，最好的性能不是最大的模型，而是在更多数据上训练的更小模型。"**

简单说：**LLaMA-13B 在 1.0T token 上训练 → 超过 GPT-3（175B 但只训了 300B token）**。

不是模型更大更好，而是**数据更多更好**。这个洞察改变了整个开源 LLM 的路线。

---

## 第二章 LLaMA 1（2023.02）

### 2.1 整体架构

LLaMA 1 采用 **Decoder-only Transformer**，核心架构选择如下：

```
输入 Token
    ↓
Token Embedding (vocab_size=32000, dim=4096/5120/6656/8192)
    ↓
┌─────────────────────────┐
│    × N 层 Decoder Block  │
│                         │
│  ├── RMSNorm            │  ← Pre-Norm（标准化在子层之前）
│  ├── RoPE Self-Attention│  ← 旋转位置编码（不是绝对位置编码）
│  │    └── x + Attn(x)   │    残差连接
│  ├── RMSNorm            │
│  ├── SwiGLU FFN         │  ← 门控前馈网络
│  │    └── x + FFN(x)    │    残差连接
│                         │
└──────────┬──────────────┘
    ↓
RMSNorm
    ↓
Linear (vocab_size)
    ↓
Softmax → 输出概率
```

### 2.2 四个关键架构选择

**1. RMSNorm（替代 LayerNorm）**

原始的 Transformer 使用 LayerNorm 做层归一化：

```
LayerNorm: LN(x) = γ · (x - μ) / √(σ² + ε) + β     ← 减均值 + 除方差
RMSNorm:   RMSNorm(x) = γ · x / √(mean(x²) + ε)      ← 只除方差，不减均值
```

RMSNorm 相当于 LayerNorm 的"简化版"——取消了均值偏移的计算。论文论证：**均值偏移对 Transformer 的贡献不大，去掉可以节省约 10% 的归一化计算时间**。LLaMA 之后，几乎所有现代 LLM 都改用 RMSNorm。

**2. RoPE（旋转位置编码，替代绝对位置编码）**

RoPE（Rotary Position Embedding，苏剑林 2021）的直觉是：

> **用旋转矩阵对 Query 和 Key 做变换，让 Attention 分数的计算自然地包含位置信息。**

关键特性：RoPE 编码的是**相对位置**而非绝对位置。这让模型在推理时可以处理比训练时更长的序列。

```
绝对位置编码：位置 1→v₁, 位置 2→v₂, ... 添加到嵌入上
  → 新位置（如 2000）如果在训练时没见过，编码就不存在

RoPE：通过旋转矩阵编码 q 和 k 各自的位置
  → q_m · k_n 的结果天然包含 m-n 的相对位置信息
  → 新位置只需要延用相同旋转公式
```

**对比**：Transformer 的原始正弦编码是固定不可学习的。RoPE 同样固定但能表示**相对位置**。后续 DeepSeek 在此基础上做了改进（部分 RoPE + 增加了旋转频率的 learnable 参数）。

**3. SwiGLU 激活函数（替代 ReLU/GELU）**

原始的 Transformer FFN：

```
FFN_ReLU(x) = ReLU(x·W₁ + b₁)·W₂ + b₂
```

SwiGLU（Shazeer 2020）引入了**门控机制**：

```
FFN_SwiGLU(x) = (Swish(x·W_gate) ⊙ (x·W_up)) · W_down
```

核心差异——**多了一个"门"**：W_gate 的输出经过 Swish 激活（`Swish(x)=x·σ(x)`）后，与 W_up 的输出做逐元素相乘。这个"门"控制有多少信息通过。

论文实验表明 SwiGLU 比 ReLU/GELU 效果好，代价是**参数多了 1/3**（因为多了 W_gate 矩阵）。LLaMA 为了维持总参数量，把 FFN 的中间维度缩小了——标准 FFN 是 `4×d_model`，SwiGLU 用的比例是 `~2.7×d_model`（实际配置见下表）。

**4. Pre-Norm / Pre-RMSNorm**

原始的 Transformer 是 Post-Norm（先子层后归一化）：
```
output = LayerNorm(x + Sublayer(x))
```

LLaMA 用的是 Pre-Norm（先归一化后子层）：
```
output = x + Sublayer(RMSNorm(x))
```

Pre-Norm 的训练稳定性更好（Post-Norm 在深层容易出现梯度爆炸），后续所有 LLM 都采用了 Pre-Norm。

### 2.3 四种规格

| 参数 | 7B | 13B | 33B | 65B |
|------|-----|------|------|------|
| d_model | 4096 | 5120 | 6656 | 8192 |
| 层数 | 32 | 40 | 60 | 80 |
| 注意力头数 | 32 | 40 | 52 | 64 |
| d_head | 128 | 128 | 128 | 128 |
| FFN 中间维度 | 11008 | 13824 | 17920 | 22016 |
| SwiGLU 比例 | ≈2.7× | ≈2.7× | ≈2.7× | ≈2.7× |
| 训练 token 数 | 1.0T | 1.0T | 1.4T | 1.4T |
| 学习率 | 3.0e-4 | 3.0e-4 | 1.5e-4 | 1.5e-4 |
| Batch 大小 | 4M tokens | 4M tokens | 4M tokens | 4M tokens |
| 优化器 | AdamW | AdamW | AdamW | AdamW |

### 2.4 训练数据

| 数据来源 | 占比 | Token 数 |
|---------|------|---------|
| CommonCrawl | 67.0% | 3.3T |
| C4 | 15.0% | 0.7T |
| GitHub | 4.5% | 0.2T |
| Wikipedia | 4.5% | 0.2T |
| Books | 4.5% | 0.2T |
| ArXiv | 2.5% | 0.1T |
| StackExchange | 2.0% | 0.1T |
| **合计** | **100%** | **~5.0T (去重后 1.0-1.4T)** |

**关键细节**：数据是**经过大量清洗和去重**的。CommonCrawl 用了 5 轮过滤：语言检测 → URL 去重 → 行级去重（MinHash） → 质量过滤（perplexity 阈值） → 文档去重。

### 2.5 效果

| Benchmark | LLaMA-13B | GPT-3 (175B) | 说明 |
|-----------|-----------|--------------|------|
| MMLU | 46.9 | 43.9 | **13B 超过 175B** |
| BoolQ | 78.1 | 77.5 | 推理 |
| RACE-h | 74.8 | 52.0 | 大幅超过 |
| HumanEval | 23.9 | 18.3 | 代码生成 |

**LLaMA-13B（1.0T token）在多数 benchmark 上超过了 GPT-3（300B token）**——不是因为它架构更好，而是因为它在**3 倍的数据上训练**。

---

## 第三章 LLaMA 2（2023.07）

### 3.1 LLaMA 1 → LLaMA 2 的改动

LLaMA 2 不是架构大改，而是**工程升级**：

| 维度 | LLaMA 1 | LLaMA 2 | 意义 |
|------|---------|---------|------|
| 上下文长度 | 2048 | **4096** | 双倍上下文 |
| 训练数据 | 1.0-1.4T | **2.0T** | 更多数据 |
| GQA | 无 (MHA) | **70B 用 GQA** | KV Cache 减半 |
| 注意力类型 | MHA | MHA (7B/13B) + **GQA (70B)** | 大模型效率优化 |
| 开源许可 | 需申请 | **商用免费** | LLaMA 2 才是真正的"开源"引爆点 |
| Chat 版 | 无 | **LLaMA 2-Chat (RLHF)** | 附带对齐版本 |

**GQA 的引入**：70B 模型使用 GQA（Grouped Query Attention），32 组 Query / 8 组 Key-Value，KV Cache 减少为原来的 1/4。这对大模型的推理效率至关重要。7B 和 13B 维持标准 MHA（因为它们的 KV Cache 占用还不算瓶颈）。

### 3.2 LLaMA 2-Chat：RLHF 对齐

LLaMA 2 附带了 Chat 版本，对齐方法：

```
SFT: 27,540 条人工标注的指令数据 → 基础对齐
    ↓
奖励模型: 人工对比偏好数据 → 训练两个独立的奖励模型
    ↓
PPO 强化学习: 用奖励模型优化策略
    ↓
多次迭代: PPO → 收集新的对比数据 → 重新训练奖励模型 → PPO...
```

LLaMA 2-Chat 的 RLHF 与 InstructGPT 的不同：
- **两个奖励模型**（一个安全性、一个有用性），最终得分为两者的调和平均
- 在优化过程中引入**对安全性的约束**（有用性得分不能以牺牲安全性为代价）

---

## 第四章 LLaMA 3 / 3.1（2024.04 / 2024.07）

### 4.1 架构改动

| 维度 | LLaMA 2 | LLaMA 3 | 意义 |
|------|---------|---------|------|
| 训练数据 | 2.0T | **15T+** | 7.5 倍增长 |
| 上下文 | 4096 | **8192** | 又翻倍 |
| Tokenizer | 32K vocab (BPE) | **128K vocab (tiktoken)** | 更高效编码 |
| GQA 范围 | 仅 70B | **8B 和 70B 都用 GQA** | 所有规格都做效率优化 |
| 注意力配置 | 8B: MHA | 8B: **8 Q / 8 KV** | 8B 也用了 GQA (实际就是 MHA, 但统一了框架) |
| | 70B: 32 Q / 8 KV | 70B: **64 Q / 8 KV** | 更多的查询头 + 固定的 KV 头 |
| 分组数 | 4 (70B) | 8 (70B) | |
| 位置编码 | RoPE (10000) | RoPE (500000) | 基频调高 50 倍——更好处理长序列 |
| RoPE theta | 10000 | **500000** | 允许更长上下文的外推 |

**Tokenizer 升级到 128K**：使用 OpenAI 的 tiktoken 分词器（和 GPT-4 同款）。好处是对代码和数字的编码效率大幅提升——之前 32K 分词器一个空格可能被拆成多个 token，128K 可以整个token表示。

### 4.2 Data Scaling

LLaMA 3 的训练数据（15T+ token）是 LLaMA 2（2T）的 7.5 倍。Meta 的论文详细描述了数据处理的关键点：

1. **网页数据过滤**：用 FastText 分类器（质量评分）和 LLM2Vec（内容相似度）过滤低质量页面
2. **去重**：URL 级 + 文档级（MinHash 3-gram）+ 行级
3. **代码数据**：从 GitHub 大规模采集，经过许可检查（仅允许 MIT/Apache 等宽松许可的仓库）
4. **多语言覆盖**：其中 5% 是非英语（主要是中文、法语、德语、西班牙语）

### 4.3 LLaMA 3.1 405B——开源 Dense 的巅峰

| 维度 | 说明 |
|------|------|
| 参数量 | **405B（纯 Dense，非 MoE）** |
| 上下文 | 128K |
| 架构 | 126 层 Decoder, d_model=16384, 128 heads (16 KV GQA) |
| 训练 | 15.6T token, 30.8T GPU hours on H100-80GB |
| 数据 | 同 LLaMA 3，但加入了更多多语言和数学/推理数据 |

**为什么 405B 选择 Dense 而非 MoE？**

Meta 的解释：Dense 模型训练更稳定、更容易扩展、推理的预测性更强。MoE 虽然推理成本低，但训练时各 expert 的 load balancing 是额外调参负担。

**LLaMA 3.1 405B 的意义**：**首个在各项指标上匹配 GPT-4 的开源模型**。这是开源社区的一个分水岭——从此"闭源比开源强"不再是共识。

---

## 第五章 LLaMA 4（2025.04）——MoE 转型

| 维度 | 说明 |
|------|------|
| 架构 | **MoE**（Meta 首次从 Dense 转向 MoE）— 128 专家 |
| Scout | 109B 总 / 17B active per token, **10M** 上下文 |
| Maverick | **402B** 总 / 17B active per token, **1M** 上下文 |
| 上下文 | Scout: **10M** / Maverick: **1M** |
| 注意力 | iRoPE（interleaved RoPE）——交错地应用 RoPE |

**为什么 LLaMA 4 转向 MoE？**

Dense 在 405B 之后 scaling 成本太高。MoE 可以通过增加 expert 数量（不增加每 token 的计算量）来扩容。这反映了行业的共识转向——**2025 年之后，旗舰模型几乎没有 Dense 了**。

---

## 第六章 LLaMA 系列的影响

### 6.1 "LLaMA 架构" = "LLM 参考架构"

LLaMA 系列最大的贡献不是模型本身，而是**定义了一套架构模板**：

```
Token Embedding
    ↓
RMSNorm ← 替代 LayerNorm
    ↓
RoPE Self-Attention ← 替代绝对位置编码
    ↓
RMSNorm
    ↓
SwiGLU FFN ← 替代 ReLU/GELU
    ↓
(可选) GQA ← KV Cache 效率优化
    ↓
以上重复 N 层
```

现在几乎所有开源 LLM（DeepSeek、Qwen、Mistral 等）都在这套模板上做修改。

### 6.2 对整个行业的影响

| 时间 | 事件 | 影响 |
|------|------|------|
| 2023.02 | **LLaMA 1** 发布（权重需申请） | 全球社区开始大量微调（Alpaca、Vicuna、Koala），证明了开源模型可以接近 GPT-3 水平 |
| 2023.03 | **LLaMA 权重泄露** | 开源运动加速——Alpaca 7B 在 LLaMA 上只用 52K 指令数据就训出了对话能力 |
| 2023.07 | **LLaMA 2** 发布（商用免费） | **真正的引爆点**。企业可以合法下载和使用。HuggingFace 上数万衍生模型 |
| 2024.04 | **LLaMA 3** 发布 | Data Scaling 路线确认——15T+ token 训练数据 |
| 2024.07 | **LLaMA 3.1 405B** 发布 | 开源首次匹配 GPT-4 |
| 2025.04 | **LLaMA 4** 发布 | 转向 MoE，10M 上下文 |

---

## 第七章 架构全景速查

```
LLaMA 1/2 → Decoder-only, RMSNorm, RoPE, SwiGLU, Pre-Norm
LLaMA 2 70B → + GQA (32Q/8KV)
LLaMA 3/3.1 → + 128K vocab (tiktoken), RoPE theta=500000, 128K ctx (3.1)
LLaMA 3.1 405B → 126层, d=16384, 128heads, 16KV (GQA 8:1)
LLaMA 4 → MoE, 10M ctx, iRoPE
```

---

**Sources:**
- [LLaMA: Open and Efficient Foundation Language Models (arXiv:2302.13971)](https://arxiv.org/abs/2302.13971)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models (arXiv:2307.09288)](https://arxiv.org/abs/2307.09288)
- [The Llama 3 Herd of Models (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer (SwiGLU, arXiv:2002.05202)](https://arxiv.org/abs/2002.05202)
- [Root Mean Square Layer Normalization (RMSNorm, arXiv:1910.07467)](https://arxiv.org/abs/1910.07467)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (arXiv:2305.13245)](https://arxiv.org/abs/2305.13245)
- [LLaMA 4 Model Card (Meta Blog)](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
