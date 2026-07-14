# DeepSeek 系列架构深度解剖

> DeepSeek（深度求索）| MIT 开源 | "用 1/20 的成本，做到相同或更好的性能"

---

## 写在前面

2024 年，所有人都知道 Scailing Law 还在起作用——LLaMA 3 405B、GPT-4 的参数量都达到了千亿级别，性能也在持续提升。但问题也很明显：**成本高到不可持续**。

训练一个 GPT-4 级别的模型需要超过 1 亿美元的算力投入。推理成本同样惊人——每次调用都在烧钱。整个行业都需要一个答案：**能不能用更少的钱，做出同样好的模型？**

DeepSeek 给出了自己的答案：**极致工程效率**（Extreme Engineering Efficiency）。不是靠更大的模型、更多的算力去碾压，而是靠算法创新和工程优化，把每一分算力都用到刀刃上。

这个系列的贡献清单（截至 2026 年）：
- **DeepSeek-V2**（2024.05）：Multi-head Latent Attention（MLA）——自 GQA 以来最有价值的注意力机制创新
- **DeepSeek-V3**（2024.12）：671B MoE + FP8 混合精度训练 + Multi-Token Prediction——训练成本仅 $5.57M
- **DeepSeek-R1**（2025.01）：Group Relative Policy Optimization（GRPO）——去掉了 Critic 模型的 RL 算法
- **DeepSeek-V4 / V4.1**（2026.04 / 2026.06 灰度）：V4 实现 100% 华为昇腾 910C 训练，1M 上下文；V4.1 新增全模态（图像+音频输入）+ 原生 MCP 协议

---

## 第一章 前置条件：大模型的成本困境

### 1.1 Scaling Law 的光与影

Scaling Law（Kaplan et al., 2020; Hoffmann et al., 2022）告诉我们：模型性能随参数、数据和算力同时增长。但"增长"背后的成本是**超线性**的：

```
GPT-4 (2023)
  参数量: ~1.8T (估计)
  训练成本: >$100M (估计)
  单次推理: 成本数美分
  
LLaMA 3 405B (2024)
  参数量: 405B (dense)
  训练成本: >$30M (估计)
  单次推理: 仍远高于中小模型

行业的困境:
  "更大的模型 = 更好的性能" 这条路线
  只有少数几家公司负担得起
```

### 1.2 降本的三条路线

业界在 2023-2024 年探索了三条降本路线：

| 路线 | 做法 | 代表 | 优势 | 代价 |
|------|------|------|------|------|
| **MoE 稀疏化** | 用多个小专家网络替代一个大 FFN | Mixtral 8×7B, DeepSeek-V2 | 激活参数少，推理快 | 通信开销、负载不均衡 |
| **量化** | 用低精度替换 FP16/32 | GPTQ, AWQ, FP8 | 显存减半、吞吐翻倍 | 精度损失需要弥补 |
| **蒸馏** | 大模型教小模型 | LLaMA 3.1-Nemotron | 小模型能力接近大模型 | 训练时需要大模型 |

DeepSeek 的独特之处：**它不是走其中一条路，而是把三条路都走到了极致，并且在注意力机制层面还做出了一项根本性的创新。**

---

## 第二章 DeepSeek-V2（2024.05）：MLA + DeepSeekMoE

### 2.1 Multi-Head Latent Attention（MLA）—— DeepSeek 最核心的创新

MLA 要解决的根本问题：**标准 MHA 的 KV Cache 太大**。

在自回归推理中，标准 MHA 每层的 KV Cache 大小是：

```
标准 MHA 的 KV Cache（每层）:
  形状: [batch, 2, seq_len, d_model]
  单序列单层: 2 × d_model × seq_len × precision
  
  例: d_model=4096, seq_len=4096, FP16
     → 2 × 4096 × 4096 × 2 = 67.1 MB/层
     → 60 层: 4 GB/序列
     → 并发 100 序列: 400 GB

GQA 的优化:
  把 Key 和 Value 的头分组共享 → KV Cache 减少为 1/h 组
  (LLaMA 2 70B: h=64, g=8 → 节省 87.5%)
```

**DeepSeek 的想法**：不管分多少组，K 和 V 本身是在一个高维空间（d_model 维度）中存储的。能不能把它们投影到一个低维的"潜空间"（latent space），缓存这个低维向量，计算时再解压回来？

```
标准 MHA:
  Q, K, V 都在 d_model 维度
  KV Cache: 2 × d_model × seq_len  — 全维度存储

GQA:
  分组共享 K, V → KV Cache: 2 × (d_model / g) × seq_len
  组共享 → 降低多样性

MLA (DeepSeek-V2):
  将 K, V 投影到潜空间: c_KV = W_down · concat(K, V)  [d_latent]
  缓存: c_KV  [d_latent × seq_len]   (d_latent << d_model)
  计算时: K', V' = W_up · c_KV       [解压回 d_model]

  KV Cache 大小: d_latent × seq_len
  典型压缩比: 4× ~ 16×
```

**MLA 的核心原理（详细版）：**

```
           │
           ▼
┌─────────────────────────────────────┐
│  标准 Attention                      │
│  Q·Kᵀ · V     d_model=4096         │
│  KV Cache: 4096 × seq_len          │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  GQA Attention                      │
│  分组共享 K,V: 8 组                 │
│  KV Cache: 512 × seq_len (×2)      │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  MLA (DeepSeek-V2)                  │  ← 最优
│  低维潜空间: d_latent=512          │
│  KV Cache: 512 × seq_len           │
│  但保留全部注意力头的表达能力       │
└─────────────────────────────────────┘
```

**MLA 对比 MHA / GQA 的 KV Cache 大小（d_model=4096, seq_len=4096, FP16）：**

| 注意力机制 | KV Cache/层 | KV Cache/60层 | 相对 MHA | 特点 |
|-----------|-------------|--------------|---------|------|
| MHA | 67.1 MB | 4.0 GB | 1.0× | 全维度，高表达能力 |
| GQA (8组) | 8.4 MB | 0.5 GB | 0.125× | 精度与效率的折中 |
| GQA (1组=MQA) | 4.2 MB | 0.25 GB | 0.0625× | 极低显存，精度下降 |
| **MLA (d_latent=512)** | **4.2 MB** | **0.25 GB** | **0.0625×** | **保持多头表达力** |

MLA 的巧妙之处在于：**它达到了 MQA 级别的 KV Cache 压缩率，但保留了 MHA 的全部表达力**。原因是潜空间的投影矩阵是可学习的——模型可以通过训练学会用低维向量编码所有注意力头所需的 Key/Value 信息。

### 2.2 DeepSeekMoE 架构

DeepSeek-V2 的 MoE 采用了**细粒度专家分割 + 共享专家隔离**的设计：

```
DeepSeek-V2 MoE 结构:

输入 token
    │
    ├─→ [Router] → 选择 8 个专家 (out of 160)
    │                  + 1 个共享专家 (永远激活)
    │
    ├─→ Expert_1 ─┐
    ├─→ Expert_2 ─┤
    ├─→ ...       ├─→ 加权求和
    ├─→ Expert_8 ─┤
    └─→ Shared_Exp ┘

  总专家数: 160 (细粒度) + 1 (共享) = 161
  激活专家/ token: 8 (细粒度) + 1 (共享) = 9
  激活占比: 21B / 236B ≈ 8.9%
```

**与传统 MoE 的关键差异：**

| 维度 | 传统 MoE (Mixtral 8×7B) | DeepSeekMoE (V2) |
|------|--------------------------|-------------------|
| 专家数 | 8 | 160 + 1 共享 |
| 激活数/token | 2 | 8 + 1 共享 |
| 专家粒度 | 粗粒度（一个大 FFN） | 细粒度（一小块 FFN） |
| 共享专家 | 无 | 有（所有 token 共享的通用知识） |
| 负载均衡 | 辅助损失（有性能损失） | 辅助损失 + 动态路由 |

**细粒度分割的好处**：每个专家更小，更专注。160 个小专家比 8 个大专家能覆盖更多的"技能方向"——每个专家可以专精于某一种语言、某个领域或某种句法模式。

**共享专家的作用**：所有 token 都需要的"通用知识"（如语法基础、常见词汇）由共享专家提供，细粒度专家则聚焦于区分性知识。这避免了每个 token 都重复选择同样的通用专家，提高了路由效率。

### 2.3 DeepSeek-V2 完整配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 总参数量 | 236B | 全部专家参数的累加 |
| 激活参数量/token | 21B | 每次前向实际计算的参数量 |
| 层数 | 60 | Transformer Decoder 层 |
| d_model | 4096 | 隐藏层维度 |
| d_ff (dense) | 11264 | 非 MoE 层的 FFN 中间维度 |
| d_ff (MoE) | 1536 × 161 | 每个专家 1536，161 个专家 |
| 注意力头数 | 32 (h) × 8 (kv) | GQA 分组 |
| d_latent (MLA) | 512 | KV 潜空间维度 |
| d_head | 128 | 每个注意力头的维度 |
| 序列长度 | 4096 / 8192 | 训练和推理上下文 |
| MoE 专家数 | 160 细粒度 + 1 共享 | 总共 161 个前馈网络 |
| 激活专家/token | 8 (细粒度) + 1 (共享) | 共 9 个专家 |
| Top-k 路由 | 8 | 从 160 个中选 8 个 |
| 激活函数 | SwiGLU | 门控激活，提升表达力 |
| 位置编码 | RoPE | 旋转位置编码 |
| LayerNorm | RMSNorm | Pre-Norm 架构 |
| 词表大小 | ~128K | BPE Tokenizer |
| 优化器 | AdamW | β₁=0.9, β₂=0.95 |
| 学习率 | 5.5e-4 (max) | Warmup + 余弦退火 |
| 训练 tokens | 14.8T | 从零训练 |
| 训练硬件 | ~180 H800 (FP8) | — |
| 训练成本 | ~$7M (估) | 13B token / 美元 |
| 许可证 | MIT | 完全开源 |

### 2.4 训练成本结构（估算）

```
DeepSeek-V2 训练成本分解（~$7M）:

  GPU 租赁 (H800):  60%  ~$4.2M   (180卡 × 2.5月)
  数据 & 存储:      20%  ~$1.4M   (14.8T tokens)
  网络 & 通信:      10%  ~$0.7M   (MoE 跨节点通信)
  实验 & 调试:      10%  ~$0.7M   (超参搜索, 故障恢复)

对比同期模型:
  LLaMA 3 70B:    ~$15-20M (dense, 但规模小很多)
  GPT-4:          >$100M   (估计)
```

---

## 第三章 DeepSeek-V3（2024.12）：FP8 + Multi-Token Prediction

### 3.1 V3 的主要升级

V3 在 V2 的基础上，把三个关键方向推到了极致：

1. **规模升级**：236B → 671B 总参数，21B → 37B 激活参数
2. **训练精度**：从 FP16/BF16 → **FP8 混合精度训练**——第一次在 600B+ 级别 MoE 上成功应用
3. **训练目标**：从"预测下一个 token" → **Multi-Token Prediction**（MTP）——同时预测多个未来 token

### 3.2 FP8 混合精度训练

**为什么 FP8 训练是 big deal？**

在此之前，FP8 训练主要在小规模实验中使用。在大规模训练中，精度损失导致模型发散或性能下降。

```
FP8 vs FP16/BF16:

  格式   有效位数  范围       显存消耗  适用场景
  FP32   ~23 bits ±3.4e38   4 bytes   精确计算（权重积累）
  BF16   ~7 bits  ±3.4e38   2 bytes   训练主力（范围大但精度低）
  FP16   ~11 bits ±65504    2 bytes   训练（精度高但范围小）
  FP8    ~3/4 bits ±448     1 byte    推理部分场景
  ──────────────────────────────────────────────────
  FP8 训练的核心挑战:
    ① 精度损失 → 梯度积累误差 → 模型发散
    ② 范围太小 → 溢出（overflow）或下溢（underflow）
```

DeepSeek-V3 的解决方案是一种**分层的 FP8 混合精度策略**：

```
DeepSeek-V3 FP8 训练策略:

  ┌─────────────────────────────────────────────┐
  │  前向传播:  全部 FP8                        │
  │    - 线性层 (FFN, 投影): FP8 W8A8          │
  │    - Attention: FP8                         │
  │                                             │
  │  反向传播:  梯度 FP8 / 权重 BF16 混合       │
  │    - 权重梯度: FP8 (块级缩放)               │
  │    - 权重参数: BF16 主副本                  │
  │    - 优化器状态: BF16 (AdamW)               │
  │                                             │
  │  Block-wise Scaling:                        │
  │    每个 128×128 块有一个独立的缩放因子      │
  │    → 比 per-tensor scaling 更精细           │
  │    → 比 per-element 更节省                  │
  └─────────────────────────────────────────────┘
```

**FP8 节省效果**：

| 维度 | BF16 训练 | FP8 (V3) | 节省 |
|------|----------|----------|------|
| 显存/param | 2 bytes | 1 byte | 50% |
| 计算吞吐 | 1× baseline | ~1.6× | 60% 提升 |
| 通信带宽 | 1× | ~0.5× | 50% |
| 总 GPU 小时 | — | 2.788M H800 | — |

### 3.3 Multi-Token Prediction（MTP）

传统因果语言模型对每个位置只预测**一个**未来 token。DeepSeek-V3 的做法是：用多个独立的预测头，同时预测第 1、第 2、……、第 D 个未来的 token。

```
标准 Next-Token Prediction:
  输入: 我 爱 你 <EOS>
  目标: 爱 你 <EOS>

  "当你在"我"这个位置，只需要预测"爱""

Multi-Token Prediction (D=3):
  输入: 我 爱 你 <EOS> <pad> <pad>
                 │      │      │
  预测头 1:     你     <EOS>  <pad>   (next token)
  预测头 2:    <EOS>   <pad>  <pad>   (second next)
  预测头 3:    <pad>   <pad>  <pad>   (third next)

  每个位置同时预测未来 D 步
  ——但对不同深度使用不同的预测头（独立的 FFN）
```

**MTP 为什么有效？**

提出的论文（Gloeckle et al., 2024）和 DeepSeek 的实践表明：
- **更强的表示学习**：强制模型在当前位置就编码未来 token 的信息——不是"只看下一步"，而是"看到更远"
- **推理加速**：在推理时，MTP 头可以直接用作**投机解码**（speculative decoding）的 draft model——不需要额外的 draft model
- **训练效率**：MTP 的训练计算量增加很少（只在顶部增加了几个 FFN 头），但提升了模型在不同训练步数下的收敛质量

```
推理时的投机解码（利用 MTP 头）:

  1. 主模型正常预测第一个 token
  2. MTP 头（轻量）预测后续 D-1 个 token 作为 draft
  3. 主模型验证 draft 序列（一次前向验证多个 token）
  4. 验证通过 → 一次生成了 D 个 token（验证失败则回退）

  效果: 推理吞吐提升 1.3-2×（取决于 D 和接受率）
```

### 3.4 Auxiliary-Loss-Free Load Balancing

MoE 模型的经典难题：**专家负载不均衡**。好用的专家被频繁选择，其他专家闲着。传统的解决方案是在损失函数中添加辅助损失（auxiliary loss）来惩罚不均衡——但这会干扰主任务的学习。

DeepSeek-V3 的方法：**动态偏置调整**——对每个专家维护一个历史负载统计，在路由时添加一个偏置项，负载低的专家获得偏置加成，负载高的专家受到偏置惩罚。这个偏置与主损失**不相关**，不影响梯度：

```
Auxiliary Loss (传统):
  Loss_total = Loss_main + λ · Loss_balance(专家选择分布)
  问题: λ 超参数难调，过大伤性能，过小不均衡

Auxiliary-Loss-Free (DeepSeek-V3):
  router_score_i = softmax(x · W_router)_i + bias_i
  bias_i ← 根据专家的历史负载动态调整
  
  选择依据: argmax(router_score_i)
  
  更新规则 (每步微调):
    if expert_i 过去 N 步负载 > 平均:
      bias_i -= ε    (减分)
    else:
      bias_i += ε    (加分)
  
  特点: 不在 Loss 中加项 → 不影响梯度计算
```

### 3.5 DeepSeek-V3 完整配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 总参数量 | 671B | 全部参数 |
| 激活参数量/token | 37B | 每次计算实际使用的参数 |
| 层数 | 61 | Transformer Decoder 层 |
| d_model | 7168 | 隐藏层维度 |
| d_ff (MoE) | 2048 × 256 | 每个专家 2048 中间维度 |
| 注意力头数 | 128 (h) × 1 (kv) | MQA 级别（但用 MLA 保持表达力） |
| d_latent (MLA) | 512 | KV 潜空间维度 |
| d_head | 128 | 每个头的维度 |
| 序列长度 | 128K | 训练上下文窗口 |
| MoE 专家数 | 256 细粒度 + 1 共享 | 总 257 个 FFN |
| 激活专家/token | 8 (细粒度) + 1 (共享) | 共 9 个 |
| Top-k 路由 | 8 | 稀疏率: 256/8 = 32:1 |
| MTP 深度 D | 1 (默认) / 可配置 | Multi-Token Prediction |
| 激活函数 | SwiGLU | — |
| 位置编码 | RoPE | — |
| LayerNorm | RMSNorm | Pre-Norm |
| 训练精度 | FP8 混合精度 | 首次在 671B MoE 级别成功 |
| 词表大小 | ~128K | — |
| 优化器 | AdamW | — |
| 训练 tokens | 14.8T | 与 V2 同样数据规模 |
| 训练硬件 | 2,048 × H800 | NVIDIA H800 GPU |
| 训练时间 | 2.788M H800 hours | 约 2 个月（2048 卡并行） |
| 训练成本 | **$5.57M** | 业界最低的千亿级训练成本 |
| 许可证 | MIT | 完全开源 |

### 3.6 性能对比

| 基准测试 | DeepSeek-V3 | LLaMA 3.1 405B | GPT-4 (2024) |
|----------|-------------|----------------|--------------|
| MMLU | 87.1 | 87.5 | 86.4 |
| MMLU (CoT) | 89.1 | 89.0 | 88.5 |
| HumanEval | 79.4 | 84.2 | 87.7 |
| GSM8K | 90.6 | 89.5 | 92.0 |
| MATH | 56.3 | 50.7 | 59.3 |
| 训练成本 | **$5.57M** | >$30M (估) | >$100M (估) |
| 激活参数 | 37B | 405B | ~300B (估) |
| 总参数 | 671B | 405B | ~1.8T (估) |
| 许可证 | MIT | 受限 | 闭源 |

V3 的结论很清晰：**在 1/5 的 LLaMA 3.1 训练成本和 1/20 的 GPT-4 训练成本下，达到了接近或超过它们的性能。**

### 3.7 推理流程——MLA 的 KV Cache 优势

```
V3 推理（生成一个 token 的过程）:

输入: "中国的首都是"
    │
  1. Token Embedding → [1, 7168]
    │
  2. 逐层计算 (61 层):
     ┌──────────────────────────────────────┐
     │  Layer i:                            │
     │    a) RoPE + 计算 Q                  │
     │    b) 从潜空间解压 K, V:              │
     │       c_KV = cache[i]                │
     │       K, V = W_up · c_KV             │
     │       → [seq_len, d_model]           │
     │    c) MLA Attention: Q · Kᵀ · V      │
     │    d) 更新 cache[i] = 新的 c_KV      │
     │       (只缓存潜空间向量, 不是完整K,V) │
     │    e) MoE FFN: 路由→8个专家→求和     │
     │    f) Residual + RMSNorm             │
     └──────────────────────────────────────┘
    │
  3. LM Head: logits → softmax → 输出 token
    │
  输出: "北京"

关键: 每一步的 KV Cache 只有 d_latent = 512
      对比同样参数量的 dense 模型:
        dense 4096 维 → KV Cache 大 8 倍
        批量 100 序列 × 128K 上下文
        → MLA: ~6.5 GB vs Dense: ~52 GB
```

---

## 第四章 DeepSeek-R1（2025.01）：GRPO + 纯 RL 推理涌现

### 4.1 推理能力的三条路线

在大语言模型中，推理能力（reasoning）——数学证明、逻辑推理、代码生成的逐步推导——怎么来的？

```
路线 1: SFT + RL
  代表: OpenAI o1
  SFT 数据（人工标注的高质量 CoT）→ 然后 PPO RLHF 优化
  问题: SFT 数据标注成本高, 依赖人类示范质量

路线 2: Pure RL → Zero
  代表: DeepSeek-R1-Zero
  完全不依赖 SFT 数据 → 只通过 RL 让模型"自学"推理
  问题: 模型学会了推理, 但输出可读性差（中英文混杂, 格式乱）

路线 3: Cold-Start + RL
  代表: DeepSeek-R1
  少量 SFT (cold-start data) → RL → 再 SFT + RL
  结合了路线 1 的稳定性和路线 2 的探索性
```

### 4.2 GRPO（Group Relative Policy Optimization）

**GRPO 是 R1 最核心的技术创新**。要理解 GRPO，先看标准的 RLHF 流程：

```
PPO RLHF (标准, OpenAI):
                                    ┌──────────────┐
                                    │  Reward Model │
                                    │  (RM)         │
                                    └──────┬───────┘
                                           │ 给每个输出打分
  ┌──────────┐  选择动作   ┌──────────┐   │
  │ Policy   │───────────►│ 输出     ├───┘
  │ (Actor)  │            │ token    │
  └──────────┘            └──────────┘
       │ ▲                    │
       │ │                    │
       │ └──── 状态更新 ──────┘
       │
       │  ┌──────────────┐
       │  │ Critic (Value)│  ← 评估 "这一步有多好"
       │  │ = 另一个模型  │      (需要和 Actor 一样大)
       │  └──────────────┘
       ▼
  Advantage = Reward - Value
  用 Advantage 更新 Policy

  问题: Critic 模型 ≈ Actor 一样大
         → 显存翻倍, 训练更不稳定
         → 训一个 671B 模型需要额外 671B 的 Critic
```

**GRPO 的改法：不要 Critic。**

```
GRPO (DeepSeek-R1):

  对同一个 prompt, 政策模型生成 G 个输出:
    {output_1, output_2, ..., output_G}
    ├── 每个 output 由 Reward Model 打分
    ├── 计算组内均值 μ = mean(reward_1, ..., reward_G)
    ├── 计算组内标准差 σ = std(reward_1, ..., reward_G)
    └── 每个输出的 Advantage:
        A_i = (reward_i - μ) / σ    ← 组内归一化的相对分数

  更新 Policy:
    Loss = -1/G · Σ[min(π_θ/π_old · A_i, clip(π_θ/π_old, 1-ε, 1+ε) · A_i)]
    
  关键: 没有 Value Model, 不需要 Critic
        Advantage 来自组内相对比较
```

**PPO vs GRPO 对比：**

```
PPO (标准):
                   ┌─────────────────┐
  Prompt → Policy  │  Policy Model   │ ← 需要更新
                   └────────┬────────┘
                            │ 输出 G 个样本
                   ┌────────▼────────┐
                   │  Reward Model   │ ← 给每个输出打分
                   └────────┬────────┘
                   ┌────────▼────────┐
                   │  Critic/Value   │ ← 也需要和 Policy 一样大
                   │  Model          │    显存占用翻倍
                   └────────┬────────┘
                   ┌────────▼────────┐
                   │  Advantage      │ = Reward - Value
                   │  (每个样本独立)  │
                   └─────────────────┘

GRPO (DeepSeek):
                   ┌─────────────────┐
  Prompt → Policy  │  Policy Model   │ ← 只需要更新这个
                   └────────┬────────┘
                            │ 输出 G 个样本
                   ┌────────▼────────┐
                   │  Reward Model   │ ← 给每个输出打分
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │  组内归一化      │ ← 不需要 Critic
                   │  A_i = (r_i-μ)/σ│    组内相对比较替代绝对价值
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │  Update Policy  │
                   │  (基于 Advantage)│
                   └─────────────────┘

  显存需求: PPO 需要 3 个模型 (Policy + Critic + RM)
           GRPO 只需要 2 个 (Policy + RM)
           → 对 671B 模型, GRPO 节省约 50% 的显存
```

### 4.3 DeepSeek-R1-Zero：纯 RL 训练

R1-Zero 的实验在学术上非常重要：**它证明推理能力可以在没有人类标注数据的情况下，仅通过 RL 从模型中涌现出来。**

```
R1-Zero 训练流程:

  Step 1: 选择一个推理基准（如 MATH, GSM8K, AIME）
  Step 2: 对原始模型（DeepSeek-V3-Base）采样 G 个输出
  Step 3: GRPO 更新 → 基于组内相对表现
  Step 4: 重复 Step 2-3

  训练初期: 输出的 CoT 短, 推理步骤简单
  训练中期: 模型自发学会更长的推理链
  训练后期: 出现"DeepSeek moment"

  "DeepSeek moment":
    在训练过程中, 模型在没有被明确教导的情况下,
    自发学会了在推理时"重新审视和纠正自己前面的错误"。
    表现为生成的 CoT 中出现类似:
      "Wait, let me reconsider..."
      "Actually, that's not right because..."
  
  这证明了: 反思和纠错能力可以从 RL 的奖励信号中
  自组织涌现, 而不需要人类教导模型"怎么反思"。
```

**R1-Zero 的问题**：输出不好读。模型可能中英文混杂、没有清晰格式、思维链杂乱——RL 奖励只关心最终答案对不对，不关心中间过程好不好看。

### 4.4 DeepSeek-R1：Cold-Start + 多阶段训练

R1 在 R1-Zero 的基础上，加入了少量的 SFT 冷启动数据来解决格式问题：

```
DeepSeek-R1 训练管线:

  Stage 1: Cold-Start SFT
    ┌──────────────────────────────────────────┐
    │  数据: ~数千条高质量 CoT 样例             │
    │  (人工标注: 清晰的推理步骤 + 最终答案)     │
    │  目的: 让模型学会"输出格式"               │
    └────────────────┬─────────────────────────┘
                     ▼
  Stage 2: Reasoning RL (GRPO)
    ┌──────────────────────────────────────────┐
    │  在数学、编程、逻辑推理任务上做 RL        │
    │  奖励信号: 答案正确性 + 格式合规奖励     │
    │  格式奖励: 是否用中文/英文统一格式       │
    │  (不惩罚内容, 只惩罚格式混乱)            │
    └────────────────┬─────────────────────────┘
                     ▼
  Stage 3: Rejection Sampling + SFT
    ┌──────────────────────────────────────────┐
    │  用当前 policy 生成大量推理轨迹           │
    │  → 用 Reward Model 筛选正确/好的轨迹      │
    │  → 用这些轨迹再做一次 SFT                 │
    │  (相当于"用 RL 后好的输出教自己")         │
    └────────────────┬─────────────────────────┘
                     ▼
  Stage 4: 全面 RL
    ┌──────────────────────────────────────────┐
    │  在全部任务上做 RL:                       │
    │    - 推理 (数学、编程、逻辑)              │
    │    - 通用对话偏好对齐 (Helpfulness)       │
    │  奖励: 融合推理正确性 + 对话质量          │
    │  (多任务 RL 联合训练)                    │
    └────────────────┬─────────────────────────┘
                     ▼
  Final: DeepSeek-R1
```

### 4.5 蒸馏：让推理能力"传播"

R1 的一个重要贡献：**把大模型的推理能力蒸馏到小模型**。使用 R1 生成的 CoT 数据（600K+ 高质量推理轨迹）来训练小模型：

```
蒸馏流程:

  DeepSeek-R1 (671B, Teacher)
        │  对推理任务采样 → 生成 600K+ CoT 轨迹
        │  每个轨迹: 中间推理过程 + 最终答案
        ▼
  ┌────────────────────────────────────┐
  │     小模型的 SFT 训练              │
  │                                    │
  │  基于 R1 输出数据, 训练:           │
  │    ├── DeepSeek-Distill-Qwen-7B    │
  │    ├── DeepSeek-Distill-Qwen-32B   │
  │    ├── DeepSeek-Distill-LLaMA-8B   │
  │    └── DeepSeek-Distill-LLaMA-70B  │
  │                                    │
  │  注意: 这里只用 SFT, 不做 RL      │
  │  但小模型学到了大模型的推理模式    │
  └────────────────────────────────────┘

蒸馏效果:

  模型                   | AIME 2024 | MATH-500
  ───────────────────────|───────────|─────────
  DeepSeek-R1 (671B)     |   79.8    |   97.3
  DeepSeek-Distill-Qwen-32B | 72.6 |   94.8
  DeepSeek-Distill-LLaMA-70B | 70.2 | 95.1
  OpenAI o1-mini         |   63.6    |   90.0

  注意: 32B 的蒸馏模型在 AIME 上超过了 o1-mini!
```

**蒸馏的价值**：
- 小模型可以在本地或低成本设备上运行
- 蒸馏比 RL 更高效（一次推理即可获取训练数据）
- 证明了推理能力的可移植性

### 4.6 R1 的完整配置

| 参数 | DeepSeek-R1 | DeepSeek-R1-Zero |
|------|-------------|-------------------|
| 基座模型 | DeepSeek-V3-Base | DeepSeek-V3-Base |
| 总参数量 | 671B | 671B |
| 激活参数量 | 37B | 37B |
| 训练方法 | Cold-start SFT + 多阶段 RL | 纯 RL (GRPO) |
| RL 算法 | GRPO | GRPO |
| 组大小 G | 64 | 64 |
| 奖励模型 | 推理正确性 + 格式 + 对话偏好 | 推理正确性 |
| Cold-start 数据 | ~数千条 | 无 |
| SFT 数据量 | ~800K (含蒸馏数据) | 0 |
| 训练 tokens | — | — |
| 许可证 | MIT | MIT |
| 发布时间 | 2025.01 | 2025.01 |

---

## 第五章 DeepSeek-V4 / V4.1（2026.04 / 2026.06）：昇腾原生 + 多模态 + Agent

### 5.1 V4：昇腾原生旗舰

### 5.1 技术独立——零 NVIDIA 依赖

V4 是最重要的战略转折：**完全基于华为昇腾 910C 芯片训练**，不依赖任何 NVIDIA GPU。

```
DeepSeek-V4 → "国产化完全体"

**发布时间**：2026 年 4 月 24 日（V4 预览版）

  训练硬件:
    ┌────────────────────────────────────┐
    │  1,000 × 华为昇腾 Ascend 910C      │
    │                                    │
    │  单个 910C: 约等于 A100 (FP16)    │
    │  HBM: 64GB/chip                    │
    │  互联: HCCS (华为自研高速互连)    │
    │                                    │
    │  对比 V3: 2,048 × H800             │
    │  V4 用一半的卡, 但算力持平         │
    └────────────────────────────────────┘

  这意味着:
    - 完全不受美国芯片出口管制影响
    - 自主可控的 AI 训练基础设施
    - 为国产芯片的大模型训练树立了标杆
```

### 5.2 1M Context Window

V4 将上下文窗口扩展到 1,048,576 (2²⁰) tokens——约等于 75 万汉字或 3 本《三体》的文本量。

```
1M 上下文的技术挑战:

  1. Attention 计算复杂度: O(n²) → 1M² = 1e12 次操作/层
     → MLA 的 KV Cache 优势在这里被推到了极限
     → d_latent = 512 → 1M seq_len 的 KV Cache:
       ~512 × 1M × 2 bytes = 1 GB / 层 × 60 层 = 60 GB
       （对比 MHA 需要约 500 GB / 层——不可行）

  2. RoPE 的外推: 训练时的 RoPE 能否 extrapolate 到更长?
     → DeepSeek 改进了 RoPE 实现（可能结合 YaRN 或 NTK-aware scaling）

  3. 显存管理: 1M tokens 的前向激活≈几百GB
     → 需要显存卸载 / 计算优化（FlashAttention 级别的优化）

  4. 数据: 包含超长序列的预训练数据
     → 代码库、书籍、长文档
```

### 5.3 Agent-Native 能力

V4 被描述为"Agent-native"——模型在架构层面被设计为可以直接调用工具、执行计划、与环境交互，而不需要外部 agent 框架：

```
Agent-Native 架构特征:

  工具调用能力:
    - 原生支持 Function Calling output
    - 输出格式直接映射到 API 调用
    - 不需要 JSON 约束解码

  长程规划:
    - 1M 上下文 → 可以维护完整的"任务记忆"
    - 在上下文内跟踪多个子任务的执行状态
    - 支持思维链 + 工具调用的交织

  自我反思:
    - 继承了 R1 的反思能力
    - 工具调用失败时自动重试/修正

  多模态 (预估):
    - 支持图像输入 + 文本推理
    - （细节待 DeepSeek 正式发布确认）
```

### 5.4 V4 配置（基于公开信息）

| 参数 | 值 | 说明 |
|------|-----|------|
| 总参数量 | — (未公开) | 预计大于 V3 |
| 激活参数量 | — (未公开) | — |
| 训练硬件 | 1,000 × 华为昇腾 910C | 100% 国产化 |
| 上下文窗口 | 1,048,576 | 1M tokens |
| Agent 能力 | Native | 原生支持工具调用 |
| 训练精度 | FP8 (昇腾原生支持) | 延续 V3 策略 |
| 许可证 | MIT | — |
| 发布时间 | 2026.04 (V4) / 2026.06 (V4.1 灰度) | — |

### 5.5 V4.1：多模态 + MCP（2026.06 灰度）

2026 年 6 月，DeepSeek 发布 V4.1 灰度版本，在 V4 基础上补齐了三项关键能力：

| 升级点 | 说明 |
|--------|------|
| **全模态输入** | 在文本基础上新增**图像和音频**理解能力（输出仍为文本）|
| **MCP 协议原生支持** | 原生适配 Model Context Protocol，无缝连接外部工具、数据库、CRM/ERP 系统 |
| **企业级工具链** | 模型微调、私有化部署、安全审计等企业功能 |

V4.1 验证了 DeepSeek 从"技术展示"全面转向"商业落地"的战略方向。

### 5.6 融资

2026 年 5 月，DeepSeek 启动首轮融资，目标 **500 亿元人民币**。创始人梁文锋个人出资 **200 亿元**，国家集成电路产业投资基金参与，腾讯拟出资约 60 亿元。投后估值突破 **3,500 亿元（约 515 亿美元）**。

---

## 第六章 训练数据与训练过程

### 6.1 预训练数据管线

DeepSeek 系列的数据策略在不同版本间有延续性：

```
数据管线 (V2/V3):

  原始数据源                   14.8T tokens (高质量)
    │
    ├── 网页抓取 (CommonCrawl 等)  ~60%
    ├── 书籍 (电子书, 学术论文)    ~20%
    ├── 代码 (GitHub, 技术文档)     ~15%
    └── 数学/科学 (arXiv, 习题集)  ~5%
    │
    ▼
  数据去重 (MinHash, Exact Dedup)
    ├── 文档级去重
    ├── 段落级去重
    └── 去重后: ~14.8T tokens
    │
    ▼
  数据过滤
    ├── 质量过滤 (基于分类器: 困惑度、语言检测)
    ├── 毒性/安全过滤
    └── 隐私过滤 (PII 去除)
    │
    ▼
  Tokenization (BPE, 词表 ~128K)
    │
    ▼
  训练数据 (已经是 tokens)
```

### 6.2 训练过程（V3 示例）

```
DeepSeek-V3 训练过程:

  Phase 1: 预训练 (从头训练)
    数据: 14.8T tokens
    精度: FP8 混合精度
    硬件: 2,048 × H800
    时长: 2.788M H800 hours (~2 月)
    成本: $5.57M
    关键: FP8 训练 + 负载均衡无辅助损失

  Phase 2: 后训练 (对齐)
    Step 1: SFT (监督微调)
      - ~800K 指令数据
      - 包括数学、代码、通用对话
    Step 2: RLHF (合并可能)
      - DeepSeek 对 RLHF 的细节披露较少
      - 已知包含数学和编程的强化学习（为 R1 铺路）
    Step 3: 安全对齐
      - 拒绝不当请求
      - 内容安全过滤

  Phase 3 (R1 专属): RL for Reasoning
    参考第四章
```

### 6.3 训练成本对比

| 模型 | 训练成本 | GPU 小时 | 数据 tokens | 每万亿 tokens 成本 |
|------|---------|----------|-------------|-------------------|
| DeepSeek-V2 | ~$7M | ~3.5M (估) | 14.8T | ~$0.47M |
| DeepSeek-V3 | **$5.57M** | **2.788M** | **14.8T** | **$0.38M** |
| LLaMA 3 405B | >$30M (估) | ~10M+ (估) | 15T+ | >$2M |
| GPT-4 | >$100M (估) | ~50M+ (估) | — | — |

DeepSeek-V3 的训练成本不到 LLaMA 3 405B 的 1/5，不到 GPT-4 的 1/20。**每万亿 tokens 的成本只有 LLaMA 3 的 1/5 左右。**

---

## 第七章 一句话总结

```
DeepSeek 的工程哲学:

  传统 AI:
    "更大 = 更好"
    GPT-4: >$100M → 性能 SOTA

  DeepSeek 的答案:
    "更聪明地大 = 更好"
    V3: $5.57M → 性能接近 GPT-4

  技术复利效应:
    MLA (更小的 KV Cache)
    → MoE (更小的激活参数)
    → FP8 (更小的计算精度)
    → GRPO (更少的训练模型)
    → 每项创新单独看是"小优化"
    → 叠加起来 = 20 倍的成本降低
```

核心贡献清单：

| 创新 | 解决的问题 | 影响范围 |
|------|-----------|----------|
| **MLA** | KV Cache 过大 | 所有长上下文 LLM |
| **DeepSeekMoE** | 专家粒度过粗 | MoE 架构设计 |
| **FP8 大规模训练** | 训练成本过高 | 大规模预训练 |
| **Multi-Token Prediction** | 表示学习效率+推理加速 | LLM 训练目标 |
| **Auxiliary-Free Load Balance** | MoE 负载均衡 | MoE 训练稳定性 |
| **GRPO** | RL 训练成本高 | RLHF 训练流程 |
| **R1 蒸馏** | 推理能力传播 | 小模型推理 |

---

## 第八章 影响：DeepSeek Moment

### 8.1 2025 年 1 月——App Store 榜首 + 全球股灾

2025 年 1 月，DeepSeek-R1 发布，在全球范围内引发了称为 **"DeepSeek Moment"** 的震动：

```
2025.01 事件时间线:

  Day 1: DeepSeek-R1 发布
         → 技术报告公开 + MIT 开源
         → 模型权重可下载

  Day 2-3: 口碑传播
         → 开发者社区发现 R1 在数学和编程上
           超过了 OpenAI o1-preview
         → 免费 + 开源 vs OpenAI 付费 API

  Day 7: DeepSeek App 登顶美国 App Store 榜首
         → 超过 ChatGPT 成为下载量第一的 AI 应用
         → 来自中国的 AI 应用首次登顶美国 App Store

  Financial Impact:
         → NVIDIA 股价单日暴跌 17%
         → 全球科技股市值蒸发 ~$1 万亿
         → 市场恐慌: "如果不需要那么多 GPU 就能做出
           顶级模型, 那 GPU 需求会不会下降?"
         → 后续: 市场反弹, 但"低成本高性能"的叙事
           已在投资者心中扎根
```

### 8.2 改写行业规则

DeepSeek 对整个 AI 行业的影响是结构性的：

| 维度 | 之前 | 之后 |
|------|------|------|
| **训练成本认知** | "做一个 GPT-4 需要 >$100M" | "做一个 GPT-4 级别模型可以 <$10M" |
| **开源 vs 闭源** | 顶级模型是闭源的 | MIT 开源的模型达到闭源水平 |
| **中国 AI 地位** | "追随者" (复制美国模型设计) | "创新者" (MLA, GRPO, FP8 MoE 都是原创) |
| **芯片依赖** | 训练只能 NVIDIA | 昇腾证明可以训练 600B+ 级别模型 |
| **API 定价** | OpenAI GPT-4: $30/百万 tokens | 被迫降价 (DeepSeek 让"每 token 定价"大幅下降) |

### 8.3 对中国 AI 的意义

DeepSeek 系列证明了三件事：

1. **中国 AI 实验室可以做原创创新**。MLA 不是对美国架构的改进——它是一个全新的注意力范式。GRPO 不是对 PPO 的调整——它是一个去掉了 Critic 的全新 RL 框架。
2. **芯片封锁可以绕开**。V4 在昇腾上的成功证明，即使被限制使用 NVIDIA 最新芯片，仍然可以训练出世界级模型。
3. **开源可以打败闭源**。MIT 许可证意味着任何企业都可以用 DeepSeek 搭建自己的 AI 系统——这对商用闭源模型构成了巨大的竞争压力。

### 8.4 对研究社区的影响

DeepSeek 的论文风格也为研究社区树立了一个值得注意的标杆：**每一篇论文都公开了足够多的技术细节，让其他团队可以复现**。MLA 的实现、FP8 训练的 block-wise scaling 策略、GRPO 的组内归一化——这些不仅是声称，而是有充分的训练配置和消融实验支持的。

---

**Sources:**

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3: A Contender in the Age of GPT-4 (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)
- [DeepSeek-V4: Ascend-Native Language Model (DeepSeek Official Blog, 2026)](https://www.deepseek.com)
- [MLA: The Hidden Innovation Behind DeepSeek-V2's Efficiency (vLLM Blog)](https://blog.vllm.ai)
- [DeepSeek-V2 技术报告解读——MLA 和 DeepSeekMoE (知乎专栏)](https://zhuanlan.zhihu.com)
- [Training DeepSeek-V3 with FP8: A Practical Guide (DeepSeek Technical Blog)](https://www.deepseek.com)
- [GRPO: Group Relative Policy Optimization Explained (Anthropic / Community Analysis)](https://www.anthropic.com)
- [DeepSeek-R1: Technical Deep Dive — Distillation and Cold-Start (Hugging Face Blog)](https://huggingface.co/blog)
- [Gloeckle et al. Better & Faster Large Language Models via Multi-Token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737)
- [LLaMA: Open and Efficient Foundation Language Models (arXiv:2302.13971)](https://arxiv.org/abs/2302.13971)
