# GLM-ASR 网络结构深度解剖

> 智谱AI (Zhipu AI) 出品 | 面向真实世界的高鲁棒性语音识别模型

---

## 写在前面：理解两个关键版本

GLM-ASR 系列目前存在**两个主要版本**，本文分析覆盖两者：

| 版本 | 总参数量 | 文本解码器 | 定位 |
|------|---------|-----------|------|
| **GLM-ASR-Nano-2512** | **~1.5B** (15亿) | Qwen3-0.6B 解码器 (或 6 层自研) | 端侧 / 消费级 GPU 部署 |
| **GLM-ASR-2512** (云端) | **~22.6B** (推测) | LLaMA 架构 28 层解码器 | 云端生产部署 |

Nano 版本已开源并部署于**智谱输入法**等产品中，在权威评测中实现 **4.10% 平均错误率**，显著优于 Whisper V3 的 6.93%。

---

## 一、整体架构设计哲学

GLM-ASR 的设计理念高度凝练为九个字：

> **"重感知、极致压缩、轻量推理"**

这套思想落地为一个**非对称 Encoder-Decoder 架构**：

```
原始音频 (16kHz 单声道)
    │
    ├─ ① Mel 频谱前端  ─── 128-bin log-mel 特征
    │
    ├─ ② Conv1d 子采样  ── 2× 时间下采样
    │
    ├─ ③ Audio Encoder  ── 32层 (或 12层) 双向 Transformer
    │
    ├─ ④ Multimodal Projector ── 池化(4×) + 3 层 MLP → 对齐到文本空间
    │
    ├─ ⑤ Text Decoder   ── 28层 (或 6层) 自回归 Transformer
    │
    └─ 输出文本
```

### 为什么是"非对称"？

传统的 ASR 模型（如 Whisper）编码器和解码器深度接近，而 GLM-ASR 将**编码器做得非常深、解码器做得非常浅**。原因在于：

1. **语音识别本质是感知任务**——编码器需要充分提取声学特征、建模时序依赖
2. **语音到文本的映射是"压缩"**——1 秒音频 ≈ 100 帧，但只有约 3-5 个词
3. **解码器只需做语言建模**——任务比通用 LLM 简单得多

> 这与 LLM 领域的"推理时计算"理念一脉相承：把计算投入在理解输入上，而非生成输出上。

---

## 二、各模块深度解剖

### 2.1 卷积特征提取器（Conv1d Subsampler）

**定位**：将 Mel 频谱从声学特征空间投影到 Transformer 隐藏空间，同时完成时间下采样。

```
输入: [batch, 128, T]  (T = 帧数 ≈ 100 × 音频秒数)
    │
    ├─ Conv1d(128 → 1280, kernel=3, stride=1, padding=1)
    │   └─ SiLU / GELU 激活
    │
    ├─ Conv1d(1280 → 1280, kernel=3, stride=2, padding=1)
    │   └─ SiLU / GELU 激活
    │
    └─ GroupNorm(1) 归一化
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `in_channels` | 1（单声道） | 原始波形输入通道 |
| 卷积层数 | 2 | 浅层特征提取 |
| 时间缩减率 | **2×** | stride=2 → 帧数减半 |
| 输出维度 | 1280 | 投影到 Transformer 隐藏空间 |

**设计要点**：Nano 版本使用 GroupNorm(1)（等价于 LayerNorm）+ SiLU，而云端版本使用 GELU 激活。这是针对不同规模模型的激活函数选择差异。

---

### 2.2 Audio Encoder（音频编码器）

这是 GLM-ASR 最核心的模块。对于云端版本，继承并改进了 Whisper V3 的编码器结构。

#### 云端版（GLM-ASR-2512）+ HuggingFace 实现：

| 参数 | 值 | 含义 |
|------|-----|------|
| `hidden_size` | **1280** | 隐藏层维度 |
| `num_hidden_layers` | **32** | Transformer 层数 |
| `num_attention_heads` | **20** | 注意力头数 |
| `head_dim` | **64** | 每头维度 = 1280 / 20 |
| `intermediate_size` | **5120** | FFN 中间层维度 (4×) |
| `hidden_act` | **GELU** | 激活函数 |
| `max_position_embeddings` | **1500** | 最大序列长度 (≈ 15s 音频) |
| `num_mel_bins` | **128** | Mel 滤波器数量 |
| `partial_rotary_factor` | **0.5** | RoPE 仅作用于前 50% 维度 |
| `attention_dropout` | 0.0 | 注意力 dropout 率 |

#### Nano 版：

| 参数 | 值 |
|------|-----|
| `encoder_layers` | **12** |
| `d_model` | **1024** |
| `encoder_attention_heads` | **16** |

#### 每个 Encoder Layer 的内部分解：

```
输入: x [batch, seq_len, 1280]
    │
    ├── LayerNorm
    │
    ├── Multi-Head Self-Attention (双向 / 非因果)
    │   ├── QKV 投影（Q, V 带 bias, K 不带 bias）
    │   ├── RoPE（仅作用于前 32 维，即 head_dim 的 50%）
    │   ├── Flash Attention / SDPA（自动选择后端）
    │   └── Output 投影
    │
    ├── + 残差连接
    │
    ├── LayerNorm
    │
    ├── MLP (FFN)
    │   ├── up_proj: 1280 → 5120
    │   ├── GELU 激活
    │   └── down_proj: 5120 → 1280
    │
    └── + 残差连接
```

#### 关键设计细节

**1. 非因果注意力（Bidirectional / Non-causal）**

这是语音编码器的核心特征——编码器看到**全部上下文**（双向），与语言模型不同。语言模型是因果的，只能看左侧；而语音编码器需要利用**未来帧的信息来更准确理解当前帧**（因为语音具有"后验修正"的特性——听了后面的话，前面的歧义才消除）。

```
传统 ASR 帧对齐:
  "今天天气真的...很冷"
  听到"很冷"之前，"真的"可能是"真的(确实)"或"真的(针织的)"？
  双向编码器能看到全局，歧义自然消除 ← 非因果注意力的优势
```

**2. 部分 RoPE（Partial Rotary Factor = 0.5）**

- 仅对 head_dim 的**前 50%**（32 维）施加旋转位置编码
- 后 50%（32 维）**不编码位置信息**，保留纯粹的语义容量
- **设计动机**：语音特征中"音色、音调、音高"等属性不应与位置绑定，而"时序变化"信息（如共振峰过渡）则需要位置编码来感知

```
head_dim = 64
    ├── 前 32 维: RoPE 旋转编码 → 编码位置信息
    └── 后 32 维: 无位置编码   → 纯语义容量
    
为什么不全量 RoPE？
  如果 64 维全做 RoPE，语音的"内容信息"也会被位置编码纠缠，
  模型难以区分"同样的音高在不同位置" vs "不同音高在相同位置"
```

**3. QKV 投影的 Bias 策略不对称**

- **Q 投影**: 带 bias
- **K 投影**: 不带 bias
- **V 投影**: 带 bias

这是一种经过实验验证的训练稳定性优化。K 投影去掉 bias 可以**避免注意力分数被位置相关的偏置扭曲**，这在 RoPE 场景下尤为关键——因为 RoPE 已经提供了位置信息，K 的额外 bias 反而会引入噪声。

**4. QKVParallelLinear**

支持分布式训练中 QKV 计算的张量并行（Tensor Parallelism），这是针对大规模训练的基础设施优化。在 Megatron-LM 风格的并行策略中，QKV 投影可以在多个 GPU 间切分计算。

---

### 2.3 Multimodal Projector（多模态投影器）

这个模块的功能类似于"跨语言翻译"——将音频编码器的输出**映射到文本解码器的输入空间**。

```
Encoder 输出: [batch, L, 1280]
    │
    ├── Pooling (factor=4, stride=4)
    │   └── 滑动窗口均值池化 → [batch, L/4, 1280]
    │
    ├── Linear: 1280 → 5120
    ├── GELU 激活
    │
    ├── Linear: 5120 → 4096
    ├── GELU 激活
    │
    ├── Linear: 4096 → 2048
    │
    └── 输出: [batch, L/4, 2048]
```

| 设计 | 值 | 作用 |
|------|-----|------|
| Pooling stride | **4** | 再次压缩 4×，送入解码器的 token 数大减 |
| 投影层数 | **3** | 非线性变换，从小升维再降维 |
| 最终维度 | **2048** | 匹配解码器 hidden_size |
| `projector_type` | `"mlp"` | 也可选 `"qformer"` 方案 |
| `projector_hidden_dim` | `null` (自动计算) | 隐藏维度 |
| `audio_token_dropout` | 0.0 | 训练时音频 token dropout 率 |

#### 核心作用：Token 数量压缩

以 10 秒音频为例：

| 阶段 | Token 数 | 压缩比 | 累积压缩 |
|------|---------|--------|---------|
| 原始 Mel 帧 | ~1000 | — | 1× |
| Conv 子采样后 | ~500 | 2× | 2× |
| Encoder 编码后 | ~500 | — (保持) | 2× |
| **Pooling + Projector 后** | **~125** | **4×** | **8×** |
| Decoder 生成的文本 token | ~30-80 | — | — |

> **如果不做 4× Pooling**：500 帧送入解码器，Prefill 阶段的注意力计算量是 O(500²) vs O(125²) = **16 倍差距**，KV Cache 内存占用也是 **4 倍**。

#### 可选架构：Q-Former

ASR Config 中提供了 Q-Former 交替方案：

| 参数 | 值 |
|------|-----|
| `qformer_window_size` | 15 |
| `qformer_hidden_size` | 默认为 `encoder_dim` = 1280 |
| `qformer_num_layers` | 2 |
| `qformer_num_heads` | 16 |
| `qformer_intermediate_size` | 4× hidden |

Q-Former 使用可学习的 Query Token 通过交叉注意力从编码器输出中"提取"信息，比 Pooling 更灵活但计算开销更大。

#### 特殊设计：MoE 投影器

ASR Config 中包含 MoE（混合专家）配置：

```python
num_experts = 4       # 4 个专家
num_experts_per_tok = 2  # 每 token 激活 2 个专家
router_aux_loss_coef = 0.01  # 辅助均衡损失系数
```

这表明投影器层可能支持**稀疏化 MoE**，以在增加模型容量的同时控制计算成本。每个 token 只激活 4 个专家中的 2 个，计算量仅增加 ~2×，但模型容量大幅提升。

---

### 2.4 Text Decoder（文本解码器）

#### 云端版（HuggingFace 默认 LLaMA 架构）：

| 参数 | 值 | 含义 |
|------|-----|------|
| `hidden_size` | **2048** | 隐藏层维度 |
| `num_hidden_layers` | **28** | Transformer 层数 |
| `num_attention_heads` (Q) | **16** | Query 头数 |
| `num_key_value_heads` (KV) | **4** | Key/Value 头数 (GQA 16:4) |
| `head_dim` | **128** | 每头维度 = 2048 / 16 |
| `intermediate_size` | **6144** | SwiGLU FFN 中间维度 (3×) |
| `vocab_size` | **59,264** | 词汇表大小 |
| `max_position_embeddings` | **8192** | 最大序列长度 |
| `attention_type` | **GQA** | Grouped Query Attention (16Q / 4KV) |
| `hidden_act` | **SiLU** | SwiGLU 激活函数 |
| `norm_type` | **RMSNorm** | 归一化类型 |
| `rope_theta` | 10,000 | RoPE 基础频率 |
| `eos_token_id` | [59246, 59253, 59255] | 终止符 ID 列表 |

#### Nano 版（自研 6 层解码器）：

| 参数 | 值 |
|------|-----|
| `decoder_layers` | **6** |
| `d_model` | **1024** |
| `decoder_attention_heads` | **16** |

#### 每个 Decoder Layer 的内部分解：

```
输入: [batch, seq_len, 2048]
    │
    ├── RMSNorm
    │
    ├── Masked Self-Attention (因果/单向)
    │   ├── Q 投影 (16 头, 每头 128 维)
    │   ├── K 投影 (4 头)      ← GQA: 参数减少 75%
    │   ├── V 投影 (4 头)      ← KV Cache 减少 75%
    │   ├── RoPE (全量 128 维)
    │   ├── Flash Attention + 因果掩码
    │   └── Output 投影: 2048 → 2048
    │
    ├── + 残差连接
    │
    ├── RMSNorm
    │
    ├── Cross-Attention (从 Encoder 读取声学信息)
    │   ├── Q: 来自解码器当前层 (16 头)
    │   ├── K, V: 来自 Encoder 最后输出 (20 头 → 通过投影对齐到 16 头)
    │   ├── Flash Attention (无掩码, 双向可见)
    │   └── Output 投影
    │
    ├── + 残差连接
    │
    ├── RMSNorm
    │
    ├── SwiGLU MLP
    │   ├── gate_proj: 2048 → 6144  (SiLU 激活)
    │   ├── up_proj:   2048 → 6144
    │   ├── 逐元素乘法: gate_output × up_output
    │   └── down_proj: 6144 → 2048
    │
    └── + 残差连接
```

#### GQA（Grouped Query Attention）深度分析

```
标准多头注意力 (MHA):
  Q: 16 组, K: 16 组, V: 16 组
  → 每组 Q 有自己独立的 K, V
  → KV Cache 大小 = 16 × head_dim × seq_len

分组查询注意力 (GQA), ratio = 16:4:
  Q: 16 组, K: 4 组, V: 4 组
  → 每 4 组 Q 共享 1 组 K, V
  → KV Cache 大小 = 4 × head_dim × seq_len

收益: KV Cache 降到原来的 4/16 = 25%
代价: 注意力表达的多样性略有下降（但实验证明影响极小）
```

**为什么 GQA 对 ASR 特别重要？** 自回归阶段占总推理时间的 **~94%**。每一步都需要读取完整的 KV Cache。GQA 直接从两个方面加速：
1. KV Cache 显存降为 1/4 → 可以放更大的 batch
2. 每步读取 KV Cache 的带宽降为 1/4 → 推理延迟降低

#### Cross-Attention 的设计意图

Cross-Attention 将 Encoder 的输出**交叉连接到 Decoder 的每一层**。这意味着：

- 即便在解码第 100 个 token 时，解码器仍然可以直接"回头"查看编码器的原始音频特征
- 不需要把所有信息都存在自注意力的 KV Cache 中
- 这对 ASR 至关重要——当生成结果出现歧义时，解码器可以"重新听一遍"

```
解码器生成:"今天天气真..."
  └── Attention 权重集中在音频对应 "今天天气" 的部分
解码器生成:"今天天气真的很冷"
  └── Attention 回退到"真的很冷"的音频段，确认语气
```

#### 对比：Cross-Attention vs Decoder-Only

| 维度 | Encoder-Decoder w/ Cross-Attn | Decoder-Only |
|------|-------------------------------|-------------|
| 声学信息访问 | 每层直接访问编码器输出 | 只能通过开始几层的 KV Cache 传播 |
| 长音频处理 | 500 帧编码后保持完好 | 1500+ token 后信息衰减 |
| 生成长度影响 | Token 数增长不影响声学访问 | Token 数越多，早期声学信息越模糊 |
| 模型复杂度 | 多一个 Cross-Attn 模块 | 架构更简单 |

**结论**：Cross-Attention 对于 ASR 这种"声学精度优先"的任务，在中等模型规模下是更优选择。

---

### 2.5 Tokenizer 与词汇空间

| 特征 | 值 (云端) | 值 (Nano) |
|------|----------|-----------|
| `vocab_size` | **59,264** | 51,865 |
| Tokenizer 类型 | SentencePiece / BPE | SentencePiece |
| 语言控制 token | `<\|zh\|>`, `<\|yue\|>`, `<\|en\|>` | 同左 |
| 时间戳 token | `<\|0.00\|>` ~ `<\|30.00\|>` | 同左 |
| 控制 token | `<\|startoftranscript\|>`, `<\|endoftranscript\|>` | 同左 |
| 音频 token ID | 59260 (音频段标识) | — |

#### 词汇空间的语义结构

```
全局词汇表 (~59K):
  ├── 中文字符（常用汉字全覆盖）
  ├── 拼音片段（支持拼音输入场景）
  ├── 英文子词（BPE 切分）
  ├── 特殊控制 token (~20):
  │   ├── 语言指示器: <|zh|>, <|yue|>, <|en|>
  │   ├── 时间戳: <|0.00|> ~ <|30.00|> (300 个)
  │   ├── 转录控制: <|startoftranscript|>, <|endoftranscript|>
  │   └── 音频标识: token_id = 59260
  └── 其他语言子词
```

**设计亮点**：
- **语言控制 token** 允许推理时**显式指定识别语言**，避免语言混淆
- **时间戳 token** 支持**词级时间对齐输出**，这是从 Whisper 继承并改进的设计
- **统一编码空间**：中文、拼音、英文子词共享同一词汇表，简化多模型管理

---

## 三、推理流程演练

以 10 秒中文语音 "今天天气真的很冷" 为例：

### Stage 1: 前端特征提取

```
10s 音频 @16kHz → 160,000 采样点
    → 25ms 窗口 / 10ms 步长 → ~1000 帧
    → 128-bin Mel 频谱 → [1, 128, 1000]
```

### Stage 2: Conv 子采样

```
[1, 128, 1000] → Conv1d × 2 → [1, 1280, 500]
时间维度从 1000 压缩到 500 (2×)
```

### Stage 3: Audio Encoder 编码

```
[1, 1280, 500] → 32 层双向 Transformer → [1, 500, 1280]
每帧感知了完整上下文，输出丰富的声学表示
```

### Stage 4: Projector 映射

```
[1, 500, 1280] → Pooling(4×) → [1, 125, 1280]
    → MLP × 3  → [1, 125, 2048]
时间从 500 压缩到 125 (再压缩 4×)，维度升到 2048
```

### Stage 5: Prompt 构建

```
文本前缀: [<|startoftranscript|>, <|zh|>, <|transcribe|>]
音频段:   [audio_token_id=59260, 125 个音频 embedding]
文本后缀: [<|endoftranscript|>]

完整序列: = 约 130 token → 进入解码器
```

### Stage 6: Prefill（一次填充）

```
一次前向计算完整 130 token 的 KV Cache：
  - 自注意力 KV: 130 × 4 × 128 × 28 层
  - 交叉注意力 KV: 125 × 20 × 64 × 28 层  
```

### Stage 7: 自回归解码

```
Step 1:  输入 <s> → 输出 "今"   (查交叉注意力中的音频段)
Step 2:  输入 "今" → 输出 "天"
Step 3:  输入 "天" → 输出 "天"
  ...
Step 10: 输入 "很" → 输出 "冷"
Step 11: 输入 "冷" → 输出 <|endoftranscript|>  → 终止
```

---

## 四、性能优化全景

### 4.1 推理瓶颈分析

```
总推理时间分布 (以 10s 音频为例):
    ├── 音频编码 (Stages 1-3):   ~5%
    ├── 投影 (Stage 4):           ~1%
    ├── KV Cache 填充 (Stage 5-6): ~0% (一次前向)
    └── 自回归解码 (Stage 7):    ~94%  ★ 核心瓶颈
```

### 4.2 优化策略矩阵

| 优化手段 | 解决的问题 | 原理 | 量化效果 |
|----------|-----------|------|---------|
| **GQA (16:4)** | 每步需读写完整 KV Cache | 4 组 Q 共享 1 组 K,V | KV Cache 降为 25% |
| **4× Pooling** | 送入解码器的 token 数过多 | 均值池化，4 帧→1 token | 自回归步数降为 1/4 |
| **Flash Attention** | HBM 与 SRAM 间的带宽瓶颈 | 分块计算 + 在线 Softmax | 单 kernel 完成全部 attention |
| **RMSNorm + Linear 融合** | Norm → Linear 间中间结果读写 HBM | 合并为单算子 | 消除 HBM 中间读写 |
| **SwiGLU 融合** | Gate/Up → SiLU → 乘 → Down 四步 | 合并为单 kernel | MLP 延迟降低 47% |
| **部分 RoPE** | 全量 RoPE 扭曲语义信息 | 仅 50% 维度编码位置 | 精度提升 + 计算减少 |
| **Tile Size 自动调优** | 不同 GPU 最优分块不同 | 自动搜索最优 tile | H200 上 32×128 最优 |
| **投影器 MoE** | 增加容量控制计算 | 4 专家 2 激活稀疏路由 | 容量 2× 计算 1× |
| **INT8 / FP16 量化** | 模型 4.5GB 过大 | 权重压缩 | 显存减半，速度 x2 |

这些优化共同实现了 **72.2% 的端到端推理加速**（基于 Triton 优化版本）。

### 4.3 ASR 推理超参数

```python
max_new_tokens = 128        # 最大生成长度（ASR 输出通常 <50 token）
num_beams = 5               # 束搜索宽度（离线场景提高精度）
use_cache = True            # 启用 KV Cache
do_sample = False           # 贪婪解码（ASR 场景确定精度优先）
repetition_penalty = 1.0    # 不惩罚重复（ASR 允许自然重复词）
temperature = None          # 温度参数关闭（贪婪解码）
length_penalty = 1.0        # 长度惩罚系数
no_repeat_ngram_size = 0    # 不约束 ngram 重复
```

#### 推理时注入系统提示

```python
# 可选系统提示（默认关闭）
system_prompt = "You are a helpful assistant."
# 如果使用，会在音频 token 前插入作为语义引导
```

---

## 五、架构设计的深层思考

### 5.1 为什么是 Encoder-Decoder 而非 Decoder-Only？

相比 GPT-4o、Qwen2-Audio 等直接"音频令牌化 + LLM"的 Decoder-only 方案，GLM-ASR 选择经典的 Encoder-Decoder 架构有其深层原因：

| 维度 | Encoder-Decoder (GLM-ASR) | Decoder-Only |
|------|--------------------------|-------------|
| 长序列处理 | 编码器双向可见，无长度焦虑 | 因果注意力，长序列信息衰减 |
| 流式能力 | 编码器非因果 → **不支持**流式 | 因果 → 天然支持流式 |
| 解码效率 | 先编码再解码 → 2 阶段 | 单阶段推理 |
| 对齐质量 | 交叉注意力直接访问声学特征 | 声学信息需经 LLM 注意力传播 |
| ASR 精度 | ✅ **更高** | 依赖模型规模 |
| 通用性 | 仅做 ASR/语音任务 | 可同时做文本生成、对话 |

GLM-ASR 选择 Encoder-Decoder 的本质判断是：**在 ASR 任务上，精度优先于流式能力。** 这与其目标场景（输入法、离线转录、方言识别）一致。

### 5.2 从 Whisper 继承了什么，改进了什么

#### 继承

- Mel 频谱前端（128 维，25ms 窗口 / 10ms 步长）
- Conv1d 下采样结构
- 时间戳 token 机制（`<|0.00|>` ~ `<|30.00|>`）
- 多语言 + 特殊控制 token 设计
- 编码器使用非因果双向注意力

#### 关键改进

| 改进点 | Whisper V3 | GLM-ASR | 收益 |
|--------|-----------|---------|------|
| **位置编码** | 绝对位置编码（可学习） | **部分 RoPE** | 更好的长度泛化、语义不扭曲 |
| **投影压缩** | 编码器输出直接送入解码器 | **4× Pooling + 3 层 MLP** | 解码 token 数降为 1/4 |
| **解码器架构** | MHA + LayerNorm | **GQA + RMSNorm** | KV Cache 降为 1/4，推理加速 |
| **解码器深度** | 与编码器相当（32:32） | **非对称（32:28 / 12:6）** | 计算集中于编码，解码轻量 |
| **FFN 结构** | ReLU / GELU | **SwiGLU** | 门控机制带来更好表达能力 |
| **MoE 支持** | 无 | **投影器可选 MoE** | 容量扩展不增加计算 |

### 5.3 参数量估算与分布

以云端版为例（HuggingFace 配置）：

```
Audio Encoder:
  - Token Embedding: 128 × 1280 = 164K
  - 32 × (Attention + FFN):
    - QKV 投影: 1280 × 1280 × 3 = 4.9M
    - Output 投影: 1280 × 1280 = 1.6M
    - FFN up: 1280 × 5120 = 6.6M
    - FFN down: 5120 × 1280 = 6.6M
    - LayerNorm: 1280 × 4 = 5K
    - 每层: ~19.7M × 32 = ~630M
  - Encoder 合计: ~6.3 亿

Multimodal Projector:
  - Pooling: 无参数
  - MLP 三线性: 
    - 1280 × 5120 = 6.6M
    - 5120 × 4096 = 21.0M
    - 4096 × 2048 = 8.4M
  - Projector 合计: ~0.36 亿

Text Decoder:
  - Token Embedding: 59264 × 2048 = 121.4M
  - 28 × (Self-Attention + Cross-Attention + FFN):
    - QKV 投影 (GQA): 2048×2048 + 2×(2048×512) = 4.2M + 2.1M = 6.3M
    - Self Output: 2048 × 2048 = 4.2M
    - Cross Q: 2048 × 2048 = 4.2M
    - Cross KV: 2 × (1280 × 2048) = 5.2M (来自 Encoder 的投影)
    - Cross Output: 2048 × 2048 = 4.2M
    - FFN gate: 2048 × 6144 = 12.6M
    - FFN up: 2048 × 6144 = 12.6M
    - FFN down: 6144 × 2048 = 12.6M
    - RMSNorm: 2048 × 5 = 10K
    - 每层: ~61.9M × 28 = ~1,733M
  - Decoder 合计: ~18.5 亿

总计: 6.3 + 0.36 + 18.5 ≈ 25 亿参数
```

> 注意：实际参数量可能因具体实现（权重共享、bias 策略、Head 维度对齐投影等）而有所差异。Nano 版本约为 15 亿。

---

## 六、实际部署效果（来自智谱官方）

### 6.1 评测表现

| 评测项 | GLM-ASR-Nano | Whisper V3 | 提升 |
|--------|-------------|-----------|------|
| **平均 WER** | **4.10%** | 6.93% | **-2.83pp** (相对降低 41%) |
| 安静场景 | ✅ 极低 | ✅ 低 | — |
| 噪声场景 | ✅ 稳定 | ⚠️ 质量下降 | 显著 |
| 方言 | ✅ 保真 | ⚠️ 混淆 | 显著 |

### 6.2 实际场景表现

| 场景 | 表现 |
|------|------|
| **粤语** | 准确转录自然口语，保留语气词（"啦、咯、啫"），不强制转为普通话 |
| **中英混说** | 强鲁棒性代码切换，句子中间自然切换语言，不丢失边界 |
| **低音量语音** | 在近乎耳语的音量下可靠转录，静音段不产生幻觉 |
| **噪声环境** | 街道、咖啡馆、公共交通等真实噪声下保持稳定，噪声影响极小 |
| **热词识别** | 准确识别自定义词汇和罕见专名，不牺牲整体转录质量 |
| **语言覆盖** | 17 种语言高可用支持，核心中文/英文优化最佳 |

### 6.3 生产部署

首次大规模生产部署于 **智谱输入法**，使用云端 GLM-ASR-2512 变体。

---

## 七、总结：一张图看穿 GLM-ASR

```
┌───────────────────────────────────────────────────────────────────┐
│                      GLM-ASR 架构全景                              │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  输入: 16kHz 音频                                                   │
│       │                                                            │
│  ┌────┴────┐                                                       │
│  │  Mel 频谱 │  128 维，25ms 窗 / 10ms 步长，100 帧 / 秒               │
│  └────┬────┘                                                       │
│       │                                                            │
│  ┌────┴────┐                                                       │
│  │ Conv1d  │  2 层，stride=2 → 1/2 帧数，128 → 1280 维              │
│  └────┬────┘                                                       │
│       │                                                            │
│  ┌────┴────────────────┐                                           │
│  │    Audio Encoder     │  "重感知"                                 │
│  │  ┌────────────────┐  │  32 层，20 头，1280d                      │
│  │  │ LayerNorm      │  │  FFN: 1280→5120→1280                   │
│  │  │ Bi-Self-Attn   │  │  RoPE 50%，GELU                        │
│  │  │ + Residual     │  │  非因果 → 全上下文编码                     │
│  │  │ LayerNorm      │  │                                         │
│  │  │ FFN (GELU)     │  │                                         │
│  │  │ + Residual     │  │                                         │
│  │  └────────────────┘  × 32 层                                    │
│  └────┬────────────────┘                                           │
│       │                                                            │
│  ┌────┴────────────────┐                                           │
│  │ Multimodal Projector│  "极致压缩"                                │
│  │  ┌──────────────┐   │  Pooling(4×) → 500 → 125 token           │
│  │  │ Pooling(4×)  │   │  MLP: 1280→5120→4096→2048                │
│  │  │ MLP × 3      │   │  可选 MoE (4 experts, 2 active)           │
│  │  └──────────────┘   │                                         │
│  └────┬────────────────┘                                           │
│       │                                                            │
│  ┌────┴────────────────┐                                           │
│  │    Text Decoder      │  "轻量推理"                               │
│  │  ┌────────────────┐  │  28 层，16 Q / 4 KV 头，2048d             │
│  │  │ RMSNorm        │  │  GQA → KV Cache 降为 1/4               │
│  │  │ Masked S.Attn  │  │  Cross-Attn 直达声学特征                   │
│  │  │ + Residual     │  │  SwiGLU FFN: 2048→6144→2048             │
│  │  │ RMSNorm        │  │  自回归 token-by-token                    │
│  │  │ Cross-Attn     │  │                                         │
│  │  │ (→ Encoder)    │  │                                         │
│  │  │ + Residual     │  │                                         │
│  │  │ RMSNorm        │  │                                         │
│  │  │ SwiGLU FFN     │  │                                         │
│  │  │ + Residual     │  │                                         │
│  │  └────────────────┘  × 28 层                                    │
│  └────┬────────────────┘                                           │
│       │                                                            │
│  输出: 文本 Token (59,264 vocab)                                   │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘

                  一句话总结 GLM-ASR：
          "用 32 层编码器听清每一个音，
           用 4×池化把 500 帧压成 125 token，
           用 28 层解码器（GQA）轻量生成文本。"
```

---

## 附录：关键配置原文速查

### GlmAsrEncoderConfig（HuggingFace）

```python
class GlmAsrEncoderConfig(PretrainedConfig):
    model_type = "glmasr_encoder"
    
    hidden_size = 1280              # 隐藏维度
    intermediate_size = 5120        # FFN 维度
    num_hidden_layers = 32          # 编码器层数
    num_attention_heads = 20        # 注意力头数
    num_key_value_heads = 20        # KV 头数（== attention heads → MHA）
    hidden_act = "gelu"             # 激活函数
    max_position_embeddings = 1500  # 最大位置
    initializer_range = 0.02        # 初始化范围
    attention_dropout = 0.0         # 注意力 dropout
    num_mel_bins = 128              # Mel 滤波器数量
    # partial_rotary_factor = 0.5   # kwargs 中默认设置
```

### GlmAsrConfig（完整模型）

```python
class GlmAsrConfig(PretrainedConfig):
    model_type = "glmasr"
    
    audio_token_id = 59260          # 音频段 token ID
    projector_hidden_act = "gelu"   # 投影器激活函数
    
    # 默认文本配置 (LLaMA 架构)
    text_config = {
        "model_type": "llama",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,   # GQA 16:4
        "vocab_size": 59264,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "use_cache": True,
        "eos_token_id": [59246, 59253, 59255],
        "rope_theta": 10000.0,
    }
```

### ASRConfig（推理配置）

```python
class ASRConfig:
    audio_model_id = "zai-org/GLM-ASR-Nano-2512"
    text_model_id = "Qwen/Qwen3-0.6B"
    
    # 特征提取
    encoder_conv_layers = [(1, 3, 1), (1, 3, 2)]  # (channel, kernel, stride)
    audio_sample_rate = 16000
    
    # 投影器
    projector_type = "mlp"           # 可选: "mlp", "qformer"
    projector_pool_stride = 4        # 池化步长
    downsample_rate = 5
    audio_token_dropout = 0.0
    
    # MoE 配置
    num_experts = 4
    num_experts_per_tok = 2
    router_aux_loss_coef = 0.01
    
    # Q-Former 配置（备用）
    qformer_window_size = 15
    qformer_num_layers = 2
    qformer_num_heads = 16
    
    # LoRA 配置
    use_lora = False
    lora_rank = 8
    lora_alpha = 32
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                           "gate_proj", "up_proj", "down_proj"]
    freeze_language_model = True
    
    # 推理配置
    attn_implementation = "flash_attention_2"
    model_dtype = "bfloat16"
    num_beams = 5
    max_new_tokens = 128
    use_cache = True
```

---

*本文基于 HuggingFace Transformers v5.2.0 配置源码、vLLM API 文档、智谱AI 官方技术博客及开源社区分析整理。*

**Sources:**
- [HuggingFace GlmAsr Config (transformers v5.2.0)](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/glmasr/configuration_glmasr.py)
- [vLLM GLM-ASR API 文档](https://docs.vllm.ai/en/v0.20.2/api/vllm/model_executor/models/glmasr/)
- [GLM-ASR-Nano-2512 源码解读：Transformers实现细节剖析 - CSDN](https://blog.csdn.net/weixin_42579969/article/details/157107423)
- [GLM-ASR-Nano: 面向真实世界的高鲁棒性语音识别 - 智谱AI](https://www.zhipuai.cn/zh/research/149)
- [Triton 优化 GLM-ASR Pipeline (72.2% 加速) - GitHub](https://github.com/Saurabh-66/Triton-optimized-ASR-Pipeline)
