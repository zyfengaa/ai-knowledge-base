# WeNet 与 Paraformer 架构深度解剖

> 生产级端到端语音识别的两条不同道路：WeNet 的两遍流式统一框架 vs Paraformer 的单轮非自回归并行解码

---

## 写在前面

本文深度解剖两个在生产环境中得到广泛验证的端到端 ASR 框架——**WeNet**（西北工业大学 & 出门问问, 2021）和 **Paraformer**（阿里巴巴达摩院, 2022）。

两者的出发点截然不同，却都回答了同一个核心问题：**如何在生产部署中同时实现高识别精度和高推理效率？**

| 维度 | WeNet | Paraformer |
|------|-------|-----------|
| 核心思路 | 两遍解码（CTC 流式 + Attention 重打分），单一模型统一流式/非流式 | 单轮非自回归（NAR），CIF 机制取代自回归解码 |
| 解码方式 | CTC 流式生成候选 → Attention 重打分 | 一次前向并行生成全部 token |
| 流式支持 | **原生支持**（动态 chunk 训练） | 支持（CIF/PIF 单调对齐） |
| 推理速度 | 较慢（两遍解码 + 自回归重打分） | **极快**（单轮并行，10×+加速） |
| 精度 | 与自回归模型持平 | 接近自回归模型（WER 差距 < 0.5%） |
| 生产部署 | LibTorch C++ runtime（x86 / ARM） | ONNX / TensorRT / vLLM |
| 社区生态 | WeNet 独立项目 | FunASR 统一工具链 |

---

## Part 1: WeNet (2021-)

### 一、背景：生产 ASR 的"双模型困境"

在 WeNet 出现之前，生产环境部署端到端 ASR 面临一个根本矛盾：

```
流式场景（语音助手、实时字幕）:
  需求: 低延迟（< 500ms）、逐帧输出
  方案: CTC / RNN-T 模型
  问题: 精度不如离线模型

离线场景（会议转录、视频字幕）:
  需求: 高精度、可以利用未来上下文
  方案: LAS / Transformer（注意力解码器）
  问题: 无法流式输出，需要等待说完

生产部署:
  只能同时部署两个模型 → 2× 训练成本 + 2× 部署成本 + 2× 维护成本
```

这一困境被形象地称为 **"一个模型不能既看着未来又流式输出"** ——直到 WeNet 的 U2 框架出现。

---

### 二、U2：Unified Two-Pass 框架

U2（Unified Two-pass）是 WeNet 的核心创新，发表于 INTERSPEECH 2021。其核心理念是：**单一模型，两遍解码，流式与非流式兼得**。

#### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      U2 (Unified Two-Pass) 架构                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                        输入音频 (16kHz)                               │
│                              │                                       │
│                          ┌────┴────┐                                 │
│                          │  Mel 频谱 │  80维 FBank, 25ms窗/10ms步长    │
│                          └────┬────┘                                 │
│                               │                                      │
│                          ┌────┴────┐                                 │
│                          │ 子采样层  │  Conv2d(→256d) → 4× 下采样      │
│                          └────┬────┘                                 │
│                               │                                      │
│                     ┌─────────┴──────────┐                           │
│                     │    Shared Encoder    │   Conformer / Transformer │
│                     │  (Transformer/Conformer Layers)  × 12/18/etc    │
│                     │  ★ Dynamic Chunk Mask                          │
│                     └─────────┬──────────┘                           │
│                               │                                      │
│              ┌────────────────┼────────────────┐                      │
│              │                │                │                      │
│       ┌──────┴──────┐  ┌─────┴──────┐  ┌──────┴───────┐              │
│       │  CTC Decoder │  │ L2R Attn   │  │ R2L Attn     │  ← U2++     │
│       │  (Linear+Soft) │  │ Decoder    │  │ Decoder      │  新增       │
│       └──────┬──────┘  └─────┬──────┘  └──────┬───────┘              │
│              │                │                │                      │
│              └────────┬───────┴────────────────┘                      │
│                       │                                              │
│                 ┌──────┴──────┐                                      │
│                 │   输出文本    │                                      │
│                 └─────────────┘                                      │
│                                                                      │
│  ★ 关键: Shared Encoder 使用 Dynamic Chunk Mask,                          │
│     训练时随机采样 chunk size (1 ~ max_len),                                │
│     推理时通过 chunk size 控制延迟                                         │
└─────────────────────────────────────────────────────────────────────┘
```

#### 2.2 三个核心组件

| 组件 | 构成 | 功能 | 解码角色 |
|------|------|------|---------|
| **Shared Encoder** | Conformer / Transformer 多层编码器 + 动态 chunk mask | 提取声学特征，同时支持流式/非流式 | 两遍共享 |
| **CTC Decoder** | Linear → LogSoftmax | 逐帧输出 CTC 概率，流式生成 n-best 候选 | **第一遍**（流式） |
| **Attention Decoder** | Transformer Decoder 多层（L2R，U2++ 增加 R2L） | 对 CTC 候选进行重打分（rescoring） | **第二遍**（离线） |

#### 2.3 联合训练损失函数

U2 框架通过联合 CTC/AED 损失进行端到端训练：

```
L = λ·L_CTC + (1-λ)·L_AED

其中:
  L_CTC    = CTC 损失（帧级对齐，无需对齐标签）
  L_AED    = Attention Decoder 交叉熵损失（标签级）
  λ        = 权重系数，典型值 0.5
```

**λ=0.5 的直觉**：如果 λ 过大（偏向 CTC），Attention Decoder 训练不充分，重打分效果差；如果 λ 过小（偏向 AED），流式第一遍精度不足。0.5 是一个稳健的平衡点。

#### 2.4 Dynamic Chunk Training（动态块训练）

这是 U2 实现"单一模型支持流式和非流式"的核心技巧。

**什么是 Chunk？**

```
完整序列:  [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, ...]
                                 
chunk_size=4 时的注意力掩码:
            t1 t2 t3 t4 | t5 t6 t7 t8 | t9 t10 ...
   t1        ◉  ◉  ◉  ◉    ✕  ✕  ✕  ✕    ✕  ✕
   t2        ◉  ◉  ◉  ◉    ✕  ✕  ✕  ✕    ✕  ✕
   t3        ◉  ◉  ◉  ◉    ✕  ✕  ✕  ✕    ✕  ✕
   t4        ◉  ◉  ◉  ◉    ✕  ✕  ✕  ✕    ✕  ✕
   t5        ◉  ◉  ◉  ◉    ◉  ◉  ◉  ◉    ✕  ✕
   t6        ◉  ◉  ◉  ◉    ◉  ◉  ◉  ◉    ✕  ✕
   ...

每个 chunk 能看到自己及之前全部 chunk 的所有帧，
但不能看到未来 chunk 的帧。
```

**动态采样策略**：

训练时，chunk size 按以下策略动态采样：

```python
if random.random() > 0.5:
    chunksize = max_len   # 非流式模式 — 看到全部上下文
else:
    chunksize = uniform(1, min(25, max_len-1))  # 流式模式 — 有限右上下文
```

- **50% 概率**使用 full chunk（全序列可见）→ 模型学习非流式识别
- **50% 概率**随机采样 1~25 的 chunk size → 模型学习各种延迟下的流式识别

**效果**：单一模型学会在任意 chunk size 下准确预测。推理时只需调整 chunk size 即可在延迟和精度之间选择。

| Chunk Size | 右上下文 | 延迟 (10ms 帧移, 4×下采样) | AISHELL-1 CER |
|-----------|---------|--------------------------|---------------|
| full | 全部 | 完整语音长度 | **4.90%** |
| 16 | 16 帧 | ~640ms | 5.33% |
| 8 | 8 帧 | ~320ms | 5.52% |
| 4 | 4 帧 | ~160ms | 5.71% |

#### 2.5 Causal Convolution

Conformer 编码器的标准卷积会同时看到左右上下文，这在流式场景中会引入额外延迟。WeNet 将 Conformer 中的卷积替换为 **causal convolution（因果卷积）**——确保卷积也只依赖左侧和当前帧：

```
标准 Conformer Conv:
  输入: [x_{t-2}, x_{t-1}, x_t, x_{t+1}, x_{t+2}]
  输出: y_t 同时依赖过去和未来 2 帧

Causal Conv (WeNet):
  输入: [x_{t-2}, x_{t-1}, x_t]
  输出: y_t 仅依赖过去和当前帧
  ★ 左 padding，右不 padding
```

---

### 三、Attention Rescoring 推理流程

Attention Rescoring 是 WeNet 生产环境的首选解码模式。它结合了 CTC 的**速度**和 Attention 的**精度**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Attention Rescoring 推理流程                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Step 1: CTC 第一遍（流式）                                           │
│   ──────────────────────────────────────────────────                  │
│   音频帧逐 chunk 进入 Shared Encoder                                     │
│       → CTC Decoder 实时输出概率分布                                     │
│       → CTC Prefix Beam Search 生成 N 条候选                             │
│       → 候选示例: "今天天气真的冷", "今天天气很冷", "今天天气真冷"             │
│                                                                      │
│   Step 2: Attention 第二遍（重打分）                                     │
│   ──────────────────────────────────────────────────                  │
│   将 N 条候选并行送入 Attention Decoder（teacher-forcing 模式）            │
│       → Decoder 对每条候选计算交叉熵损失（分数）                            │
│       → 分数 = λ·S_CTC + (1-λ)·S_Attention                              │
│                                                                      │
│   Step 3: 输出最优结果                                                  │
│   ──────────────────────────────────────────────────                  │
│   选择总分最高的候选作为最终输出                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**为什么 Attention Rescoring 比纯 Attention 解码快？**

- 纯 Attention Decoder 是自回归生成（逐 token 输出），无法并行
- Attention Rescoring 对**已经生成完整的候选序列**做并行打分——所有候选的序列已完成，只需一次前向即可计算所有位置的分数
- 实际测量：Attention Rescoring 的 RTF 比纯 Attention 解码降低约 **2.4 倍**

---

### 四、四种 Decoding 模式对比

| 解码模式 | 方法 | 速度 | 精度 | 流式 | 生产使用 |
|----------|------|------|------|------|---------|
| **ctc_greedy_search** | CTC 贪心搜索 | ★★★★★ 最快 | ★★★☆☆ 一般 | ✅ 是 | 极少（仅对延迟极度敏感的场景） |
| **ctc_prefix_beam_search** | CTC 前缀束搜索 | ★★★★☆ 快 | ★★★★☆ 较高 | ✅ 是 | 偶用（作为 Attention Rescoring 的前置步骤） |
| **attention_rescoring** | CTC 前缀束搜索 → Attention 重打分 | ★★★☆☆ 中等 | ★★★★★ 最高 | 🟡 流式第一遍 + 离线第二遍 | ✅ **生产首选** |
| **attention** | 纯 Attention 自回归束搜索 | ★★☆☆☆ 慢 | ★★★★☆ 高 | ❌ 否 | 极少（仅离线场景，且精度无显著优势） |

**推荐配置**：
- **生产环境（通用）**：`attention_rescoring`，chunk_size=16
- **超低延迟（语音助手）**：`ctg_greedy_search`，chunk_size=4~8
- **最高精度（离线转录）**：`attention_rescoring`，chunk_size=full

---

### 五、U2++：双向 Attention 重打分

U2++（2021年6月）在 U2 基础上增加了一个关键改进：**Right-to-Left (R2L) Attention Decoder**。

#### 5.1 架构变化

```
U2:                                  U2++:
┌────────────────────────┐          ┌────────────────────────┐
│  Shared Encoder        │          │  Shared Encoder        │
│      │                 │          │      │                 │
│  ┌───┴────┐            │          │  ┌───┴────┐            │
│  │CTC Dec │            │          │  │CTC Dec │            │
│  └───┬────┘            │          │  └───┬────┘            │
│      │                 │          │      │                 │
│  ┌───┴────┐            │          │  ┌───┴────┐  ┌───┴────┐│
│  │L2R Attn│  ← 只有 L2R│          │  │L2R Attn│  │R2L Attn││  ← 新增
│  └───┬────┘            │          │  └───┬────┘  └───┬────┘│
│      │                 │          │      │           │     │
│  输出文本               │          │    ╔══╧═══════════╧══╗  │
└────────────────────────┘          │    ║  加权融合评分    ║  │
                                    │    ╚══╤═══════════╤══╝  │
                                    │       │           │     │
                                    │    输出文本        │     │
                                    └────────────────────────┘
```

**为什么需要 R2L？**

L2R（左到右）解码器只能利用已生成的左侧上下文进行预测。但对于某些口语表达，只有"听了后面的词"才能确定前面的正确结果：

```
"我今天去了银行（hang2 / yin2）..."
  → L2R: 看到"银"时，前面已确定"行"→ 无法回退
  → R2L: 从后往前看，能感知"行"的右侧上下文 → 更好消歧
```

#### 5.2 训练损失

```
L_AED(x, y) = (1 - reverse_weight)·L_L2R(x, y) + reverse_weight·L_R2L(x, y)

其中 reverse_weight 典型值 = 0.3（R2L 作为辅助，L2R 为主）
```

**为什么 reverse_weight=0.3 而非 0.5？** R2L 解码器在训练中作为"辅助角色"，主解码器仍是 L2R。推理时通常只使用 L2R 做重打分（因为 R2L 需要整句完成后才能开始），R2L 的作用是：通过双向训练信号，让 Shared Encoder 学习到更鲁棒的声学表示。

#### 5.3 精度提升

| 数据集 | 语言 | 指标 | U2 | U2++ | 提升 |
|--------|------|------|-----|------|------|
| AISHELL-1 | 中文 | CER | 4.97% | **4.63%** | -6.8% |
| AISHELL-2 | 中文 | CER | 6.08% | **5.39%** | -11.3% |
| LibriSpeech test-clean | 英文 | WER | 2.85% | **2.66%** | -6.7% |
| LibriSpeech test-other | 英文 | WER | 7.24% | **6.53%** | -9.8% |
| GigaSpeech dev | 英文 | WER | 11.30% | **10.70%** | -5.3% |

---

### 六、流式缓存机制

WeNet 的 Shared Encoder 在流式推理时通过两种缓存实现增量计算：

#### 6.1 att_cache（注意力缓存）

```
当前 chunk 输入 → Shared Encoder 第 i 层

  att_cache[L-1] (来自之前所有 chunk 的 K, V)
        │
        ▼
  Concat([[K_cache, V_cache], K_new, V_new])
        │
        ▼
  注意力计算（能看到完整的左侧上下文 + 当前 chunk）
        │
        ▼
  更新 att_cache（将新的 K, V 追加到缓存）
```

- 缓存的粒度：每个 Encoder Layer 存储其自注意力层的 K, V 序列
- 避免重复计算历史帧的 K, V 投影
- 实现位置：`encoder.py:forward_chunk()`

#### 6.2 cnn_cache（CNN 缓存）

Conformer 中的 causal CNN 模块在逐 chunk 推理时也需要缓存左侧上下文：

```
第 i 层 CausalConv 的 cnn_cache:
  存储前一 chunk 的最后因果卷积核大小 - 1 个 timestep 的输出
  用于保证卷积在 chunk 边界处的连续性

  例如 kernel_size=7, causal padding=3:
    cnn_cache 长度 = 3
    新 chunk 推理时，将 cache 与当前输入拼接
```

#### 6.3 下采样层的特殊处理

下采样 CNN（Conv2d with stride=2）的左上下文 / 右上下文 / 步长各不相同，直接实现缓存逻辑较为复杂。WeNet 采用 **输入重叠（overlap）** 的方式：

```
前一 chunk 的最后若干帧: [..., t-2, t-1, t]
当前 chunk 的前若干帧:   [t-2, t-1, t, t+1, t+2, ...]
                          ^^^^^^^
                          重叠部分（重新计算，但下采样 CNN 计算量极小）
```

这种方案虽然引入了少量冗余计算，但因为下采样层的计算量在整个编码器中占比极小（约 3-5%），额外开销可以忽略。

---

### 七、小结：一张图看懂 WeNet

```
┌──────────────────────────────────────────────────────────────────────┐
│                         WeNet 架构全景                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌──────────┐                                                        │
│   │ 输入音频   │  16kHz → 80维 FBank → Conv2d 子采样 (4×)                │
│   └─────┬────┘                                                        │
│         │                                                             │
│   ┌─────┴──────────────────────────────┐                              │
│   │       Shared Encoder                │  Conformer × 12/18          │
│   │  ┌──────────────────────────────┐   │  ★ Dynamic Chunk Mask        │
│   │  │  Multi-Head Self-Attention   │   │  ★ Causal Convolution         │
│   │  │  + Causal Conv (Conformer)   │   │                             │
│   │  │  + Feed Forward              │   │  ← att_cache + cnn_cache    │
│   │  └──────────────────────────────┘×N │  支持逐 chunk 增量推理       │
│   └─────┬──────────────────────────────┘                              │
│         │                                                             │
│   ┌─────┴───────────┐     ┌───────────┴─────────────┐                │
│   │  CTC Decoder     │     │  Attention Decoder(s)    │                │
│   │  (Linear+Softmax)│     │  ┌──────────────────┐   │                │
│   └─────┬───────────┘     │  │  Masked Self-Attn │   │                │
│         │                 │  │  Cross-Attn(→Enc) │   │                │
│         │                 │  │  Feed Forward     │   │                │
│         │                 │  └──────────────────┘×N │                │
│         │                 │  U2: L2R only           │                │
│         │                 │  U2++: L2R + R2L        │                │
│         ▼                 └───────────┬─────────────┘                │
│   ┌────────────┐                      │                             │
│   │  第一遍     │ 流式 CTC Prefix      │  第二遍                      │
│   │  n-best 候选│ Beam Search         │  Attention Rescoring        │
│   └────────────┘                      ▼                             │
│                                ┌──────────┐                         │
│                                │  最终文本  │                         │
│                                └──────────┘                         │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘

            一句话总结 WeNet：
    "共享编码器 + 动态 chunk —— 一个模型跑遍流式与非流式，
     CTC 快速出候选，Attention 精打重评分。"
```

---

## Part 2: FunASR / Paraformer (2022-)

### 一、背景：自回归模型的"慢"瓶颈

WeNet 解决了"流式 vs 非流式"的统一问题，但它仍然依赖**自回归（Autoregressive, AR）解码**——无论是 Attention Rescoring 还是纯 Attention 解码，都是逐 token 生成的：

```
自回归解码 (Whisper / WeNet Attention Decoder):
  Step 1: <sos> → "今"
  Step 2: "今" → "天"
  Step 3: "天" → "天"
  Step 4: "天" → "气"
  ...
  Step 15: "冷" → <eos>
  ★ 共 15 步前向计算
  ★ 每步都需读取完整的 KV Cache (memory-bound)
  ★ 序列越长，速度越慢

非自回归解码 (Paraformer):
  ┌──────────────────┐
  │ 一步并行生成全部   │  ← 一个前向输出全部 token
  └──────────────────┘
  ★ 只需 1 步前向计算
  ★ 速度与序列长度无关
```

**非自回归 ASR 的核心难题**：

1. **Token 数量未知** — AR 模型逐 token 生成，长度自然确定；NAR 需要在解码前就知道"这句话有几个字"
2. **条件独立假设** — NAR 模型通常假设 token 间条件独立（如 CTC），但自然语言有强上下文依赖
3. **替换错误率高** — 没有自回归的逐步修正能力，NAR 模型的替换错误（substitution）显著高于 AR 模型

Paraformer 通过 **CIF 机制** 和 **GLM Sampler** 系统地解决了这三个问题。

---

### 二、CIF：Continuous Integrate-and-Fire

CIF（连续整合-发放机制）由中国科学院自动化研究所董林昊和徐波提出（ICASSP 2020），是 Paraformer 的基石。

#### 2.1 生物启发的直觉

CIF 的名称和灵感来自**脉冲神经网络（SNN）**中的整合-发放模型：

```
生物神经元的工作原理:
  不断接收来自突触的输入信号（权重×脉冲）
  当膜电位累积超过阈值 → 发放一个脉冲（fire）
  发放后膜电位重置

CIF 的工作方式:
  不断接收编码器每帧的输出（声学特征）
  当累积权重超过阈值 β=1.0 → 发放一个 token 级别的 embedding
  发放后剩余权重用于下一个 token
```

#### 2.2 CIF 工作机制详解

```
输入: 编码器输出 h = [h_1, h_2, ..., h_T]  (T 帧)
      权重预测 α = WeightPredictor(h)  (每帧一个 0~1 的权重)

过程:
  Step t=1: α_1 = 0.2
    累积权重 = 0.2 < 1.0 → 继续累积
    c_1 暂存 = 0.2 × h_1

  Step t=2: α_2 = 0.9
    累积权重 = 0.2 + 0.9 = 1.1 ≥ 1.0 → **发放!**
    分割: α_2用于c_1的部分 = 1.0 - 0.2 = 0.8
          α_2用于c_2的剩余 = 0.9 - 0.8 = 0.1
    c_1 = 0.2·h_1 + 0.8·h_2  ← 第一个 token 的声学 embedding
    重置累积权重 = 0.1

  Step t=3: α_3 = 0.6
    累积权重 = 0.1 + 0.6 = 0.7 < 1.0 → 继续累积
    c_2 暂存 = 0.1·h_2 + 0.6·h_3

  Step t=4: α_4 = 0.6
    累积权重 = 0.7 + 0.6 = 1.3 ≥ 1.0 → **发放!**
    分割: α_4用于c_2的部分 = 1.0 - 0.7 = 0.3
          α_4用于c_3的剩余 = 0.6 - 0.3 = 0.3
    c_2 = 0.1·h_2 + 0.6·h_3 + 0.3·h_4
    重置累积权重 = 0.3
  ...

输出: c = [c_1, c_2, ..., c_U]  (U 个 token 级别声学 embedding)
```

**CIF Predictor 的神经网络结构**：

```
编码器输出 h [T, d_model]
    │
    ├─ Conv1d (kernel=1)  ← 通道变换
    ├─ ReLU
    ├─ Linear → 1 维
    ├─ Sigmoid
    │
    └─ α [T]  ← 每帧权重，取值 (0, 1)
```

CIF Predictor 本身只有极少的可学习参数（一个 Conv1d + 一个 Linear），主要依赖累积 + 发放的逻辑（无参数）。

#### 2.3 Quantity Loss（数量损失）

CIF 需要知道每句话实际有多少个 token 才能训练。Quantity Loss 用来监督 Predictor 学习正确的 token 数量：

```
L_QUA = |Σα_t - S_GT|

其中:
  Σα_t = 所有帧权重的总和（预期 = 目标 token 数）
  S_GT = 目标文本的真实 token 数量
```

**缩放策略（Scaling Strategy）**：训练时，将权重乘以 `S_GT / Σα_t`，确保 CIF 精确发放 S_GT 个 token，实现与目标文本的一一对应交叉熵训练。

**尾部处理（Tail Handling）**：推理时，末尾的残差权重 > 0.5 时触发一次额外发放，避免漏掉最后一个 token。

---

### 三、Paraformer 架构

Paraformer（INTERSPEECH 2022）将 CIF 与 Conformer 编码器、并行 Transformer 解码器、GLM Sampler 组合为完整的非自回归 ASR 系统。

#### 3.1 整体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Paraformer 架构全景                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│                         输入音频 (16kHz)                                    │
│                              │                                            │
│                     ┌────────┴────────┐                                   │
│                     │  Conformer Encoder│   SAN-M / Conformer              │
│                     │  (双向, 非因果)     │   多层编码器                      │
│                     └────────┬────────┘                                   │
│                              │ h = [h_1, ..., h_T]                        │
│                              │                                            │
│                     ┌────────┴────────┐                                   │
│                     │   CIF Predictor  │   ★ 非自回归核心创新                 │
│                     │   (权重预测+累积)  │   预测 token 数 + 抽取声学 embedding │
│                     └────────┬────────┘                                   │
│                              │ c = [c_1, ..., c_U]                        │
│                              │                                            │
│              ┌───────────────┴────────────────┐                           │
│              │           GLM Sampler           │  ★ 仅在训练时激活             │
│              │  (随机替换部分 acoustic embed)    │  解决替换错误问题              │
│              └───────────────┬────────────────┘                           │
│                              │ c_sampled                                   │
│                              │                                            │
│              ┌───────────────┴────────────────┐                           │
│              │   Parallel Transformer Decoder  │  ★ 双向（非因果）注意力       │
│              │   (非自回归, 一步生成全部 U 个 token) │  无掩码, 无因果关系         │
│              └───────────────┬────────────────┘                           │
│                              │                                            │
│                    ┌─────────┴──────────┐                                 │
│                    │     输出文本         │  一次并行生成全部 token            │
│                    └────────────────────┘                                 │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 3.2 各模块详解

| 模块 | 结构 | 功能 | 特点 |
|------|------|------|------|
| **Conformer Encoder** | SAN-M 或 Conformer 多层编码器 | 提取声学特征，与 WeNet 的 Shared Encoder 类似 | 双向、非因果，可以看到全部上下文 |
| **CIF Predictor** | Conv1d + ReLU + Linear + Sigmoid + CIF累积逻辑 | ① 预测输出 token 数量 ② 抽取 token 级声学 embedding | 无参数逻辑部分，仅 ~2 层 NN |
| **GLM Sampler** | 随机替换 + embedding 查找 | 训练时将部分声学 embedding 替换为目标文本 embedding | **推理时不使用**，零额外开销 |
| **Parallel Decoder** | Transformer Decoder × N 层（非因果） | 基于声学 embedding 并行生成全部 token | 无因果掩码，双向可见 |
| **Loss** | CE + CTC + L_QUA | 联合优化 | L = 0.3·L_CTC + 0.7·L_CE + L_QUA |

#### 3.3 两遍训练策略（Two-Pass Training）

Paraformer 的 GLM Sampler 采用独特的两遍训练策略：

```
Pass 1: 无梯度（前向计算参考）
  ┌─────────────────────────────────────┐
  │  完整前向: 编码器 → CIF → Decoder   │
  │  记录 Decoder 输出与 GT 的差异位置    │
  └──────────────┬──────────────────────┘
                 │
                 ▼
  Pass 2: 有梯度（训练主路径）
  ┌─────────────────────────────────────┐
  │  编码器 → CIF → GLM Sampler → Decoder  │
  │  Sampler 替换 Pass 1 中错误位置的    │
  │  声学 embedding 为目标文本 embedding  │
  │  计算损失 ← 反向传播                 │
  └─────────────────────────────────────┘
```

**为什么需要两遍训练？**

NAR 模型的替换错误（substitution）高的一个原因是：Decoder 只看到了"可能正确的声学 embedding"，没有见过"正确的文本 embedding"作为上下文信号。GLM Sampler 的策略是在训练时用 GT embedding **替换**部分声学 embedding，让 Decoder 学会在部分"正确上下文"下预测其余位置：

```
训练时的两种 embedding:
  acoustic_embed[i] = CIF 从声学帧池化的 embedding  (可能不准确)
  text_embed[i]     = 从目标文本 token 查表的 embedding  (完全正确)

GLM Sampler 的输出 embedding:
  c_input[i] = 
    ┌─ acoustic_embed[i]   (概率 1-p)
    └─ text_embed[i]       (概率 p)  ← 替换为正确文本

p 的取值策略: 根据 Pass 1 的预测误差动态决定
  误差大 → 替换概率 p 大（给更多正确的上下文信号）
  误差小 → 替换概率 p 小（让模型自己学会从声学特征预测）
```

#### 3.4 推理时的工作流

推理时，GLM Sampler 被完全移除，流程简化为一条直线：

```
音频 → Encoder → CIF Predictor → Parallel Decoder → 输出文本
                                        ↑
                              ★ 一步前向，全部 token 并行生成
```

这与自回归模型的 N 步前向形成鲜明对比：

```
推理速度对比 (10 秒音频, 15 个输出 token):
  自回归 (Whisper):  Encoder(1步) + Decoder(15步) = 16 步前向
  自回归 (WeNet):    Encoder(1步) + CTC(1步) + Decoder(1步) = 3 步前向
                     （但 Decoder 步需要 teacher-forcing 并行打分）
  Paraformer:        Encoder(1步) + CIF(1步) + Decoder(1步) = 3 步前向
                     ★ 无需逐 token 生成
                     ★ 实际 RTF = 0.0168~0.0251 (比 AR 模型快 10×+)
```

---

### 四、E-Paraformer：PIF 并行整合-发放

E-Paraformer（Efficient Paraformer, Interspeech 2024）在原始 Paraformer 的基础上，用 **PIF（Parallel Integrate-and-Fire）** 取代了 CIF 的递归累积计算，实现了进一步加速。

#### 4.1 CIF 的计算瓶颈

原始 CIF 的累积逻辑本质上是**递归的**——必须按时间顺序逐帧处理，无法并行：

```python
# CIF: O(T) 串行
accum = 0
for t in range(T):
    accum += alpha[t]
    if accum >= 1.0:
        # fire: 无法提前知道 t 时刻是否会 fire
```

这与 GPU 的并行计算范式相冲突——GPU 擅长矩阵乘法等可并行操作（O(log T)），不擅长递归累加（O(T)）。

#### 4.2 PIF 的并行对齐机制

PIF 的核心思路：**将递归的累积-判断逻辑转换为可并行的距离矩阵计算**。

```
Step 1: 预测帧级权重 α [T]
  ┌──────────────────────────────┐
  │  α = WeightPredictor(h)      │  ← 与 CIF 相同的预测网络
  └──────────────────────────────┘

Step 2: 构建对齐位置向量
  ┌──────────────────────────────┐
  │  pos[t] = Σα[0:t]            │  ← 可并行计算的累积和（prefix sum）
  │  ★ GPU 上 O(log T) 完成      │
  └──────────────────────────────┘

Step 3: 构建对齐矩阵
  ┌──────────────────────────────┐
  │  M[u, t] = distance(center_u, pos[t])  │
  │  基于 token 中心位置与帧位置的距离      │
  │  高斯核: exp(-(p - c)^2 / (2σ^2))     │
  └──────────────────────────────┘

Step 4: 并行抽取 token embedding
  ┌──────────────────────────────┐
  │  c_u = Σ_t M[u, t] · h_t    │  ← 可并行计算的矩阵乘法
  └──────────────────────────────┘
```

**关键改进**：

| 特性 | CIF（原始） | PIF（E-Paraformer） |
|------|-----------|-------------------|
| 计算方式 | **递归**累积 + 判断 | **并行**前缀和 + 距离矩阵 |
| 上下文 | 有限局部上下文 | **全局上下文**——每 token 可看到所有帧 |
| 训练速度 | 1× | ~1.35× |
| 推理速度 | 1× | ~**2×** |
| 可学习参数 | 无（权重预测网络之外） | 可训练的 σ 和 δ 参数 + 多头机制 |
| AISHELL-1 Test CER | 5.11% (Paraformer base) | **4.79%** |

#### 4.3 PIF 的多头机制

E-Paraformer 还引入了**多头对齐**：每个 token 使用多个对齐中心（类比 Multi-Head Attention 的多头），每个头学习不同的对齐模式：

```
标准 PIF:    c_u = Σ_t M[u, t] · h_t      (单一对齐)
多头 PIF:    c_u = Concat(head_1, ..., head_H) · W_O
             head_i = Σ_t M_i[u, t] · h_t  (每个头有不同的 σ_i)
```

---

### 五、Paraformer-v2：基于 CTC 的 Token 提取

Paraformer-v2（arXiv:2409.17746, NCMMSC 2024 最佳论文）进一步简化了架构——**用 CTC 替代 CIF** 进行 token 位置预测，提升了多语言和噪声场景的鲁棒性。

#### 5.1 设计动机

CIF 的权重预测虽然机制优雅，但在**严重噪声**和**多语言混合**场景下，权重的准确度下降明显。CTC 经过多年验证，在各种声学条件下都有稳定的对齐能力。

#### 5.2 架构变化

```
Paraformer v1:                  Paraformer-v2:
┌────────────────────┐          ┌────────────────────┐
│  Conformer Encoder  │          │  Conformer Encoder  │
└────────┬───────────┘          └────────┬───────────┘
         │                               │
┌────────┴───────────┐          ┌────────┴───────────┐
│  CIF Predictor      │          │  CTC Decoder        │  ← CIF → CTC
│  (权重累积 + 发放)    │          │  (argmax 提取位置)   │
└────────┬───────────┘          └────────┬───────────┘
         │                               │
┌────────┴───────────┐          ┌────────┴───────────┐
│  Parallel Decoder   │          │  Parallel Decoder   │
└────────────────────┘          └────────────────────┘
```

CTC 到 token embedding 的转换：使用 CTC 的 argmax 路径确定每个 token 的边界位置，然后在这些位置处池化编码器输出作为 token-level embedding。

#### 5.3 精度提升

| 场景 | Paraformer v1 | Paraformer-v2 | 提升 |
|------|--------------|--------------|------|
| 英文清洁语音 (LibriSpeech) | 基线 | **14%+ WER 降低** | 显著 |
| 噪声语音 | 质量下降 | 稳定 | 显著 |
| 多语言混合 | 混淆 | 稳定 | 显著 |

---

### 六、FunASR 工具链

Paraformer 作为 **FunASR** 工具包的核心模型发布。FunASR 提供了一个完整的生产级语音识别工具链：

```
┌──────────────────────────────────────────────────────────────────────┐
│                      FunASR 完整 Pipeline                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  输入长音频 (会议录音 / 直播流 / 电话录音)                                 │
│       │                                                               │
│  ┌────┴────┐                                                          │
│  │   VAD    │  FSMN-VAD → 分割为独立的语音段 [[t1,t2], [t3,t4], ...]   │
│  │  (语音检测)│  可选: 按长度排序以实现高效批处理                          │
│  └────┬────┘                                                          │
│       │                                                               │
│  ┌────┴──────────┐                                                    │
│  │  ASR 识别      │  Paraformer / SenseVoice / Fun-ASR-Nano            │
│  │  (核心模型)    │  支持热词定制: hotwords=["Kubernetes", "张三"]       │
│  └────┬──────────┘                                                    │
│       │                                                               │
│  ┌────┴──────────┐                                                    │
│  │  时间戳对齐    │  Paraformer-TP / TP-Aligner → 词级时间戳              │
│  └────┬──────────┘                                                    │
│       │                                                               │
│  ┌────┴──────────┐                                                    │
│  │  标点恢复      │  CT-Transformer → 插入逗号、句号、问号                  │
│  └────┬──────────┘                                                    │
│       │                                                               │
│  ┌────┴──────────┐                                                    │
│  │  说话人分离    │  CAM++ → "谁在什么时候说话"                           │
│  └────┬──────────┘                                                    │
│       │                                                               │
│  ┌────┴──────────┐                                                    │
│  │  句子级修正    │  规则或 NGram 语言模型 → 修正常见识别错误               │
│  └────┬──────────┘                                                    │
│       │                                                               │
│  ┌────┴──────────┐                                                    │
│  │  最终输出      │  {"speaker": "A", "text": "今天天气真冷。",         │
│  │               │   "timestamp": [1200ms, 3200ms]}                   │
│  └────────────────┘                                                   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

**一行代码调用完整 Pipeline**：

```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",       # ASR 模型
    vad_model="fsmn-vad",        # 语音活动检测
    punc_model="ct-punc",        # 标点恢复
    spk_model="cam++",           # 说话人分离
)

res = model.generate(
    input="meeting.wav",
    batch_size_s=300,
    hotwords=["Kubernetes", "Docker", "FunASR"],
)

# 输出示例
# [
#   {"sentence_info": [
#     {"spk": "spk1", "text": "大家好，今天我们来讨论一下Kubernetes部署方案。",
#      "timestamp": [0, 3200]},
#     {"spk": "spk2", "text": "好的，我先介绍一下整体架构。",
#      "timestamp": [3500, 6100]},
#   ]}
# ]
```

#### 工具链各模块一览

| 模块 | 模型 | 功能 | 是否可选 |
|------|------|------|---------|
| **VAD** | FSMN-VAD | 语音活动检测，分割长音频 | ✅ 推荐长音频使用 |
| **ASR** | Paraformer / SenseVoice / Fun-ASR-Nano | 语音文本识别 | 核心必需 |
| **热词** | Contextual Paraformer | 提升专名、术语召回率 | ✅ 可选 |
| **时间戳** | Paraformer-TP / TP-Aligner | 词级时间对齐 | ✅ 可选 |
| **标点** | CT-Transformer | 插入标点符号 | ✅ 可选 |
| **说话人分离** | CAM++ | 区分不同说话人 | ✅ 可选 |
| **句子修正** | NGram / 规则 | 修正常见识别错误 | ✅ 可选 |

#### 推理后端支持

| 后端 | 平台 | 格式 | 特点 |
|------|------|------|------|
| **PyTorch** | CPU / GPU | 原生 | 最灵活 |
| **ONNX** | CPU / GPU | ONNX 量化 | ~2× 加速 |
| **TensorRT** | GPU | TensorRT 引擎 | 最大吞吐 |
| **LibTorch** | CPU/GPU, Android, iOS | TorchScript | 跨平台 |
| **vLLM** | GPU (多卡 TP) | — | LLM 模型支持 |
| **WebSocket** | 服务端 | — | 实时流式服务 |

---

### 七、WER vs 速度权衡全表

| 模型 | 类型 | AISHELL-1 CER | 相对推理速度 | RTF | 流式 | 部署难度 |
|------|------|:------------:|:----------:|:---:|:----:|:-------:|
| **Whisper Large-v3** | 自回归（Encoder-Decoder） | 6.93% | 1× (最慢) | ~0.3 | ❌ | 中等 |
| **WeNet (chunk=full)** | 两遍（CTC+Attention） | **4.90%** | ~3× | ~0.05 | ❌ | 低 |
| **WeNet (chunk=16)** | 两遍 + 流式 | 5.33% | ~4× | ~0.04 | ✅ | 低 |
| **WeNet U2++ (chunk=full)** | 两遍 + 双向 | **4.63%** | ~2.5× | ~0.06 | ❌ | 低 |
| **Paraformer (base)** | 单轮非自回归 | 5.11% | ~10× | ~0.025 | ✅ | 低 |
| **Paraformer (large)** | 单轮非自回归 | **1.95%** | ~8× | ~0.03 | ✅ | 低 |
| **E-Paraformer** | 单轮非自回归（PIF） | **4.79%** | ~**20×** | ~**0.012** | ✅ | 低 |
| **Paraformer-v2** | 单轮非自回归（CTC提取） | — | ~10× | ~0.025 | ✅ | 低 |

> **解读**：
> - 追求**极致精度**（CER < 2%）：Paraformer (large)
> - 追求**流式 + 高精度**：WeNet U2++ (chunk=16, attention_rescoring)
> - 追求**极致速度**（10×+ 加速）：E-Paraformer / Paraformer (base)
> - 追求**单一模型统一流式/非流式**：WeNet（独特优势）
> - 追求**完整生产工具链**（VAD + 标点 + 分离）：FunASR 全栈

---

### 八、架构设计的深层思考

#### 8.1 为什么是 CIF 而非 CTC？(Paraformer)

CTC 本身也是非自回归的，为什么 CIF 能做得更好？

| 维度 | CTC | CIF |
|------|-----|-----|
| **对齐方式** | 帧级独立，假设帧间条件独立 | 帧级连续性累积，保留帧间依赖 |
| **输出表示** | 概率分布（含 blank token） | **连续向量 embedding** |
| **下游解码** | 需要额外语言模型或 CTC beam search | 天然输出 token 级向量，可直接送入 Decoder |
| **token 数量** | 通过 blank 压缩确定 | 通过累积阈值确定 |
| **表达能力** | 仅输出 token ID 概率 | 输出连续声学空间向量（信息更丰富） |

CIF 最重要的一点贡献是：**从声学帧到 token 的"桥接 embedding"**——这是 CTC 无法直接提供的。有了这个桥接 embedding，Decoder 可以直接在声学空间和文本空间之间做转换，不需要逐帧累积信息。

#### 8.2 为什么 WeNet 没有选择 NAR 路线？

WeNet（2021）发布于 Paraformer（2022）之前，当时 NAR ASR 的精度还远落后于 AR 方法。WeNet 的设计者选择了更务实的路径：

```
WeNet 的设计选择:
  ✅ 复用成熟的 CTC + Attention 架构
  ✅ 通过"动态 chunk"解决流式问题
  ✅ 通过"两遍解码"兼顾速度和精度
  ⚠️ 保留自回归解码 → 速度有天花板

对比 Paraformer:
  ❌ NAR 在 2021 年还不成熟
  ✅ 2022 年 CIF + GLM Sampler 大幅缩小了精度差距
  ✅ 10× 加速对生产部署的吸引力巨大
```

#### 8.3 生产部署建议

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| **实时语音助手（<200ms 延迟）** | WeNet (chunk=4~8, ctc_prefix_beam_search) | 流式延迟最低，精度可接受 |
| **云端实时会议转录** | FunASR (SenseVoice/Paraformer + VAD) | 完整工具链，支持热词和说话人分离 |
| **高并发离线批量转录** | Paraformer (E-Paraformer) | 10×+ 加速，GPU 利用率高 |
| **最高精度（字幕/听写）** | WeNet (chunk=full, U2++, attention_rescoring) | CER 最低（AISHELL-1: 4.63%） |
| **移动端/嵌入式** | WeNet (LibTorch, INT8 量化) | 成熟稳定的 C++ Runtime |
| **多语言/噪声环境** | Paraformer-v2 | CTC 提取更鲁棒 |

---

### 九、总结：WeNet vs Paraformer — 两条技术路线的对比

```
┌──────────────────────────────────────────────────────────────────────┐
│                    WeNet 与 Paraformer 的技术哲学                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  WeNet:               Paraformer:                                     │
│  "两遍总比一遍好"       "一遍能做完，绝不两遍"                              │
│                                                                       │
│  ┌────────────────┐    ┌────────────────┐                              │
│  │  Shared Encoder │    │  Conformer Enc │  ← 编码器双方都使用 Conformer │
│  └───────┬────────┘    └───────┬────────┘                              │
│          │                     │                                       │
│  ┌───────┴────────┐    ┌───────┴────────┐                              │
│  │  CTC 第一遍     │    │  CIF/PIF        │  ← 两者都预测 token 位置     │
│  │  (流式 n-best)  │    │  (并行提取)      │                             │
│  └───────┬────────┘    └───────┬────────┘                              │
│          │                     │                                       │
│  ┌───────┴────────┐    ┌───────┴────────┐                              │
│  │  Attn 第二遍    │    │  Parallel Dec   │  ← 关键分歧:                  │
│  │  (自回归重打分)  │    │  (非自回归解码)  │     WeNet 串行 vs Paraformer 并行 │
│  └───────┬────────┘    └────────────────┘                              │
│          │                                                             │
│  ★ 精度更高 ★          ★ 速度更快 ★                                    │
│  ★ 流式/非流式统一 ★   ★ 工具链更完整 ★                                 │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

**一句话总结**：
- **WeNet** 用"共享编码器 + 动态 chunk"打破了流式与非流式的壁垒，**两遍解码**在精度上建立了难以逾越的优势
- **Paraformer** 用"CIF + GLM Sampler"打破了自回归的串行枷锁，**单轮并行**在速度上实现了数量级的碾压

两者并非竞争关系，而是互补的——一个追求**精度上限**，一个追求**速度极限**。在实际生产中，根据场景需求选择最合适的方案，或许是最好的策略。

---

## 附录：关键概念速查

| 术语 | 全称 | 解释 |
|------|------|------|
| **U2** | Unified Two-pass | WeNet 的统一两遍框架：CTC 第一遍 + Attention 第二遍 |
| **U2++** | — | U2 的增强版：增加 R2L Attention Decoder 实现双向重打分 |
| **Dynamic Chunk** | 动态块训练 | 训练时随机采样 chunk size，使单一模型同时学会流式和非流式 |
| **Attention Rescoring** | 注意力重打分 | CTC 快速生成候选，Attention Decoder 并行打分 |
| **CIF** | Continuous Integrate-and-Fire | 连续整合-发放机制，从声学帧累积提取 token 级 embedding |
| **PIF** | Parallel Integrate-and-Fire | CIF 的并行加速版本，用距离矩阵替代递归累积 |
| **GLM** | Glancing Language Model | 训练时用部分 GT 文本替换声学 embedding，减少替换错误 |
| **NAR** | Non-Autoregressive | 非自回归解码，一步并行生成全部 token |
| **Causal Conv** | Causal Convolution | 因果卷积，只依赖左侧和当前帧 |
| **GQA** | Grouped Query Attention | 分组查询注意力，多组 Q 共享一组 KV |
| **FSMN** | Feedforward Sequential Memory Network | 前馈序列记忆网络，用于 VAD |
| **CAM++** | — | 说话人嵌入模型，用于说话人分离 |

---

## ==Sources==

- [WeNet: Production Oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit (arXiv:2102.01547)](https://arxiv.org/abs/2102.01547)
- [U2++: Unified Two-pass Bidirectional End-to-end Model for Speech Recognition (arXiv:2106.05642)](https://arxiv.org/abs/2106.05642)
- [WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit (arXiv:2203.15455)](https://arxiv.org/abs/2203.15455)
- [WeNet GitHub](https://github.com/wenet-e2e/wenet)
- [CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition (arXiv:2005.04390)](https://arxiv.org/abs/2005.04390)
- [Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition (arXiv:2206.08317)](https://arxiv.org/abs/2206.08317)
- [FunASR: A Fundamental End-to-End Speech Recognition Toolkit (INTERSPEECH 2023)](https://arxiv.org/abs/2305.11013)
- [E-Paraformer (INTERSPEECH 2024)](https://www.isca-archive.org/interspeech_2024/zou24_interspeech.html)
- [Paraformer-v2: An Improved Non-autoregressive Transformer for Noise-Robust Speech Recognition (arXiv:2409.17746)](https://arxiv.org/abs/2409.17746)
- [FunASR Documentation on ModelScope](https://modelscope.github.io/FunASR/)
- [FunASR GitHub](https://github.com/modelscope/FunASR)

---

*本文基于 WeNet 论文、Paraformer 家族论文、FunASR 官方文档及开源代码综合分析整理。*
