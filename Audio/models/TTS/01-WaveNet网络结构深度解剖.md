# WaveNet 网络结构深度解剖

> DeepMind (Google) 出品 | 首次用深度生成模型直接合成原始波形的奠基之作

---

## 写在前面：理解三个版本

WaveNet 系列存在**三个主要版本形态**：

| 版本 | 用途 | 输入 | 输出 | 关键差异 |
|------|------|------|------|---------|
| **无条件 WaveNet** | 语音/音乐生成（探索性） | 纯噪声/初始 seed | 原始波形 | 无外部条件，仅学习数据分布 |
| **条件 WaveNet（局部）** | TTS 声码器（Tacotron 2 标配） | Mel 频谱（帧级对齐） | 原始波形 | 以 Mel 谱为条件，逐帧引导生成 |
| **条件 WaveNet（全局+局部）** | 多说话人 TTS 声码器 | Mel 频谱 + 说话人嵌入 | 原始波形 | 额外全局条件控制音色/风格 |

> 本文分析以**条件 WaveNet（局部 + 全局）** 为主，这是对 TTS 影响最大的版本。

---

## 一、整体架构设计哲学

WaveNet 的设计理念可以用三条原则概括：

> **"因果、膨胀、深度"**

- **因果**：输出只依赖当前和过去的输入——时序建模的基本约束
- **膨胀**：指数级扩大感受野——用少量层覆盖长时依赖（30 层覆盖数千采样点）
- **深度**：堆叠 30 层门控卷积——从局部到全局的分层特征提取

### 架构总览

```
原始音频 x_t (16kHz, 16-bit)
    │
    ├── ① μ-law 量化 (256 路)
    │
    ├── ② Causal Conv1x2 (输入投影)
    │
    ├── ③ Dilated Conv Stack × 30 (残差 + 跳跃连接)
    │   ├── [dilations: 1,2,4,8,16,32,64,128,256,512] × 3 堆叠
    │   ├── 每层: Gated Activation (tanh · σ)
    │   └── 条件注入 (全局/局部)
    │
    ├── ④ 后处理: ReLU → Conv1x1 → ReLU → Conv1x1 → Softmax
    │
    └── 输出: p(x_t | x_<t, c) → 采样下一个采样点 x_{t+1}
```

### 为什么是"因果膨胀卷积"而非 RNN？

| 维度 | 因果膨胀卷积（WaveNet） | RNN（LSTM） |
|------|------------------------|-------------|
| 训练并行度 | ✅ **全序列可并行**（固定感受野内计算独立） | ❌ 逐时间步串行 |
| 感受野控制 | ✅ **精确控制**（层数 × 膨胀率决定） | ❌ 隐式（靠时序传播，末端衰减） |
| 梯度传播 | ✅ 无梯度消失（残差结构） | ⚠️ 有梯度截断风险 |
| 长时依赖 | ✅ 膨胀率指数增长 → 30 层覆盖 3000+ 采样点 | ✅ 理论上无限，但实际受限 |
| 参数效率 | ❌ 每层独立参数，参数量随深度线性增长 | ✅ 参数共享 |

WaveNet 选择卷积路线的核心判断：**训练时并行 = 更快的实验迭代。** 虽然推理时自回归逐点生成（类似 RNN），但训练时完整序列可以一次前向完成——这是当时 TTS 研究中最重要的效率提升。

---

## 二、各模块深度解剖

### 2.1 μ-law 量化编码器

**定位**：将 16-bit 音频（65536 个可能值）压缩为 256 个离散类别，使输出层从 65536 路分类降为 256 路。

```
输入: 16-bit PCM 采样点 s_t ∈ [-32768, 32767]
    │
    ├── μ-law 变换:
    │   f(x_t) = sign(x_t) · ln(1 + μ|x_t|) / ln(1 + μ)
    │   其中 μ = 255 (标准值)
    │
    ├── 线性量化至 [0, 255] —— 256 个整数类别
    │
    └── 输出: one-hot 向量 [batch, 256] 或标量索引
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `quantization_channels` | **256** (8-bit) | 量化等级数 |
| `mu` | 255 | μ-law 参数（标准电话语音压缩值） |
| 输入范围 | [-32768, 32767] | 16-bit PCM 标准动态范围 |
| 有效位 | ~8-bit | 量化后等效精度 |

**为什么不用 16-bit 直接预测？** 65536 路分类 → 计算量太大，且相邻值的概率分布平滑（通常相邻 `idx=i` 和 `idx=i+1` 的概率接近）。256 路是效果和计算量的平衡点。

**μ-law 的非线性特性**：信号幅值小时量化细（保留弱音细节），幅值大时量化粗（高幅值区不需要太高精度）。这符合人耳的感知特性——人对弱音的变化比强音更敏感。

---

### 2.2 Causal Convolution（因果卷积）

**定位**：保证输出只在时间上"看向过去"，不泄漏未来信息。这是自回归模型的时序约束。

#### 因果卷积的实现原理

```
标准卷积 (kernel=2, stride=1):
  y_t = w_0 · x_{t} + w_1 · x_{t+1}
  → y_t 用到了 x_{t+1} —— 这是未来的信息

因果卷积 (kernel=2, stride=1, padding='causal'):
  y_t = w_0 · x_{t-1} + w_1 · x_{t}
  → y_t 只用到 x_{t-1} 和 x_{t} —— 没有未来信息
```

**实现技巧**：在 TensorFlow 中，因果卷积通过 `padding='causal'` 实现——实际是在序列左侧填充 `kernel_size - 1` 个 0，然后做标准卷积。等价于：

```
# 左填充 kernel-1 个 0
padded = pad_left(x, [kernel_size - 1, 0])
# 标准 1D 卷积
y = conv1d(padded, kernel_size, padding='valid')
```

#### 输入投影层

```
输入: one-hot [batch, T, 256]
    │
    └── Causal Conv1d(kernel=2, 256 → residual_channels)
        └── 输出: [batch, T, residual_channels]   ← 进入膨胀栈
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `filter_width` (kernel) | **2** | 最小因果窗口——只看当前和前一个采样点 |
| `residual_channels` | **16~128**（取决于配置） | 残差路径的通道数 |

---

### 2.3 Dilated Convolution Stack（膨胀卷积堆栈）

**定位**：WaveNet 的核心。通过指数增长的膨胀率，在有限的层数内达到巨大的感受野。

#### 膨胀卷积 vs 标准卷积

```
标准卷积 (kernel=2, dilation=1):
  y_t = w_0 · x_{t-1} + w_1 · x_{t}
  → 感受野 = 2

膨胀卷积 (kernel=2, dilation=4):
  y_t = w_0 · x_{t-4} + w_1 · x_{t}
  → 感受野 = 5 (跳过了中间的 3 个点)
```

#### 30 层堆叠结构

```python
dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # 10 层，感受野 1023
           × 3  # 重复 3 次堆叠
           = 30 层
```

| 堆叠周期 | 层索引 | 膨胀率 | 累计感受野（kernel=2） |
|---------|--------|--------|--------------------|
| 堆叠 1 | 0-9 | 1,2,4,…,512 | 1023 |
| 堆叠 2 | 10-19 | 1,2,4,…,512 | 2047 |
| 堆叠 3 | 20-29 | 1,2,4,…,512 | **3071** |

**感受野 = (kernel - 1) × ∑dilation_i + 1 = 1 × (1023 + 1024 + 1024) + 1 = 3072**

以 16kHz 采样率计算：3072 / 16000 ≈ **192ms** 的上下文窗口——足够覆盖一个音节的时长（汉语一个音节约 200-300ms）。

**为什么用 3 次堆叠而不是直接 30 层膨胀到 2^29？**

```
方案 A: 三堆叠 [1..512]×3,    max dilation=512,  感受野=3072
方案 B: 单堆叠 [1..2^29],     max dilation=5.4亿, 感受野=10亿+

问题：方案 B 中，太高膨胀率的层，感受野跨越数秒音频，局部细节建模能力趋近于零
      卷积核只看到相距极远的两个点，中间的信息全部丢失
解决：重复堆叠低膨胀率周期 → 每个位置都被不同粒度的膨胀率覆盖多次
```

#### 每层 Residual Block 内部分解

```
输入: x  [batch, T, residual_channels=128]
    │
    ├── Gated Activation Unit（门控激活单元）
    │   ├── filter_conv: Dilated Causal Conv → tanh  →┐
    │   ├── gate_conv:   Dilated Causal Conv → σ(sigmoid) → ⊗ (逐元素乘)
    │   │                                            │
    │   │   张量形状: [batch, T, dilation_channels=128]
    │   └── 输出: z = tanh(W_f ∗ x) ⊙ σ(W_g ∗ x)
    │
    ├── + 条件注入（可选）
    │   ├── 局部条件: Mel 频谱上采样 → 加到 filter 和 gate 的卷积输出上
    │   └── 全局条件: 说话人嵌入 → 广播加到所有时间步
    │
    ├── 1×1 Conv → skip_connection: [batch, T, skip_channels]
    │   └── 送入后处理层（所有层的 skip 求和）
    │
    └── 1×1 Conv → residual_connection: [batch, T, residual_channels]
        └── + 输入残差 → 输出到下一层
    
    每层输出: skip_out + residual_out（两条路径）
```

#### 门控激活单元（Gated Activation Unit）

```
z = tanh(W_{f,k} ∗ x) ⊙ σ(W_{g,k} ∗ x)

其中:
  W_{f,k}: 第 k 层的 filter 卷积核（tanh 路径）
  W_{g,k}: 第 k 层的 gate 卷积核（sigmoid 路径）
  ∗: 膨胀因果卷积操作
  ⊙: 逐元素乘法
  k: 膨胀率
```

这个门控机制受 LSTM 门控理念启发——**filter 路径决定"提取什么特征"，gate 路径决定"这些特征多重要"**。sigmoid 输出 0~1 的门控值，控制 tanh 提取的特征通过多少。

与 ReLU 的对比：

| 激活函数 | 表达式 | 特性 |
|---------|--------|------|
| ReLU | `max(0, x)` | 硬门控（要么完全通过，要么完全阻断） |
| Gated Activation | `tanh(x_f) · σ(x_g)` | **软门控**（0~1 的连续控制） |
| GELU | `x · Φ(x)` | 软门控的平滑近似 |

**Gated Activation 对 WaveNet 的意义**：语音信号的局部结构极其丰富——同一个滤波器在清音段（高频噪声为主）和浊音段（周期性波形为主）需要完全不同的门控策略。软门控允许模型对每个时间步、每个通道自适应地调节信息流。

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `residual_channels` | 128 | 残差路径通道数 |
| `dilation_channels` | 128 | 膨胀卷积滤波器数（= residual_channels） |
| `skip_channels` | 128 | 跳跃连接通道数 |
| `filter_width` | 2 | 卷积核宽度（最小因果窗口） |
| `use_biases` | False | 不使用偏置（实验证明无偏置训练更稳定） |

**为什么不使用偏置（bias）？** 门控激活中原有的 `tanh` 和 `sigmoid` 函数本身包含偏移能力，额外的偏置参数可能造成冗余。实验表明去掉 bias 不影响收敛速度和最终音质。

---

### 2.4 条件注入机制（Conditioning）

**定位**：让 WaveNet 从不控制"说什么"的纯生成模型，变成"你说什么我就说什么"的条件 TTS 声码器。

#### 局部条件（Local Conditioning）—— Mel 频谱

```
Mel 频谱 [batch, frames, 80]          ← 帧率 ~80Hz
    │
    ├── Transposed Conv2d 上采样
    │   ├── 步长 1: → [batch, frames×8, 80]  (~640Hz)
    │   └── 步长 2: → [batch, frames×200, 80] (~16kHz，对齐采样率)
    │
    ├── Conv1x1 投影到 dilation_channels
    │
    └── 逐元素加到 gated activation 的 filter 和 gate 输出上:
        z' = tanh(W_f ∗ x + V_f ∗ y) ⊙ σ(W_g ∗ x + V_g ∗ y)
        
        其中 y 是上采样后的 Mel 特征，V_f 和 V_g 是条件投影矩阵
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `cin_channels` | 80 | 条件特征维度（Mel 谱 band 数） |
| 上采样方式 | **转置卷积**（可学习） | 比线性插值更灵活 |

#### 全局条件（Global Conditioning）—— 说话人嵌入

```
说话人嵌入 [batch, 16]                 ← 单个向量，不随时间变化
    │
    ├── 广播到所有时间步: [batch, T, 16]
    │
    ├── Conv1x1 投影到 dilation_channels
    │
    └── 逐元素加到 gate/filter 输出上:
        z' = tanh(W_f ∗ x + V_f ∗ y + G_f ∗ h) ⊙ σ(W_g ∗ x + V_g ∗ y + G_g ∗ h)
        
        其中 h 是广播后的说话人嵌入，G_f, G_g 是投影矩阵
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `gin_channels` | 16 (典型值) | 说话人嵌入维度 |
| `n_speakers` | 7~100+ | 训练集中的说话人数量 |

**为什么全局条件加在每层而非仅入口？** 和 Cross-Attention 的理由类似——生成过程的每一层都可能需要 speaker 信息来指导"音色"相关的声学细节生成。只加在入口时，深层网络可能丢失条件信号。

---

### 2.5 后处理与输出层

**定位**：将所有层的跳跃连接汇总，通过非线性变换映射到 256 路分类输出。

```
所有 30 层的 skip_out 求和:
    sum_skip = Σ(skip_i) for i in 0..29
    shape: [batch, T, skip_channels=128]
    │
    ├── ReLU 激活
    │
    ├── Conv1d(skip_channels → skip_channels): [batch, T, 128]  ← 保持维度
    ├── ReLU 激活
    │
    ├── Conv1d(skip_channels → quantization_channels): [batch, T, 256]
    │
    ├── Softmax
    │
    └── 输出: p(x_t | x_{<t}, c) —— 当前采样点在 256 个类别上的概率分布
```

**为什么用跳连求和而不是级联？** 30 层级联的话，最后一层的输入需要包含前面所有层的信息——参数量爆炸。跳连求和将每层的输出独立投影到 skip_channels 再求和，总参数只是 30 × (dilation_channels × skip_channels) 的投影矩阵。

---

## 三、推理流程演练

以生成 1 秒（16000 个采样点）的 "a" 发音为例：

### Stage 1: 条件准备（针对 TTS 场景）

```
10 秒 Mel 频谱 (80 dims, 12.5ms hop = 800 frames)
    → Transposed Conv 上采样至 16kHz 对齐采样率
    → [800, 80] → 投影 → [800, dilation_channels=128]
```

### Stage 2: 逐点自回归生成（核心循环）

```
Step 0:  输入: [<BOS>] (μ-law 128，静音态)
         → 前向 30 层 → 输出 256 概率分布 → 采样得到 x_1

Step 1:  输入: [x_0, x_1] (已生成的 1 个采样点)
         → 前向 30 层 → 输出 p(x_2) → 采样 x_2

Step 2:  输入: [x_0, x_1, x_2]
         → ...

Step 15999: 输入: [x_0 ... x_15999]
             → 输出 x_16000
             → 得到 1 秒完整音频
```

### Stage 3: 条件对齐（每步）

```
每个采样步 t 中:
  1. 确定当前采样的时间位置: t / 16000 = t_sec
  2. 找到对应的 Mel 帧索引: floor(t_sec / 0.0125)
  3. 从条件缓存中读取该帧的嵌入向量
  4. 该向量加入每层 gated activation 的 bias 项
```

### 各阶段数据维度变化

```
| 阶段 | 输入形状 | 操作 | 输出形状 |
|------|---------|------|---------|
| 量化 | [1, 1] (标量采样点) | μ-law + one-hot | [1, 256] |
| 输入投影 | [1, 256] | Causal Conv1x2(256→128) | [1, 128] |
| 膨胀栈 ×30 | [1, 128] | Gated Act + 残差 + 跳连 | 每层 [1,128] skip |
| 跳连求和 | 30 × [1, 128] | Σ(skip_i) | [1, 128] |
| 后处理 Conv1 | [1, 128] | ReLU + Conv1d(128→128) | [1, 128] |
| 后处理 Conv2 | [1, 128] | ReLU + Conv1d(128→256) | [1, 256] |
| Softmax + 采样 | [1, 256] | argmax / 随机采样 | [1] (标量) |
```

---

## 四、性能优化全景

### 4.1 推理瓶颈分析

```
生成 1 秒音频 (16kHz = 16000 采样点):
    ┌────────────────────────────────────────────┐
    │  每步计算量:                                │
    │    30 层 × (gate + filter conv + 投影 + 激活) │
    │    ≈ 30 × 5 = 150 个 Op/步                   │
    │                                              │
    │  总计: 16000 × 150 = 2,400,000 次 Op        │
    │  (对比: 实时需在 1 秒内完成)                  │
    │                                              │
    │  → CPU 实现: 1 秒音频需 ~3-5 分钟            │
    │  → GPU 实现: 1 秒音频需 ~30-60 秒            │
    └──────────────────────────────────────────────┘
```

核心瓶颈：**自回归逐点生成不可并行**。每一步待上一步完成后才能开始，GPU 在这类 workload 上利用率极低。

### 4.2 后续优化方向

WaveNet 的低速促使了一系列加速工作：

| 优化方案 | 核心思路 | 加速比 | 音质损失 |
|----------|---------|--------|---------|
| **Parallel WaveNet** (2017) | 教师蒸馏 + Inverse Autoreressive Flow | **1000×** | 几乎无损 |
| **ClariNet** (2018) | 高斯分布假设 + 并行波形生成 | **1000×** | 略低于 WaveNet |
| **WaveGlow** (2018) | Normalizing Flow 完全并行 | **500×** | MOS ~4.5 接近 |
| **LPCNet** (2019) | LPC + 稀疏 RNN，CPU 实时 | CPU 实时 | MOS ~3.8 略降 |

**Parallel WaveNet** 用 WaveNet 作为教师模型，训练一个并行（非自回归）的学生模型。教师 WaveNet 生成概率分布，学生通过学习这些分布来实现并行生成。这证明了 WaveNet 的**蒸馏价值**可能大于其直接使用价值。

---

## 五、架构设计的深层思考

### 5.1 为什么 WaveNet 做声码器而非端到端 TTS？

WaveNet 本身能生成语音，但**不能控制"说什么"**。它的原始版本是无条件生成——给它一段 seed 音频，它接续生成风格相似的语音/音乐，但内容不可控。

条件 WaveNet（以 Mel 频谱为条件）才解决了"说指定内容"的问题。但即使如此，WaveNet 也只做 Mel → 波形这一步。**为什么不从文本直接到波形？** 因为：
1. 文本到波形的映射太长（几个字符 → 数万采样点），对齐困难
2. 文本信息密度远低于波形——让 WaveNet 边读文本边生成波形，注意力分散

所以 WaveNet 在 TTS 中的角色是**声码器**而非完整 TTS 系统——它接收已经压缩好的 Mel 频谱，专注于"还原高质量波形"这一件事。

### 5.2 WaveNet 与后续声码器的关系

```
WaveNet (2016) — 奠基
    │
    ├── Distillation → Parallel WaveNet (2017) — 速度提升 1000×
    │
    ├── Flow-based → WaveGlow (2018) — 完全并行
    │   └── Glow-TTS (2020) — 将 Flow 扩展到声学模型
    │
    └── GAN-based → MelGAN (2019) — 极速
                    └── HiFi-GAN (2020) — 音质接近于 WaveNet，速度快 1000×
                        └── VITS Decoder (2021) — HiFi-GAN 作为 VITS 的解码器
```

HiFi-GAN 是目前实际取代 WaveNet 的主力声码器——音质 MOS 接近 WaveNet，但速度快 1000× 以上。

### 5.3 WaveNet 参数估算

以标准配置（residual_channels=128, skip_channels=128, dilation_channels=128）：

```
输入投影层:
  Causal Conv1d: 256 × 128 = 32,768 (权重)

膨胀栈 × 30 层：
  每层:
    2 个膨胀卷积 (filter + gate): 2 × (2 × 128 × 128) = 65,536
    1×1 Conv (dense→residual): 128 × 128 = 16,384
    1×1 Conv (skip): 128 × 128 = 16,384
    每层合计: ~98,304
  30 层合计: ~2,949,120

后处理层:
  Conv1: 128 × 128 = 16,384
  Conv2: 128 × 256 = 32,768
  合计: ~49,152

总计: ~303 万参数

(条件注入投影矩阵: ~256 × 128 × 2 × 30 = ~196 万，按需加载)
```

> WaveNet 的参数规模比常想象的小得多（约 300 万）。它的计算瓶颈不在参数量，而在**自回归生成过程中每一步都要完整走一遍 30 层网络**。

---

## 六、实际部署效果

### 6.1 主观评测（MOS）

| 系统 | MOS | 数据来源 |
|------|-----|---------|
| **WaveNet（输入语言学特征）** | 4.341 | 英语 TTS |
| **WaveNet（输入 Mel 频谱）** | **4.53** | 英语 TTS（LJSpeech） |
| **Parallel WaveNet** | 4.41 | 近似无损 |
| **HiFi-GAN** | 4.35~4.52 | 接近 WaveNet |
| **人类录音（Ground Truth）** | 4.58 | — |

### 6.2 实际场景表现

| 场景 | 表现 |
|------|------|
| **有声书/朗读风格** | 自然度极高，MOS >4.3，无明显人工感 |
| **音乐生成** | 可以生成钢琴等乐器的原始音频（但内容不可控） |
| **多说话人** | 全局条件 + 说话人嵌入可合成不同音色 |
| **静音段处理** | 无输入时随机生成噪声（条件模式的呼吸声效果自然） |

### 6.3 生产部署

- **Google Assistant**（2019-2022 间）：Tacotron 2 + WaveNet 组合用于部分语音合成流量，后切换至更高效的 Tacotron 2 + WaveRNN/LPCNet
- **Google Cloud Text-to-Speech**：WaveNet 语音是其最高品质档位（WaveNet voices），至今仍是 Google Cloud TTS 的旗舰选项

---

## 七、总结：一张图看穿 WaveNet

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WaveNet 架构全景 (条件 TTS 模式)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Mel 频谱 (80 dims, ~80Hz)    文本编码器 (Tacotron 2) 输出         │
│       │                                                             │
│  ┌────┴────┐                                                       │
│  │ 上采样  │  Transposed Conv → 16kHz 对齐采样率                    │
│  └────┬────┘                                                       │
│       │                                                             │
│  ┌────┴────────────────────────────────────────────────────┐        │
│  │              Dilated Conv Stack (30 层)                  │        │
│  │                                                          │        │
│  │  输入采样点 x_t (μ-law 256)                               │        │
│  │       │                                                   │        │
│  │  ┌────┴─────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐   │        │
│  │  │Causal Proj│  │ d=1  │→│ d=2  │→│ ...  │→│d=512 │→... │  × 3   │
│  │  └────┬─────┘  │filter│  │filter│  │filter│  │filter│        │
│  │       │         │ gate │  │ gate │  │ gate │  │ gate │        │
│  │       │         │skip→ │  │skip→ │  │skip→ │  │skip→ │        │
│  │       └────────→│resi→ │→│resi→ │→│resi→ │→│resi→ │        │
│  │                 └──────┘  └──────┘  └──────┘  └──────┘   │        │
│  │                                                          │        │
│  │  条件注入: Mel ↑ + Speaker Emb → Gate Bias (每层)         │        │
│  └────┬────────────────────────────────────────────────────┘        │
│       │                                                             │
│  ┌────┴──────────┐                                                  │
│  │  Σ skip_connections (30 层求和)  │                                  │
│  └────┬──────────┘                                                  │
│       │                                                             │
│  ┌────┴──────────┐                                                  │
│  │ ReLU → Conv   │  ReLU → Conv → Softmax → 256 路分类              │
│  └────┬──────────┘                                                  │
│       │                                                             │
│  ┌────┴──────────┐                                                  │
│  │ p(x_t|x_<t,c) │  → 采样 → x_t (下一个采样点)                      │
│  └───────────────┘                                                  │
│                                                                     │
│  一句话总结 WaveNet：                                               │
│  "30 层门控膨胀卷积等效于 ~3000 个采样点的因果感受野，               │
│   逐点自回归代价巨大，但音质至今是 SOTA 上限。"                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 附录：关键配置与实现速查

### ibab/tensorflow-wavenet 配置参数

```python
class WaveNetModel:
    def __init__(self,
                 batch_size=1,
                 dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512] * 3,
                 filter_width=2,
                 residual_channels=16,     # 标准论文用 128
                 dilation_channels=32,     # 标准论文用 128
                 skip_channels=16,         # 标准论文用 128
                 quantization_channels=256,
                 use_biases=False,
                 scalar_input=False,
                 initial_filter_width=32,
                 global_condition_channels=None,
                 global_condition_cardinality=None):
```

### r9y9/wavenet_vocoder 配置（Tacotron 2 版本）

```python
# 16kHz 配置示例（LJSpeech 标准）
sample_rate = 16000
hop_length = 200  # 12.5ms per frame
num_mels = 80

model = {
    'out_channels': 10,            # MoL 分布的组件数（不是 μ-law 256）
    'layers': 30,                  # 总层数
    'stacks': 3,                   # 堆叠数
    'residual_channels': 128,      # 
    'gate_channels': 256,          # (filter + gate) = 2 × residual_channels
    'skip_channels': 128,
    'cin_channels': 80,            # Mel 条件维数
    'gin_channels': -1,            # 全局条件（无）
    'upsample_conditional_features': True,
    'upsample_scales': [4, 4, 4, 4],  # 4 层转置卷积上采样
    # freq_axis_kernel_size：频谱方向的卷积核宽度（用于上采样）
    # 16kHz / 200 hop = 80Hz → 目标 16kHz = 80 × 200 → 总共 200× 上采样
    # [4, 4, 4, 4] 乘积 = 256 > 200，略超但够用
}
```

### 关键超参数选择逻辑

| 参数 | 选择 | 原因 |
|------|------|------|
| 输出分布 | **MoL (Mixture of Logistics)** | 连续分布比 256 路离散分类更精细，音质更高 |
| 层数 30 | 3 × [1..512] | 感受野 3072 点 ≈ 192ms，覆盖一个音节的长度 |
| kernel=2 | 2 个采样点 | 最小因果窗口 + 膨胀后感受野线性叠加 |
| bias=False | 去掉 | 门控激活天然含偏置，额外 bias 冗余 |

---

*本文基于 WaveNet 原始论文 (van den Oord et al., 2016)、r9y9/wavenet_vocoder 实现源码 (MIT License, GitHub)、ibab/tensorflow-wavenet 实现源码 (MIT License, GitHub) 整理分析。*

**Sources:**
- [WaveNet: A Generative Model for Raw Audio - DeepMind (arXiv 2016)](https://arxiv.org/abs/1609.03499)
- [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions - Google (arXiv 2017)](https://arxiv.org/abs/1712.05884)
- [r9y9/wavenet_vocoder - GitHub (MIT)](https://github.com/r9y9/wavenet_vocoder)
- [ibab/tensorflow-wavenet - GitHub (MIT)](https://github.com/ibab/tensorflow-wavenet)
- [Parallel WaveNet: Fast High-Fidelity Speech Synthesis - DeepMind (ICML 2017)](https://arxiv.org/abs/1711.10433)
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis - NeurIPS 2020](https://arxiv.org/abs/2010.05646)
