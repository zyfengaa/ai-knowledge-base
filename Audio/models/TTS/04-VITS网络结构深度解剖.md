# VITS 网络结构深度解剖

> KAIST（韩国科学技术院）出品 | 条件变分自编码器 + 对抗学习 + 端到端 TTS
> 论文发表于 ICML 2021 | GitHub 12K+ Stars | 社区衍生生态最广的 TTS 模型

---

## 写在前面：VITS 的定位

VITS 是第一个将**单阶段端到端 TTS**（文本→波形，无需中间 Mel 频谱）做到实用水平的模型。在它之前：

| 范式 | 代表模型 | 管线段数 |
|------|---------|---------|
| 两阶段流水线 | Tacotron 2 + WaveNet | 文本→(声学模型)→Mel→(声码器)→波形 |
| 两阶段联合 | Tacotron 2 + HiFi-GAN | 同上，声码器被优化但仍是两段 |
| **VITS** | **单阶段端到端** | **文本→(一个模型)→波形** |

VITS 把 VAE（变分自编码器）、Normalizing Flow（归一化流）、GAN（生成对抗网络）、MAS（单调对齐搜索）融合为一个模型。**它不是首创其中任何一个技术，而是第一次把它们组合在一起达到了实用音质。**

---

## 一、整体架构设计哲学

### 核心思想

> **"把 Mel 频谱扔掉，让模型从噪声直接生成波形；用 VAE 做分布对齐，用 MAS 做对齐，用 GAN 保音质。"**

详细解读：
1. **扔掉 Mel 频谱**：不输出中间频谱，直接生成波形。声学模型和声码器合并为一个网络
2. **VAE 架构**：从真实波形提取隐变量 z（后验），让文本生成与之对齐的先验分布
3. **MAS 替代 Attention**：用动态规划找到音素和帧的最优对齐——稳定、不跳词
4. **GAN 提升音质**：HiFi-GAN 解码器 + 对抗训练减少"生成模型模糊"的问题

### 架构总览

```
训练阶段:
真实波形 (Ground Truth Audio)
    │
    ├── Posterior Encoder (非因果 WaveNet)
    │   └── 输出: 隐变量 z ~ q(z|x)
    │
文本序列 (Phoneme)
    │
    ├── Text Encoder (Transformer)
    │   └── 输出: 先验参数 μ_θ, σ_θ
    │
    ├── Normalizing Flow f_θ
    │   └── 将先验和后验对齐: p_θ(z|c)
    │
    ├── MAS (Monotonic Alignment Search)
    │   └── 找到音素和帧的最优对齐 A
    │
    ├── Stochastic Duration Predictor
    │   └── 预测每个音素的持续时间 d ~ P(d|h_text)
    │
    └── HiFi-GAN Decoder
        └── z → 波形

推理阶段:
文本 → Text Encoder → Flow → 从先验采样 z
     → Duration Predictor → Length Regulator → Decoder → 波形
```

---

## 二、各模块深度解剖

### 2.1 Posterior Encoder（后验编码器）

**定位**：训练时从真实波形中提取"隐变量 z"，作为"参考答案"提供给模型。

```
输入: 真实波形 [batch, T_audio]
    │
    ├── 16-bit PCM → μ-law 变换（可选）/ 或原始浮点
    │
    ├── WaveNet 残差块 × 16 层（非因果）
    │   ├── Dilated Conv (dilation 1,2,4,...,256)
    │   ├── Gated Activation: tanh(W_f∗x) ⊙ σ(W_g∗x)
    │   ├── 条件注入: 无（后验编码器不接收外部条件）
    │   ├── 残差连接 + 1×1 Conv 投影
    │   └── Skip Connections 累积
    │
    ├── 线性投影 → μ_z, log_σ_z (隐变量 z 的后验分布参数)
    │   └── z ~ N(μ_z, σ_z)
    │
    └── 输出: z [batch, z_channels, T_frame]  (T_frame = T_audio / hop_length)
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `residual_channels` | 192 | WaveNet 残差块通道数 |
| `dilation_channels` | 192 | 门控激活中 filter/gate 的通道数 |
| `skip_channels` | 192 | 跳跃连接通道数 |
| `num_layers` | 16 | WaveNet 残差块层数 |
| `z_channels` | 192 | 隐变量 z 的维度（= residual_channels） |
| `filter_width` | 3 | 卷积核宽度 |
| 因果性 | **非因果** | 每个位置可以看到前后帧——提取更准确的全局声学特征 |

**非因果（Non-Causal）设计**：后验编码器在训练时看到完整的波形，因此使用非因果（双向）膨胀卷积。这不同于 WaveNet 声码器的因果设计（只能看过去）。训练时能看到未来，推理时模仿未来——这是 VAE 训练的自然属性。

---

### 2.2 Text Encoder（文本编码器）

**定位**：将音素序列编码为隐藏状态序列，作为先验分布的输入。

```
输入: 音素序列 [batch, T_phone]
    │
    ├── Embedding(vocab_size, hidden_channels=192)
    │
    ├── Transformer Encoder × 6 层
    │   ├── Multi-Head Self-Attention (2 heads, 192 dims)
    │   ├── Conv1D (kernel=3, 192 → 768 → 192)
    │   ├── LayerNorm + Residual
    │   └── Dropout(0.1)
    │
    └── 输出: h_text [batch, T_phone, 192]
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `hidden_channels` | 192 | 隐藏维度（较小） |
| `num_layers` | 6 | Transformer 层数 |
| `num_heads` | 2 | 注意力头数（每头 96 维） |
| `filter_channels` | 768 | FFN 中间维度（4×） |
| `kernel_size` | 3 | Conv1D 卷积核宽度 |

---

### 2.3 Normalizing Flow（归一化流）

**定位**：桥接先验分布（文本条件）和后验分布（从真实波形提取）。这是 VITS 能工作的关键数学模块。

#### Flow 解决了什么问题？

```
没有 Flow 的情况:
  后验分布 q(z|x) = N(μ_z, σ_z)  ← 从真实波形提取
  先验分布 p(z|c) = N(μ_θ, σ_θ)  ← 从文本条件预测
  KL(q||p) 在简单高斯假设下很难降到 0 —— 因为文本->波形的映射
  太复杂了，一个简单高斯分布拟合不了

有 Flow 的情况:
  后验 z ~ q(z|x)  ← 从真实波形提取
  z' = f_θ(z)      ← 用 Flow 将后验 z 变换为简单分布
  KL(f_θ(z), N)     ← 在简单分布空间计算 KL，更容易优化
  推理时: 从 N 采样 → f_θ^{-1}(sample) → 得到隐变量 z
```

#### Flow 的具体结构

```
输入: z [batch, z_channels=192, T_frame] (从后验编码器采样)
    │
    ├── 仿射耦合层 (Affine Coupling Layer) × 4
    │
    │   每层 (WaveNet Block):
    │   ├── 将 z 沿通道轴分为 z_a, z_b (各 96 维)
    │   ├── z_a 保持不变
    │   ├── z_b 通过 WaveNet Block (dilated conv + gate) → 输出 scale, shift
    │   │   └── WaveNet Block 与后验编码器的残差块结构相同
    │   ├── z_b' = z_b * exp(scale) + shift (仿射变换)
    │   └── 拼接 [z_a, z_b'] → 输出
    │
    └── 输出: f_θ(z) [batch, z_channels, T_frame]
```

| 参数 | 值 | 含义 |
|------|-----|------|
| Flow 层数 | 4 | 仿射耦合层堆叠数 |
| 每层 WaveNet 块 | 4 个膨胀卷积 + gate + 1×1 | 用于计算 scale/shift |
| 通道分裂 | 50/50 | z_a, z_b 各 half |

**为什么是 4 层 Flow 而不是更多？** 论文消融实验显示：1 层 Flow → KL 降不下来；4 层 Flow → KL 收敛，音质好；8 层 Flow → 音质不再提升，但训练更慢。4 层是精度和效率的平衡点。

---

### 2.4 MAS（Monotonic Alignment Search）

**定位**：找到文本音素和语音帧之间的最优单调对齐。这是 VITS 替代 Attention 的组件。

```
问题: 文本 "h e l l o" (5 个音素)
      音频帧 [x_1, x_2, ..., x_T] (T 帧)
      每个音素对应连续的若干帧:
        h → x_1~x_20, e → x_21~x_45, l → x_46~x_70, ...

MAS 求解: 找到这个映射使得:
  A* = argmax_A log p_θ(z | c_text, A)
  = 使得在给定对齐 A 下，后验 z 被先验预测得最准确
```

**数学实现**：用动态规划求解。这是一个标准的"已知序列 A 和序列 B，找到最优单调匹配"的问题。

```
状态定义: dp[i][j] = 前 i 个音素、使用到第 j 帧时的最优 log-likelihood
转移方程: dp[i][j] = max(dp[i][j-1], dp[i-1][j-1]) + log_likelihood(i, j)

其中 log_likelihood(i, j) 表示音素 i 对齐到第 j 帧的"匹配度"
这是一个完全确定性的算法——没有可学习参数
```

| 方法 | 可学习参数？ | 训练时使用？ | 推理时使用？ |
|------|------------|------------|------------|
| **MAS** | ❌ 无参数（纯动态规划） | ✅ 提供对齐标签 | ❌（由 Duration Predictor 替代） |
| **Attention (Tacotron 2)** | ✅ 有参数 | ✅ 自学习 | ✅ 自回归 |
| **Duration Predictor (FastSpeech 2)** | ✅ 有参数 | ✅ 从 MFA 学习 | ✅ 预测 |

---

### 2.5 Stochastic Duration Predictor（随机时长预测器）

**定位**：将时长建模为概率分布（而非 FastSpeech 2 的固定值），推理时采样得到自然的节奏变化。

```
输入: h_text [batch, T_phone, 192]
    │
    ├── Flow-based Duration Predictor:
    │   ├── Conv1D × 2 (kernel=3)
    │   ├── 条件输入: h_text + Speaker Embedding
    │   ├── 训练时: 从真实时长 d (来自 MAS) 学习分布 p(d|h)
    │   └── 推理时: 从 p(d|h) 采样时长, 而不是取期望值
    │
    └── 输出: d_i ~ P(d|h_i) = 音素 i 的持续帧数
```

**为什么"随机"比"固定"好？**

```
FastSpeech 2: 对同一文本，每次生成节奏完全相同
  → 听起来"机械"——因为真人每次说话都有自然的节奏变化

VITS: 对同一文本，每次生成可能会有微小节奏差异
  → 听起来"自然"——因为采样带来的多样性
	
"我今天很开心" (FastSpeech 2):
  每遍: "我(5帧) 今(3) 天(4) 很(6) 开(8) 心(10)" (相同)
	
"我今天很开心" (VITS):
  第1遍: "我(5帧) 今(3) 天(4) 很(6) 开(8) 心(10)" 
  第2遍: "我(6帧) 今(4) 天(3) 很(5) 开(9) 心(11)"
  第3遍: "我(5帧) 今(4) 天(5) 很(7) 开(7) 心(9)" 
  → 每次微妙的节奏变化更接近真人
```

**Flow-based Duration Predictor 的具体实现**：与先验的 Normalizing Flow 类似，它也是一个 Flow 模型——用仿射耦合层将"真实时长"映射到"标准正态分布"，再从标准正态采样解码为"预测时长"。

---

### 2.6 HiFi-GAN Decoder（解码器）

**定位**：将隐变量 z 直接解码为原始波形。这是 VITS 的"声码器"。

**HiFi-GAN 结构摘要**（完整结构见 HiFi-GAN 论文）：

```
输入: z [batch, z_channels=192, T_frame]
    │
    ├── 转置卷积上采样 (×2 或 ×4) — 从帧率到采样率
    │   过程: 192 → 512 → ... → 1 (单通道音频)
    │
    ├── Multi-Receptive Field Fusion (MRF)
    │   ├── 并行多组残差块
    │   ├── kernel 大小: 3, 5, 7 (不同感受野)
    │   └── 输出求和 → 多尺度特征融合
    │
    └── 输出: [batch, 1, T_audio] (原始波形)
```

**对抗训练**：除生成器外，还有一个多周期判别器（MPD，Multi-Period Discriminator），对生成波形和真实波形做判别。生成器损失 = L1 Mel loss + 对抗 loss + 特征匹配 loss。

| 损失项 | 来源 | 作用 |
|------|------|------|
| L1 Mel loss | 预测 vs 目标 Mel 频谱 | 保证频谱一致性 |
| KL 散度 | q(z|x) vs p(z|c) | 后验和先验分布对齐 |
| 对抗 loss (生成器) | 判别器对生成音频的评分 | 提升感知音质 |
| 特征匹配 loss | 判别器中间特征 | 稳定 GAN 训练 |

---

## 三、推理流程演练

以下以 "hello"（2 个音节）为例：

### Stage 1: 文本→音素→先验分布

```
"hello" → G2P → [hh, ax, l, ow] (4 个音素)
    → Text Encoder (6× Transformer) → h_text [4, 192]
    → Flow → 先验分布 N(μ_θ, σ_θ) 的参数 [4, 192] × 2
```

### Stage 2: 时长采样

```
随机时长预测器:
  d_hh ~ P(d|h_hh) = 采样得到 12 帧
  d_ax ~ P(d|h_ax) = 6 帧
  d_l  ~ P(d|h_l)  = 14 帧
  d_ow ~ P(d|h_ow) = 16 帧
  
总帧数: 12 + 6 + 14 + 16 = 48 帧
```

### Stage 3: 先验采样

```
从先验分布 N(μ_θ, σ_θ) 采样 z:
  z [batch, 192, 4] — 4 个音素的先验采样
    → Length Regulator（按时长展开）
    → z [batch, 192, 48] — 48 帧的隐藏状态
```

### Stage 4: HiFi-GAN 解码

```
z [batch, 192, 48]
    → 转置卷积上采样 × 4
    → MRF 残差块融合
    → 波形 [batch, 1, 48×256=12288 采样点] (@ 16kHz ≈ 0.77 秒)
```

---

## 四、训练与推理对比

| 阶段 | 后验编码器 | Flow | MAS | Duration Predictor | HiFi-GAN |
|------|-----------|------|-----|-------------------|---------|
| 训练 | ✅ 从真实波形提取 z | ✅ 将 z 映射到简单分布 | ✅ 计算最优对齐 | ✅ 从 MAS 对齐学习 | ✅ 对抗训练 |
| 推理 | ❌ 移除 | ✅ 逆变换: N→z | ❌ 移除 | ✅ 采样生成时长 | ✅ 解码 z→波形 |

**关键差别**：后验编码器和 MAS 只在训练时使用——推理时完全不需要。Flow 在训练和推理时都用（训练时前向映射，推理时逆向采样）。

---

## 五、VITS 参数量估算

```
Posterior Encoder:
  16 × WaveNet Block:
    Dilated Conv (3×192×192) × 2 (filter+gate) = ~221K
    1×1 Conv (192×192) × 2 (dense+skip) = ~74K
  每层: ~295K × 16 = ~4.7M
  Posterior Encoder 合计: ~5M

Text Encoder:
  Embedding: 80 × 192 = 15K
  Transformer × 6:
    Attn: 3 × (192×192) + 192×192 = ~147K
    FFN: 192×768 + 768×192 = ~295K
  每层: ~442K × 6 = ~2.7M
  Text Encoder 合计: ~2.8M

Normalizing Flow:
  4 × Affine Coupling:
    WaveNet Block × 4: 4 × (3×192×192 ≈ 110K) × 2+... ≈ ~3M
  Flow 合计: ~3M

HiFi-GAN Decoder:
  Transposed Conv: ~1M
  MRF Blocks: ~2M
  Decoder 合计: ~3M

Duration Predictor:
  Flow-based: ~1M

判别器 (只训练用, 推理不用):
  MPD: ~2M

总参数量 (训练): ~17M
总参数量 (推理): ~15M (不含判别器)
```

> VITS 的参数量（~15M）在 2021 年属于中等水平，远小于 Tacotron 2（~30M）但比 FastSpeech 2（~27M）小。

---

## 六、VITS 的生态影响——真正的力量不在论文里

VITS 论文本身 MOS 4.43，音质好但并不是革命性提升。它的真正影响力来自开源的**社区衍生生态**：

| 衍生项目 | 改进方向 | 影响力 |
|---------|---------|--------|
| **Bert-VITS2** | BERT 替换 Text Encoder + VQ | 中文社区最广泛使用的 TTS |
| **GPT-SoVITS** | GPT + VITS + 少样本克隆 | ⭐ **2024 年开源语音合成热度最高** |
| **SoVITS-SVC** | VITS 变体做唱歌声音转换 (SVC) | B 站翻唱视频大量使用 |
| **MoeGoe** | VITS 推理引擎 (ONNX/C++) | 端侧部署 |
| **DDSP-SVC** | VITS + 可微数字信号处理 | 高质量歌声合成 |

**为什么 VITS 的衍生生态这么活跃？**

1. **单阶段端到端**：一个模型直接生成波形——部署比两阶段（Tacotron 2 + 声码器）简单得多
2. **MIT 许可**：完全开放，允许商用和改写
3. **结构模块化**：替换 Text Encoder（Bert-VITS2）、修改 Duration Predictor（GPT-SoVITS）、换解码器（DDSP-SVC）——每个模块可以独立改进
4. **训练门槛低**：在单卡 V100/A100 上可以在 2-3 天内完成 LJSpeech 训练

---

## 七、消融实验——VITS 的关键设计

VITS 论文的消融实验回答了以下问题：

| 去掉的模块 | 效果影响 | 说明 |
|-----------|---------|------|
| **Normalizing Flow** | MOS ↓ ~0.4 | 先验分布拟合不了复杂的 q(z|x)，音质大幅下降 |
| **对抗训练 (GAN)** | MOS ↓ ~0.3 | 生成音质模糊，HiFi-GAN 的对抗训练对感知质量的提升至关重要 |
| **随机时长预测器** | MOS ↓ ~0.1~0.15 | 固定时长的节奏较机械，但影响不大 |
| **文本 Encoder 层数减半 (6→3)** | MOS ↓ ~0.1 | 浅层编码器语义理解略差 |
| **Flow 层数减半 (4→2)** | KL 散度 ↑ | 更少的 Flow 层无法充分对齐分布 |

**结论**：Flow 和 GAN 训练是 VITS 音质的两大支柱，缺一不可。

---

## 八、VITS vs FastSpeech 2 vs Tacotron 2 对比

| 维度 | Tacotron 2 | FastSpeech 2 | VITS |
|------|-----------|-------------|------|
| 架构 | Enc-Dec + Attention | FFT Block + V.Adaptor | **VAE + Flow + GAN** |
| 对齐 | Attention（隐式） | Duration Predictor（显式） | **MAS（动态规划）** |
| 输出 | Mel 频谱 | Mel 频谱 | **波形（端到端）** |
| 声码器 | WaveNet（分离） | HiFi-GAN（分离） | **内置 HiFi-GAN** |
| MOS | 4.53 | 4.45 | 4.43 |
| 推理速度 | 慢（自回归） | **最快（并行）** | 中（Flow + GAN） |
| 训练复杂度 | 低（两阶段独立） | 中 | **高（VAE+GAN 联合）** |
| 社区衍生 | 少 | 少 | **极多（生态最广）** |

---

## 九、总结：一张图看穿 VITS

```
┌────────────────────────────────────────────────────────────────────────┐
│                      VITS 架构全景                                     │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  训练: (实线)                    推理: (虚线部分跳过)                   │
│                                                                        │
│  真实波形                             文本                              │
│      │                                  │                              │
│  ┌───┴──────┐                     ┌─────┴──────┐                      │
│  │Posterior  │                     │  Text Enc  │                      │
│  │ Encoder   │                     │ (Trans ×6) │                      │
│  │(WN 16lay) │                     └─────┬──────┘                      │
│  └───┬──────┘                             │ h_text                     │
│      │ z (隐变量)                         │                            │
│      │                                    │                            │
│  ┌───┴───────────┐               ┌──────────┴──────────┐              │
│  │  Flow (4层)    │               │  Flow (4层, 逆向)    │              │
│  │  q(z)→简单分布  │               │  简单分布→先验 z     │              │
│  └───┬───────────┘               └──────────┬──────────┘              │
│      │  KL(后验||先验)                      │                            │
│      │                                      │                            │
│  ┌───┴──────────┐  ┌─────────────────┐     │                            │
│  │   MAS        │→│ Duration Pred   │     │                            │
│  │  (动态规划)   │  │ (随机时长采样)   │     │                            │
│  └──────────────┘  └────────┬────────┘     │                            │
│                             │              │                            │
│                       ┌─────┴─────┐        │                            │
│                       │ Length Reg │        │                            │
│                       └─────┬─────┘        │                            │
│                             │              │                            │
│                       ┌─────┴───────┐      │                            │
│                       │HiFi-GAN Dec│←─────┘ (流部分拼成完整 z)           │
│                       │ (转置Conv)  │                                    │
│                       └─────┬───────┘                                    │
│                             │ 波形                                      │
│                             │                                           │
│  ┌─────────────────────────────┐                                        │
│  │ Multi-Period Discriminator  │ ← 对抗训练（仅训练）                    │
│  └─────────────────────────────┘                                        │
│                                                                        │
│  一句话总结 VITS：                                                      │
│  "VAE 做分布桥梁（后验→先验），MAS 做单调对齐（音素↔帧），               │
│   Flow 提升 KL 散度表达能力，HiFi-GAN 做波形解码——单阶段端到端。"        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 附录：关键配置速查

### VITS 完整超参数表（LJSpeech 版本）

```python
# 音频参数
sample_rate = 22050          # 采样率
hop_length = 256             # 帧移 (~11.6ms)
win_length = 1024            # 窗口大小
n_fft = 1024                 # FFT 点数
n_mels = 80                  # Mel 带数
mel_fmin = 0.0
mel_fmax = None

# 编码器
hidden_channels = 192        # 全局隐藏维度
inter_channels = 192         # Flow 中中间通道数
filter_channels = 768        # FFN 层中间维度
kernel_size = 3              # Conv1D 卷积核
p_dropout = 0.1              # Dropout 率
n_layers_enc = 6             # Text Encoder Transformer 层数
n_layers_flow = 4            # Normalizing Flow 层数

# 后验编码器 (WaveNet 解码器)
residual_channels = 192      # 后验编码器残差通道
dilation_channels = 192      # 膨胀卷积通道
skip_channels = 192          # 跳跃连接通道
n_layers_post = 16           # WaveNet 块层数

# HiFi-GAN
upsample_rates = [8, 8, 2, 2]  # 上采样率
upsample_kernel_sizes = [16, 16, 4, 4]  # 上采样卷积核
resblock_kernel_sizes = [3, 7, 11]     # MRF 卷积核
resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]  # 膨胀率

# 训练
batch_size = 16
learning_rate = 2e-4
adam_betas = [0.8, 0.99]
num_epochs = 2000
segment_size = 8192          # 训练时随机裁剪的音频长度
```

---

*本文基于 VITS 论文 (Kim et al., ICML 2021)、jaywalnut310/vits 官方实现 (MIT License, GitHub)、Bert-VITS2 及 GPT-SoVITS 社区衍生项目整理分析。*

**Sources:**
- [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech - KAIST (ICML 2021)](https://arxiv.org/abs/2106.06103)
- [jaywalnut310/vits - GitHub (MIT)](https://github.com/jaywalnut310/vits)
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis - NeurIPS 2020](https://arxiv.org/abs/2010.05646)
- [Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search - NeurIPS 2020](https://arxiv.org/abs/2005.11129)
- [Bert-VITS2 - GitHub](https://github.com/fishaudio/Bert-VITS2)
- [GPT-SoVITS - GitHub (MIT)](https://github.com/RVC-Boss/GPT-SoVITS)
