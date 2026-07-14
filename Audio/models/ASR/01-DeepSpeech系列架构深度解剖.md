# DeepSpeech 系列架构深度解剖

> 百度研究院（Baidu SVAIL）出品 | 端到端语音识别的奠基之作

---

## 写在前面

DeepSpeech 系列包含两个版本，分别发布于 2014 年和 2015 年。理解它俩的递进关系，比单独看任何一个都更重要：

| 版本 | 论文 | 发布时间 | 架构 | 语言 |
|------|------|---------|------|------|
| **DeepSpeech 1** | Deep Speech: Scaling up end-to-end speech recognition | 2014.12 | FC + Bi-RNN + CTC | 英语 |
| **DeepSpeech 2** | Deep Speech 2: End-to-End Speech Recognition in English and Mandarin | 2015.12 | Conv + Bi-GRU + CTC | **英语 + 普通话** |

DS1 验证了"端到端 ASR 可行"，DS2 证明了"中英文都行，且达到商用水平"。DS2 入选 2016 年《麻省理工科技评论》十大突破技术。

---

## 第一章 前置条件：GMM-HMM 体系的困境

在 DeepSpeech 之前，语音识别是 GMM-HMM 的天下。一个系统由四个独立组件拼成：

```
原始音频
    │
    ├── 声学模型 (GMM)
    │   └── 每一帧属于哪个音素的概率
    │
    ├── 发音词典
    │   └── 词 → 音素映射
    │
    ├── 语言模型 (n-gram)
    │   └── 词序列的概率
    │
    └── 解码器 (WFST)
        └── 上述三者在编译图中搜索最优路径
```

每个组件用不同的算法、不同的人、不同的框架训练。声学团队调 GMM 混合数，语言团队调 n-gram 裁剪阈值，出了问题根本分不清是谁的锅。**整个行业在等一个模型把这一切统一起来。**

CTC（Connectionist Temporal Classification，2006 年提出）在学术上证明了"不需要帧级对齐也能训 ASR"，但直到 DeepSpeech 1 之前，没有人在大规模数据上验证过这条路能打到商用水平。

---

## 第二章 DeepSpeech 1（2014）：浅层架构验证路线

### 2.1 整体架构

DeepSpeech 1 是一个**5 隐藏层**的神经网络，直接从频谱图映射到字符序列：

```
原始音频 (16kHz)
    │
    ├── 频谱特征提取
    │   └── 每个时间帧 + 左右 C 帧上下文 → 频谱上下文窗
    │
    ├── Layer 1: 全连接
    ├── Layer 2: 全连接      ← 逐帧独立，非递归
    ├── Layer 3: 全连接
    │
    ├── Layer 4: 双向 RNN    ← 核心时序建模层
    │   ├── 前向 RNN: h⁽ᶠ⁾_t = g(W h³_t + W_r⁽ᶠ⁾ h⁽ᶠ⁾_{t-1} + b)
    │   └── 后向 RNN: h⁽ᵇ⁾_t = g(W h³_t + W_r⁽ᵇ⁾ h⁽ᵇ⁾_{t+1} + b)
    │
    ├── Layer 5: 全连接
    │   └── h⁵_t = g(W (h⁽ᶠ⁾_t + h⁽ᵇ⁾_t) + b)
    │
    └── Softmax 输出
        └── 字符概率分布 (a-z + 空格 + 撇号 + blank 共 29 符号)
```

### 2.2 关键技术选择

**1. Clipped ReLU 激活函数**

```
g(z) = min(max(0, z), 20)
```

普通的 ReLU 是 `max(0, z)`，没有上界。DeepSpeech 发现 RNN 中 ReLU 的激活值会随着时间步增长而爆炸。加一个上界 20 之后训练稳定了。后来很多 RNN 工作沿用了这个 trick。

**2. 上下文窗输入**

每帧不是只送入当前频谱切片，而是连同左右 C 帧一起送入：
```
输入: [x_{t-C}, ..., x_t, ..., x_{t+C}]
```
C 通常取 5-10。序列越长上下文越广，但计算量线性增长。这个"手工设定上下文窗"的做法后来被 Attention 彻底替代。

**3. CTC 损失**

CTC 引入了一个特殊的 **blank** 符号（表示"不确定输出什么"），允许模型在输出序列中插入 blank，从而让长度可变的音频输入映射到长度可变的文本输出。方法是在所有可能的对齐路径上求和：

```
P(Y|X) = Σ_{路径π: B(π)=Y} P(π|X)
```

其中 B 是路径压缩函数：先合并连续重复字符，再删除 blank。这个前向后向求和算法是 CTC 的核心，也是 DeepSpeech 训练成功的关键。

**4. 训练数据**

| 配置 | 数据量 | 数据来源 | 噪声增强 |
|------|-------|---------|---------|
| 小规模 | 300h | Switchboard | — |
| 中规模 | 2,300h | SWB + Fisher | — |
| **大规模** | **7,000h → 100,000h** | 自有数据 | **15 类环境噪声叠加** |

大规模配置的训练数据通过添加 15 种环境噪声（交通、餐厅、地铁等）在不同 SNR 下叠加，从 7,000 小时合成到 100,000 小时。

### 2.3 三种配置

| 配置 | 层数 | 隐藏单元 | 参数量 |
|------|------|---------|-------|
| Switchboard 300h | 5 hidden | 2,304 | ~40M |
| SWB+Fisher 2300h | 4 RNN × 5 hidden | 2,304 | ~160M |
| **100K 增强** | **6 RNN × 5 hidden** | **2,560** | **~250M** |

### 2.4 效果

| 测试集 | WER | 对比 |
|-------|-----|------|
| Switchboard Hub5'00 | **16.0%** | — |
| CallHome（更难） | 19.3% | — |
| 噪声 10dB SNR | **19.1%** | 商用系统 30.5% |

**噪声场景下比当时最好的商用系统（Google/Apple）低 10-13 个百分点**。

### 2.5 局限

1. **模型太浅**：只有 1 层双向 RNN，时序建模能力有限，中文场景更是完全不可用
2. **非端到端特征**：输入是手工设计的频谱上下文窗，仍然保留了大量传统信号处理的痕迹
3. **训练太慢**：用 GPU 集群训 100K 小时数据也需要数周

---

## 第三章 DeepSpeech 2（2015）：深度架构的全面升级

### 3.1 整体架构

```
原始音频 (16kHz)
    │
    ├── 频谱特征 (log-spectrogram, 160 维)
    │
    ├── Conv2d #1: (1 → 32, kernel 41×11, stride 2×2)
    │   └── BatchNorm + HardTanh(0, 20)
    │
    ├── Conv2d #2: (32 → 32, kernel 21×11, stride 2×1)
    │   └── BatchNorm + HardTanh(0, 20)
    │
    ├── Bi-GRU #1: (672 → 2560)
    ├── Bi-GRU #2: (2560 → 2560)   ← 5~7 层双向 GRU 堆叠
    ├── Bi-GRU #3: (2560 → 2560)
    ├── Bi-GRU #4: (2560 → 2560)
    ├── Bi-GRU #5: (2560 → 2560)
    │
    ├── BatchNorm(2560)
    │
    ├── Linear: 2560 → vocab_size (字符数 + blank)
    │
    └── CTC loss (训练) / CTC beam search (推理)
```

### 3.2 DS1 → DS2 的四大跃迁

#### 跃迁一：从 FC 到 Conv —— 更好的频谱特征提取

DS1 用的是全连接层直接处理频谱上下文窗，每个频率位置独立学习权重。DS2 用 2 层 Conv2d：

- Conv1: kernel (41,11), stride (2,2) —— 频率轴 41 → 压缩，时间轴 11 → 局部上下文融合
- Conv2: kernel (21,11), stride (2,1) —— 频率轴再压缩，时间轴保持

**卷积的优势**：权重共享 + 平移不变性。语音信号的频谱模式（如共振峰）在不同位置形态类似，卷积可以复用检测器。FC 做不到这一点，每个位置都需要自己学。

```
Conv vs FC 处理频谱的差异:
  FC:  每个时间-频率点独立学权重 → 学到的是"位置相关的模式"
  Conv: 卷积核在整个频谱上滑动 → 学到的是"位置无关的模式"
```

#### 跃迁二：从 RNN 到 GRU —— 更强、更深的时序建模

DS1 用 1 层双向 RNN，DS2 用 **5-7 层双向 GRU**（隐藏单元 2,560，总参数量 ~208M）。

GRU（Gated Recurrent Unit）是 LSTM 的简化版——把遗忘门和输入门合并为一个"更新门"，参数更少、计算更快：

```
GRU 内部:
  z_t = σ(W_z x_t + U_z h_{t-1})     ← 更新门: 保留多少旧信息、接受多少新信息
  r_t = σ(W_r x_t + U_r h_{t-1})     ← 重置门: 丢弃多少旧信息
  h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}))  ← 候选新状态
  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t     ← 最终状态
```

#### 跃迁三：BatchNorm for RNN —— 训练加速器

DS2 率先将 Batch Normalization 引入 RNN 训练。BatchNorm 在每一层的输入上做标准化：

```
BN(x) = γ · (x - μ_B) / √(σ²_B + ε) + β
```

这对 RNN 训练有两重意义：
1. **解决梯度消失/爆炸**：每层的输入分布在训练中保持稳定，深层 RNN 不再是不可训的
2. **允许更大学习率**：稳定了梯度的尺度，学习率可以设得更高 → 收敛更快

#### 跃迁四：SortaGrad —— 针对 CTC 的课程学习

DS2 提出的 SortaGrad 是一个针对 CTC 训练的特殊优化：**按音频长度对 batch 排序，先训短样本、再训长样本**。

为什么 CTC 需要这个？CTC 的对齐路径数和序列长度呈指数关系。长样本的对齐可能性比短样本多得多，早期模型还没学会对齐就遇到长样本，训练非常困难。SortaGrad 让模型"先走稳再跑"。

### 3.3 小创新：长步幅跳跃

DS2 在 GRU 的输入侧使用长步幅——每隔几帧才送入一次 GRU，单条样本的计算量减少约 **3 倍**。这建立在"相邻帧信息高度冗余"的观察上。可以看作是后来的 pool/stride 操作的早期雏形。

### 3.4 训练基础设施

DS2 用 **16 GPU 同步 SGD + 自定义 all-reduce 通信**，单 GPU 达到约 **3 TFLOPS**（50% 峰值利用率）。在当时这是大规模分布式训练的一个标杆工程。

### 3.5 流式推理机制

DS2 虽然整体是双向 RNN（需要整句），但推理时可以用 **Lookahead** 技巧做流式：

```
Lookahead: 
  当前帧解码时，往后多看 21 帧（约 210ms）
  → 这 21 帧的卷积/GRU 特征帮助消歧翻译
  → 代价是推理延迟增加了 210ms
```

### 3.6 效果

| 测试集 | DS2 | DS1 | 相对提升 |
|-------|-----|-----|---------|
| 英文标准测试 | **~5-7%** | ~13.2% | **-43%** |
| 中文 AISHELL-1 | **~6.8%** | 不支持 | — |
| 英文短语 | **3.7% WER** | — | — |

**中英文在同一架构下首次同时达到 SOTA。** 入选 2016 MIT 十大突破技术。

---

## 第四章 参数量估算

### DeepSpeech 1（大规模配置）

```
RNN 部分 (6 Bi-RNN × 2560 hidden):
  单向前向: 26,651 + 26,651 → 1 组
  双向: 1 组 × 2 = 53,302 参数
  6 组 × 53,302 = ~320K
  但全连接层远大于此...
  
实际总参数量: ~250M（6 Bi-RNN × 5 hidden × 2560）
```

### DeepSpeech 2

```
卷积部分:
  Conv1: 1 × 32 × 41 × 11 + 32 = 14,464
  Conv2: 32 × 32 × 21 × 11 + 32 = 236,576
  合计: ~251K

GRU 部分 (双向, 5 层, 隐藏 2560):
  GRU #1: I=672, H=2560 → 每方向 3 × 2560 × (672 + 2560 + 1) ≈ 24.8M
          双向 × 2 ≈ 49.6M
  GRU #2-5: I=2560, H=2560 → 每方向 3 × 2560 × (2560 + 2560 + 1) ≈ 39.3M
            双向 × 2 × 4 ≈ 314.5M
  GRU 合计: ~364M+

输出层: 
  Linear: 2560 × vocab_size ≈ 2560 × 30 ≈ 77K

总计: ~365M 参数
```

---

## 第五章 总结：为什么要理解 DeepSpeech

DeepSpeech 系列在今天的标准下已经不再是最优模型（WER 远高于 Whisper/Conformer），但它的**历史地位无法绕过**：

1. **定义了端到端 ASR 的基本范式**：频谱 → Conv 子采样 → 时序 Encoder → CTC 输出
2. **SortaGrad + BatchNorm for RNN** 是深度学习训练方法论的重要贡献
3. DS2 证明了**中英文可以在同一个端到端架构下达到 SOTA**，这一结论影响了后来所有 ASR 模型的多语言设计
4. Mozilla DeepSpeech（2017）基于 DS2 架构开源并发布预训练权重，成为 2017-2022 年间唯一可离线部署的开源 ASR 引擎，孕育了整整一代 ASR 开发者

```
DeepSpeech 1 (2014)
  └── "端到端可行" → 1 层 Bi-RNN，高噪声刚性
      │
DeepSpeech 2 (2015)
  └── "中英文都行，商用级" → Conv + 7 层 Bi-GRU
      │
Mozilla DeepSpeech (2017)
  └── "人人都能下载部署" → C++ runtime + 预训练权重
      │
... 自此 ASR 全面进入端到端时代 ...
```

---

**Sources:**
- [Deep Speech: Scaling up end-to-end speech recognition (arXiv:1412.5567)](https://arxiv.org/abs/1412.5567)
- [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin (arXiv:1512.02595)](https://arxiv.org/abs/1512.02595)
- [Mozilla DeepSpeech GitHub (Archived)](https://github.com/mozilla/DeepSpeech)
- [Baidu Research Blog - Deep Speech 2](https://research.baidu.com/Blog/index-view?id=85)
- [NVIDIA OpenSeq2Seq DeepSpeech2 Implementation](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html)
- [Torchaudio DeepSpeech Model Documentation](https://pytorch.org/audio/stable/generated/torchaudio.models.DeepSpeech.html)
