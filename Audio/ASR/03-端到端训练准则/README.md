﻿﻿﻿﻿# 03 — 三种端到端范式

## 为什么一个问题，会有三种答案？

讲端到端 ASR 之前，得先明白一个问题：到底"对齐"这件事有多麻烦。

在 CTC 之前，做语音识别需要先训练一个 GMM-HMM 系统做强制对齐——也就是说，你得告诉模型"这段语音的第 3 到第 8 帧对应音素 /h/，第 9 到 14 帧对应 /e/……"。但这个对齐本身是有问题的：你需要的对齐越准，你就需要分类器越强；但分类器越强，它就越来越不需要你给它指定对齐——所以说这是个鸡生蛋蛋生鸡的问题。

CTC 的天才之处就在于：**它直接绕过了对齐。**

### CTC：取消对齐，但付出独立假设的代价

CTC 引入了一个叫 "Blank" 的特殊符号——你可以把它理解为"模型在说"我还没决定好现在说啥，先出个 Blank 占位""。这样模型输出不再直接对应每一帧，而是一段可以被任意压缩和拉伸的序列。训练时前向后向算法把所有可能的对齐路径都枚举出来求平均——模型自己学会了什么时候该说话、什么时候该等待。

但 CTC 有一个硬伤：它假设每一帧的预测是**条件独立**的。模型输出"h"的时候，不知道刚才自己说了什么，也不知道接下来打算说什么。在安静环境下这不是大问题，但一旦有噪音或者口音，模型就很容易"改口"——明明说的是"hello"，听到一点卡车声就变成了"yellow"。

这个缺陷在 CTC 诞生那会儿（2006）就已经被意识到了，但直到 2014 年才有人用工业数据验证了 CTC 路线真正可行——这就是 Deep Speech 1。而在那之后一年，就有了一个更彻底的替代方案。

### AED：用 Attention 打破独立假设，但牺牲了流式

AED 的直觉很简单：如果你想让模型知道"我已经说了什么"，那就给它一个能看到全部输入的注意力机制——解码器每一步都回头看整个编码器的输出，结合历史已生成的信息来预测下一个词。这叫 Attention。

LAS（Listen, Attend and Spell）是最经典的 AED 架构。它的名字其实很形象：Listen 就是编码器把语音编码成特征序列；Attend 就是解码器每一步都在"听"编码器输出的不同位置；Spell 就是逐步生成字符或词片。

精度比 CTC 好很多，因为它不假设独立——干净环境和噪音环境下都有明显提升。但代价是：**模型必须听完整句语音才能开始解码。** 所以 LAS 不能做实时语音识别——你对着手机说"Hey Siri"，手机必须等你把整句说完才开始处理。

### RNNT：兼顾端到端和流式，但训练不太稳定

RNNT 的核心想法是：用一个小型的 Prediction Network 来建模"我已经输出了什么文本"，然后通过一个 Joiner 网络把声学信息和文本信息融合起来。每一帧都可以独立决策——不需要等整句话结束——同时又有 Prediction Network 提供历史信息的帮助。

这样说可能有点抽象，换个角度理解：CTC 是"只看声学，不看文本"；AED 是"同时看声学和全部文本，但不能流式"；RNNT 是"同时看声学和历史文本，而且能流式"——Prediction Network 相当于一个轻量的"语言模型"，告诉你"就目前已经说出来的话来看，合理的下一个词应该什么方向"。

但 RNNT 也有自己的问题：训练不稳定，Prediction Network 和 Joiner 之间的交互有时候比较玄学，而且推理速度比 CTC 慢。Google 把 RNNT 做进了手机 ASR，但也是花了大量工程精力去优化的。

### Hybrid CTC/Attention：为什么工程上最常用？

这几条路线不是互斥的。Watanabe 2017 的想法很务实：把 CTC 做前端对齐加速训练收敛，然后用 Attention 做解码提升精度。ESPnet 框架把这个思路做了开源实现，所以如果你现在去翻 ESPnet 的 recipe，大部分配置都是 Hybrid CTC/Attention。它不追求"最极致"的某种能力（比如最流式或最高精度），但它在大多数实际场景下都能给出不错的折中。

---

你这三种范式都接触过后，应该能建立这样的直觉：

- **做离线、高精度**（录完再转文字）：AED / Hybrid CTC/Attention
- **做在线、低延迟**（实时语音转录）：RNNT
- **做快速原型、简单场景**：CTC
- 以上所有都可以用同一套编码器（比如 Conformer），**编码器和训练范式是正交的**

## 学习目标

读完你能：

- 默画 CTC / AED / RNNT 的结构差异图——不用多精美，关键是说清楚三个组件的区别：声学编码器、对齐机制、文本依赖建模
- 解释 CTC Blank 为什么是个巧妙的 trick——以及它的代价是什么
- 用一句话说清 LAS 为什么不能流式（答案：Decoder 每一步需要看 Encoder 的全部输出）
- 用一句话说清 RNNT 为什么能流式（答案：Joiner 是帧同步的，Prediction Network 只依赖已输出文本）
- 知道在实际项目中怎么选：延迟要求？精度要求？有没有语言模型？

> 注意：这节是 ASR 所有内容里最核心的——花时间把三种范式的优缺点和适用场景想清楚，后面的编码器和学习范式都建立在这个基础上。

## 精选论文

**Graves et al. (2006) "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks" [ACM](https://dl.acm.org/doi/10.1145/1143844.1143891)**

这篇是端到端 ASR 的理论起点。读它的重点不是技术实现细节（Blank 和前向后向的推导），而是理解它"为什么不需要对齐"这个核心 insight。如果你时间紧，读摘要 + 引言 + 结论就可掌握精神。

**Hannun et al. (2014) "Deep Speech: Scaling up End-to-End Speech Recognition" [[arXiv](https://arxiv.org/abs/1412.5567)]**

这篇起了一个关键的桥接作用——CTC 2006 年就提出了理论，但直到 Deep Speech 1 才有人验证了它在工业数据上真正跑得通。它的贡献不是新的理论（核心算法仍然是 CTC），而是证明了"这套方法在真实场景下能工作"。如果你时间紧，读它的实验设置和结果分析就够了。

**Chan et al. (2016) "Listen, Attend and Spell" [[arXiv](https://arxiv.org/abs/1608.08087)]**

把 Attention 引入 ASR 的代表作。你如果熟悉 NLP 里的 Attention，读起来会很顺畅。重点是理解 Listen + Attend + Spell 三条 pipe 的协作方式。LAS 也是后来 Whisper 架构的前身（Whisper 基本上就是 LAS 的大规模版本）。

**Graves (2012) "Sequence Transduction with Recurrent Neural Networks" [[arXiv](https://arxiv.org/abs/1211.3711)]**

这篇是 RNNT 的原始论文，发表时间早于 LAS（2012 vs 2016），但 RNNT 的实用化是很多年以后的事了（Google 在 2018-2019 年才把它推到生产环境）。读的时候重点放在 Prediction Network + Joiner 的设计上，它是 RNNT 和 CTC/AED 最大的差异所在。

**Watanabe et al. (2017) "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition" [[arXiv](https://arxiv.org/abs/1703.03506)]**

这篇是 ESPnet 的奠基论文，也是目前工程上最常用的方案。核心想法很简洁：CTC 加速收敛 + Attention 提升精度 = 最好的折中。如果你工作中用 ESPnet，这篇是必读。

## 拓展阅读

- **Amodei et al. (2016) "Deep Speech 2" [[arXiv](https://arxiv.org/abs/1512.02595)]** — 和 Deep Speech 1 同一路线，但规模更大（多 GPU 训练 + 数据增强）。如果你想了解 CTC 路线在"更大规模"下怎么做，可以翻翻。
---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with RNNs | CTC () | — |
| Deep Speech: Scaling up End-to-end Speech Recognition | DeepSpeech () | [arXiv](https://arxiv.org/abs/1412.5567) |
| Hybrid CTC/Attention Architecture for End-to-End Speech Recognition | HybridCTCAttention () | [arXiv](https://arxiv.org/abs/1703.03506) |
| Listen, Attend and Spell | LAS () | [arXiv](https://arxiv.org/abs/1608.08087) |
| Sequence Transduction with Recurrent Neural Networks | RNNT () | [arXiv](https://arxiv.org/abs/1211.3711) |

---
