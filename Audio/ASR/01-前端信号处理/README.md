﻿# 01 — 信号输入

## 波形到特征，是怎么过来的？

你拿到一个 WAV 文件，把它读到内存里——那是一串 float 数组，44.1 kHz 采样率的话一秒钟就有 44100 个点。你不能直接把这一维数组丢给 CNN 或者 Transformer，原因有二：一是太长（一秒就四万多点），二是这 44100 个点之间没有很明确的"特征"意义——它们只是声波的振幅采样。

所以要做特征提取。ASR 领域过去四十年最通用的做法是：把人耳听声音的原理用数学模拟出来。耳朵里的基底膜对不同频率的响应不同——低频分辨率高、高频分辨率低，大致是对数关系的。梅尔滤波器组就是模拟这个：用一组三角滤波器把线性频率映射到梅尔刻度上，得到 FBank（滤波器组特征）。

FBank 的维度之间是相关的（毕竟相邻滤波器有重叠），如果后面接的是 GMM-HMM 这种对特征独立性比较敏感的模型，就要再做一步 DCT（离散余弦变换）来解相关——这就是 MFCC。但到了 DNN/端到端时代，模型本身能处理相关特征，FBank 就够用了，所以现在大部分系统直接用 80 维 FBank，不再转 MFCC。

至于 SpecAugment——它大概是 ASR 领域性价比最高的技巧。一句话描述：对 FBank 特征图做随机的"时间掩码"和"频率掩码"。听起来简单得不像个顶会论文，但它就是管用，而且几乎零成本。现在所有主流 ASR 系统都把它当标配。

## 学习目标

读完你要能回答这几个问题：

- 从一个 WAV 文件到 80 维 FBank 特征图，中间经过了哪几个步骤？
- 为什么梅尔刻度是对数分布的？这和人类听觉有什么关系？
- MFCC 和 FBank 的区别在哪？DCT 解相关是什么时候需要的？
- SpecAugment 为什么管用？（提示：它强迫模型从"全局轮廓"而非"局部细节"做识别）
- 你知道 VAD（语音活动检测）这回事，但现阶段可以先不管细节——它属于工程优化不是学术问题

## 精选论文

**Davis & Mermelstein (1980) "Comparison of Parametric Representations for Monosyllabic Word Recognition" — MFCC**

这篇是 MFCC 的原始论文。技术上很简单：预加重 → 分帧 → 加窗 → FFT → 梅尔滤波器组 → log → DCT → MFCC。但它的影响之大，四十年后所有 ASR 系统（包括 Whisper）依然在用这个 pipeline。写个几千字的综述都讲不完这条线，但核心思想其实就是上面那几行字。愿意的话了解一下具体计算过程就行，不需要背公式。

**Park et al. (2019) "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition" [[arXiv](https://arxiv.org/abs/1904.08779)]"

这篇就是 SpecAugment。说实话它的论文写得比我上面那段描述复杂得多（做了大量消融实验、还有时域扭曲 warping 的变体），但核心就是时间和频率掩码。你如果赶时间，读完摘要和算法描述就够了。

## 阅读建议

这部分其实是整个 ASR 里最"可跳过"的一节——因为你现在大概率已经在用 FBank 了。快速翻一下 MFCC 的计算链路，确认自己能画出来就行。后面 03 的三种端到端范式才是需要花时间的硬东西。
---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Comparison of Parametric Representations for Monosyllabic Word Recognition | MFCC () | — |
| SpecAugment: A Simple Data Augmentation Method for ASR | SpecAugment () | [arXiv](https://arxiv.org/abs/1904.08779) |

---
