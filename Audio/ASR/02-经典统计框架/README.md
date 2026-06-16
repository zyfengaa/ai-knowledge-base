﻿﻿# 02 — HMM 的遗产

## 在端到端之前，ASR 是怎么装的？

现在你做一个 ASR 系统可能就是"对着 FBank 跑一个 Conformer，接 CTC 或 RNNT，训一个模型完事"。但回到 2012 年以前，事情远没这么简单——当时整个 ASR 系统得拼好几块：

1. **声学模型**（当时是 GMM，后来变成 DNN）——负责说"这段语音听起来像是音素 /a/"
2. **发音词典**——负责说"这个词 /hello/ 对应 h-h-e-l-l-ou 这一串音素"
3. **语言模型**（n-gram）——负责说"hello 后面跟着 world 的概率比 hello 后面跟着 elephant 高"
4. **解码器**（WFST 图）——负责把上面三个东西融在一起，找到一条最优路径

这四个东西是各自独立训练、独立调参的，然后用 WFST 的 HCLG 组合编译成一张大的解码图。每一步都有专门的团队在维护：声学团队调 GMM 的混合数，语音团队调词典的读音变体，LM 团队调 n-gram 的裁剪阈值……过程极其复杂，每个组件都要专家，出了问题很难定位。

DNN 的介入其实最开始不是要"推翻这套框架"，而是一种更温和的替代。2012 年 Dahl 在微软的论文其实就问了一个很简单的问题："我们把 GMM 换成 DNN 来做音素后验概率估计，会不会更好？" 结果证明 DNN 全面超越了 GMM——从此混合系统从 GMM-HMM 变成了 DNN-HMM。

但直到端到端出现之前，这套"混合"的本质没有变：声学模型和语言模型还是两个不同的模型，各自独立调优。端到端 ASR 的突破在于：**一个模型，一口吃进来**——不再需要 WFST，不再需要独立的发音词典（端到端系统自己学音素到字的映射），甚至不再需要显式的语言模型（虽然加一个还是有用）。

**那为什么还要学这套老东西？** 两个现实原因：第一，很多工具链里做强制对齐（Forced Alignment）还是用 GMM-HMM 的传统流程——比如你手里有一堆语音和对应文本，要用对齐来生成 CTC 训练的帧级标签；第二，你理解了对齐为什么曾经是个麻烦事，才能理解 CTC 为什么是个突破。

## 学习目标

读完你要能：

- 画出一张图：语音 → GMM-HMM 声学模型 → 词典 → n-gram LM → WFST 解码 → 文本
- 说清楚"GMM 做声学建模"和"DNN 做声学建模"的本质区别在哪（判别 vs 生成）
- 理解 WFST 的 HCLG 直觉——不需要会写代码，知道 H→C→L→G 分别是什么就行
- 知道 DNN-HMM 和端到端的根本区别：一个是一堆散件拼起来，一个是一口吃掉

## 精选论文

**Rabiner (1989) "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition" [IEEE](https://ieeexplore.ieee.org/document/18626)**

这是一个 70,000+ 引用的论文。你读完前两章（HMM 的三个基本问题和前向后向算法），就已经掌握了传统 ASR 框架最核心的 80%。第三章的 Baum-Welch 可以后面再用到再细看。坦率地说，这篇读起来有些枯燥（毕竟 1989 年的排版和语言风格），但它是这个领域的"必修课"。

**Dahl et al. (2012) "Context-Dependent Pre-Trained Deep Neural Networks for Large-Vocabulary Speech Recognition" [[arXiv](https://arxiv.org/abs/1207.0580)]**

这篇是 DNN 正式取代 GMM 的里程碑。你不需要深究它的预训练细节（现在看已经过时了），重点读它的对比实验——DNN-HMM 比最好的 GMM-HMM 系统提升了多少，为什么会提升。这篇直接告诉你：判别式模型替代生成式模型，是那一轮最大的推动力。

## 拓展阅读

- **Hinton et al. (2012) "Deep Neural Networks for Acoustic Modeling in Speech Recognition" [[arXiv](https://arxiv.org/abs/1207.0580)]** — 这是微软和 Google 联合写的 DNN-HMM 综述，如果你对 DNN-HMM 这段历史感兴趣可以翻翻。
- **Povey et al. (2011) "The Kaldi Speech Recognition Toolkit" [IEEE](https://ieeexplore.ieee.org/document/6163935)** — Kaldi 的论文不是学术突破，但它几乎是传统 ASR 工程的事实标准。如果你工作中需要做强制对齐或 WFST 相关的事情，Kaldi 是绕不开的。
---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Context-Dependent Pre-trained Deep Neural Networks for Large-Vocabulary Speech Recognition | DNN () | [arXiv](https://arxiv.org/abs/1202.0445) |
| A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition | HMM () | — |

---
