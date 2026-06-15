# 04 — 编码器进化

## 从 GRU 到 Conformer，ASR 编码器经历了什么？

回到你熟悉的 CNN+GRU 时代。GRU/LSTM 做序列建模有一个根本性的问题：它是串行的。处理一段 30 秒的语音（960 帧 @ 25ms 帧移），GRU 需要老老实实跑 960 步——每步依赖上一步的状态。这就意味着你不能用 GPU 的并行能力来加速计算，而且反向传播梯度经过 960 步传到前面，要么爆炸要么消失。

2017 年 Transformer 在机器翻译上横空出世，ASR 研究者们很快想到一件事：Self-Attention 是高度并行的——输入 T 帧，一次性算出 T×T 的 Attention 矩阵，而且任意两帧之间的距离都是 1，不存在长程依赖问题。

### Speech-Transformer：第一步尝试

Dong 等人在 2018 年直接把 Transformer Encoder 搬来做了 ASR 的声学编码器。论文里做的事情其实很简单：把 Transformer 的 Positional Encoding 从 NLP 改成语音型（因为语音帧之间不像 NLP 词之间有明显的边界），然后接 CTC 训练。结果证明并行编码确实能工作，这是 Transformer 在 ASR 的"投石问路"。

### Conformer：CNN + Attention 的融合

纯 Self-Attention 有一个被忽视的问题：它没有"局部归纳偏置"。CNN 天生知道相邻的像素/频率是相关的——3×3 卷积核的视野就是一个局部窗口。但 Attention 是对序列里所有位置一视同仁的，你需要在 Positional Encoding 上做很多文章才能让它"理解"局部关系。对于语音信号，这一点特别重要——语音的底层特征（共振峰、能量跳变、过零率）都是高度局部的，你丢了局部性就丢了大量信息。

Conformer 的解决方案是：**不选了，两个都要。**

它的结构是这样的——一个 Macaron 风格的 Sandwich：FF → (Conv Module + Self-Attention) → FF。中间的 Conv Module 做了一个精巧的复合：Pointwise Conv → GLU 激活 → Depthwise Conv → BatchNorm。Depthwise Conv 负责在时间和频率维度上做局部特征提取（类似 CNN 的角色），Self-Attention 负责建模长程依赖。两个并行，最后串接，一次性拿到"局部细节 + 全局上下文"。

这个设计的巧妙之处在于：它不是简单地把"CNN 和 Attention 两个模块"串在一起，而是在特征维度上让他们做了互补——卷积核感受野有限但平移不变，Attention 覆盖全序列但对位置信息敏感有限。这两者互为补充，不是谁替代谁。

## 学习目标

读完你能：

- 用一句话说清楚为什么 Transformer 比 RNN 更适合做声学编码
- 对比 Conv Module 和 Self-Attention 在"做同一件事"时的不同方式——一个抓局部，一个抓全局
- 理解 Conformer 的 Macaron 结构长什么样——不需要默写每一层的维度，但能画出大致块图
- 知道 Speech-Transformer 和 Conformer 的关系是"引路人和解决方案"的关系

## 精选论文

**Dong et al. (2018) "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition"**

这篇的工作量不大，但它问了一个很重要的问题：Transformer 到底能不能用在 ASR 上？答案是能。它就是那个"投石问路"的工作。如果你对 Transformer 已经很熟悉了，这篇可以快读——重点是它的实验设置和结果分析。

**Gulati et al. (2020) "Conformer: Convolution-augmented Transformer for Speech Recognition"**

Conformer 是 2020 年至今 ASR 编码器的事实标准——LibriSpeech 上的 WER 刷新、ESPnet 里的默认配置、WeNet / WenetSpeech 等框架都在用它。你在工作中应该已经接触到了，这一篇是帮你理解"为什么 Conformer 能成为标准"的。

重点读它的 Attention + Conv Module 的融合设计。如果你能理解"为什么把卷积嵌入 Transformer 里这么管用"，那这篇就算读透了。

## 拓展阅读

- **Peng et al. (2022) "Branchformer: Parallel MLP-CNN-Attention Hybrid Architecture"** — Conformer 的改进版本，把串行的 Macaron 结构改成双分支并行的"全局分支 + 局部分支"。效果有一点提升，但思路比 Conformer 更清晰——如果你想了解"Conformer 之后还有什么"，可以翻翻。

> 注意：Squeezeformer、Zipformer、Emformer 这些主要是效率优化，不在"突破性 idea"的范畴内。
---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Conformer: Convolution-augmented Transformer for Speech Recognition | Conformer () | [arXiv](https://arxiv.org/abs/2005.08100) |
| Speech-Transformer: A No-Recurrence Seq2Seq Model for Speech Recognition | SpeechTransformer () | [arXiv](https://arxiv.org/abs/1804.06993) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| Branchformer: A Novel Encoder Architecture for Speech Processing | [arXiv](https://arxiv.org/abs/2207.02971) |
| Conformer: Convolution-augmented Transformer for Speech Recognition | [arXiv](https://arxiv.org/abs/2005.08100) |
| Speech-Transformer: A No-Recurrence Seq2Seq Model for Speech Recognition | [arXiv](https://arxiv.org/abs/1804.06993) |
