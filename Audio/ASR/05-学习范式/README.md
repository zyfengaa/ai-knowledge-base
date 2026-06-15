# 05 — 数据问题

## 数据不够怎么办？三种路线的选择

前三节讲的 CTC、AED、RNNT 加 Conformer，效果确实好——但它们有一个共同的前提：你有足够的标注数据。具体地说，在 1000+ 小时的标注语音上，Conformer + RNNT 可以达到令人满意的 WER。

问题在于，大部分实际场景并没有 1000 小时标注。你可能就只有几十个小时的普通话标注，想做一个特定领域的 ASR（比如会议纪要或者医疗口述）。这时候你应该怎么做？

过去三四年的进展给出了三条路线。

### 路线一：自监督预训练（wav2vec 2.0）

你的情况是：有大量无标注语音（录起来便宜），只有少量（10h~100h）标注语音。

wav2vec 2.0 的思路是：先用大量的无标注语音训练一个特征提取器，然后在少量的标注语音上 fine-tune 做 ASR。具体来说，预训练阶段做的事情是：把输入语音的某些片段随机 Mask 掉（类似 BERT 的做法），然后模型要做的是"区分真正应该落在 Mask 位置的编码和随机采样的编码"——这是一个对比学习任务。

为什么这个思路特别吸引人？因为无标注语音几乎不要钱——YouTube 上、播客上、会议上，到处都是语音数据，只是没有对应的文本标注。而 fine-tune 阶段只需要极少量的标注——wav2vec 2.0 在 10 分钟标注上的效果就能达到传统系统在 10 小时标注上的水平。

### 路线二：聚类伪标签（HuBERT）

wav2vec 2.0 能工作，但对比学习的设计比较复杂——你不仅要选负例的数量，还得设计量化的方式。HuBERT 换了一个思路：先对语音做 k-means 聚类，生成离散化的伪标签，然后让模型去预测这些伪标签（"预测被 Mask 段落的聚类 ID"）。这个过程可以迭代优化——第一轮生成的伪标签可能质量一般，但模型学完后生成的第二轮伪标签就更好了。

比 wav2vec 2.0 更简洁，效果通常也更好。所以现在自监督 ASR 的主流方案其实已经转向 HuBERT 了。

### 路线三：弱监督大规模训练（Whisper）

Whisper 走了一条和前两条完全不同的路：不要无标注数据做自监督，直接拿 68 万小时 YouTube 自动字幕做"弱标注"——自动字幕的质量远不如人工标注，但量级足够大。Whisper 的 Encoder-Decoder 直接在这么大规模的数据上训，结果一个模型统一了 ASR（多语言）、翻译（X→English）、语言识别三个任务。

Whisper 告诉 ASR 社区一个很可能影响未来几年的结论：**标注质量可以不用那么高，只要数据量级大两个数量级。** 对很多工业场景来说，收集海量弱标注数据可能比精标少量数据更容易落地。

### 怎么选？

你手里有：

- **10h 以下标注** → wav2vec 2.0 或 HuBERT 预训练 + fine-tune
- **10h-100h 标注** → HuBERT（效果通常更好）或者直接用 Whisper（零样本）
- **100h-1000h 标注** → 自监督预训练有收益，但直接训 Conformer + RNNT 已经可以
- **1000h+ 标注 + 不在乎多语言** → Whisper 的方案最省事；或者用自监督提升鲁棒性

## 学习目标

读完你能：

- 用不太学术的语言讲清楚"对比学习预训练"和"聚类伪标签"的区别
- 理解为什么 HuBERT 比 wav2vec 2.0 更简洁但仍然有效
- 知道弱监督和自监督的区别在哪（答案：弱监督有标注但质量低大数量，自监督完全不需要标注）
- 面对实际项目能给出"选哪个方法"的决策建议

## 精选论文

**Baevski et al. (2020) "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"**

这篇是自监督 ASR 的里程碑。核心创新就是对比学习 + 量化 + Transformer 的预训练框架。如果你之前不太了解对比学习，这篇可能需要多花点时间在"对比损失函数"上。如果你了解，重点看它的量化设计（为什么需要量化，以及怎么做的）和 fine-tune 阶段的实验——在 10 分钟、1 小时、10 小时标注数据上的提升。

**Hsu et al. (2021) "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"**

HuBERT 的论文写得非常清楚（作者来自 Meta FAIR）。重点读它两个部分：（1）迭代伪标签的生成方法——先跑一次 k-means 聚类生成第一轮标签 → 训模型 → 用模型提取特征再做聚类 → 生成更高质量的标签；（2）为什么简单的 MSE 损失在聚类伪标签上就管用，不需要对比学习的复杂设计。

**Radford et al. (2022) "Robust Speech Recognition via Large-Scale Weak Supervision" (OpenAI Whisper)**

Whisper 的论文篇幅不长，但实验量很大。不需要逐字读完——读它的数据集构建方法（怎么筛选 68 万小时的弱标注数据、怎么处理噪声数据）和模型的设计选择（为什么用 Encoder-Decoder 而不是纯 Encoder、为什么做多任务训练）就够了。它的"弱监督"结论对所有从业者有启发意义。
---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Unit... | HuBERT () | [arXiv](https://arxiv.org/abs/2106.07447) |
| Robust Speech Recognition via Large-Scale Weak Supervision | Whisper () | [arXiv](https://arxiv.org/abs/2212.04356) |
| wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | wav2vec2 () | [arXiv](https://arxiv.org/abs/2006.11477) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units | [arXiv](https://arxiv.org/abs/2106.07447) |
| Robust Speech Recognition via Large-Scale Weak Supervision | [arXiv](https://arxiv.org/abs/2212.04356) |
| wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | [arXiv](https://arxiv.org/abs/2006.11477) |
