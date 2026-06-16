# 02 — 自回归路线

> Tacotron + WaveNet — 深度学习 TTS 的第一个成功范式

## 学习目标

| 学前 | 学后 |
|------|------|
| "深度学习 TTS 从 WaveNet 开始" | 理解 WaveNet 的因果卷积 + 门控扩张卷积 (Gated Dilated CNN) 结构 |
| — | 理解 Tacotron 的 Encoder-Attention-Decoder + CBHG 模块 |
| — | 能解释 Tacotron 2 如何把两个阶段串联为完整管线：文本→mel→波形 |
| — | 知道自回归路线的核心痛点：推理慢、注意力对齐不稳定 |

## 精选论文

| # | 论文 | 为什么算突破 |
|---|------|------------|
| 1 | **van den Oord et al. (2016) "WaveNet: A Generative Model for Raw Audio" [[arXiv](https://arxiv.org/abs/1609.03499)]** | 深度学习 TTS 的起源性工作，第一个用神经网络生成高质量原始音频波形，被引 5000+ |
| 2 | **Wang et al. (2017) "Tacotron: Towards End-to-End Speech Synthesis" [[arXiv](https://arxiv.org/abs/1703.10135)]** | 第一个从文本直接预测 mel 谱的端到端声学模型，包含 CBHG + Attention 结构 |
| 3 | **Shen et al. (2018) "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" [[arXiv](https://arxiv.org/abs/1712.05884)]** | 把 Tacotron 和 WaveNet 串联为完整端到端管线，音质达到接近人类水平 |

## 架构总结

```
文本 → Tacotron → mel谱 → WaveNet → 波形
       (Encoder+Attn+Decoder)    (Causal Conv+Gate)
```



---



## 论文参考



| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions | Tacotron2 () | [arXiv](https://arxiv.org/abs/1712.05884) |
| Tacotron: Towards End-to-End Speech Synthesis | Tacotron () | [arXiv](https://arxiv.org/abs/1703.10135) |
| WaveNet: A Generative Model for Raw Audio | WaveNet () | [arXiv](https://arxiv.org/abs/1609.03499) |



---



## 论文参考



| 论文 | 链接 |
|---|---|
| Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions | [arXiv](https://arxiv.org/abs/1712.05884) |
| Tacotron: Towards End-to-End Speech Synthesis | [arXiv](https://arxiv.org/abs/1703.10135) |
| WaveNet: A Generative Model for Raw Audio | [arXiv](https://arxiv.org/abs/1609.03499) |

