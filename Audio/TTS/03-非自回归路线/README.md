# 03 — 非自回归路线

> FastSpeech + HiFi-GAN — 并行生成的效率革命

## 学习目标

| 学前 | 学后 |
|------|------|
| "Tacotron 太慢了" | 理解 FastSpeech 用时长预测器 (Duration Predictor) + 位置对齐解决自回归问题 |
| — | 理解 FastSpeech 2 去掉知识蒸馏改用真实时长 + VAE 的改进 |
| — | 理解 HiFi-GAN 的多周期判别器 (MPD) + 多尺度判别器 (MSD) 设计 |
| — | 能对比两条路线的优劣：自回归（质高但慢）vs 非自回归（快但韵律偏弱） |

## 精选论文

| # | 论文 | 为什么算突破 |
|---|------|------------|
| 1 | **Ren et al. (2019) "FastSpeech: Fast, Robust and Controllable TTS" [[arXiv](https://arxiv.org/abs/1905.09263)]** | 非自回归声学模型的首创，时长预测器 + 位置对齐 + 知识蒸馏，推理速度提升 270 倍 |
| 2 | **Ren et al. (2021) "FastSpeech 2: Fast and High-Quality End-to-End TTS" [[arXiv](https://arxiv.org/abs/2006.04558)]** | 去掉知识蒸馏，直接预测真实时长 + VAE + 音高/能量预测器，质量显著提升 |
| 3 | **Kong et al. (2020) "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" [[arXiv](https://arxiv.org/abs/2010.05646)]** | GAN vocoder 的里程碑，多周期判别器实现高质量波形，快速且可控 |

## 架构总结

```
文本 → FastSpeech → mel谱 → HiFi-GAN → 波形
      (Duration Predictor + FFT Blocks) (Generator + MPD/MSD)
```



---



## 论文参考



| 论文 | 作者(年份) | 链接 |
|---|---|---|
| FastSpeech 2: Fast and High-Quality End-to-End Text to Speech | FastSpeech2 () | [arXiv](https://arxiv.org/abs/2006.04558) |
| FastSpeech: Fast, Robust and Controllable Text to Speech | FastSpeech () | [arXiv](https://arxiv.org/abs/1905.09263) |
| HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | HiFiGAN () | [arXiv](https://arxiv.org/abs/2010.05646) |



---



## 论文参考



| 论文 | 链接 |
|---|---|
| FastSpeech 2: Fast and High-Quality End-to-End Text to Speech | [arXiv](https://arxiv.org/abs/2006.04558) |
| FastSpeech: Fast, Robust and Controllable Text to Speech | [arXiv](https://arxiv.org/abs/1905.09263) |
| HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | [arXiv](https://arxiv.org/abs/2010.05646) |

