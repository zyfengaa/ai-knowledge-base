﻿# 03 �� ���Իع�·��

> FastSpeech + HiFi-GAN �� �������ɵ�Ч�ʸ���

## ѧϰĿ��

| ѧǰ | ѧ�� |
|------|------|
| "Tacotron ̫����" | ��� FastSpeech ��ʱ��Ԥ���� (Duration Predictor) + λ�ö������Իع����� |
| �� | ��� FastSpeech 2 ȥ��֪ʶ���������ʵʱ�� + VAE �ĸĽ� |
| �� | ��� HiFi-GAN �Ķ������б��� (MPD) + ��߶��б��� (MSD) ��� |
| �� | �ܶԱ�����·�ߵ����ӣ��Իع飨�ʸߵ�����vs ���Իع飨�쵫����ƫ���� |

## ��ѡ����

| # | ���� | Ϊʲô��ͻ�� |
|---|------|------------|
| 1 | **Ren et al. (2019) "FastSpeech: Fast, Robust and Controllable TTS" [[arXiv](https://arxiv.org/abs/1905.09263)]** | ���Իع���ѧģ�͵��״���ʱ��Ԥ���� + λ�ö��� + ֪ʶ���������ٶ����� 270 �� |
| 2 | **Ren et al. (2021) "FastSpeech 2: Fast and High-Quality End-to-End TTS" [[arXiv](https://arxiv.org/abs/2006.04558)]** | ȥ��֪ʶ����ֱ��Ԥ����ʵʱ�� + VAE + ���/����Ԥ������������������ |
| 3 | **Kong et al. (2020) "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" [[arXiv](https://arxiv.org/abs/2010.05646)]** | GAN vocoder ����̱����������б���ʵ�ָ��������Σ������ҿɿ� |

## �ܹ��ܽ�

```
�ı� �� FastSpeech �� mel�� �� HiFi-GAN �� ����
      (Duration Predictor + FFT Blocks) (Generator + MPD/MSD)
```

---

## ���Ĳο�

| ���� | ����(���) | ���� |
|---|---|---|
| FastSpeech 2: Fast and High-Quality End-to-End Text to Speech | FastSpeech2 () | [arXiv](https://arxiv.org/abs/2006.04558) |
| FastSpeech: Fast, Robust and Controllable Text to Speech | FastSpeech () | [arXiv](https://arxiv.org/abs/1905.09263) |
| HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | HiFiGAN () | [arXiv](https://arxiv.org/abs/2010.05646) |

---

