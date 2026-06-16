# 02 �� �Իع�·��

> Tacotron + WaveNet �� ���ѧϰ TTS �ĵ�һ���ɹ���ʽ

## ѧϰĿ��

| ѧǰ | ѧ�� |
|------|------|
| "���ѧϰ TTS �� WaveNet ��ʼ" | ��� WaveNet �������� + �ſ����ž�� (Gated Dilated CNN) �ṹ |
| �� | ��� Tacotron �� Encoder-Attention-Decoder + CBHG ģ�� |
| �� | �ܽ��� Tacotron 2 ��ΰ������׶δ���Ϊ�������ߣ��ı���mel������ |
| �� | ֪���Իع�·�ߵĺ���ʹ�㣺��������ע�������벻�ȶ� |

## ��ѡ����

| # | ���� | Ϊʲô��ͻ�� |
|---|------|------------|
| 1 | **van den Oord et al. (2016) "WaveNet: A Generative Model for Raw Audio" [[arXiv](https://arxiv.org/abs/1609.03499)]** | ���ѧϰ TTS ����Դ�Թ�������һ�������������ɸ�����ԭʼ��Ƶ���Σ����� 5000+ |
| 2 | **Wang et al. (2017) "Tacotron: Towards End-to-End Speech Synthesis" [[arXiv](https://arxiv.org/abs/1703.10135)]** | ��һ�����ı�ֱ��Ԥ�� mel �׵Ķ˵�����ѧģ�ͣ����� CBHG + Attention �ṹ |
| 3 | **Shen et al. (2018) "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" [[arXiv](https://arxiv.org/abs/1712.05884)]** | �� Tacotron �� WaveNet ����Ϊ�����˵��˹��ߣ����ʴﵽ�ӽ�����ˮƽ |

## �ܹ��ܽ�

```
�ı� �� Tacotron �� mel�� �� WaveNet �� ����
       (Encoder+Attn+Decoder)    (Causal Conv+Gate)
```

---

## ���Ĳο�

| ���� | ����(���) | ���� |
|---|---|---|
| Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions | Tacotron2 () | [arXiv](https://arxiv.org/abs/1712.05884) |
| Tacotron: Towards End-to-End Speech Synthesis | Tacotron () | [arXiv](https://arxiv.org/abs/1703.10135) |
| WaveNet: A Generative Model for Raw Audio | WaveNet () | [arXiv](https://arxiv.org/abs/1609.03499) |

---

