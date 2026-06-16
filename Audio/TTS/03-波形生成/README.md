﻿# 03 �� �������ɣ�Waveform Generation / Vocoders��

## һ�仰����

> mel �ױ�������һ��"80 ά������ͼ"����������λ��Ϣ����ô����������ͼ��ԭ��ÿ�� 16000 ��������Ŀɲ��Ų��Σ�

## ���ģ�����ʽ��⣨3-5 �Σ�

**��һ�㣺���ⶨ�塣** �����������񣺽� mel �׵���ѧ����ת��Ϊ������ 16-bit PCM ���Ρ�������ս�� 200-300 �����ϲ�������mel ��ÿ��Լ 86 ֡��ÿ֡ 80 ά��������ÿ�� 16000-24000 �������㡣ģ�Ͳ���Ҫ��ȫ��Ƶϸ�ڣ���Ҫ�ָ��� mel �׶�������λ��Ϣ��

**�ڶ��㣺����ֱ����** ���"�ڰ��߸���ɫ"����mel �����߸壨��Ƶ�Ǽܣ���������Ҫ��ȫ��ɫ����Ƶϸ�� + ��λ����WaveNet ��������"������Ϳɫ"���ϸ����������HiFi-GAN �����������ɶԿ�����������һ�λ�������ͼ���б���������ﲻ��ʵ��

**�����㣺����ϸ�ڡ�** �����������ĺ�����ƣ�

| ���� | ���� | ���ɷ�ʽ | ������� |
|------|------|---------|---------|
| ��һ�����Իع� | WaveNet (2016) | ������� | Causal Dilated CNN + Gated Activation |
| �ڶ�����Flow | WaveGlow (2018) | һ��ǰ�� | Glow-based invertible network |
| ��������GAN | HiFi-GAN (2020) | һ������ + �Կ� | Generator: Transposed Conv + MRF; Discriminator: MPD + MSD |

WaveNet �Ĺؼ���ƣ�32 ��������ž���ѵ���dilation 1��2��4��...��512����ÿ���� Gated Activation��tanh �� sigmoid��+ Skip Connection��
HiFi-GAN �Ĺؼ���ƣ�Generator ��ת�þ���ϲ��� + MRF �ں϶�·�����MPD �Ӷ�����ά�ȿ����Σ�period=2,3,5,7,11������׽��ͬ���ȵ������ԡ�

**���Ĳ㣺��ͬ������Ȩ�⡣**

| ά�� | WaveNet | WaveGlow | HiFi-GAN |
|------|---------|----------|----------|
| ���� | ������ | ����� | ������ |
| �����ٶ� | �ʵʱ��0.02�� | ��ʵʱ��5�� | ����ʵʱ��100+�� |
| ѵ���ȶ��� | ������ | ���������� | ���� |
| ������ | 5M | 88M | 13M |
| ʵʱ���� | ? ������ | ?? ��ǿ | ? ���� |

**����㣺�ܽ�������** WaveNet ������"�������ģ����������"�ķ��򣬵��Իع�Ĵ���ʹ���޷�����HiFi-GAN ���ٶȺ�������������ǰ�����ҵ������Ž⣬�� 2020 ������Ĺ�ҵ��ʵ��׼������ VITS �� HiFi-GAN Generator ��Ϊ�������˵���ģ�ͣ��������Ӷ���ģ�����˶˵���ģ�͵�һ���֡�

---

## ѧϰĿ��

�������ܣ�

- ��һ�仰˵�� WaveNet �� HiFi-GAN ���ɷ�ʽ��ʵ�������� AR vs һ�� GAN��
- ���� WaveNet ����ṹ��Causal Conv + Gated Activation + Skip Connection
- ���� HiFi-GAN �� Multi-Period Discriminator Ϊʲô����ͨ�б������ʺ�����
- ���ʵʱ TTS ��Ʒ������ѡ�����������ľ��߼�����

---

## ��ѡ����

**van den Oord et al. (2016) "WaveNet: A Generative Model for Raw Audio" [[arXiv](https://arxiv.org/abs/1609.03499)]**

- **һ�仰��λ**�����ѧϰ TTS ����Դ����һ�������������ɸ�����ԭʼ��Ƶ���Σ����� 8000+
- **�Ķ��ص�**���� 2-3 �ڣ�Causal Dilated CNN + Gated Activation��
- **ʱ����佨��**�������� 2 �ڼܹ���ƣ����Ǳ��ʴ��£�����������½�ɨ������
- **�뱾ģ��Ĺ�ϵ**�����������������㣬���к��� vocoder ���������������Ļ�����

**Kong et al. (2020) "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" [[arXiv](https://arxiv.org/abs/2010.05646)]**

- **һ�仰��λ**��GAN vocoder ����̱����������б���ʵ�ָ�����+�����ܣ���ҵ����׼
- **�Ķ��ص�**���� 2-3 �ڣ�Generator �� MRF + Multi-Period/Multi-Scale Discriminator��
- **ʱ����佨��**�������� 2 ���б�����ƣ�MPD �Ǻ��Ĵ��£����� 3 ������ʵ���Ƽ���
- **�뱾ģ��Ĺ�ϵ**�������˹�ҵ�������������ŷ�����Ҳ�� VITS �����������

---

## ��չ�Ķ�

- **Prenger et al. (2019) "WaveGlow: A Flow-based Generative Network for Speech Synthesis" [[arXiv](https://arxiv.org/abs/1811.00002)]** �� Flow-based �������Ĵ������������ɸ�Ϊһ��ǰ��������"�� AR Ҳ�� GAN"��·�߸���Ȥ���Է�����

> ��չ���Ĳ��Ƴ������ڸ�ģ��� ��չ/ �ļ����¡�����������ģ���Ŀ¼��

---

## ģ�������

- **ǰ������**�������ȶ� **02-��ѧ��ģ**����� mel ����ʲô������ STFT/spectrogram ������ֱ�Ӷ�
- **�����ν�**��������� **04-ͳһ�˵���**����� VITS ��ΰ� HiFi-GAN ����˵���ģ��
- **��ģ������Щģ������**���� 01���ı�ǰ�ˣ���ȫ�������������������Ķ�������


---

## ���Ĳο�

| ���� | ����(���) | ���� |
|---|---|---|
| HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | HiFiGAN () | [arXiv](https://arxiv.org/abs/2010.05646) |
| WaveNet: A Generative Model for Raw Audio | WaveNet () | [arXiv](https://arxiv.org/abs/1609.03499) |

---

