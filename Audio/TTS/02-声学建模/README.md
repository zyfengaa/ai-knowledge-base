﻿﻿# 02 �� ��ѧ��ģ��Acoustic Modeling��

## һ�仰����

> ����һ������ /h??lo? w??ld/������ô����ʮ������ɢ���ű��һ���������� 80 ά mel ��ͼ��

## ���ģ�����ʽ��⣨3-5 �Σ�

**��һ�㣺���ⶨ�塣** ��ѧģ�͵����񣺸��������������У����ء����ɱ�ǣ���Ԥ���Ӧ����ѧ������ͨ���Ƕ��� mel �ף�������ì�ܣ���������ɢ�ġ�������֪�ķ��ţ�����������ġ���ά�ġ�����δ֪����ѧ֡�����"����δ֪"�Ķ������⣬��������ѧ��ģ����ĵļ����ѵ㡣

**�ڶ��㣺����ֱ����** ���Ϊ"���� �� ����"����ͬһ�����׿��Ե��ÿ����������ء���ͬ���ɶԴ��в�ͬ���裺�Իع���Ϊ"��֡���ɣ�ÿ֡����ǰһ֡"����һ��һ��˵���������Իع���Ϊ"�ȿ�ȫ�ֽṹ����һ��������"����һ�����������ף������߶��������������������� Flow-based �ȵ�����·�ߡ�

**�����㣺����ϸ�ڡ�** ����·�ߵĺ�����ƣ�

| ·�� | ���� | ���뷽ʽ | ������� |
|------|------|---------|---------|
| �Իع� AR | Tacotron 1/2 | Attention ��ʽѧϰ | Encoder-Attention-Decoder + CBHG/GRU |
| ���Իع� NAR | FastSpeech 1/2 | Duration Predictor ��ʽ | FFT Blocks + Duration Predictor + Variance Adaptor |
| Flow-based | Glow-TTS | MAS ������������ | Transformer + Flow Decoder |

Tacotron 1��2017���״� CBHG + Attention�������벻�ȣ�Tacotron 2��2018���򻯽ṹ����λ��� Attention��������Ծ��FastSpeech 1��2019���� Teacher ����ѧʱ����FastSpeech 2��2021��ȥ�����������ʵʱ�� + VAE��

**���Ĳ㣺��ͬ������Ȩ�⡣**

| ά�� | �Իع飨Tacotron�� | ���Իع飨FastSpeech�� | Flow-based��Glow-TTS�� |
|------|-------------------|----------------------|----------------------|
| ��Ȼ�� | ������ | ����� | ������ |
| �����ٶ� | ���1�� | �������270�� | �����2-5�� |
| �����ȶ��� | ����Ʈ�ƣ� | ������ȶ��� | �����ȶ��� |
| ���ɶ����� | ������ | ���ƫƽ���� | ����� |
| ʵ�ָ��Ӷ� | ���� | ���� | ������ |

**����㣺�ܽ�������** ��ѧ��ģ�� TTS pipeline ����"ѧ��"��һ����AR ����Ȼ�ȵ������� NAR ȡ����NAR �쵫����ƫƽ�������ڱ� Flow/VAE �ֲ���02 �� 03���������ɣ���ƽ���ݻ�����������ģ�飬ֱ�� 04��ͳһ�˵��ˣ������Ǻϲ���

---

## ѧϰĿ��

�������ܣ�

- ��һ�仰˵�� AR �� NAR ��ʵ����𣺶��뷽ʽ������ʽ Attention vs ��ʽ Duration Predictor
- ���� Tacotron 2 �� FastSpeech 2 �������ܹ���ͼ
- ���� FastSpeech 1 Ϊʲô��Ҫ Teacher ����FastSpeech 2 Ϊʲô����Ҫ
- ��Բ��𳡾�������ѡ Tacotron vs FastSpeech �ľ��߽���

---

## ��ѡ����

**Wang et al. (2017) "Tacotron: Towards End-to-End Speech Synthesis" [[arXiv](https://arxiv.org/abs/1703.10135)]**

- **һ�仰��λ**����һ���˵����ı���mel ��ѧģ�ͣ��춨�� Encoder-Attention-Decoder �� TTS ������ʽ
- **�Ķ��ص�**���� 2-3 �ڣ�CBHG Encoder + Attention ���ƣ�
- **ʱ����佨��**�������� 2-3 �ڽṹ��ƣ��� 4 ��ʵ��ɨ��
- **�뱾ģ��Ĺ�ϵ**���ش���"�ܲ�����һ����������洫ͳ��׶���ѧ��ģ"����֤������

**Shen et al. (2018) "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions" [[arXiv](https://arxiv.org/abs/1712.05884)]**

- **һ�仰��λ**��Tacotron 2 �� Tacotron + WaveNet ����Ϊ�������ߣ������ӽ�����ˮƽ
- **�Ķ��ص�**���� 2 �ڣ������ Tacotron 1 �ļ򻯸Ľ���
- **ʱ����佨��**������� Tacotron 1 ���ն����ص��ע�Ľ���
- **�뱾ģ��Ĺ�ϵ**���ش���"�Իع���ѧģ���ܲ���������ҵ������"�������ԣ��������ٶȳ�ƿ��

**Ren et al. (2019) "FastSpeech: Fast, Robust and Controllable Text-to-Speech" [[arXiv](https://arxiv.org/abs/1905.09263)]**

- **һ�仰��λ**�����Իع���ѧģ�͵��״���Duration Predictor + FFT Blocks ʵ�ֲ�������
- **�Ķ��ص�**���� 2-3 �ڣ�Duration Predictor + Teacher ���󷽰���
- **ʱ����佨��**�������� 2 �ڲ������ɻ��ƣ��� 3 ��ɨ��
- **�뱾ģ��Ĺ�ϵ**���ش���"���Իع��ܲ�������ѧ��ģ"�����ܣ��ٶ����� 270 ��

**Ren et al. (2021) "FastSpeech 2: Fast and High-Quality End-to-End Text-to-Speech" [[arXiv](https://arxiv.org/abs/2006.04558)]**

- **һ�仰��λ**��ȥ����������ʵʱ�� + VAE + Variance Adaptor��������������
- **�Ķ��ص�**���� 2 �ڣ��� FastSpeech 1 �Ĳ��죩
- **ʱ����佨��**��ʱ���ֱ�Ӷ��� FastSpeech 1 �����첿��
- **�뱾ģ��Ĺ�ϵ**���ش���"���Իع��ܷ�׷���Իع�����"����ͨ�� VAE + ��ʵʱ���������С���

---

## ��չ�Ķ�

- **Kim et al. (2020) "Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search" [[arXiv](https://arxiv.org/abs/2005.11129)]** �� Flow-based ��ѧ��ģ�Ĵ��������� MAS ��� Attention��������"���� Duration Predictor Ҳ���� Attention �ĵ����ַ���"����Ȥ���Է�����

> ��չ���Ĳ��Ƴ������ڸ�ģ��� ��չ/ �ļ����¡�����������ģ���Ŀ¼��

---

## ģ�������

- **ǰ������**�������ȶ� **01-�ı�ǰ��**�������������Ա�����
- **�����ν�**��������� **03-��������** �� **04-ͳһ�˵���**
- **��ģ������Щģ������**���� 03���������ɣ�����ˮ�����ڵ���ƶ���������ѧģ�͹�ע"����"����������ע"�ʸ�"


---

## ���Ĳο�

| ���� | ����(���) | ���� |
|---|---|---|
| FastSpeech 2: Fast and High-Quality End-to-End Text to Speech | FastSpeech2 () | [arXiv](https://arxiv.org/abs/2006.04558) |
| FastSpeech: Fast, Robust and Controllable Text to Speech | FastSpeech () | [arXiv](https://arxiv.org/abs/1905.09263) |
| Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions | Tacotron2 () | [arXiv](https://arxiv.org/abs/1712.05884) |
| Tacotron: Towards End-to-End Speech Synthesis | Tacotron () | [arXiv](https://arxiv.org/abs/1703.10135) |

---

