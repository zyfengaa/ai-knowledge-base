﻿# 04 �� ͳһ�˵��ˣ�Unified End-to-End��

## һ�仰����

> ���׶ι��ߣ���ѧģ�͡����������е� mel �׶�������λ��Ϣ������ģ�ͷֿ�ѵ������������ۻ������ܲ���һ��ģ��ֱ�Ӵ��ı����ɲ��Σ�

## ���ģ�����ʽ��⣨3-5 �Σ�

**��һ�㣺���ⶨ�塣** ͳһ�˵��˵�Ŀ�꣺һ��ģ��ֱ�Ӵ��ı����ɲ��Σ���������ʽ���м� mel �ס���Ҫͬʱ������������⣺�ı�����ѧ�ռ�ӳ�䣨02 �����񣩡���ѧ�����������ϲ�����03 �����񣩡��Լ���������ͳһĿ���»�����Ӧ������ì���ǣ���ѧģ�ͺ���������"��"�ı�׼��ͬ������ѧģ��׷������׼ȷ��������׷������ʵ��

**�ڶ��㣺����ֱ����** ���"�˵��˻�������� pipeline���ִʡ��䷨�����ɣ��� Transformer ��ת��"�����м������mel �ף���"������ʽ���"�����"��ʽ��ģ���ڲ�����"��VITS ����������һ�� VAE ��ģ��ѧϰǱ����ѧ�ռ� z����� mel �ף����� Normalizing Flow �� z �ķֲ�����Ȼ���� HiFi-GAN Generator �� z ֱ�Ӻϳɲ��Ρ������ڵ�һ��ʧ�����������Ż���

**�����㣺����ϸ�ڡ�** VITS �ܹ���⣺

```text
ѵ��ʱ���ı� + ��ʵ mel �� Posterior Encoder �� z �� Flow �� KL(Prior||Posterior)
                                   �� z �� HiFi-GAN Generator �� ����
����ʱ���ı� �� Prior Encoder �� z �� HiFi-GAN Generator �� ����
```

�������������
- **VAE ���**��Posterior Encoder������ʵ mel ����� z��+ Prior Encoder�����ı�Ԥ�� z����KL ɢ��Լ���ֲ�����
- **Normalizing Flow**����������任Ϊ���ֲ���ʹ z �Ľ�ģ����׼
- **HiFi-GAN Generator**���� z ֱ��ӳ�䵽���Σ��б�����Ϊ������ʧ

NaturalSpeech��΢���2022�����ò�ͬ·�ߣ����� Transformer + VAE + WaveNet Decoder��ǿ�� Scaling��

**���Ĳ㣺��ͬ������Ȩ�⡣**

| ά�� | Pipeline��02+03�� | VITS | NaturalSpeech |
|------|------------------|------|--------------|
| ѵ������ | ���׶ζ��� | ���׶ζ˵��� | ���׶ζ˵��� |
| ģ�ʹ�С | 2 ���е�ģ�� | 1 ���е�ģ�� | 1 ����ģ�� |
| ���� | ����� | ������ | ������ |
| �����ٶ� | ���� | ������ģ�ͣ� | ���� |
| �ɿ��� | ������ɽ�� | ������ռ���ϣ� | ���� |
| ʵ���Ѷ� | ��� | ����� | ����� |

**����㣺�ܽ�������** VITS �����˴�ͳ pipeline ��"�ռ���̬"������ģ�Ͷ˵��ˣ������ߡ�����졣������Ȼ��Ҫ����������ݣ��ҿɿ��Բ������� pipeline��2023 ��� TTS ����ת���� LLM + Codec ��ʽ��05�������Ǽ������ͳһ�˵���·�ߡ�

---

## ѧϰĿ��

�������ܣ�

- ��һ�仰˵�� VITS �� VAE �� Encoder �� Decoder ���Ե�����
- ���� VITS ��ѵ������ͼ����������ͼ��ע�����ߵĲ��죩
- ���� Normalizing Flow �� VITS �а��ݵĽ�ɫ
- �Ա� pipeline ������ͳһ�˵��˷��������ӣ�˵��������ó���
- ˵�� 2023 ��� TTS ����Ϊʲôת�� LLM ��ʽ

---

## ��ѡ����

**Kim et al. (2021) "VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" [[arXiv](https://arxiv.org/abs/2106.06103)]**

- **һ�仰��λ**��VAE + Flow + HiFi-GAN ����һ�ĵ�ģ�Ͷ˵��� TTS��GitHub 10k+ star������Ӱ�������
- **�Ķ��ص�**���� 2 �ڣ�VAE ��ʽ + Posterior/Prior Encoder + Flow ��ƣ�
- **ʱ����佨��**�������� 2 �ڣ�VAE ѵ��/������졢Flow ���ã�����¼�ܹ�ͼһ��Ҫ��
- **�뱾ģ��Ĺ�ϵ**��ͳһ�˵��˷�ʽ�ĺ��Ĵ��������������к������������û���� VITS

**Tan et al. (2022) "NaturalSpeech: End-to-End Text-to-Speech Synthesis with Naturalness" [[arXiv](https://arxiv.org/abs/2205.04421)]**

- **һ�仰��λ**��΢��Ĵ��ģ�˵��� TTS��չʾ Scaling up ·�ߵĳɹ��������ӽ�����¼��
- **�Ķ��ص�**���� 2 �ڣ�VAE + Transformer + WaveNet �Ĵ��ģѵ��˼·��
- **ʱ����佨��**���� VITS ���ն�����ע Scaling ���Ǽܹ����£���ʱ����ժҪ�ͽ��ۼ���
- **�뱾ģ��Ĺ�ϵ**��չʾ���� VITS ��ͬ�ļ���·�ߡ��������ģ�͡����������

---

## ��չ�Ķ�

- **Donahue et al. (2021) "EATS: End-to-end Adversarial Text-to-Speech" [[arXiv](https://arxiv.org/abs/2105.xxxxx)]** �� Google �Ķ˵��� TTS���� GAN + Duration Predictor ֱ�Ӵ��ı����ɲ��Ρ������� VITS ֮��Ķ˵���·�߸���Ȥ���Է�����

> ��չ���Ĳ��Ƴ������ڸ�ģ��� ��չ/ �ļ����¡�����������ģ���Ŀ¼��

---

## ģ�������

- **ǰ������**�������ȶ� **02-��ѧ��ģ** �� **03-��������**��VITS ���������ߵ��ںϣ�
- **�����ν�**��������� **05-��ģ����������**�����Ϊʲô TTS ����ת������ȫ��ͬ�� LLM ��ʽ
- **��ģ������Щģ������**���� 01���ı�ǰ�ˣ��������������Ƿ�˵��ˣ��ı�ǰ�˵����ⲻ��


---

## ���Ĳο�

| ���� | ����(���) | ���� |
|---|---|---|
| NaturalSpeech: End-to-End Text to Speech Synthesis with Naturalness Guarantees | NaturalSpeech () | [arXiv](https://arxiv.org/abs/2205.04421) |
| VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to... | VITS () | [arXiv](https://arxiv.org/abs/2106.06103) |

---

