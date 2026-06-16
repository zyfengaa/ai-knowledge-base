﻿# 05 �� ��ģ������������Large Model & Zero-shot��

## һ�仰����

> ���� 3 ���ϰ��¼���Ȼ�����������������ʼ�����2023 ��֮ǰ����Ҫ��Сʱ¼�����ݣ�2023 ��֮������һ�� TTS ģ�͵�Ĭ��������

## ���ģ�����ʽ��⣨3-5 �Σ�

**��һ�㣺���ⶨ�塣** ��ģ�� TTS Ҫ����ĺ������⣺�ڼ�С������Ŀ��˵���˱�ע���ݵ�����£��ϳ�����˵���˵���Ȼ�������Ҫ��ģ��ͬʱ���㷢��׼ȷ����ɫ���̡�������Ȼ����ͳ TTS �� Scaling ������"ģ��Խ��Խ��Ȼ"������Ȼ��Ҫ������ݡ���ģ�ͷ�ʽ�ĸ���ͻ�����ڣ�**�� TTS ��"��ѧ�����Ļع�����"���¶���Ϊ��"���� token ��������������"**��

**�ڶ��㣺����ֱ����** ����˼�뼫��ֱ�ۣ���Ȼ LLM ���԰��ı� tokenize Ȼ��Ԥ����һ�� token��������Ҳ���Ա� tokenize����ɢ���� token����Ȼ����ͬ���� next-token prediction ��ʽ��ѵ�����ؼ������Ҷ�"���� tokenizer"�����񾭱��������Neural Audio Codec������������ѹ������ɢ codec token��һ���������� token������ LLM �����䣨Scaling��In-context Learning��Prompting����ȫ���������ˡ�

**�����㣺����ϸ�ڡ�** ���������·�ߣ�

| ���� | ���ķ�ʽ | ������� | �ؼ����� |
|------|---------|---------|---------|
| VALL-E (2023) | AR+NAR Next-token Prediction | EnCodec �޼ල codec | ����� GPT��AR ������ + NAR ϸ���� |
| VoiceBox (2023) | Flow Matching + Infilling | EnCodec codec | ͳһ TTS/ȥ��/�༭��һ����ܶ����� |
| CosyVoice (2024) | LLM + CausalFlow Matching | SST �мල���� token | ����/��ѧ���Σ�SST �������ݺ���ɫ |

VALL-E��EnCodec tokenize �� AR ģ�����ɴ� token �� NAR ģ�Ͳ�ȫϸ token �� EnCodec decoder ��ԭ���Ρ�
VoiceBox����� mask ����Ƭ�� �� Flow Matching ���ı���������� mask �� һ�����ͬʱ�� TTS/ȥ��/�༭��
CosyVoice��SST token���мලѧϰ���壩+ LLM ���� SST �� CausalFlow Matching ������ѧ���� �� vocoder �ϳɲ��Ρ�

**���Ĳ㣺��ͬ������Ȩ�⡣**

| ά�� | VALL-E | VoiceBox | CosyVoice |
|------|--------|----------|-----------|
| ���������� | ����� | ������ | ������ |
| ������ | ����Ӣ�ģ� | ����6 �֣� | ������Ӣ�� |
| �����ٶ� | ��AR ���� | ���� | ����� |
| ѵ������ | 60k hrs | 50k hrs | ~10k hrs |
| �ɿ��� | ��Prompt ������ | ���Text-guided�� | ����SFT�� |
| �þ����� | ���϶ࣩ | ���� | �������٣� |

**����㣺�ܽ�������** ��ģ�� TTS ���� WaveNet ���� TTS ������ķ�ʽת�䡣�����ĺô��������� + ����չ + ͳһ��ܡ����������鷳��LLM ��ͨ�����þ����߳ɱ������ɿ��ԣ�Ҳ�����̳С�2025-2026 �꿴����ģ�� TTS �ѳ�Ϊѧ���Ͳ�ҵ�ľ�����������ͳ pipeline TTS ������"�ͳɱ���Ե����"����ֻ���

---

## ѧϰĿ��

�������ܣ�

- ��һ�仰˵�� VALL-E �ĺ���˼�룺���� tokenization + next-token prediction = ����� GPT
- �����񾭱��������Neural Codec���ڴ�ģ�� TTS �а��ݵĽ�ɫ������������� Tokenizer
- �Ա� VALL-E��VoiceBox��CosyVoice ��"��ν�ģ����"�ϵĺ��Ĳ���
- ˵����ģ�� TTS ��ȴ�ͳ���������ƣ�������/����չ/ͳһ�����������ޣ��þ�/�ɱ�/�ɿ��ԣ�
- ���һ�� TTS ��Ŀ���ж��ʺϴ�ͳ��ʽ���Ǵ�ģ�ͷ�ʽ

---

## ��ѡ����

**Wang et al. (2023) "VALL-E: Neural Codec Language Model for Zero-Shot Text-to-Speech" [[arXiv](https://arxiv.org/abs/2301.02111)]**

- **һ�仰��λ**���״����"�񾭱���� + ����ģ��"�� TTS �·�ʽ�������������¡��ͻ���Թ���
- **�Ķ��ص�**���� 2-3 �ڣ�EnCodec tokenization + AR+NAR ���㽨ģ��
- **ʱ����佨��**�������� 2 �� AR+NAR ģ����ƣ������ idea������ 4 ��ɨ��
- **�뱾ģ��Ĺ�ϵ**����ģ�� TTS ·�ߵĿ���֮���������� Codec + LM ��ʽ

**Le et al. (2023) "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale" [[arXiv](https://arxiv.org/abs/2306.15687)]**

- **һ�仰��λ**��Meta ������ʽ����ģ�ͣ�Flow Matching + Infilling һ�����ʵ�� TTS/ȥ��/�༭/������
- **�Ķ��ص�**���� 2-3 �ڣ�Flow Matching ѵ��Ŀ�� + Text-Guided Infilling ���ԣ�
- **ʱ����佨��**�������˽� Flow Matching �������ص���� 2 �� Infilling ����
- **�뱾ģ��Ĺ�ϵ**��չʾ��"���Իع��ģ�� TTS"·�ߵĿ����ԣ��� VALL-E ����

**Du et al. (2024) "CosyVoice: A Scalable Multilingual Zero-shot Text-to-Speech based on Supervised Semantic Tokens" [[arXiv](https://arxiv.org/abs/2407.05407)]**

- **һ�仰��λ**������ͨ��Ķ����������� TTS��LLM + �мල���� token + CausalFlow Matching���������²�ҵ����
- **�Ķ��ص�**���� 2-3 �ڣ�SST token ��� + CausalFlow Matching �ܹ���
- **ʱ����佨��**�������� 2 �� SST ��ƣ��� VALL-E �� EnCodec token ���ն���
- **�뱾ģ��Ĺ�ϵ**��չʾ��"���� + ��ѧ����"�Ľ��˼·���ȴ� codec LM ���ȶ�

---

## ��չ�Ķ�

- **Kharitonov et al. (2023) "SpearTTS: Speaker-Fourier Transformer for Text-to-Speech" [[arXiv](https://arxiv.org/abs/2305.xxxxx)]** �� Meta ������ speaker-conditioned TTS��չʾ�� speaker ����˼·��������"˵������Ϣ��ν�ģ"����Ȥ���Է�����

> ��չ���Ĳ��Ƴ������ڸ�ģ��� ��չ/ �ļ����¡�����������ģ���Ŀ¼��

---

## ģ�������

- **ǰ������**�������ȶ� **01-�ı�ǰ��** �� **02-��ѧ��ģ**����⴫ͳ��ʽ����ܶԱ� LLM ��ʽ
- **�����ν�**��������� **06-�ɿ�������Ի�**����������з�ʽ�Ŀ����ڴ�ģ��ʱ������ս
- **��ģ������Щģ������**���� 03���������ɣ���һ���̶��϶���������ģ�� TTS Ҳ���� vocoder/codec decoder ��������ؽ�


---

## ���Ĳο�

| ���� | ����(���) | ���� |
|---|---|---|
| CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer | CosyVoice () | [arXiv](https://arxiv.org/abs/2407.05407) |
| VALL-E: Neural Codec Language Models for Zero-Shot Text to Speech Synthesis | VALL-E () | [arXiv](https://arxiv.org/abs/2301.02111) |
| VoiceBox: Text-Guided Multilingual Universal Speech Generation at Scale | VoiceBox () | [arXiv](https://arxiv.org/abs/2306.15687) |

---

