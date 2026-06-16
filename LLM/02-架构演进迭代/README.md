# 02 �� �ܹ��ݽ�����

> ԭʼ Transformer ��ÿ������������õ������á�����6 ��ʱ�䣬5 ���ؼ��Ľ����Խ��һ���������⡣

## ���ģ�����ʽ���

**��һ�㣺���ⶨ�塣** Vaswani 2017 ��ԭʼ Transformer �� 5 �������õ����������������λ�ñ����ù̶����Ҳ�����������ע�������� O(n2) �� IO Ч�ʵ͡�ע����ͷ�����˷Ѽ��㡢LayerNorm ���˺ܶ಻��Ҫ�����㡢ReLU ��������������졣�� 5 �����⻥�����������Ը��Զ����ݽ���

**�ڶ��㣺����ֱ����** ������ô�룺
- **RoPE**����ÿ��λ��һ������ת�Ƕȡ�������ԽԶ�ǶȲ�Խ�󡪡�ģ�ʹӡ���λ�ñ�š���ɡ�����Ծ��롹
- **FlashAttention**���ѡ����Դ�ȡ���� �� �� �� д���Դ桹������̸ĳɡ��ֿ�������д���������ⷴ�����Դ棬���ø���� SRAM
- **GQA**������ע����ͷ���� K/V��ֻ�� Q ͷ��һ����������ʱֻ�軺��һ�� K/V���Դ����
- **RMSNorm**��ȥ�� LayerNorm �� mean centering������ֵ������ֻ���� scaling�����Ա�׼���ʡ��һ��ͳ��������
- **SwiGLU**��ReLU �ڸ�����ֱ�ӿ�����Ϣ �� SwiGLU �� sigmoid �ſر���һ���ָ�ֵ �� ��Ϣ�������ḻ

**�����㣺����ϸ�ڡ�** ����Ľ�������

| �Ľ� | ���Ĺ�ʽ / ���� | ���Ӷȱ仯 | Ч�� |
|------|----------------|-----------|------|
| RoPE | ����ת����� Q/K ���룬�ڻ���Ȼ�������λ�� | O(n2) ���� | ���������� 1x �� 8x+ |
| FlashAttention | �ֿ� tiling + ���� softmax �ؼ��� | O(n2) ���������� | ѵ������ 2-4x |
| GQA | ����ͷ�� K/V �ػ�Ϊ n �� | O(n2) ���䣬KV Cache /n | �����Դ���� |
| RMSNorm | ֻ�� Var(x) ��һ����ȥ�� Mean(x) | O(d) һ����ʡһ�� | ѵ�����ȶ� |
| SwiGLU | Swish(x��W) �� (x��V)�����ſ����Ե�Ԫ | ������Լ 30% | ѵ����ʧ���� |

**���Ĳ㣺��ͬ������Ȩ�⡣**

λ�ñ���Աȣ�

| ���� | �������� | ѵ���Ѻ� | ʵ�ָ��Ӷ� | ����ģ�� |
|------|---------|---------|-----------|---------|
| Absolute | ? �������� | ? �� | �� | GPT-2 |
| RoPE | ? ǿ��+NTK �ɴ� 128K�� | ? | �� | **LLaMA/Qwen/DeepSeek** |
| ALiBi | ?? �е� | ? | �� | MPT, Bloom |
| YaRN | ? �������� | ?? ������� | �� | ���� RoPE �����ϼ� |

ע����ͷ��ƶԱȣ�

| ���� | KV Cache ��С | ������ʧ | ���� |
|------|--------------|---------|------|
| MHA��ȫ�������� | ��� | �� | ԭʼ Transformer |
| MQA������һ�� K/V�� | ��С | ?? һ�� | PaLM |
| **GQA�����鹲���** | һ�� | ��С | **LLaMA 2/3, Qwen 2.5** |
| MLA������ѹ���� | ��С | ��С | DeepSeek V2 |

**����㣺�ܽ�������** �ִ� LLM �ı�׼������ **RoPE + GQA + SwiGLU + RMSNorm + FlashAttention**��������ϴ� 2023 �� LLaMA ȷ����������û�д�ı仯�������Ľ���MLA / Mamba / BitNet��Ҫô��Ч���Ż�Ҫô���·�ʽ���Ե���δ��Ϊ��һ����׼������� 5 ���Ľ����������˵����������� LLM �ġ��Ǽܡ���

---

## ѧϰĿ��

�������ܣ�

- ��һ�仰˵�� RoPE��GQA��FlashAttention��RMSNorm��SwiGLU ���Խ��ʲô����
- �г���ǰ���� LLM �ġ���׼���á��嵥��RoPE + GQA + SwiGLU + RMSNorm + FlashAttention��
- ���µ� LLM �Ƴ�ʱ��ͨ��������������Щ������ж����ġ�Ѫͳ��
- ˵������ 3 �� RoPE �����������ALiBi / Absolute / YaRN�������������
- ���Ϊʲô FlashAttention �ǡ�IO ��֪�����ǡ������Ż����������ĺ��Ĺ����ڷô�ģʽ�����㷨

---

## ��ѡ����

**Su et al. (2021) "RoFormer: Rotary Position Embedding" [[arXiv](https://arxiv.org/abs/2104.09864)]**

- **һ�仰��λ**��RoPE ��Ŀǰ��������λ�ñ��룬LLaMA/Qwen/DeepSeek ȫϵ������
- **�Ķ��ص�**���� 3 �ڡ���RoPE ����ת�����Ƶ������λ�ñ�������֤�������� insight �ǣ�����ת��������ӷ�����λ�ñ���
- **ʱ����佨��**����ʽ�Ƶ������������˼�뼴�ɣ����ص㿴 Figure 2 ��λ�ñ�����ӻ��Ա�

**Dao et al. (2022) "FlashAttention: Fast and Memory-Efficient Exact Attention" [[arXiv](https://arxiv.org/abs/2205.14135)]**

- **һ�仰��λ**��IO ��֪ע�����㷨�������ִ�ѵ��/�����ܱ���
- **�Ķ��ص�**���� 2 �ڣ�IO ���Ӷȷ��� + tiling ���� + �ؼ���˼�룩���Ⱦ����㷨ʵ�ָ���Ҫ
- **ʱ����佨��**��Section 3 �Ŀ鼶ʵ��ϸ�ڣ�Algorithm 1������������⡸Ϊʲô IO ��ƿ������������ֿ鷽������Ҫ

**Ainslie et al. (2023) "GQA: Training Generalized Multi-Query Transformer Models" [[arXiv](https://arxiv.org/abs/2305.13245)]**

- **һ�仰��λ**��MHA �� MQA �� GQA ���ݻ�·����LLaMA 2/3 ���õ�ע����ͷ���
- **�Ķ��ص�**���� 2 �ڣ�������ƵĶԱȣ���ʵ��������GQA ���������� MHA �൱���ٶ��� MQA �൱
- **ʱ����佨��**������� LLaMA �ܹ���Ϥ����ֱ�ӿ� Table 1 �ĶԱ��ܽ�

**Zhang & Sennrich (2019) "Root Mean Square Layer Normalization" [[arXiv](https://arxiv.org/abs/1910.07467)]**

- **һ�仰��λ**��RMSNorm��ȥ�� mean centering �ļ� LayerNorm��LLaMA ϵ�еı�׼��һ��
- **�Ķ��ص�**������˼��ܼ򵥡���ֻ�����Ա�׼���������ֵ������ 2 �ڹ�ʽ����
- **ʱ����佨��**��ȫ�ḷ́ܶ�~4 ҳ��������ͨ����ʵ�鲿�ֿ�������

**Shazeer (2020) "GLU Variants Improve Transformer" [[arXiv](https://arxiv.org/abs/2002.05202)]**

- **һ�仰��λ**��SwiGLU �������LLaMA/Qwen/DeepSeek/PaLM �ı�׼ѡ��
- **�Ķ��ص�**���� 2 �ڣ�GLU �������ʽ�����壩�͵� 3 �ڣ�ʵ��Աȣ�
- **ʱ����佨��**������Լ������Ϥ��ֻ�� Table 1 �ĶԱȽ����SwiGLU = Swish �� Linear ��ֱ���ȹ�ʽ��Ҫ

---

## ģ�������

- **ǰ������**�������ȶ� **01-Transformer ��Դ**����ģ�����۵� 5 ���Ľ����Ƕ�ԭʼ Transformer �������������������ԭʼ�ܹ���λ�ñ��� / ע���� / ��һ�� / ������ֱ���ʲô���޷����Ľ�������
- **�����ν�**�����걾ģ�����Խ����������ģ�顪��03 ��עѵ�����̡�04 ��ע�������05 ��עӦ�á�06 ��עǰ��
- **��ģ������Щģ������**���� 05-Ӧ�ü�����RAG / CoT����ȫ�������� 03-ѵ������뷶ʽ���������Ľ��ļܹ���Ӱ��ѵ�����Ե�ȡ�ᣩ


---

## ���Ĳο�

| ���� | ����(���) | ���� |
|---|---|---|
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | FlashAttention () | [arXiv](https://arxiv.org/abs/2205.14135) |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | GQA () | [arXiv](https://arxiv.org/abs/2305.13245) |
| Root Mean Square Layer Normalization | RMSNorm () | [arXiv](https://arxiv.org/abs/1910.07467) |
| RoFormer: Enhanced Transformer with Rotary Position Embedding | RoFormer () | [arXiv](https://arxiv.org/abs/2104.09864) |
| GLU Variants Improve Transformer | SwiGLU () | [arXiv](https://arxiv.org/abs/2002.05202) |

---

