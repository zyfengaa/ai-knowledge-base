# 02 — 声学建模（Acoustic Modeling）

## 一句话开场

> 你有一串音素 /həˈloʊ wɜːld/，但怎么从这十几个离散符号变成一长串连续的 80 维 mel 谱图？

## 正文：渐进式理解（3-5 段）

**第一层：问题定义。** 声学模型的任务：给定语言特征序列（音素、韵律标记），预测对应的声学参数（通常是对数 mel 谱）。核心矛盾：输入是离散的、长度已知的符号；输出是连续的、高维的、长度未知的声学帧。这个"长度未知"的对齐问题，是整个声学建模最核心的技术难点。

**第二层：核心直觉。** 类比为"乐谱 → 演奏"——同一张乐谱可以弹得快或慢、轻或重。不同流派对此有不同假设：自回归认为"逐帧生成，每帧依赖前一帧"（像一句一句说话）；非自回归认为"先看全局结构，再一次性生成"（像一次演奏完整首）。两者都不完美，所以衍生出了 Flow-based 等第三种路线。

**第三层：方案细节。** 三大路线的核心设计：

| 路线 | 代表 | 对齐方式 | 核心组件 |
|------|------|---------|---------|
| 自回归 AR | Tacotron 1/2 | Attention 隐式学习 | Encoder-Attention-Decoder + CBHG/GRU |
| 非自回归 NAR | FastSpeech 1/2 | Duration Predictor 显式 | FFT Blocks + Duration Predictor + Variance Adaptor |
| Flow-based | Glow-TTS | MAS 单调对齐搜索 | Transformer + Flow Decoder |

Tacotron 1（2017）首创 CBHG + Attention，但对齐不稳；Tacotron 2（2018）简化结构、定位敏感 Attention，质量飞跃。FastSpeech 1（2019）用 Teacher 蒸馏学时长，FastSpeech 2（2021）去掉蒸馏改用真实时长 + VAE。

**第四层：不同方案的权衡。**

| 维度 | 自回归（Tacotron） | 非自回归（FastSpeech） | Flow-based（Glow-TTS） |
|------|-------------------|----------------------|----------------------|
| 自然度 | ★★★★★ | ★★★★ | ★★★★★ |
| 推理速度 | ★（×1） | ★★★★★（×270） | ★★★（×2-5） |
| 对齐稳定性 | ★★（易飘移） | ★★★★★（稳定） | ★★★★（稳定） |
| 韵律多样性 | ★★★★★ | ★★★（偏平均） | ★★★★ |
| 实现复杂度 | ★★★ | ★★★ | ★★★★★ |

**第五层：总结升华。** 声学建模是 TTS pipeline 中最"学术"的一环。AR 高自然度但慢→被 NAR 取代，NAR 快但韵律偏平均→正在被 Flow/VAE 弥补。02 和 03（波形生成）是平行演化的两个独立模块，直到 04（统一端到端）将它们合并。

---

## 学习目标

读完你能：

- 用一句话说清 AR 和 NAR 最本质的区别：对齐方式——隐式 Attention vs 显式 Duration Predictor
- 画出 Tacotron 2 和 FastSpeech 2 的完整架构框图
- 解释 FastSpeech 1 为什么需要 Teacher 蒸馏、FastSpeech 2 为什么不需要
- 面对部署场景，给出选 Tacotron vs FastSpeech 的决策建议

---

## 精选论文

**Wang et al. (2017) "Tacotron: Towards End-to-End Speech Synthesis"**

- **一句话定位**：第一个端到端文本→mel 声学模型，奠定了 Encoder-Attention-Decoder 的 TTS 基础范式
- **阅读重点**：第 2-3 节（CBHG Encoder + Attention 机制）
- **时间分配建议**：精读第 2-3 节结构设计；第 4 节实验扫读
- **与本模块的关系**：回答了"能不能用一个神经网络代替传统多阶段声学建模"——证明可以

**Shen et al. (2018) "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"**

- **一句话定位**：Tacotron 2 将 Tacotron + WaveNet 串联为完整管线，质量接近人类水平
- **阅读重点**：第 2 节（相较于 Tacotron 1 的简化改进）
- **时间分配建议**：建议和 Tacotron 1 对照读，重点关注改进点
- **与本模块的关系**：回答了"自回归声学模型能不能做到工业级质量"——可以，但推理速度成瓶颈

**Ren et al. (2019) "FastSpeech: Fast, Robust and Controllable Text-to-Speech"**

- **一句话定位**：非自回归声学模型的首创，Duration Predictor + FFT Blocks 实现并行生成
- **阅读重点**：第 2-3 节（Duration Predictor + Teacher 蒸馏方案）
- **时间分配建议**：精读第 2 节并行生成机制；第 3 节扫读
- **与本模块的关系**：回答了"非自回归能不能做声学建模"——能，速度提升 270 倍

**Ren et al. (2021) "FastSpeech 2: Fast and High-Quality End-to-End Text-to-Speech"**

- **一句话定位**：去掉蒸馏，用真实时长 + VAE + Variance Adaptor，质量显著提升
- **阅读重点**：第 2 节（与 FastSpeech 1 的差异）
- **时间分配建议**：时间紧直接对照 FastSpeech 1 读差异部分
- **与本模块的关系**：回答了"非自回归能否追上自回归质量"——通过 VAE + 真实时长，大幅缩小差距

---

## 拓展阅读

- **Kim et al. (2020) "Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search"** — Flow-based 声学建模的代表作，用 MAS 替代 Attention。如果你对"不用 Duration Predictor 也不用 Attention 的第三种方案"感兴趣可以翻翻。



> 拓展论文不移除，放在各模块的 拓展/ 文件夹下。核心论文在模块根目录。

---

## 模块间连接

- **前置依赖**：建议先读 **01-文本前端**（理解输入的语言表征）
- **后续衔接**：读完进入 **03-波形生成** 或 **04-统一端到端**
- **本模块与哪些模块正交**：与 03（波形生成）是流水线相邻但设计独立——声学模型关注"内容"，声码器关注"质感"





---



## 论文参考



| 论文 | 作者(年份) | 链接 |
|---|---|---|
| FastSpeech 2: Fast and High-Quality End-to-End Text to Speech | FastSpeech2 () | [arXiv](https://arxiv.org/abs/2006.04558) |
| FastSpeech: Fast, Robust and Controllable Text to Speech | FastSpeech () | [arXiv](https://arxiv.org/abs/1905.09263) |
| Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions | Tacotron2 () | [arXiv](https://arxiv.org/abs/1712.05884) |
| Tacotron: Towards End-to-End Speech Synthesis | Tacotron () | [arXiv](https://arxiv.org/abs/1703.10135) |



---



## 论文参考



| 论文 | 链接 |
|---|---|
| FastSpeech 2: Fast and High-Quality End-to-End Text to Speech | [arXiv](https://arxiv.org/abs/2006.04558) |
| FastSpeech: Fast, Robust and Controllable Text to Speech | [arXiv](https://arxiv.org/abs/1905.09263) |
| Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions | [arXiv](https://arxiv.org/abs/1712.05884) |
| Tacotron: Towards End-to-End Speech Synthesis | [arXiv](https://arxiv.org/abs/1703.10135) |

