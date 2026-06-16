# 03 — 波形生成（Waveform Generation / Vocoders）

## 一句话开场

> mel 谱本质上是一张"80 维的描述图"，它丢了相位信息。怎么从这张描述图还原出每秒 16000 个采样点的可播放波形？

## 正文：渐进式理解（3-5 段）

**第一层：问题定义。** 声码器的任务：将 mel 谱等声学参数转换为可听的 16-bit PCM 波形。核心挑战是 200-300 倍的上采样——mel 谱每秒约 86 帧（每帧 80 维），波形每秒 16000-24000 个采样点。模型不仅要补全高频细节，还要恢复被 mel 谱丢弃的相位信息。

**第二层：核心直觉。** 类比"黑白线稿上色"——mel 谱是线稿（低频骨架），声码器要补全颜色（高频细节 + 相位）。WaveNet 的做法像"逐像素涂色"（最精细但最慢），HiFi-GAN 的做法像生成对抗——生成器一次画完整张图，判别器检查哪里不真实。

**第三层：方案细节。** 三代声码器的核心设计：

| 代际 | 代表 | 生成方式 | 核心设计 |
|------|------|---------|---------|
| 第一代：自回归 | WaveNet (2016) | 逐采样点 | Causal Dilated CNN + Gated Activation |
| 第二代：Flow | WaveGlow (2018) | 一次前向 | Glow-based invertible network |
| 第三代：GAN | HiFi-GAN (2020) | 一次生成 + 对抗 | Generator: Transposed Conv + MRF; Discriminator: MPD + MSD |

WaveNet 的关键设计：32 层因果扩张卷积堆叠（dilation 1→2→4→...→512），每层用 Gated Activation（tanh × sigmoid）+ Skip Connection。
HiFi-GAN 的关键设计：Generator 用转置卷积上采样 + MRF 融合多路卷积；MPD 从多周期维度看波形（period=2,3,5,7,11），捕捉不同粒度的周期性。

**第四层：不同方案的权衡。**

| 维度 | WaveNet | WaveGlow | HiFi-GAN |
|------|---------|----------|----------|
| 音质 | ★★★★★ | ★★★★ | ★★★★★ |
| 推理速度 | ★（实时×0.02） | ★★（实时×5） | ★★★★（实时×100+） |
| 训练稳定性 | ★★★★★ | ★★（难收敛） | ★★★ |
| 参数量 | 5M | 88M | 13M |
| 实时部署 | ❌ 不可能 | ⚠️ 勉强 | ✅ 可行 |

**第五层：总结升华。** WaveNet 开创了"深度生成模型做声码器"的方向，但自回归的代价使其无法部署；HiFi-GAN 在速度和质量的帕累托前沿上找到了最优解，是 2020 年至今的工业事实标准。后续 VITS 把 HiFi-GAN Generator 作为组件融入端到端模型，声码器从独立模块变成了端到端模型的一部分。

---

## 学习目标

读完你能：

- 用一句话说清 WaveNet 和 HiFi-GAN 生成方式最本质的区别（逐点 AR vs 一次 GAN）
- 画出 WaveNet 单层结构：Causal Conv + Gated Activation + Skip Connection
- 解释 HiFi-GAN 的 Multi-Period Discriminator 为什么比普通判别器更适合语音
- 面对实时 TTS 产品，给出选哪种声码器的决策及理由

---

## 精选论文

**van den Oord et al. (2016) "WaveNet: A Generative Model for Raw Audio"**

- **一句话定位**：深度学习 TTS 的起源，第一个用神经网络生成高质量原始音频波形，被引 8000+
- **阅读重点**：第 2-3 节（Causal Dilated CNN + Gated Activation）
- **时间分配建议**：精读第 2 节架构设计（这是本质创新）；后续变分下界扫读即可
- **与本模块的关系**：波形生成领域的起点，所有后续 vocoder 工作都建立在它的基础上

**Kong et al. (2020) "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"**

- **一句话定位**：GAN vocoder 的里程碑，多周期判别器实现高质量+高性能，工业级标准
- **阅读重点**：第 2-3 节（Generator 的 MRF + Multi-Period/Multi-Scale Discriminator）
- **时间分配建议**：精读第 2 节判别器设计（MPD 是核心创新）；第 3 节消融实验推荐读
- **与本模块的关系**：代表了工业界声码器的最优方案，也是 VITS 的生成器组件

---

## 拓展阅读

- **Prenger et al. (2019) "WaveGlow: A Flow-based Generative Network for Speech Synthesis"** — Flow-based 声码器的代表，将逐点生成改为一次前向。如果你对"非 AR 也非 GAN"的路线感兴趣可以翻翻。

> 拓展论文不移除，放在各模块的 拓展/ 文件夹下。核心论文在模块根目录。

---

## 模块间连接

- **前置依赖**：建议先读 **02-声学建模**（理解 mel 谱是什么）；有 STFT/spectrogram 基础可直接读
- **后续衔接**：读完进入 **04-统一端到端**，理解 VITS 如何把 HiFi-GAN 融入端到端模型
- **本模块与哪些模块正交**：与 01（文本前端）完全独立——输出侧和输入侧的独立问题


---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | HiFiGAN () | [arXiv](https://arxiv.org/abs/2010.05646) |
| WaveNet: A Generative Model for Raw Audio | WaveNet () | [arXiv](https://arxiv.org/abs/1609.03499) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | [arXiv](https://arxiv.org/abs/2010.05646) |
| WaveNet: A Generative Model for Raw Audio | [arXiv](https://arxiv.org/abs/1609.03499) |
