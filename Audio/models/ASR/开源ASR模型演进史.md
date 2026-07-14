# 开源 ASR 模型演进史（2014-2026）

> 只收录有可下载权重/代码的**真正开源**模型。每个模型的前置条件是理解它"为什么出现在这个时间点"的关键。

---

## 2014 — DeepSpeech 1 · 百度

### 前置条件：GMM-HMM 体系的统治

在 DeepSpeech 1 之前，ASR 是一个复杂的多组件管线：

- **声学模型**：GMM（高斯混合模型），算每一帧属于哪个音素的概率
- **语言模型**：n-gram，算词序列的概率
- **发音词典**：词 → 音素映射表
- **解码器**：WFST（加权有限状态转换器）把上述所有组件编译到一张图中搜索最优路径

每个组件用不同的算法、不同的人来训练——声学团队调 GMM 混合数，语言团队调 n-gram 裁剪阈值，出了问题根本分不清是谁的锅。整个行业在等一个模型把这一切统一起来。

### DeepSpeech 1

- **架构**：3 层全连接 → 1 层双向 RNN → 全连接 → softmax + CTC
- **训练数据**：7,000 小时纯净语音 + 15 类噪声叠加合成 → 100,000 小时
- **改进点**：用 CTC 代替 HMM 做对齐——模型自己学每帧对应哪个字符，不再需要手工指定对齐
- **效果**：噪声环境下 WER 比当时的 Google/Apple 商用系统低 10-13%
- **局限**：架构太浅（只有 1 层 Bi-RNN），中文准确率远不够商用
- **开源**：❌ 论文公开，权重和代码均未开源

---

## 2015 — DeepSpeech 2 · 百度

### 前置条件：DS1 验证了路线，但精度不够

DS1 证明了"端到端可行"，但只能做英文、训练太慢、模型太浅。市场需要：(1) 中英文双语言；(2) 更深更强的时序建模；(3) 能商用的推理延迟。

### DeepSpeech 2

| 升级点 | DS1 | DS2 | 原因 |
|--------|-----|-----|------|
| 架构 | 3FC+1Bi-RNN | **2-3 Conv + 7+ GRU** | 卷积层提频谱特征更鲁棒；GRU 时序建模更强 |
| 参数 | 较小 | **~300M** | 更大容量才能覆盖中英 |
| 语言 | 仅英语 | **英语 + 普通话** | 第一个中英同架构 SOTA |
| 训练 | 数据并行 | **SortaGrad + BatchNorm for RNN + 16 GPU 同步 SGD** | BatchNorm 加速收敛；SortaGrad 解决 CTC 长度偏差 |
| 数据 | 7K→100K增强 | **11,940h 英文 + 9,400h 中文人工标注** | 有声书/朗读数据，质量远高于 DS1 的合成数据 |
| 英文 WER | ~13.2% | **~5-7%** | 相对降低 43% |
| 中文 CER | 不支持 | **~6.8%（AISHELL-1）** | 首次中文端到端 SOTA |
| 延迟 | 未优化 | **98分位 67ms** | 达到生产部署标准 |

**入选 2016 年《麻省理工科技评论》十大突破技术**。证明了端到端深度学习可以在双语上达到商用水平。

- **开源**：✅ 代码 Apache 2.0（PaddlePaddle），预训练权重 Wolfram 仓库可获取

---

## 2017 — Mozilla DeepSpeech（v0.1 起）

### 前置条件：没有"下载就能用的 ASR"

百度 DeepSpeech 2 论文开源了架构，但代码用 PaddlePaddle 写、数据是百度内部不公开的。其他公司和个人要做端到端 ASR 只有两条路：(1) 自己从零训——需要几千到一万小时数据 + GPU 集群；(2) 用 Kaldi——GMM-HMM 老路，学习曲线极陡。**没有预训练权重可下载、没有离线 runtime 可用。**

### Mozilla DeepSpeech

- **架构**：基于百度 DeepSpeech 论文的 TensorFlow 实现，5 层 GRU + CTC
- **预训练权重**：✅ 英文 WER ~7.5%，中文等 10+ 语言权重均可直接下载
- **部署能力**：提供 C++ runtime + Python/Node.js/Rust 绑定，**Raspberry Pi 4 能离线实时跑**
- **历史定位**：在 Whisper 出现之前（2022），它是唯一可离线部署、可下载权重、真正开源的端到端 ASR 引擎
- **状态**：2025 年 6 月正式归档 discontinued。继承者 Coqui STT 已在 2024 年停止维护

- **开源**：✅ 完整开源，MPL-2.0，GitHub `mozilla/DeepSpeech`

---

## 2020 — wav2vec 2.0 · Meta

### 前置条件：ASR 被"标注数据量"卡住了脖子

2015-2020 年间，所有端到端 ASR 都靠大量标注数据推动。做一个新语言的 ASR 系统，最少需要 1,000 小时人工标注音频。地球上绝大多数语言根本没有这个量级的标注数据。结果：ASR 只能覆盖约 100 种主要语言，剩下数千种语言/方言没有 ASR。

### wav2vec 2.0

**核心想法**：能不能让模型从**无标注音频**上先学会"语音的一般特征"，再用极少量标注数据微调？

- **架构**：CNN Encoder → Transformer Context Network → **对比学习目标**
- **训练**：无标注音频，随机 mask 部分帧，模型从可见帧中预测被 mask 帧的潜表征（对比学习：区分正负样本）
- **效果**：预训练后只需 **10 分钟标注数据微调**就能达到之前 1,000 小时训练的效果
- **改进点**：**范式级别的转变**——解决了 ASR 的"标注瓶颈"，低资源语言 ASR 第一次变得现实
- **局限**：对比学习依赖负样本选取策略；编码器学到了太多对 ASR 无用的声学细节

- **开源**：✅ MIT License，HuggingFace `facebook/wav2vec2-base/large/large-lv60`

---

## 2020 — Conformer · Google

### 前置条件：纯 Self-Attention 做 ASR 编码器不够

2018 年 Speech-Transformer 把 Transformer 引入 ASR，但很快发现一个问题：ASR 编码器需要同时做好两件事——**局部建模**（相邻音素的过渡细节，卷积擅长）和**全局建模**（整句结构/说话人习惯，Self-Attention 擅长）。纯 Transformer 全局有余、局部不足。

### Conformer

每个 Encoder Block 内并联两条路：
- **Self-Attention** → 全局关系
- **Convolution Module**（Pointwise Conv + GLU + Depthwise Conv + BN）→ 局部模式

然后用 **Macaron 结构**（FFN → Self-Attn+Conv → FFN）把两者粘在一起。

- **影响**：**ASR 编码器的事实标准**。此后的所有现代 ASR 模型（Whisper、SenseVoice、GLM-ASR、Qwen2-Audio、Qwen3-ASR 的 AuT）编码器都是 Conformer 或其变体
- **开源**：⚠️ Google 未发布原生权重，但 Espnet / NeMo 提供官方实现代码

---

## 2021 — HuBERT · Meta

### 前置条件：wav2vec 2.0 好，但不稳定

wav2vec 2.0 有两个实际痛点：
1. **对比学习不稳定**——负样本的选取质量直接影响训练效果。太难或太容易都不行
2. **编码器学了很多对 ASR 无用的东西**——对比学习只是"区分帧表征"，结果说话人音色、设备特征都被学进去了

### HuBERT

**思路**：把"区分"换成"预测"。

1. 对音频帧做 **k-means 聚类**，得到离散"伪标签"
2. 让模型做 **掩码预测**——mask 一部分帧，预测其簇归属（和 BERT 的 Masked LM 一致）

为什么更好？
- 分类目标（cross-entropy）比对比学习稳定——没有负样本选取的麻烦
- **离散聚类天然过滤了无用的声学细节**——说话人/背景/设备差异被聚类抹平
- **迭代式聚类**：第一轮用 MFCC 聚类 → 训 HuBERT → 用深层表征做第二轮聚类 → 再训，精度持续提升

- **效果**：LibriSpeech 全面超越 wav2vec 2.0，训练更稳定收敛更快
- **开源**：✅ MIT License，HuggingFace `facebook/hubert-base/large/xlarge`

---

## 2021 — WeNet

### 前置条件：流式和离线必须二选一

生产环境面临两难：
- **流式方案**（CTC/RNN-T）：延迟低，能实时出结果，但精度不够（缺未来上下文）
- **离线方案**（LAS/Transformer）：精度高，但不能流式

大多数公司只能部署两套模型——流式的做实时显示、离线的做精度修正。维护两套训练/部署/监控管线成本很高。

### WeNet

**U2 统一框架**：训练时同时优化 CTC + Attention 两个目标。推理时：
- CTC 做**流式 prefix beam search** → 实时出结果
- Attention Decoder 对候选路径做**二次 rescoring** → 精度接近完全离线

**U2++** 进一步把前帧（past）和后帧（future）上下文分开建模，流式精度与离线差距缩小到 1% 以内。

- **配套**：C++ runtime + WebSocket 服务端 + 模型导出管线，完整生产级方案
- **落地**：中文社区部署最广的开源 ASR 之一（腾讯、美团、字节）
- **开源**：✅ Apache 2.0，GitHub Release 提供中英文预训练模型

---

## 2022 — Whisper · OpenAI

### 前置条件：零样本 ASR 还没人做到

2022 年的 ASR 有两件事没解决：
1. 每个新语言/场景仍然需要 fine-tune。全球绝大部分语言没有几千小时的标注数据来做这件事
2. 最强模型（Conformer 系列）针对的是 LibriSpeech 的干净朗读数据，到真实场景骤降

### Whisper

**赌注——"标注质量不如数据量"**：

从互联网爬取 **68 万小时**（large-v2）已有字幕的 YouTube 音频，不做数据清洗。错词、噪声、语言混合的原始网络数据直接拿去训练。

**多任务输出格式**（一个模型做四件事）：

```
<|startoftranscript|> <|lang|> <|transcribe|> <|timestamps|> text <|endoftranscript|>
```

ASR + 翻译 + 语种识别 + 时间戳，全部在一个模型内完成。

**六个规格覆盖全场景**：

| 规格 | 参数 | 适用场景 |
|------|------|---------|
| tiny | 39M | 端侧/嵌入式 |
| base | 74M | 手机端 |
| small | 244M | 边缘设备 |
| medium | 769M | 低算力服务器 |
| large-v2 | 1.55B | 云端最高精度 |
| large-v3 | 1.55B | 128bins Mel + 500万小时数据 |
| turbo | ~800M | Decoder 32→4 层，速度优先 |

**large-v3 增量变化**（2023）：
- 训练数据：68 万 → **500 万小时**（含 large-v2 生成的 4M 伪标注）
- 输入 Mel：80 bins → **128 bins**（高频细节更丰富，噪声场景 +12% 精度）
- 新增粤语 token：vocab 51,865 → 51,866
- WER 再降 10-20%

**影响**：ASR 变成了 `pip install openai-whisper` 就能做的事。零样本 99 种语言。**开源 ASR 影响力最大的模型，没有之一。**

**留下的麻烦——幻觉**：弱监督数据的粗糙导致模型在静音段编造文本。large-v3 数据越大幻觉越明显，OpenAI 自己的 API 至今主要用 large-v2。

- **开源**：✅ MIT License，HuggingFace `openai/whisper-tiny` ~ `-large-v3`

---

## 2022 — FunASR / Paraformer · 阿里达摩院

### 前置条件：自回归推理太慢了

Whisper 很好用，但两个问题：(1) **自回归太慢**——Whisper 逐 token 解码在高并发下撑不住；(2) **只有模型、没有管线**——生产环境需要 VAD、标点、热词、说话人日志整条链路，Whisper 只做"波形→文本"。

### Paraformer 的核心创新——CIF

非自回归 ASR 的主要障碍是没有一个好的可微分的"对齐"机制。CTC 做了帧间独立假设（精度吃亏），RNN-T 的 Joint Network 很难训。

**CIF（Continuous Integrate-and-Fire）**：在编码器输出上持续积分声学分数，当累计分数超过阈值时"激发"输出一个 token。本质——用可微分的累加器替代了 CTC 的帧间独立预测。CIF + 非自回归解码器 = 推理速度比自回归 **快 5-10 倍**、精度接近自回归。

### FunASR 是全链路工具包

Paraformer 只是可选后端之一。完整管线：

```
VAD → 离线/流式解码 → 句级纠错 → 标点恢复 → 说话人日志 → 热词定制 → ONNX/C++ runtime
```

中文社区最活跃的开源 ASR 项目，社区模型下载量百万级。

- **开源**：✅ Apache 2.0，ModelScope / HuggingFace 提供全系列预训练模型

---

## 2024 — SenseVoice · 阿里通义实验室（FunAudioLLM）

### 前置条件：ASR 不转文字，还错过了大量信息

2024 年几乎所有 ASR 模型的共同目标都是把语音转成文字。情感和背景音被视为"需要过滤掉的噪声"。但真实场景中——"BGM 响起时说话"、"用户笑声"、"咖啡厅背景"——这些信息和文字本身一样重要。

另一个问题：推理延迟。Whisper-Large 在 GPU 上处理 10 秒音频约需 1.3 秒。高并发撑不住，端侧更不可能。

### SenseVoice——砍掉 Decoder

**Encoder-only + 单 CTC 头**。70 层 SANM（Self-Attention Network with Memory，包含 FSMN depthwise conv kernel=11，比 Conformer 更轻量），隐藏维度 512，4 头注意力，234M 参数。

**多任务靠 4 个 Query Embedding**：

输入音频前 concat 4 个 learnable query embedding。CTC 输出格式：

```
<LID_token> <EMO_token> <EVENT_token> <ITN_token> transcription_text
```

LID（语种识别）、SER（情感：HAPPY/SAD/ANGRY）、AED（背景事件：BGM/掌声/哭声）、ITN（文本归一化标记）。**一模型出四件事，当前开源唯一。**

**延迟对比**（10 秒音频）：

| 模型 | 架构 | 参数 | 延迟 |
|------|------|------|------|
| SenseVoice-Small | Encoder-only + CTC | 234M | **70ms** |
| Whisper-Small | Enc-Dec | 244M | 518ms |
| Whisper-Large-V3 | Enc-Dec | 1.55B | 1.28s |

比 Whisper-Large-V3 快 15 倍。同时发布 CosyVoice（语音生成），两者组成完整语音交互框架。

- **局限**：纯 CTC 精度天花板低于自回归；输出格式固定（`<lang><emo><event>text`）；长文本 CTC 对齐退化
- **开源**：✅ Apache 2.0，HuggingFace `FunAudioLLM/SenseVoiceSmall`

---

## 2024 — GLM-ASR-Nano · 智谱 AI

### 前置条件：能不能让 LLM 做 ASR 解码器？

2024 年 LLM 的能力已经很成熟。自然想法：LLM 比传统 Transformer Decoder 多了大量预训练语义知识——做语音识别时，听到模糊音可以根据上下文做出更合理的判断。但 LLM 自回归推理更重，怎么在 ASR 上跑起来？

### 关键设计——非对称 Encoder-Decoder

- **Encoder**：12 层（Nano）/ 32 层（Cloud）**双向** Conformer——可以同时看过去和未来的帧，对消歧义至关重要
- **Decoder**：6 层 LLaMA（Nano）/ 28 层（Cloud）**因果注意力**——只能看过去 token。ASR 解码是自左向右的，因果注意力天生适配

### 工程优化组合拳——让 LLM Decoder 在 ASR 上实时运行

**1. 4× Pooling Projector**

Encoder 输出 500 帧（10s 音频）→ 池化为 125 个 token。Decoder 自注意力 O(500²) → O(125²)，16 倍差距。这一刀决定了 LLM 做 ASR 解码有没有工程可行性。

**2. GQA（16 Query / 4 KV）**

KV Cache 压缩到普通 MHA 的 25%。Cloud 版 28 层 Decoder 没有 GQA 的话 KV Cache 太大，GPU 放不下。

**3. 部分 RoPE（rotary factor = 0.5）**

50% 的 head_dim 加位置编码，50% 留给纯语义容量。ASR 对绝对位置的依赖不如 NLP 强，这一刀砍得精准。

**4. 跨层 Cross-Attention**

每个 Decoder 层都通过 Cross-Attention 直接访问 Encoder 的声学特征，不是只在入口做一次跨模态投影。

**中文 WER 4.10%**（Wenet Meeting / Aishell-1 基准）vs Whisper V3 的 6.93%。这是第一个在中文场景上明确超越 Whisper 的开源 ASR 模型。方言（粤语）、噪声、耳语场景都做了针对性优化。

- **配套**：智谱 AI 输入法（小凹）——语音转文字 + 智能改写/翻译/Vibe Coding
- **开源**：✅ Apache 2.0，HuggingFace `zai-org/GLM-ASR-Nano-2512`，GitHub `zai-org/GLM-ASR`

---

## 2024 — Qwen2-Audio · 阿里通义

### 前置条件：ASR 的两条路线

2024 年 ASR 分化为两条路线：
- **纯 ASR 路线**（GLM-ASR、Whisper）——LLM 只做解码，目标是降 WER
- **音频 LLM 路线**（GPT-4o）——ASR 被吸收为 LLM 的子能力，目标是"理解音频而不只是转文字"

Qwen2-Audio 选择走音频 LLM 路线，而且是**开源的**。

### 架构

**Whisper-large-v2 Encoder + Qwen-7B LLM**

与 GLM-ASR 的关键区别：GLM-ASR 的 LLM 只做解码（输入压缩后的音频 token，输出文本）。Qwen2-Audio 的 Qwen-7B 是完整 LLM，能做推理、对话、音频事件分析。

**两类交互模式**：
- **语音对话**：用户说话，LLM 直接理解音频含义并生成回复
- **音频分析**：用户问"这段录音的背景声是什么？说话人情绪怎样？"

ASR 精度：LibriSpeech test-clean 1.6%，AISHELL-2 Mic 3.0%，中文 Common Voice 6.9%（Whisper-V3 12.8%）。

- **局限**：7B 模型推理延迟高，部署硬件要求高；HuggingFace 版本转换有轻微 WER 退化
- **开源**：✅ HuggingFace `Qwen/Qwen2-Audio-7B`

---

## 2026 — Qwen3-ASR · 阿里通义

### 前置条件：前代模型留下的三个未解决问题

**问题 1：流式和离线为什么要两套模型？**

WeNet U2 做了统一框架，但推理时流式和离线仍然是不同解码模式。Qwen3-ASR 的 **动态 Flash Attention 窗口（1-8 秒可调）** 让单一模型在流式模式用短窗口（低延迟），高精度时自动切到长窗口（全上下文）。真正的一体化。

**问题 2：ASR 的方言覆盖为什么一直这么差？**

Whisper 覆盖 99 种语言，但方言几乎不专门优化（粤语 WER ~10.9%）。Qwen3-ASR 支持 **52 种语种与方言（含 22 种中文方言）**——开源 ASR 方言覆盖最广。

**问题 3：ASR 能识别唱歌吗？**

所有 ASR 训练数据都是说话语音，没人做过。Qwen3-ASR 做到了 **整歌识别错误率 <8%**，独此一家。

### 附加创新

**Contextual Biasing**：给任意格式背景文本（关键词列表、整份文档），模型在推理时自动引导识别结果——不需要额外部署热词增强模块。

**GSPO（Group Sequence Policy Optimization）**：把强化学习引入 ASR 精调，约 5 万条语音做对齐。ASR 首次公开应用 RL。

### 架构

- **AuT Encoder**：Audio Transformer，32 层 Self-Attn + 3 层 Conv2D，8× 下采样（100Hz → 12.5Hz）
- **Projector**：桥接音频编码器与 LLM 嵌入空间
- **Qwen3 LLM Decoder**：8 层，Cross-Attn + Self-Attn
- **参数量**：1.7B / 0.6B

**四阶段训练**：
1. AuT 预训练：4,000 万小时伪标注 ASR 数据（中英文为主）
2. Omni 多任务预训练：3 万亿 token（音频/视觉/文本），训练 Qwen3-Omni 基座
3. ASR SFT：多语言、流式增强、上下文偏置数据
4. GSPO RL：5 万条语音精调

- **注意**：Qwen3-ASR-Flash（2025.09）是 API only 不开源。开源的是 2026.01 的 1.7B / 0.6B
- **开源**：✅ Apache 2.0，HuggingFace `Qwen/Qwen3-ASR-1.7B` / `0.6B`

---

## 全览：一张图

```
2014 ── DeepSpeech 1 ───────────── "端到端 ASR 可行了"
2015 ── DeepSpeech 2 ───────────── "中英文都行，且达商用水平"
2017 ── Mozilla DeepSpeech ─────── "人人都能下载部署了"
2020 ── wav2vec 2.0 ────────────── "10 分钟标注就够了"
2020 ── Conformer ──────────────── "CNN+Attention 混合才是编码器正解"
2021 ── HuBERT ─────────────────── "聚类伪标签比对比学习更稳"
2021 ── WeNet ──────────────────── "流式离线一个模型搞定"
2022 ── Whisper ────────────────── "零样本 99 语种，pip install 即用"
2022 ── FunASR/Paraformer ──────── "非自回归快 10 倍 + 全链路工具包"
2024 ── SenseVoice ─────────────── "70ms 极速 + 情感/事件多任务"
2024 ── GLM-ASR-Nano ───────────── "LLM 做 ASR 解码器 + 中文超 Whisper"
2024 ── Qwen2-Audio ────────────── "ASR 被 LLM 吸收为音频理解子能力"
2026 ── Qwen3-ASR ──────────────── "52 方言 + 歌声识别 + 动态流式窗口"
```

---

## 附录：各模型开源状态速查

| 模型 | 年份 | 代码许可 | 权重可下载 | 快速获取 |
|------|------|---------|-----------|---------|
| DeepSpeech 2 | 2015 | Apache 2.0 | ⚠️ 绕道 Wolfram | PaddlePaddle 仓库 |
| Mozilla DeepSpeech | 2017 | MPL-2.0 | ✅ | GitHub Release |
| wav2vec 2.0 | 2020 | MIT | ✅ | HuggingFace |
| Conformer | 2020 | Apache 2.0 | ⚠️ 无原生权重 | Espnet / NeMo |
| HuBERT | 2021 | MIT | ✅ | HuggingFace |
| WeNet | 2021 | Apache 2.0 | ✅ | GitHub Release |
| Whisper | 2022 | MIT | ✅ | HuggingFace / pip |
| FunASR/Paraformer | 2022 | Apache 2.0 | ✅ | ModelScope / HuggingFace |
| SenseVoice | 2024 | Apache 2.0 | ✅ | HuggingFace |
| GLM-ASR-Nano | 2024 | Apache 2.0 | ✅ | HuggingFace / GitHub |
| Qwen2-Audio | 2024 | 开源 | ✅ | HuggingFace |
| Qwen3-ASR | 2026 | Apache 2.0 | ✅ | HuggingFace |

> **注意**：标注为 ⚠️ 或 ❌ 的表示权重获取不直接或不可用，代码仍然可用。
