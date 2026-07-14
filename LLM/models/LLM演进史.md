# LLM 演进史（2017-2026）

> 从 Transformer 到万亿参数 MoE，大语言模型九年的进化之路

---

## 写在前面

这份演进史追踪的是**大语言模型（LLM）**的发展脉络。和 ASR 演进史的写法一致——分阶段、每个模型交代"前置条件"和"核心改进"。

整条时间线可以浓缩为五个阶段、五个节点：

```
2017 ── Transformer ────────── "所有 LLM 的技术底座"
2020 ── GPT-3 ──────────────── "越大越强——Scaling Law 验证"
2022 ── ChatGPT + RLHF ────── "AI 变成全民产品"
2023 ── LLaMA 开源浪潮 ────── "创新权力不再只在巨头手里"
2024-25 ── o1 / DeepSeek-R1 ── "从拼大到拼会思考"
```

---

## 第一阶段：底座搭建（2017-2019）——"能不能做大"

### 前置条件

在 Transformer 之前，NLP 的主流架构是 RNN/LSTM。它的问题：
- **串行计算**——每步依赖前一步的隐状态，无法并行，训练慢
- **梯度问题**——长序列下梯度消失/爆炸严重，即使 LSTM 也只能缓解到数百步
- **长程依赖弱**——"文档开头的某个实体"和"文档结尾的指代"之间的关系很难捕捉

Attention 机制（Bahdanau 2015）在机器翻译中证明了"让解码器在每一步动态选择要看源端的哪些位置"确实有效，但当时 Attention 只是 RNN 的附属模块。

---

### 2017.06 — Transformer（Google）· 论文开源

**"Attention Is All You Need"——不要 RNN，只要注意力就够了。**

| 维度 | 说明 |
|------|------|
| 架构 | **Encoder-Decoder，纯 Self-Attention**（无 RNN/CNN） |
| 核心组件 | 多头注意力 + 残差连接 + LayerNorm + 位置编码（正弦/余弦） |
| 核心改进 | **自注意力替代 RNN，并行计算 + 长程依赖一次解决**。计算复杂度 O(n²·d)，但在 GPU 上可以完全并行。n=512 的序列 Transformer 训练速度比同等 RNN 快数十倍 |
| 参数量 | Base 65M / Big 213M |
| 影响 | **所有后续 LLM 的技术骨架**。GPT 用了它的 Decoder（因果注意力），BERT 用了它的 Encoder（双向注意力），T5 用了完整的 Encoder-Decoder |

---

### 2018.06 — GPT-1（OpenAI）· 开源

| 维度 | 说明 |
|------|------|
| 架构 | 12 层 Transformer **Decoder**（因果注意力，只能看左侧 token） |
| 参数量 | **117M** |
| 核心改进 | **生成式预训练路线的先驱**。用大量无标注文本做语言模型预训练（预测下一个词），再在有标注数据上微调。证明了"无监督预训练 → 有监督微调"这个范式可行。在 9 个 NLP 任务上超越了当时使用手工特征的 SOTA |
| 与 BERT 的路线差异 | **单向（从左到右）→ 适合生成 / 双向（上下文都看）→ 适合理解**。这个路线分歧至今存在 |

---

### 2018.10 — BERT（Google）· 开源

| 维度 | 说明 |
|------|------|
| 架构 | **双向** Transformer **Encoder** |
| 参数量 | Base 110M / Large 340M |
| 核心改进 | **Masked Language Model（随机遮住 15% 的词，让模型预测被遮住的词）+ Next Sentence Prediction**。双向上下文让模型"读完整个句子再做判断"。在 11 项 NLP 任务上全面碾压此前 SOTA。BERT 统治了 2018-2022 年间的 NLP 理解类任务（分类、标注、问答、搜索排序） |
| 局限 | BERT 是编码器，不能生成文本。它做的 NLU，不是 NLG |

---

### 2019.02 — GPT-2（OpenAI）· ⚠️ 延迟发布

| 维度 | 说明 |
|------|------|
| 架构 | 48 层 Transformer Decoder |
| 参数量 | **1.5B**（比 GPT-1 大 13 倍） |
| 核心改进 | **零样本能力涌现**。GPT-2 在**没有微调**的情况下能完成翻译、摘要、问答——只是给一个 prompt，模型自动完成任务。这个发现让行业意识到：**"规模增大会带来能力的质的飞跃"** |
| 争议 | OpenAI 因担心滥用（生成假新闻）延迟发布 9 个月，引发了对开源与安全的大讨论 |

---

### 2019.10 — T5（Google）· 开源

| 维度 | 说明 |
|------|------|
| 架构 | Encoder-Decoder（Transformer 完整结构） |
| 参数量 | 11B（当时最大的纯 NLP 模型） |
| 核心改进 | **"Text-to-Text"统一框架**。所有任务——翻译、分类、摘要、QA——全部表示为"输入文本 → 输出文本"。翻译："translate English to German: That is good" → "Das ist gut"。极大简化了模型设计和推理 |
| 意义 | T5 的"统一格式"思路后来被 GPT-3 的 in-context learning 继承并发扬光大 |

---

## 第二阶段：规模化爆发（2020-2022）——"能不能听懂人类"

### 前置条件

GPT-2 展示了"更大 = 更强"的趋势，但行业还没意识到"继续放大"会带来**质的**飞跃。当时的主流认知还是"模型要针对每个任务精调"。APIs 和产品的时代还没到来。

---

### 2020.05 — GPT-3（OpenAI）· ❌ API only

| 维度 | 说明 |
|------|------|
| 架构 | 96 层 Transformer Decoder |
| 参数量 | **175B**（比 GPT-2 大 116 倍） |
| 核心改进 | **Scaling Law 的验证里程碑**。GPT-3 带来的不是架构创新，而是规模创新的非线性回报——参数量增大 100 倍后，**上下文学习（In-context Learning）** 能力涌现：给 prompt 里写几个例子就能做任务，不需要更新任何参数。这彻底改变了"使用模型"的方式：从"下载权重微调"变成了"写 prompt 调 API" |
| **Scaling Law 的含义** | Kaplan et al. (2020) 的论文《Scaling Laws for Neural Language Models》发现：模型性能与参数量、数据量、计算量呈**幂律关系**——增加 10 倍计算量，loss 降低一个固定比例。这意味着"无限堆参数就有无限提升"（至少到某个上限之前）。这个结论直接引发了 GPT-4 和全球大模型的军备竞赛 |
| 局限 | 训练成本极高（估计 $12M），仅通过 API 提供，开源社区无法复现。这为后来的开源运动埋下了伏笔 |

---

### 2022.01 — InstructGPT（OpenAI）· ❌

| 维度 | 说明 |
|------|------|
| 核心改进 | **RLHF（Reinforcement Learning from Human Feedback）**——三阶段训练：1️⃣ **SFT**：用人工写的高质量"指令→回复"数据微调 GPT-3 2️⃣ **奖励模型**：让人类对模型的多组输出排序，训练一个奖励模型打分 3️⃣ **PPO 强化学习**：用奖励模型的分数作为 reward，PPO 算法优化策略模型。结果：**模型学会"按人类意图办事"**。1.3B 的 InstructGPT 在人类评估中优于 175B 的 GPT-3。 |
| 意义 | 这是 ChatGPT 的底层技术。没有 RLHF，模型再大也只是"高级汉字接龙" |

---

### 2022.11 — ChatGPT（GPT-3.5）· ❌

| 维度 | 说明 |
|------|------|
| 核心改进 | **对话界面 + RLHF = 全民产品**。2 个月 1 亿用户——史上增长最快的消费级应用。不是模型层面的突破（GPT-3.5 基于 GPT-3 的改进版），而是**产品层面**的突破：把 RLHF 对齐的模型包装成一个好用的聊天界面，人人都能对话 |
| 影响 | AI 从"实验室工具"变成"大众消费品"。全球所有科技公司从此启动了 LLM 战略 |

---

## 第三阶段：开源浪潮 + 多元竞争（2023）——"谁能把能力扩散出去"

### 前置条件

ChatGPT 引爆了全球需求，但所有能力掌握在 OpenAI 一家手里。企业想做 LLM 应用只有一条路：调 OpenAI API，又贵又受制于单一供应商（"vendor lock-in"）。全球开发者想要"自己能控制、能部署、能修改"的模型。市场需要开源替代。

---

### 2023.02 — LLaMA（Meta）· ✅ 需申请权重

| 维度 | 说明 |
|------|------|
| 参数量 | 7B / 13B / 33B / 65B |
| 核心改进 | **高质量开源路线的引爆点**。核心洞察：**"小模型训更多 token"优于"大模型训少数据"**。LLaMA-13B 在多数 benchmark 上超越了 GPT-3（175B）。原因：LLaMA 用了 1.0T~1.4T token 训练，而 GPT-3 只用了 300B token。**更多的训练数据可以补偿模型容量的不足** |
| 架构选择 | 后续成为"LLM 最佳实践"的架构要素在 LLaMA 中初步定型：Pre-RMSNorm（替代 LayerNorm）、SwiGLU 激活函数、Rotary Position Embedding（RoPE） |
| 影响 | 权重泄露后全球社区大量微调（Alpaca、Vicuna、Koala 等）。**LLaMA 证明了"开源模型可以达到接近 GPT-3 的水平"**，彻底点燃了开源 LLM 运动 |

---

### 2023.03 — GPT-4（OpenAI）· ❌

| 维度 | 说明 |
|------|------|
| 核心改进 | **多模态（文本+图像输入）**，考试超越 90% 人类（Bar Exam 前 10%）。可以理解图片内容（图表、截图、照片）并做推理。模型能力质的飞跃。参数量和架构未公开 |

---

### 2023.03 — Claude 1（Anthropic）· ❌

| 维度 | 说明 |
|------|------|
| 核心改进 | **Constitutional AI（CAI）**——不需要人类标注偏好的 RLHF。原理：给定一套"宪法"规则（如"不要有害"），让模型自己生成偏好数据来训练奖励模型。同样三阶段（SFT + CAI 训练 + RL），但不依赖人工标注偏好。长上下文（当时最长） |

---

### 2023.07 — LLaMA 2（Meta）· ✅ 商用免费

| 维度 | 说明 |
|------|------|
| 参数量 | 7B / 13B / 70B |
| 核心改进 | **开源进入可商用时代**。许可证明确允许商业使用。训练数据 2T token，上下文扩展到 4K。附带 LLaMA 2-Chat（RLHF 对齐版）。HuggingFace 上数万微调变体。全球社区从此有了"免费且合法可用"的基础模型 |

---

### 2023.09 — Qwen（通义千问）（阿里）· ✅ Apache 2.0

| 维度 | 说明 |
|------|------|
| 参数量 | 7B / 14B / 72B |
| 核心改进 | **国内最早全面开源的大模型之一**。架构选择：GQA（Grouped Query Attention）、SwiGLU、RMSNorm、RoPE——和 LLaMA 类似的"最佳实践"，但针对中文做了分词优化。从一开始就走**开源 + 中文 + 多模态 + Agent** 的完整路线。72B 版本在当时是中文模型中最强的开源选择 |

---

### 2023.10 — ChatGLM-3（智谱 AI）· ✅ Apache 2.0

| 维度 | 说明 |
|------|------|
| 核心改进 | **国内最早持续开源的 LLM 系列之一**。从 2021 年的 GLM-10B 开始就已开源。GLM（General Language Model）采用 Encoder-Decoder 混合架构——不同于 GPT 的纯 Decoder，也不完全是 T5 的 Encoder-Decoder，而是用 Autoregressive Blank Infilling 做训练目标。130B 版本是中国第一个达到千亿参数级别的开源模型 |

---

### 2023.12 — Gemini 1.0（Google）· ❌

| 维度 | 说明 |
|------|------|
| 核心改进 | **原生多模态（文本+图像+音频+视频）**——从一开始就设计为多模态，不是后期加功能。三层规格：Nano / Pro / Ultra。Google 自 BERT 之后在 LLM 领域最重要的产品发布。训练使用了 TPUv5 超大规模集群 |

---

### 2023.12 — Mixtral 8x7B（Mistral AI）· ✅ Apache 2.0

| 维度 | 说明 |
|------|------|
| 参数量 | 46.7B MoE（12.9B active per token） |
| 核心改进 | **"小参数 MoE"路线的标杆**。8 个专家 × 7B，每 token 激活 2 个专家。性能超过 LLaMA 2 70B 和 GPT-3.5，推理成本不到它们的 1/5。Sliding Window Attention（注意力窗口 = 4096）。**证明了 MoE 是"花小钱办大事"的有效路线**，为 2024 年 DeepSeek-V2 的极致 MoE 工程铺平了道路 |

---

## 第四阶段：MoE + 推理革命（2024）——"谁更会思考"

### 前置条件

2023 年的竞争焦点是"模型多大、能做什么"。到了 2024 年，大家发现"一味堆参数"收益递减——LLaMA 2 70B 和 GPT-3.5 差距在缩小，但需要的计算量仍在线性增长。两个新方向出现：
1. **MoE（混合专家）**——稀疏激活，用更少的计算量达到同等的效果
2. **推理模型**——不直接给答案，先在内部"思考"再输出

---

### 2024.02 — Claude 3（Anthropic）· ❌

| 维度 | 说明 |
|------|------|
| 规格 | **Opus / Sonnet / Haiku 三级梯队** |
| 核心改进 | Opus 在多个 benchmark 上超越 GPT-4。200K 上下文。Vision 能力（理解图像/文档）。Sonnet 性价比突出，成为编程场景的首选之一。三级规格的设计后来被多家厂商效仿 |

---

### 2024.04 — LLaMA 3（Meta）· ✅ 开源

| 维度 | 说明 |
|------|------|
| 参数量 | 8B / 70B |
| 核心改进 | 训练数据从 LLaMA 2 的 2T 升级到 **15T+ token**。大幅提升数据质量和过滤（去重、去低质量）。使用 GQA（8B 用 8 KV heads / 8 queries，70B 用 8 KV heads / 64 queries）。Tokenizer 扩展到 128K vocab。开源质量首次**接近**闭源前沿 |

---

### 2024.05 — GPT-4o（OpenAI）· ❌

| 维度 | 说明 |
|------|------|
| 核心改进 | **原生多模态——文本+语音+图像实时交互**。延迟减半、价格减半。不只是"支持图像输入"，而是真正的"跨模态融合"——模型同时理解语音语调、图像和文字，生成带情感的语音回复。GPT-4o 的"o"代表"omni"（全向）。后续发布的 GPT-4o mini 以极低的价格（推理成本降低 90%+）成为最广泛使用的 API 模型之一 |

---

### 2024.05 — DeepSeek-V2（深度求索）· ✅ MIT

| 维度 | 说明 |
|------|------|
| 参数量 | 236B MoE（21B active per token） |
| 核心改进 | **中国在效率上开始领先的标志**。两个关键架构创新：**1️⃣ MLA（Multi-head Latent Attention）**——把 Key 和 Value 映射到低维潜空间再解压，KV Cache 压缩约 75%，推理时显存需求大幅降低。**2️⃣ MoE 极致工程**——每 token 只激活 21B 参数（总 236B 的 9%）。效果接近 LLaMA 3 70B，但推理成本不到 1/3。训练成本也显著低于同期模型 |
| 意义 | 这条"低成本+高性能"路线在 2024 年底被 DeepSeek-V3 推到了极致 |

---

### 2024.07 — LLaMA 3.1 405B（Meta）· ✅ 开源

| 维度 | 说明 |
|------|------|
| 参数量 | **405B（Dense，非 MoE）** |
| 核心改进 | **首个在各项指标上匹配 GPT-4 的开源模型**。128K 上下文。这是 Meta 选择做的"大赌注"——当大家都在转向 MoE 时，Meta 选择了 Dense 架构，证明在足够大的规模下 Dense 也能达到 GPT-4 水平。**开源/闭源的差距在此基本抹平** |

---

### 2024.09 — o1（OpenAI）· ❌

| 维度 | 说明 |
|------|------|
| 核心改进 | **推理模型新范式——"慢思考"**。o1 的独特性在于它**不要立刻回答**。模型在生成最终答案之前先产生一条内部"思维链"（chain-of-thought），自我推导、自我纠错、分解复杂问题。训练方式是通过强化学习让模型学会"思考"——不是简单的 prompt 技巧（"let's think step by step"），而是内化在模型权重里的推理能力。在数学（AIME）、编程（Codeforces）、科学（GPQA）上达到 SOTA，远超 GPT-4 |
| 影响 | 行业的重心从"做大"正式切换到"做会思考" |

---

### 2024.12 — DeepSeek-V3（深度求索）· ✅ MIT

| 维度 | 说明 |
|------|------|
| 参数量 | **671B MoE（37B active per token）** |
| 核心改进 | **训练成本仅 $5.6M——同期 GPT-4 级别模型 > $100M。全球震动。** 关键技术：MoE 继续优化 + **FP8 混合精度训练**（首次在大规模 MoE 上成功应用）+ **Multi-Token Prediction**（一次预测多个未来 token）。性能接近 GPT-4，但训练成本只有同级别的 1/20。**彻底改写了"高性能 = 高成本"的公式** |

---

## 第五阶段：开源推理 + Agent 化（2025-2026）——"谁能真正干活"

### 前置条件

2024 年两条主线并行：
- **推理模型**——o1 证明了"慢思考"范式，但它是闭源的。DeepSeek-R1 要证明"开源也能做到"
- **Agent**——光会聊天不够了。模型要能调用工具、写代码、操作浏览器、自动完成任务分解
- **中国模型全面崛起**——从"追赶者"变成"规则重构者"。全球 Token 消耗中国占比达 30%

---

### 2025.01 — DeepSeek-R1（深度求索）· ✅ MIT

| 维度 | 说明 |
|------|------|
| 架构 | 671B MoE（基于 V3），但增加推理训练 |
| 核心改进 | **开源推理模型匹配 o1，成本低 20-50 倍**。核心技术：**GRPO（Group Relative Policy Optimization）**——不需要 Critic 模型的强化学习。传统 PPO 需要两个网络（Actor + Critic），GRPO 只用一组输出的优势值做策略优化。大幅简化了 RL 训练。另一个技术：**DeepSeek-R1-Zero——没有 SFT 数据，纯 RL 从零训出推理能力**。这对行业的核心含义是：推理能力不需要人类数据，纯 RL 就能涌现 |
| 影响 | **DeepSeek 应用登顶美区 App Store 榜首**——第一次有中国 AI 应用在美区超越 ChatGPT。全球科技股震荡。全球重新评估"闭源高价路线"的合理性。大量企业开始部署开源推理模型 |

---

### 2025.02 — Moonlight（月之暗面/Kimi）· ✅ MIT

| 维度 | 说明 |
|------|------|
| 参数量 | 16B MoE（活跃 2.24B / 总 15.29B） |
| 核心改进 | **Muon 优化器——替代 AdamW，计算效率翻倍**。论文《Muon is Scalable for LLM Training》。核心思路：在优化器中引入牛顿-舒尔茨迭代做动量正交化，每次更新时让参数沿曲率较低的方向走更远。用 5.7T token 验证了 Muon 在大规模 LLM 训练上的可行性。**这是少有的底层优化器级别的创新** |

---

### 2025.03 — Qwen 2.5 / QwQ-32B（阿里）· ✅ Apache 2.0

| 维度 | 说明 |
|------|------|
| 参数量 | 32B / 72B / 110B / 236B MoE |
| 核心改进 | **QwQ-32B 以极小参数实现了接近 R1 的推理能力**。通过强化 RL 训练让 32B 的模型学会"思考"，在多个推理 benchmark 上匹配甚至超越 DeepSeek-R1。**证明推理能力也可以被"蒸馏"到小模型**。Qwen 2.5 系列覆盖了从 0.5B 到 236B 的全尺寸，成为全球最完整的开源模型家族之一 |

---

### 2025.04 — LLaMA 4（Meta）· ✅ 开源

| 维度 | 说明 |
|------|------|
| 参数量 | 402B~109B MoE（均 17B active per token） |
| 核心改进 | **MoE 架构——Meta 首次从 Dense 转向 MoE**。**128 专家**。Scout（109B 总/17B active, **10M 上下文**）和 Maverick（**402B** 总/17B active, 1M 上下文, 原生多模态）两种规格。开源旗舰继续推进 |

---

### 2025.04 — Kimi-VL / Kimi-VL-Thinking（月之暗面）· ✅ MIT

| 维度 | 说明 |
|------|------|
| 架构 | MoE（16B 总 / 2.8B active）|
| 核心改进 | **MoonViT 原生分辨率视觉编码器 + 长思维链 VLM**。不同于 CLIP 式 ViT 需要把图像缩放到固定分辨率，MoonViT 可以处理任意分辨率的图像输入，保留更多细节。**开源首个支持长链推理的视觉语言模型**。在 MMMU、MathVista 等多模态推理任务中刷新小模型记录 |

---

### 2025.05 — Claude Opus 4（Anthropic）· ❌

| 维度 | 说明 |
|------|------|
| 核心改进 | 编程/Agent/工具调用全面达新高度。SWE-bench 等编程基准领先。注册为"Level 3"风险级别（Anthropic 的内部安全评级体系） |

---

### 2025.07 — Kimi K2（月之暗面）· ✅ MIT

| 维度 | 说明 |
|------|------|
| 参数量 | **1.04T MoE（32B active per token）**——万亿参数开源模型 |
| 架构细节 | 61 层（1 dense + 60 MoE），384 专家，每 token 激活 8+1 共享专家，**MLA 注意力**（64 heads），SwiGLU，160K vocab |
| 核心改进 | **(1) MuonClip 优化器**——在 Muon 基础上增加 QK-Clip（注意力分数裁剪），实现**零损失尖峰**的万亿参数训练。(2) **极高稀疏度**——48:1（384 专家/8 active），是目前开源模型中稀疏度最高的。(3) **Agent 能力原生内置**——不像其他模型是"语言模型 + 后加工具调用"，K2 从训练数据开始就注入了大量 Agent 交互数据。SWE-bench 65.8%，多项 Agent 任务超越 GPT-4.1 |
| 论文 | 《Kimi K2: Open Agentic Intelligence》（arXiv: 2507.20534） |

---

### 2025.08 — GPT-5（OpenAI）· ❌

旗舰级统一多模态。快速/慢速思考自动路由、永久记忆。详细架构未公开。

---

### 2025.08 — GPT-OSS 120B（OpenAI）· ✅ 开源

| 维度 | 说明 |
|------|------|
| 参数量 | 120B |
| 核心改进 | **OpenAI 自 GPT-2（2019）以来首次开源权重**。闭源巨头开始拥抱开源。意义不在于性能（不如 GPT-5），而在于态度 |

---

### 2026.01 — GLM-5 / 智谱港股上市（智谱 AI）· ✅ Apache 2.0

| 维度 | 说明 |
|------|------|
| 意义 | **港股上市——全球首家上市的大模型公司**。GLM-5 系列（5.1/5.2）在开源智能指数上排名全球第一（51 分）。走 Agent 工程路线，支持复杂推理和长时间 Agent 任务。GLM-5.2 全面支持 1M 上下文。所有模型支持国产芯片（昇腾、摩尔线程、壁仞）|

---

### 2026.01 — Kimi K2.5（月之暗面）· ✅ MIT

| 维度 | 说明 |
|------|------|
| 参数量 | 1.04T MoE（32B active），256K 上下文 |
| 核心改进 | **(1) 早期视觉-文本融合**——训练数据 10% 视觉 + 90% 文本，图像和文本在模型浅层就开始融合，而非在最后拼接。(2) **Zero-Vision SFT**——只用纯文本 SFT 数据就激活了模型的视觉推理能力。(3) **Agent Swarm**——多 Agent 协作框架。**17 个视觉基准中拿 9 个第一**。BrowseComp 78.4% 超越 GPT-5.2 Pro |

---

### 2026.04 — Kimi K2.6（月之暗面）· ✅ MIT

K2.5 的进一步升级。**300 个并行子 Agent（Agent Swarm）**，262K 上下文，支持视频输入。单任务 4,000+ 步骤，跨 Python/Rust/Go 多语言。SWE-Bench Pro **58.6** 超越 GPT-5.4（57.7）。开源 MIT。

---

### 2026.04 / 2026.06 — DeepSeek-V4 / V4.1（深度求索）· ✅ MIT

**V4（2026.04）**：**完全在华为昇腾 910C 芯片上训练（1,000 张卡，零 NVIDIA 依赖）**。1M 上下文。Agent 路线继续深化。

**V4.1（2026.06 灰度）**：V4 基础上新增 **全模态（图像+音频输入）+ 原生 MCP 协议 + 企业级工具链**。标志 DeepSeek 从技术展示全面转向商业落地。同月启动 500 亿元融资。

---

### 2026.02 — Qwen3.5 Plus + Qwen3-Max-Thinking（阿里）· ✅ Apache 2.0

**Qwen3.5 Plus（2026.02）**：397B MoE（17B active），原生多模态（文本+图像+视频，最多 256 张图片），1M 上下文，推理吞吐提升 19 倍。登顶全球最强开源模型。

**Qwen3-Max-Thinking**：新一代推理机制。Qwen3 系列（1.7B / 32B / 235B）覆盖全尺寸。延续 Apache 2.0 开源路线。

---

### 2026.03 — Attention Residuals（月之暗面论文）· ✅ 论文+代码

| 维度 | 说明 |
|------|------|
| 核心改进 | **用注意力机制替代残差连接的固定"+"操作**——Transformer 中每层的残差连接一直是 `output = layer(input) + input`，这个"相加"是固定的。Attention Residuals 让每一层**能从前面的所有层中"选择性"获取信息**，而非只和上一层相加。Block AttnRes 的 Scaling Law 等效于**多花 25% 的算力**。在 Kimi Linear 48B 上测试，GPQA+7.5、Math+3.6、HumanEval+3.1。Elon Musk 点赞，Karpathy 参与讨论 |
| 意义 | 这是**改变 Transformer 底层设计**的论文。残差连接是 Transformer 自 2017 年以来的不可变组件之一，这是第一次有人系统性论证"固定相加"可以被取代 |

---

## 六大模型家族一览

```
GPT 家族（OpenAI）：
  2018 GPT-1 → 2019 GPT-2 → 2020 GPT-3 → 2023 GPT-4 → 2024 GPT-4o → o1 → 2025 GPT-5
  路线：闭源 → 2025 GPT-OSS 首次开源

LLaMA 家族（Meta）：
  2023 LLaMA → LLaMA 2 → 2024 LLaMA 3 → LLaMA 3.1 405B → 2025 LLaMA 4 MoE
  路线：完全开源，推动开源生态

Claude 家族（Anthropic）：
  2023 Claude 1 → 2024 Claude 3 → 2025 Claude Opus 4
  路线：闭源，安全对齐先行者

DeepSeek 家族（深度求索）：
  2024 DeepSeek-V2 → V3 → 2025 R1 → 2026 V4
  路线：完全开源 MIT，低成本高效率颠覆者

Kimi 家族（月之暗面）：
  2025 Moonlight → Kimi-VL → Kimi K2 → K2.5 → K2.6
  路线：完全开源 MIT，万亿参数先驱，底层创新（MuonClip、AttnRes）

Qwen 家族（阿里）：
  2023 Qwen → 2024 Qwen2 → 2025 Qwen 2.5 / QwQ → 2026 Qwen3 / Max-Thinking
  路线：完全开源 Apache 2.0，全球最完整开源模型家族

GLM 家族（智谱 AI）：
  2021 GLM-10B → 2023 ChatGLM → 2024 GLM-4 → 2025 GLM-5
  路线：完全开源 Apache 2.0，国内最早，Agent 工程先行
```

---

## 五个改变行业的关键节点

| 节点 | 时间 | 为什么关键 |
|------|------|-----------|
| **① Transformer** | 2017 | 所有现代 LLM 的技术底座。没有 Transformer 就没有后续一切 |
| **② GPT-3** | 2020 | 证明 Scaling Law——"越大越强"成为行业信仰。In-context learning 改变使用范式 |
| **③ ChatGPT + RLHF** | 2022 | AI 从实验室走向大众。2 个月 1 亿用户。大模型产品化元年 |
| **④ LLaMA 开源浪潮** | 2023 | 高质量开源模型释放了全球社区的创新能力。创新权力不再只在巨头手里 |
| **⑤ o1 / DeepSeek-R1** | 2024-2025 | 从"拼大"到"拼会思考"。推理模型 + 低成本颠覆了行业路线。证明了"小而精"可以胜过"大而全" |

---

## 总结：九年的范式演变

| 时期 | 核心问题 | 代表 | 驱动力量 |
|------|---------|------|---------|
| 2017-2020 | 能不能做大？ | GPT-3 175B | Scaling Law，算力堆叠 |
| 2022 | 能不能听懂人类？ | InstructGPT + RLHF | 对齐技术 |
| 2023 | 能不能扩散出去？ | LLaMA 开源浪潮 | 社区力量 |
| 2024 | 能不能更高效、更会思考？ | DeepSeek-V3 / o1 | MoE + 推理模型 |
| 2025-2026 | 能不能真正干活？ | Agent + 工具调用 | 工程落地能力 |

> **大模型竞争已从"参数军备竞赛"变成"系统工程竞赛"。** 赢家由谁的 Agent 最可靠、谁的工具调用最精准、谁的部署成本最低决定——而不只是谁的模型最大。

---

**Sources:**
- [Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — Transformer
- [GPT-1: Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers (arXiv:1810.04805)](https://arxiv.org/abs/1810.04805)
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3: Language Models are Few-Shot Learners (arXiv:2005.14165)](https://arxiv.org/abs/2005.14165)
- [Scaling Laws for Neural Language Models (arXiv:2001.08361)](https://arxiv.org/abs/2001.08361)
- [Training language models to follow instructions with human feedback (InstructGPT, arXiv:2203.02155)](https://arxiv.org/abs/2203.02155)
- [LLaMA: Open and Efficient Foundation Language Models (arXiv:2302.13971)](https://arxiv.org/abs/2302.13971)
- [GPT-4 Technical Report (arXiv:2303.08774)](https://arxiv.org/abs/2303.08774)
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models (arXiv:2307.09288)](https://arxiv.org/abs/2307.09288)
- [Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073)
- [Mixtral of Experts (arXiv:2401.04088)](https://arxiv.org/abs/2401.04088)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434)
- [The Llama 3 Herd of Models (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)
- [Qwen2.5 Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Kimi K2: Open Agentic Intelligence (arXiv:2507.20534)](https://arxiv.org/abs/2507.20534)
- [Kimi-VL Technical Report (arXiv:2504.07491)](https://arxiv.org/abs/2504.07491)
- [Muon is Scalable for LLM Training (Moonlight, arXiv:2502.09891)](https://arxiv.org/abs/2502.09891)
- [Attention Residuals (arXiv:2603.XXXXX)](https://arxiv.org/abs/2603.xxxxx)
- GLM-5 / 智谱上市公开报道
- Kimi K2.5 / DeepSeek-V4 / Qwen3-Max-Thinking 官方技术报告
