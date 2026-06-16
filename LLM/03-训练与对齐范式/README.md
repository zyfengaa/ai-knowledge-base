﻿﻿﻿# 03 — 训练与对齐范式

> 怎么让一个「背了 100TB 互联网文本」的模型学会「好好说话」而不是「复读数据」？

## 正文：渐进式理解

**第一层：问题定义。** GPT-3 展示了惊人的文本能力和 Scaling Law，但它的输出并不「好用」——模型倾向于延续输入风格、编造事实、拒绝指令。问题根源是：**预训练目标（next token prediction）和「使用目标（服从指令、说真话）」不是同一个东西。** 训练与对齐要解决的就是这个 gap。

**第二层：核心直觉。** 训练流程像三阶段教育：
1. **预训练（学前阅读）**：读海量文本，学语法/事实/世界知识——什么都不懂的裸模型
2. **SFT（行为规范）**：教模型「别人问问题你要回答，而不是继续往下编」
3. **RLHF/DPO（价值观养成）**：不只要回答，还要回答得好——有用、无害、诚实

**第三层：方案细节。**

**GPT-3 与 Scaling Law（2020）：** Brown et al. 训练了 175B 参数模型，发现三个规律：① 模型性能随参数量/数据量/计算量呈幂律增长（Scaling Law）；② 模型变大后自动涌现 In-Context Learning 能力（无需梯度更新的小样本学习）；③ 足够大的模型在 Zero-shot 下也显著优于小模型。**核心贡献**：证明了「更大就是更好」——LLM 时代的真正起点。

**InstructGPT / RLHF（2022）：** 三阶段管线：

`
① SFT：收集人工编写的指令-回答对，微调预训练模型
    ↓
② Reward Model：对上一步的输出做人工排序，训练一个评分模型
    ↓
③ PPO 优化：用 Reward Model 的评分作为奖励，进一步优化 SFT 模型
`

**DPO（2023）：** 发现不需要显式训练 Reward Model，直接用偏好对（好/坏回答）构造损失函数来优化。**核心区别**：RLHF = 3 模型（SFT + RM + Policy），DPO = 1 模型（直接用偏好数据优化）。

**第四层：不同方案的权衡。**

| 维度 | RLHF (InstructGPT) | DPO | KTO |
|------|-------------------|-----|-----|
| 模型数 | 4 个（Base + SFT + RM + PPO） | 1 个 | 1 个 |
| 训练稳定性 | ❌ PPO 超参数敏感 | ✅ 稳定 | ✅ 稳定 |
| 需要的数据 | 人工排序对（ranked pairs） | 偏好对（chosen/rejected） | 仅需二进制反馈（好/坏） |
| 对齐效果 | ✅ 最强 | ✅ 接近 RLHF | ⚠️ 略差 |
| 当前地位 | 质量天花板 | **实际首选** | 数据受限时替代 |

**第五层：总结升华。** 训练与对齐的本质是：**预训练学的是「语料中的统计分布」，对齐学的是「人希望模型怎么使用这个分布」**。Scaling Law 告诉我们越大越好，对齐告诉我们越大越需要对齐——这两个规律互为因果，构成了当前 LLM 发展的核心张力。

---

## 学习目标

读完你能：

- 画出 RLHF 的完整流程图：SFT → Reward Model → PPO 的三阶段和每阶段的输入/输出
- 用一句话说清 DPO 和 RLHF 的核心区别（DPO 去掉了 Reward Model，直接用偏好对优化）
- 解释 Scaling Law 的三个维度（参数 / 数据 / 计算量）和它们的幂律关系
- 面对实际对齐项目，决定用 RLHF 还是 DPO 并给出理由
- 理解为什么 GPT-3 的 In-Context Learning 是「涌现」而非「设计」——训练目标里没有这个目标但它自动出现了

---

## 精选论文

**Brown et al. (2020) "Language Models are Few-Shot Learners" (GPT-3) [[arXiv](https://arxiv.org/abs/2005.14165)]**

- **一句话定位**：175B 参数展示 Scaling Law + In-Context Learning，LLM 时代真正的开端
- **阅读重点**：第 3 节（Scaling Law 的实验验证——Figure 3.1 的幂律曲线是核心）和第 4 节（In-Context Learning 的涌现现象）
- **时间分配建议**：时间紧只读第 3 节了解 Scaling Law + 看 Table 3.2 的模型规格表；时间充裕精读第 4-5 节了解 Few-shot / Zero-shot 表现

**Ouyang et al. (2022) "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT) [[arXiv](https://arxiv.org/abs/2203.02155)]**

- **一句话定位**：RLHF 范式的确立论文，ChatGPT 的技术基础
- **阅读重点**：第 2 节（RLHF 三阶段方法——Figure 2 是整个管线的流程图）和第 3.1-3.3 节（PPO 优化细节）
- **时间分配建议**：PPO 的训练细节（Section 3.4+）可以跳读，理解「三个阶段分别做什么」比理解「PPO 的 KL 散度惩罚」更重要

**Rafailov et al. (2023) "Direct Preference Optimization" (DPO) [[arXiv](https://arxiv.org/abs/2305.18290)]**

- **一句话定位**：去掉 Reward Model，直接优化偏好，对齐方法的更优选
- **阅读重点**：第 2 节（DPO 的公式推导——为什么可以跳过 RM）和第 4 节（实验对比）
- **时间分配建议**：公式推导（从 Bradley-Terry 模型到 DPO 损失函数）可以跳读，核心是理解「去掉了 RM 这个中间层」

---

## 模块间连接

- **前置依赖**：建议先读 **01-Transformer 起源**（理解模型架构）和 **02-架构演进迭代**（理解现代架构配置）。本模块讨论的「训练什么」依赖对架构的理解
- **后续衔接**：读完本模块后推荐进入 **04-推理与部署优化**——训好的模型怎么部署
- **本模块与哪些模块正交**：与 05-应用技术（RAG / CoT）完全正交——训练怎么训和应用怎么用是两个独立问题


---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model | DPO () | [arXiv](https://arxiv.org/abs/2305.18290) |
| Language Models are Few-Shot Learners | GPT3 () | [arXiv](https://arxiv.org/abs/2005.14165) |
| Training language models to follow instructions with human feedback | InstructGPT () | [arXiv](https://arxiv.org/abs/2203.02155) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model | [arXiv](https://arxiv.org/abs/2305.18290) |
| Language Models are Few-Shot Learners | [arXiv](https://arxiv.org/abs/2005.14165) |
| Training language models to follow instructions with human feedback | [arXiv](https://arxiv.org/abs/2203.02155) |
