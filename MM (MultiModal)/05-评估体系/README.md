# 05 — 评估体系

## 一句话开场

> 一个 VLM 说它"会看图说话"，但具体有多好？它认识路牌上的字吗？能分清"猫在桌子下面"和"猫在桌子上面"的区别吗？知道一张图里有几个苹果吗？不同的 benchmark 测不同能力——如果你的 VLM 在 MMBench 上高分，不代表它在 OCR 场景下好用。

## 正文：渐进式理解（5 层）

**第一层：问题定义。** 评估解决的根本问题是"怎么系统性地测量 VLM 在各个能力维度的真实水平"。多模态评估比纯语言评估难得多，因为"理解图像"这件事本身就有 100 种以上不同能力需要分别测量——从简单的物体识别到复杂的空间推理、OCR、图表理解、数学推理、时序理解。

**第二层：核心直觉。** 评估 VLM 和给一个学生出考试题是一样的：单一科目的高分不意味着全能。MMBench 是"选择题考试"（60 道题覆盖 20 种能力），MMMU 是"大学专业考试"（涵盖 6 个学科的多选题），MMVP 是"专门挑刺考试"——专门找 VLM 最容易犯错的场景（空间关系、数量判断、属性绑定）。好的评估体系是多个考试的组合，而不是一个万能指标。

**第三层：方案细节。** VLM benchmark 的演化经历了三个阶段：① VQA 时代（2016-2022）——以 VQA 2.0 为代表，简单的图像问答，弊端是语言 bias（模型可以不看图仅靠问题分布和语言先验就答对）。② 综合 Benchmark 时代（2023-2024）——MMBench / MMMU / MMLMM 等覆盖 20+ 能力维度，引入对抗样本和人工校验。③ 低饱和 Benchmark 时代（2024-）——MMMU-Pro / MMLMM-Pro 在已有 benchmark 上增加 harder 样本，MMVP 聚焦 VLM 的典型失败模式。当前最大的问题：大部分 benchmark 的饱和速度很快（新模型快速满分），需要持续更新。

**第四层：不同评估维度的权衡。**

| 评估维度 | 代表Benchmark | 测什么 | 饱和速度 |
|---------|-------------|-------|---------|
| 通用图文理解 | MMBench (2023) | 20 种能力抽样 | 高（已饱和） |
| 学术/领域知识 | MMMU (2024) | 6 学科知识 | 中 |
| 细粒度诊断 | MMVP (2024) | 空间/数量/属性 | 低 |
| 幻觉 | POPE / MMHal-Bench | 物体是否存在等问题 | 中 |
| OCR/文档 | OCRBench / DocVQA | 文字识别+理解 | 中 |

**第五层：总结升华。** 评估是一个"一直在追赶、永远追不上"的领域——模型在 MMBench 上满分了，不代表它真正解决了视觉理解。两个根本性的问题还未解决：① benchmark contamination（模型在训练数据中见过测试样本），开源的 benchmark 尤甚；② 人类偏好评估成本高，自动化评估（如用 GPT-4V 做 judge）有偏。所以"不要只看一个 benchmark，要看多个 benchmark + 具体 case 的 error analysis"。

---

## 学习目标
- 列出 VLM 评估的 5 个主要能力维度及其对应 benchmark
- 用一句话说清 MMBench / MMMU / MMVP 各自测什么、为什么需要三个不同的
- 理解 benchmark contamination 问题，知道为什么"在 MMMU 上高分"不一定说明模型真强
- 能判断一个 VLM 在实际项目中是否好用（不依赖单个 benchmark 数字）
- 面对一个新的 VLM paper，知道应该看哪些 benchmark 和 error analysis 才能判断真正水平

---

## 精选论文

**Liu et al. (2023) 'MMBench: Is Your Multi-modal Model an All-around Player?'**
- 一句话定位：第一个系统化的 VLM benchmark，覆盖 20 种能力维度，首创"circular evaluation"避免 bias
- 阅读重点：§3 Abilities Definition and Questions（20 种能力定义）、§4 Circular Evaluation（评估方法）
- 时间分配建议：30 分钟精读能力分类体系，知道 VLM 应该被测评哪些方面
- 与本模块的关系：综合 VLM benchmark 的开创者

**Yue et al. (2024) 'MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI'**
- 一句话定位：大学级别多学科多模态评测集（6 学科 30 个细分领域），测试 VLM 的知识深度而非广度
- 阅读重点：§3 Data Construction（数据构建流程和人工校验）
- 时间分配建议：30 分钟精读数据构建方法和评估协议
- 与本模块的关系：VLM 的"高考卷"，测试真正的理解和推理

**Tong et al. (2024) 'Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs' (MMVP)**
- 一句话定位：系统揭示 VLM 在细粒度视觉理解上的系统性失败——空间关系、物体计数、属性绑定
- 阅读重点：§3 Visual Shortcomings（9 类失败模式分析）、§4.2 CLIP Failure Modes
- 时间分配建议：45 分钟精读失败模式分析，可以跳过部分实验
- 与本模块的关系：展示 benchmark 测不出的"盲点"，提示评估体系的局限性

---

## 拓展阅读
- **Li et al. (2024) 'Evaluating Object Hallucination in Multimodal Large Language Models' (POPE)** — 评估 VLM 幻觉的标准方法。如果你要实际部署 VLM 做图片问答，建议看看
- **Wang et al. (2024) 'MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark'** — MMMU 的加强版

---

## 模块间连接
- **前置依赖**：02-跨模态对齐、03-多模态融合架构（理解 VLM 架构后才能理解评估设计中的针对性）
- **后续衔接**：06-视频多模态与前沿挑战（benchmark 的局限性也是开放问题）
- **正交于**：01-视觉编码器、04-训练数据与规模化

---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| MMBench: Is Your Multi-modal Model an All-around Player? | MMBench () | [arXiv](https://arxiv.org/abs/2307.06281) |
| MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expe... | MMMU () | [arXiv](https://arxiv.org/abs/2311.16502) |
| MMVP: A Multimodal Mosai Puzzle for Evaluating Vision-Language Models | MMVP () | [arXiv](https://arxiv.org/abs/2401.02577) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| MMBench: Is Your Multi-modal Model an All-around Player? | [arXiv](https://arxiv.org/abs/2307.06281) |
| MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI | [arXiv](https://arxiv.org/abs/2311.16502) |
| MMVP: A Multimodal Mosai Puzzle for Evaluating Vision-Language Models | [arXiv](https://arxiv.org/abs/2401.02577) |
