﻿# 03 — 多模态融合架构

## 一句话开场

> 假设你已经有了一套强大的视觉编码器（ViT）和一个强大的语言模型（LLM）。现在的问题是：怎么把"看到的东西"告诉语言模型？直接在 LLM 前面开一个"视觉输入接口"就行——但接口怎么设计，决定了 LLM 能多好地理解图像。

## 正文：渐进式理解（5 层）

**第一层：问题定义。** 多模态融合架构解决的核心问题是"视觉特征以什么形式、在什么位置、以什么机制进入 LLM"。形式（projected embedding vs Q-Former queries vs 原生 ViT token）、位置（输入层 vs 中间层 vs 所有层）、机制（cross-attention vs prefix concatenation vs 可学习query）的不同组合，决定了模型的视觉理解能力和训练效率。

**第二层：核心直觉。** 想象你有一个只会读文字的助手（LLM），和一个视力很好的观察员（ViT）。让助手理解图像有三种方式：① 观察员用文字把看到的东西写下来给助手读（LLaVA 的 MLP projector 方案）；② 让助手直接问观察员问题，观察员专门回答（BLIP-2 的 Q-Former 方案）；③ 把观察员和助手用同一个语言训练，让他们自然理解彼此（InternVL 的动态分辨率方案）。

**第三层：方案细节。** 三种主要融合方式：① MLP Projector（LLaVA 系列）：ViT 输出的视觉 token 序列经过 2 层 MLP 投影到 LLM 的 embedding 空间，拼接到文本 embedding 前。计算最简单，视觉信息保留最多。② Q-Former（BLIP-2）：一组可学习的 query tokens 通过 cross-attention 从冻结 ViT 的输出中提取信息，再输入冻结 LLM。query 数量（默认 32）远少于 ViT token 数量（256+），信息被压缩。③ 原生融合（InternVL / Qwen2-VL）：ViT 和 LLM 共享 embedding 空间，端到端训练特定 layer 的参数。支持动态分辨率和多图输入，但训练成本最高。

**第四层：不同方案的权衡。**

| 方案 | 信息保留 | 训练成本 | LLM兼容性 | 代表工作 |
|------|---------|---------|----------|---------|
| MLP Projector | ★★★（全token） | 低（仅训练projector+LLM） | 高（任何LLM可接） | LLaVA (2023) |
| Q-Former | ★★（压缩后） | 极低（冻结ViT+LLM） | 高（需改造LLM输入） | BLIP-2 (2023) [[arXiv](https://arxiv.org/abs/2301.12597)] |
| Cross-Attention | ★★★★（多层交互） | 中 | 低（需修改LLM架构） | Flamingo (2022) |
| 原生多模态 | ★★★★★（全端到端） | 极高 | 低（从头训练） | InternVL (2024) |

**第五层：总结升华。** 融合架构的演进遵循一条清晰的路径：从"尽量不动 LLM"（LLaVA / BLIP-2 的冻结 LLM 方案）到"LLM 和 ViT 一起训练"（InternVL / Qwen-VL 的端到端方案），本质是从"把视觉挤进语言的壳里"到"语言和视觉共用一套参数空间"。当前事实标准是 LLaVA 式 MLP Projector，前沿趋势是原生多模态。

---

## 学习目标
- 画出 LLaVA 和 BLIP-2 两种融合架构的完整计算链路
- 用一句话说清 MLP Projector 和 Q-Former 在信息处理方式上的本质区别
- 能判断在资源受限时该选哪种融合方案（训练预算 vs 性能要求 vs LLM 可更换性）
- 理解为什么 LLaVA 的 MLP Projector 成为事实标准——不是因为它最好，而是因为它最"不碍事"
- 理解原生多模态（InternVL）为什么是趋势：去掉信息瓶颈

---

## 精选论文

**Liu et al. (2023) 'Visual Instruction Tuning' (LLaVA)**
- 一句话定位：定义 "ViT + MLP Projector + LLM" 这个最简融合范式，开源自 VLM 的事实标准
- 阅读重点：§3 Architecture（MLP projector 设计）、§3.1 Visual Instruction Tuning（GPT-4 生成指令数据）
- 时间分配建议：40 分钟精读架构，数据生成流程了解即可
- 与本模块的关系：最简融合方案的定义者

**Liu et al. (2024) 'Improved Baselines with Visual Instruction Tuning' (LLaVA 1.5)**
- 一句话定位：LLaVA 的实用改进版——数据配比、MLP 加层、分辨率提升，ACL 2024
- 阅读重点：§2.3 Training Recipes（数据配比和训练技巧）
- 时间分配建议：30 分钟精读训练细节
- 与本模块的关系：展示了"简单架构 + 正确数据 + 合理训练"的实用效果

**Li et al. (2023) 'BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs'**
- 一句话定位：Q-Former 用可学习 query 从冻结 ViT 提取特征再喂给冻结 LLM，训练效率极高
- 阅读重点：§3 Method（Q-Former 架构和两阶段训练）
- 时间分配建议：45 分钟精读 Q-Former 核心设计
- 与本模块的关系：另一种重要的融合思路（压缩式 query）

**Chen et al. (2024) 'How Far Are We to GPT-4V?' (InternVL 1.5/2.0)**
- 一句话定位：端到端训练大规模 ViT (6B) + LLM，动态分辨率，开源 VLM 逼近 GPT-4V 水平
- 阅读重点：§3 Architecture（ViT-LLM 共享训练策略）、§4 Dynamic Resolution
- 时间分配建议：60 分钟精读架构和动态分辨率机制
- 与本模块的关系：展示了"原生多模态"范式的可行性

---

## 拓展阅读
- **Alayrac et al. (2022) 'Flamingo'** — DeepMind 的 interleaved cross-attention VLM，展示"visual tokens 插入 LLM 各层"的融合方式
- **Bai et al. (2023) 'Qwen-VL'** — 阿里 Qwen-VL 系列，MLP projector + 位置编码合成，支持多图输入
- **Dai et al. (2024) 'LLaMA-Adapter V2'** — 参数高效微调方案，用少量可学习参数为 LLM 增加视觉能力

---

## 模块间连接
- **前置依赖**：01-视觉编码器、02-跨模态对齐
- **后续衔接**：04-训练数据与规模化、05-评估体系
- **正交于**：06-视频多模态

---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs | BLIP2 () | [arXiv](https://arxiv.org/abs/2301.12597) |
| InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic T... | InternVL () | [arXiv](https://arxiv.org/abs/2312.14238) |
| Improved Baselines with Visual Instruction Tuning | LLaVA15 () | [arXiv](https://arxiv.org/abs/2310.03744) |
| Visual Instruction Tuning | LLaVA () | [arXiv](https://arxiv.org/abs/2304.08485) |
| Qwen-VL: A Versatile Vision-Language Model | QwenVL () | [arXiv](https://arxiv.org/abs/2308.12966) |

---
