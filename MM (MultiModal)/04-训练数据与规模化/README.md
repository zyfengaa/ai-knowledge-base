# 04 — 训练数据与规模化

## 一句话开场

> CLIP 的成功不是因为架构有多聪明，而是因为它有 4 亿图文对。但 4 亿对里有多少是噪声？数据的质量更重要还是数量更重要？当模型从 1B 参数 Scale 到 100B 参数时，性能会一直线性增长吗？

## 正文：渐进式理解（5 层）

**第一层：问题定义。** 多模态模型的性能由两个因素决定：架构（怎么学）和数据（学什么）。本模块关注后者——图文数据从哪来、怎么清洗、怎么配比、当数据规模扩大时性能如何变化。核心问题是：给定有限的 budget，应该投资数据质量还是数据数量？

**第二层：核心直觉。** 想象你在教一个学生看图说话。你给他 100 张高质量的图+精心写的描述，他学得很好但只见过 100 种场景。你给他 1 亿张从互联网随便爬的图+粗糙的描述，他接触了大量场景但学到了大量噪声。好的策略是：先用大量噪声数据预训练（获得宽度），再用少量高质量数据精调（获得精度）。这就是 BLIP 的 CapFilt 策略和 LLaVA 的数据配比策略的核心直觉。

**第三层：方案细节。** 数据工作流：① 数据采集——从互联网爬取图文对（LAION-5B / COYO-700M），需要URL 过滤、NSFW 过滤、文本质量过滤。② 数据清洗——用 CLIP 相似度过滤（图文不匹配的低分样本）、用类别平衡策略（防止长尾分布）。③ 数据增强——BLIP 的 CapFilt：先训练一个生成器（为图像生成描述），再训练一个过滤器（去掉低质量描述）。LLaVA 用 GPT-4 生成高质量指令数据。④ 数据配比——LLaVA 1.5 发现"GQA + OCR-VQA + VG"等混合数据效果优于单一数据源。⑤ Scaling——VLM 的 scaling law 研究显示：更多数据 x 更大模型 的乘积关系适用于 VLM，但视觉端的数据需求远大于文本端。

**第四层：不同方案的权衡。**

| 策略 | 数据成本 | 质量增益 | 适用阶段 |
|------|---------|---------|---------|
| 纯爬取+CLIP过滤 | 低（自动化） | 中 | 预训练初期 |
| BLIP CapFilt | 中（需训练生成器） | 高 | 预训练数据增强 |
| GPT-4 合成指令数据 | 高（API成本） | 极高 | SFT 阶段 |
| 数据集混合配比 | 低（组合现有数据） | 高 | SFT / 微调 |

**第五层：总结升华。** 数据的核心矛盾是"质量 vs 数量"。2021-2023 年的共识是数量为王（CLIP/LAION），2024 年的共识转向质量+配比（DataComp/LLaVA 1.5 证明了数据配方比模型架构更能影响 VLM 的最终性能）。一个被广泛证明的经验法则：**先 Scale 数据量，再优化数据质量，最后用高质量指令数据微调**。

---

## 学习目标
- 说出从互联网爬图到训练 VLM 的完整数据工作流
- 用一句话说清 CapFilt 为什么有效——生成器的多样性 + 过滤器的精确性互补
- 理解为什么数据配比（data mixing ratio）比数据总量更能影响微调效果
- 知道 VLM Scaling Law 的主要发现：视觉端的数据需求远大于文本端
- 在实际项目中，能判断当前阶段应该投资数据量还是数据质量

---

## 精选论文

**Gadre et al. (2023) 'DataComp: In search of the next generation of multimodal datasets'**
- 一句话定位：系统研究图文数据配比对 CLIP 训练的影响，提供不同 budget 下的最优数据策略
- 阅读重点：§3-4 Experimental Setup and Results（不同数据策略的消融实验）
- 时间分配建议：45 分钟精读实验设计和方法论，了解"什么数据最有用"
- 与本模块的关系：数据策略的参考标准

**Li et al. (2022) 'BLIP: Bootstrapping Language-Image Pre-training'**
- 一句话定位：CapFilt 机制用生成+过滤提升数据质量，展示了"数据质量 > 数据数量"的一个有力证据
- 阅读重点：§3.1 CapFilt（核心数据增强机制）
- 时间分配建议：30 分钟精读 CapFilt 流程
- 与本模块的关系：生成式数据增强的代表工作

**Liu et al. (2024) 'Improved Baselines with Visual Instruction Tuning' (LLaVA 1.5)**
- 一句话定位：虽然不是专门的数据论文，但其数据配比实验是关键参考——混合数据源好于单一数据源
- 阅读重点：§2.2 Data Mixture（数据配比实验）
- 时间分配建议：20 分钟阅读数据配比部分
- 与本模块的关系：展示了数据配比如何影响 VLM 的微调效果

---

## 拓展阅读
- **Schuhmann et al. (2022) 'LAION-5B: An open large-scale dataset for training next generation image-text models'** — 5B 图文对的开放数据集，爬虫流程和数据过滤的参考范本
- **Cherti et al. (2023) 'Reproducible scaling laws for contrastive language-image learning'** — CLIP 训练的 Scaling Law 研究，资源允许时可以复现

---

## 模块间连接
- **前置依赖**：02-跨模态对齐（理解 CLIP 才能理解数据清洗中的 CLIP filtering）
- **后续衔接**：03-多模态融合架构（数据配比影响融合架构的训练效果）
- **正交于**：01-视觉编码器、05-评估体系

---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| DataComp: In search of the next generation of multimodal datasets | DataComp () | [arXiv](https://arxiv.org/abs/2304.14108) |
| Reproducible scaling laws for contrastive language-image learning | ScalingLaws () | [arXiv](https://arxiv.org/abs/2303.16199) |

---
