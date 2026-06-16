# 04 — 可控生成与适配

## 一句话开场

> 你有一个已经训练好的 Stable Diffusion，它能按 prompt 画出不错的图，但你想：① 精确控制人物的姿势（用骨架图约束）；② 把自己画成某个风格的角色（个人化）；③ 让模型学会一个新的物体（如你的宠物）。这时候基础模型不够用了——你需要的是**围绕预训练模型的外围控制工具**。

---

## 正文：渐进式理解

**第一层：问题定义。** 基础扩散模型（如 SD / SDXL / Flux）只能通过文本 prompt 做粗粒度的控制，面对精确的空间约束（边缘/深度/姿态）、个人化（风格/人脸/物体）、多条件组合等需求时力不从心。**如何在不破坏基础模型能力的前提下，添加新的控制能力并保持高效？** 这个问题的核心约束是：不希望每次需求变化都重新训练整个模型。

**第二层：核心直觉。** 这个问题的解决思路类似于"给一个熟练的画家加辅助线"：

- **ControlNet**：不修改画家的技巧，但在画纸上叠加辅助线（边缘/深度/骨架），让画家按辅助线画
- **LoRA**：不是改造整个画家，而是给画笔加上一个"轻量级适配器"，让下笔的风格微调
- **Textual Inversion**：不是教画家画新东西，而是告诉画家"这个词代表了这个新事物"——只学一个 token
- **IP-Adapter**：给画家一张参考图，说"用这张图的风格和氛围来画"

**第三层：方案细节。** 四种主流方法的技术本质：

**方法一：ControlNet——空间条件控制的瑞士军刀**
- 复制 U-Net 的编码器作为"可训练副本"，用零初始化卷积连接原模型
- 输入空间控制图（Canny 边缘 / 深度图 / OpenPose 骨架 / 语义分割图等）
- 核心创新：zero convolution 确保训练开始时副本输出为零，不破坏原模型分布
- 训练时只训练副本，原模型参数冻结

**方法二：LoRA——低秩适配微调**
- 对模型中的权重矩阵 W ∈ R^{d×k}，将其更新 ΔW 分解为低秩矩阵 BA（B∈R^{d×r}, A∈R^{r×k}）
- 推理时：W' = W + αBA，其中 r（秩）控制表达能力（通常 r=4~64）
- 可以在不同数据集上训练多个 LoRA，推理时组合使用
- 存储开销极小（数十 MB vs 数 GB 全量模型）

**方法三：Textual Inversion——新概念学习**
- 不修改模型权重，而是在文本 embedding 空间学一个新的伪 token S*
- 用 3-5 张参考图优化该 token 的 embedding（使 S* 在 U-Net Cross-Attention 中能召回该概念）
- 推理时在 prompt 中使用该 token

**方法四：IP-Adapter——图像条件控制**
- 在 U-Net 中新增一个 Cross-Attention 层（与文本 Cross-Attention 并行）
- 用 CLIP 图像编码器提取参考图特征作为 K/V，注入去噪过程
- 核心分离文本条件路径和图像条件路径，各自独立

**第四层：不同方案的权衡。**

| 维度 | ControlNet | LoRA | Textual Inversion | IP-Adapter |
|------|-----------|------|-------------------|------------|
| **控制类型** | 空间条件（边缘/深度/骨架） | 风格/人物/物体适配 | 新概念学习 | 以图生图 |
| **是否改权重** | 是（新训练副本） | 是（低秩矩阵） | 否（只改 embedding） | 是（新 Cross-Attn） |
| **参数量** | ~350M（SD1.5） | ~3-50M | < 1K（仅一个 token） | ~20M |
| **训练数据** | 1k~50k 条件-图对 | 10~200 张图 | 3~5 张图 | ~100M 图对 |
| **合成能力** | 可叠加多个条件 | 可叠加多个 LoRA | 单一概念 | 可叠加其他条件 |
| **推理开销** | 略增（额外编码器） | 几乎无 | 无 | 略增（图像编码） |

**第五层：总结升华。** 可控生成与适配的生态围绕一个原则构建：**基础模型不变，外围控制工具独立开发**。这层"可插拔"架构是 SD 生态繁荣的根本原因——一个模型（SD1.5 / SDXL）能衍生出成百上千种定制版本。随着 DiT/MDiT 成为主流，控制方法也在向 Transformer 适配（如 DiT-Based ControlNet），但"冻结基础模型 + 外围适配"的设计哲学没有变。

---

## 学习目标

读完你能：

- **为一个实际需求选择正确的控制方法**：给出"为什么 ControlNet 而不是 LoRA"或反之的决策理由
- **解释 ControlNet 的 zero convolution 为什么能让训练不破坏原模型**
- **解释 LoRA 的低秩分解原理**：写出 ΔW = BA 的公式并解释秩 r 的作用
- **区分 Textual Inversion 和 LoRA 的本质差异**：一个学 embedding，一个学权重
- **设计一个组合方案**：例如"用 ControlNet 控制姿势 + LoRA 控制风格 + Textual Inversion 控制物体"

---

## 精选论文

**Zhang et al. (2023) "Adding Conditional Control to Text-to-Image Diffusion Models" [[arXiv](https://arxiv.org/abs/2302.05543)]**

- **一句话定位**：ControlNet 的提出，将空间条件控制接入扩散模型的标准框架，极大扩展了可控性
- **阅读重点**：Section 3（方法：零初始化 + 可训练副本 + 条件输入）、Figure 3（架构图：条件编码器如何插入 U-Net）
- **时间分配建议**：Section 3（约 4 页）为必读核心，理解 zero convolution 的设计动机。Section 4（实验）看 Figure 6-10 即可
- **与本模块的关系**：回答了"如何在不破坏基础模型的前提下添加空间条件控制"

**Hu et al. (2022) "LoRA: Low-Rank Adaptation of Large Language Models" [[arXiv](https://arxiv.org/abs/2106.09685)]**

- **一句话定位**：LoRA 提出低秩适配范式，虽在 LLM 提出，但在 SD 社区是影响力最大的微调方法
- **阅读重点**：Section 2（低秩分解：ΔW = BA）、Section 3（在 Transformer 上的应用）
- **时间分配建议**：全文仅 5 页，建议通读。重点理解秩 r 如何影响表达能力和参数量之间的 trade-off
- **与本模块的关系**：回答了"如何用最小的参数量修改模型行为"

**Gal et al. (2022) "An Image is Worth One Word: Textual Inversion for Personalized Image Generation" [[arXiv](https://arxiv.org/abs/2208.01618)]**

- **一句话定位**：提出用伪 token 学习新概念，仅需 3-5 张图即可让模型学会新物体/风格
- **阅读重点**：Section 3（方法：在 embedding 空间优化伪 token）、Figure 2（伪 token 的优化过程）
- **时间分配建议**：Section 3（约 3 页）理解伪 token 的优化目标即可
- **与本模块的关系**：回答了"最小干预的方式让模型学会新概念"

---

## 拓展阅读

- **Ruiz et al. (2023) "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation" [[arXiv](https://arxiv.org/abs/2208.12242)]** — 全量/部分微调的代表工作，用少量图让模型记住特定对象。与 LoRA 对比，DreamBooth 更强大但更重。
- **Ye et al. (2023) "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models" [[arXiv](https://arxiv.org/abs/2308.06721)]** — 图像 prompt 控制的标准方案，解决了"以图生图"条件下高保真度控制的问题。
- **Mou et al. (2024) "T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models" [[arXiv](https://arxiv.org/abs/2302.08453)]** — ControlNet 的轻量级替代方案。如果你关心"有没有比 ControlNet 更轻量的选择"，值得一看。

> 拓展论文不移除，放在 `拓展/` 文件夹下。

---

## 模块间连接

- **前置依赖**：建议先完成 **02-骨干架构演进**（理解 U-Net 架构）和 **03-条件注入机制**（理解 Cross-Attention 和 CFG），因为 ControlNet 和 IP-Adapter 都在这些机制上构建。
- **后续衔接**：读完本模块后，可以进入 **05-采样加速与蒸馏**（让生成更快的技术）或 **06-多模态扩展**（视频/3D 应用）。
- **本模块与哪些模块正交**：本模块与 01（扩散理论）正交——不用理解 ELBO 推导也能使用 ControlNet。

