# 06 — 前沿方向

> LLM 的下一步：是让它更大（MoE）、更主动（Agent）、还是感知更丰富（MLLM）？

## 正文：渐进式理解

**第一层：问题定义。** 当前的 LLM 有三大天花板：① **参数天花板**——Dense 模型在 70B-400B 参数后边际收益递减，继续增大训练成本不可持续；② **交互天花板**——LLM 只能问答，不能主动执行任务、调用工具、从反馈中修正行为；③ **模态天花板**——LLM 只能处理文本，无法理解图像/音频/视频。三个方向——MoE、Agent、MLLM——各自回答一个独立问题。

**第二层：核心直觉。** 三个方向的类比：

`
MoE  = 一群人各有所长（每次只叫最懂的那几个人干活）
       → 解决「Dense 模型太浪费」的问题
Agent = 一个会动手的人（不只是回答，而是去查、去算、去验证）
       → 解决「模型只能嘴说不能动手」的问题
MLLM = 一个人同时有眼睛、耳朵、嘴巴
       → 解决「模型只能读文本」的问题
`

**第三层：方案细节。**

**MoE（Mixture of Experts）**：把一个大模型拆成多个「专家」（Expert），每个 token 通过 Router 只激活 Top-2 专家。Mixtral 8×7B 有 46.7B 总参数但每 token 只激活 12.9B，以 LLaMA 2 13B 的激活参数达到 70B 的性能。**关键挑战**：路由不均衡（有些专家总是被选到，有些闲置）、通信开销（分布式场景下 All-to-All 通信）。

**ReAct Agent**：Reasoning + Acting 的循环：

`
Observation（感知当前状态）
    ↓
Thought（思考下一步该做什么）
    ↓
Action（调用工具 / 搜索 / 计算）
    ↓
Observation（观察工具返回结果）
    ↓
...（循环直到任务完成）
`

ReAct 是后续 ChatGPT Plugins / Function Calling / MCP（Model Context Protocol）的范式基础。

**MLLM（Multimodal LLM）**：在文本 LLM 基础上增加视觉编码器（Vision Encoder），将图像特征投影到文本 embedding 空间。典型架构：Vision Encoder（ViT）+ Projection Layer（Q-Former / MLP）+ LLM Backbone。代表：GPT-4V / LLaVA / Qwen-VL。

**第四层：不同方案的权衡。**

MoE vs Dense 模型：

| 维度 | MoE（稀疏） | Dense（稠密） |
|------|------------|--------------|
| 总参数量 | 大（8×7B = 56B） | 可控 |
| 单 token 激活 | 小（~12.9B） | 全部 |
| 训练效率 | ⚠️ 需额外处理路由 + Load Balance | ✅ 稳定 |
| 推理吞吐 | ✅ 高（更少 FLOP/token） | ❌ 低 |
| 部署复杂度 | ❌ 分片通信开销大 | ✅ 简单 |
| 代表 | Mixtral, DeepSeek MoE, Qwen MoE | LLaMA, GPT-4 |

单 Agent vs 多 Agent 编排：

| 维度 | 单 Agent（ReAct） | 多 Agent 编排 |
|------|----------------|--------------|
| 设计复杂度 | 低 | 高（协调 / 通信 / 共识） |
| 任务容错 | 低（错误无法恢复） | 中（可通过投票/交叉验证恢复） |
| 适合场景 | 简单工具调用 | 复杂多步任务 |
| 代表方案 | Function Calling, MCP | AutoGen, CrewAI |

**第五层：总结升华。** MoE、Agent、MLLM 三个方向不是竞争关系而是**互补**——MoE 让模型更大的同时控制推理成本，Agent 让模型从「问答工具」变成「自主助手」，MLLM 让模型的感知超越文本。未来 LLM 的真实形态很可能是三者的融合：**MoE 作为骨干、MLLM 提供多模态感知、Agent 框架提供行动能力**。

---

## 学习目标

读完你能：

- 用一句话说清 MoE 和 Dense 模型的本质区别（稀疏激活 vs 全部激活）
- 画出 ReAct 的循环流程图（Observe → Think → Act → Observe）并解释每一环
- 给一个实际场景判断用单 Agent 还是多 Agent 编排
- 解释 MLLM 的基本架构（Vision Encoder → Projection Layer → LLM）
- 理解三个方向不是竞争而是互补——它们各自解决 LLM 当前的不同天花板

---

## 精选论文

**Jiang et al. (2024) "Mixtral of Experts"**

- **一句话定位**：MoE 在 LLM 中的成功实践，8×7B 只有 12.9B 激活参数但超越 LLaMA 2 70B
- **阅读重点**：第 2 节（MoE 层结构和 Router 设计——Figure 1 的架构图）和第 3 节（实验对比）
- **时间分配建议**：实验细节可以跳读，核心是理解「总参数 vs 激活参数的 trade-off」以及 Mixtral 为什么选择 Top-2 路由

**Yao et al. (2023) "ReAct: Synergizing Reasoning and Acting in Language Models"**

- **一句话定位**：Agent 范式的起点，推理 + 工具调用的组合，后续 ChatGPT Plugins / MCP 的基础
- **阅读重点**：第 2 节（ReAct 的定义 + Figure 1 的对比实验——ReAct vs CoT vs Standard 在 HotpotQA 上的差异非常直观）
- **时间分配建议**：时间紧只看 Figure 1（三种方法怎么回答同一个问题的对比）和 Table 1（实验结果）；时间充裕精读第 4 节（消融实验——CoT 加推理效果更好吗）

---

## 模块间连接

- **前置依赖**：建议先读 **01-Transformer 起源**（理解模型基础）和 **05-应用技术**（本模块的 Agent 是 RAG/CoT 的自然延伸——Agent = CoT 推理 + RAG 知识 + Tool Use 行动）
- **后续衔接**：读完本模块意味着完成了整个 LLM 知识体系的主干学习。接下来可以：① 深入某个感兴趣的前沿方向深入阅读该方向的论文；② 回到某个具体模块的精读；③ 开始动手实践（通过 vLLM 部署模型或通过 LangChain 搭建 Agent）
- **本模块与哪些模块正交**：与 04-推理与部署优化完全正交——「下一步去哪」不影响「怎么部署当前模型」


---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Mixtral of Experts | Mixtral () | [arXiv](https://arxiv.org/abs/2401.04088) |
| ReAct Yao2023 | ReAct () | [arXiv](https://arxiv.org/abs/2210.03629) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| Mixtral of Experts | [arXiv](https://arxiv.org/abs/2401.04088) |
| ReAct Yao2023 | [arXiv](https://arxiv.org/abs/2210.03629) |
