# 01 — Agent 基础范式

## 一句话开场

> 你让 LLM 查天气，它回复"我不能浏览互联网"——怎么让它学会自己调 API、查数据库、算数学，然后把结果告诉你？这就是 Agent 的基础范式：让 LLM 从"只会说话"变成"会干活"。

## 正文：渐进式理解

**第一层：问题定义。** LLM 本质上是一个文本生成模型——输入文本，输出文本。但如果我们要它执行实际任务（查天气、算数据、发邮件），纯文本输出是不够的。我们需要一种机制：让 LLM 输出**结构化指令**（不是对人类说的，而是对系统说的），然后由系统执行这些指令，把结果反馈给 LLM。这就是 Agent 的起点。

**第二层：核心直觉。** 想象你有一个超级聪明的助手，但他只会说不会做。你需要给他一套"工具说明书"：每样工具叫什么、能干什么、怎么用。然后他告诉你"用工具 A 做 X"——你去执行，把结果告诉他。这个循环就是 Agent 的核心模式：**思考 → 行动 → 观察 → 再思考**。

**第三层：方案细节。** Agent 基础范式的三个关键步骤：
1. **工具定义**：把每个 API 描述成 LLM 能理解的 JSON Schema（工具名、参数、用途）
2. **决策输出**：LLM 在对话中自主决定是否调用工具，输出结构化 JSON（`{"tool": "get_weather", "args": {"city": "北京"}}`）
3. **执行反馈**：系统调用 API 后把结果注入对话，LLM 继续推理

第三种模式 **ReAct（Reasoning + Acting）** 是目前事实标准：思考链和行动交错输出，每一步 LLM 都可以选择"继续推理"或"调用工具"或"给出最终答案"。

**第四层：不同方案的权衡。** 实现 Tool Use 有三条路线：

| 方案 | 代表作 | 核心优点 | 核心代价 |
|------|-------|---------|---------|
| **Prompt 驱动**（In-Context） | ReAct | 无需微调，任何 LLM 可用 | 依赖 LLM 指令跟随能力，不稳定 |
| **微调驱动**（Fine-tune） | Toolformer / Gorilla | 更稳定可控，推理成本低 | 需训练数据，模型更新后需重训 |
| **检索增强**（Retrieval-augmented） | Gorilla (API 文档检索) | 工具文档自动更新，不依赖训练 | 检索质量决定上限，增加一次检索调用 |

**第五层：总结升华。** Agent 基础范式是整个 AI Agent 领域的基石——没有 Tool Use，Agent 永远是"纸上谈兵"。ReAct 模式在 2023 年出现后迅速成为标准，后续所有工作（推理增强、多智能体、记忆系统）都是在 ReAct 循环的基础上增加能力。如果你只能理解 Agent 领域的一个概念，那就是 ReAct。

---

## 学习目标

读完你能：

- **用 30 行代码手写一个 ReAct 循环**（LLM 思考 → 调用工具 → 观察结果 → 最终回答）
- **用一句话说清 ReAct 和普通 Prompt 的区别**：ReAct 让 LLM 在"推理"和"行动"之间切换，普通 Prompt 只有文本输出
- **当面对一个需要调用外部 API 的需求时，能给出 Toolformer 和 Function Calling 两种方案并说出各自的适用场景**
- **能画出 ReAct 循环的数据流图**：User Input → Thought → Action → Observation → ... → Final Answer
- **能判断一个任务是否适合 Agent 范式**：是否需要外部信息？是否需要多步操作？

> 每一条学习目标都能被客观检验——你可以清楚地说"我做到了"或"我还没做到"。

---

## 精选论文

**Yao et al. (2023) "ReAct: Synergizing Reasoning and Acting in Language Models"**

- **一句话定位**：提出「思考→行动→观察」交替循环，Agent 范式的起源工作
- **阅读重点**：第 2-3 节（ReAct 框架和它与 CoT 的对比）。Table 1 非常直观
- **时间分配建议**：精读全文（约 15 页），这是整个领域最基础的一篇。附录的可视化案例值得一看
- **与本模块的关系**：它定义了 Agent 基础范式的核心模式

**Schick et al. (2023) "Toolformer: Language Models Can Teach Themselves to Use Tools"**

- **一句话定位**：LLM 自举学会"什么场景调用什么工具"，证明了工具使用能力可以自我进化
- **阅读重点**：第 3 节（自监督数据生成流程）。理解"采样 → 执行 → 过滤"的自动化 pipeline
- **时间分配建议**：精读第 3 节，其余快速浏览。核心贡献是训练方法，不是推理架构
- **与本模块的关系**：回答了"工具使用能力从哪来"的问题——不需要人工标注

**Patil et al. (2023) "Gorilla: Large Language Model Connected with Massive APIs"**

- **一句话定位**：大规模 API 调用 + 检索增强，解决 Agent 落地的核心工程问题
- **阅读重点**：第 4 节（检索式 API 文档更新机制）。理解为什么"训练时记住 API"不如"推理时检索 API"
- **时间分配建议**：第 4 节精读，其余略读。重点在工程创新而非模型创新
- **与本模块的关系**：回答了"工具太多、文档经常更新，Agent 怎么跟上"的问题

**OpenAI (2023) "Function Calling Capability" (Blog / API Docs)**

- **一句话定位**：ReAct 的产品化实现，使 Tool Use 在 API 层面原生支持
- **阅读重点**：API 文档中的 `tools` 参数定义和调用流程
- **时间分配建议**：运行一个 demo 比读文档更快。建议直接看 `code/function_calling_openai.py`
- **与本模块的关系**：让 Tool Use 从学术工作变成可产品化的 API 能力

---

## 拓展阅读

- **Qin et al. (2023) "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs"** — 在 ToolBench 数据集上微调的 Tool-Use 专用模型。如果你想了解"大规模工具调用的全流程"可以看这篇。
- **Chen et al. (2023) "OpenAGI: When LLM Meets Domain Experts"** — 将 Agent 范式扩展到科学/医疗领域。如果你对 "Agent 在垂直领域的应用"感兴趣可以翻翻。

> 拓展论文不移除，放在 `01-Agent基础范式/拓展/` 文件夹下。核心论文在模块根目录。

---

## 模块间连接

- **前置依赖**：熟悉基本的 LLM API 调用（Chat Completion）。可以先去复习 LLM 的 05-应用技术 中的 Prompt Engineering 基础
- **后续衔接**：读完本模块后，建议进入 **02-推理与规划**（让 Agent 能处理多步任务）或 **04-Agent 记忆与知识**（让 Agent 有记忆）
- **本模块与哪些模块正交**：本模块与 03-多智能体系统、05-工程框架与协议、06-安全与前沿 没有知识依赖，可以并行阅读

---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| Gorilla: Large Language Model Connected with Massive APIs | Gorilla (2023) | [arXiv](https://arxiv.org/abs/2305.15334) |
| ReAct: Synergizing Reasoning and Acting in Language Models | ReAct (2023) | [arXiv](https://arxiv.org/abs/2210.03629) |
| Toolformer: Language Models Can Teach Themselves to Use Tools | Toolformer (2023) | [arXiv](https://arxiv.org/abs/2302.04761) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| Gorilla: Large Language Model Connected with Massive APIs | [arXiv](https://arxiv.org/abs/2305.15334) |
| OpenAGI: When LLM Meets Domain Experts | [arXiv](https://arxiv.org/abs/2304.04370) |
| ReAct: Synergizing Reasoning and Acting in Language Models | [arXiv](https://arxiv.org/abs/2210.03629) |
| Toolformer: Language Models Can Teach Themselves to Use Tools | [arXiv](https://arxiv.org/abs/2302.04761) |
