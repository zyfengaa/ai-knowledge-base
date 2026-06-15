# 05 — 工程框架与协议

## 一句话开场

> 你实现了 ReAct 循环、接上了记忆系统、添加了多 Agent 协作——但你的工具接口和别人的不通用，评估指标是你自己定义的，日志基本靠 print——怎么让 Agent 系统可维护、可扩展、可评估？这就是工程框架与协议要解决的标准化问题。

## 正文：渐进式理解

**第一层：问题定义。** Agent 的研究进展很快，但工程落地一直在追赶。2023-2024 年，每个团队都在造自己的轮子：定义工具的方式不同、Agent 通信格式不同、评估标准不同。这种碎片化严重阻碍了 Agent 从研究到产品的转化。工程框架与协议要解决的根本问题是：**怎么让 Agent 系统的各组件（LLM、工具、记忆、Agent 间通信）有标准接口，能够像拼积木一样组合？**

**第二层：核心直觉。** 想想 USB 标准出现之前的外设：每个设备都要专用的接口和驱动。MCP（Model Context Protocol）就是 Agent 领域的 USB 标准——它定义了"工具"的通用接口，让任何 Agent 能使用任何工具，只要双方都支持 MCP。同样，评估标准（AgentBench/WebArena）就像考试的标准化试卷——没有它，你说你的 Agent 好用、我说我的好用，谁都没法比较。

**第三层：方案细节。** Agent 工程化的三个核心层次：

1. **工具标准化（MCP）**：MCP 的核心是"工具即服务"。工具端实现一个 MCP Server（注册工具名、参数 Schema、执行逻辑），Agent 端通过 MCP Client 发现和调用工具。关键设计点：JSON-RPC 协议、生命周期管理（状态）、安全边界（权限声明）。
2. **Agent 框架（Orchestration）**：AutoGen / LangChain / CrewAI 等框架提供：Agent 生命周期管理、对话流程控制、工具注册、记忆管理等基础设施。选型时核心考虑：灵活性 vs. 开箱即用。
3. **评估基准（Benchmark）**：Agent 评估比 LLM 评估复杂得多——需要"环境+任务+交互"的三元组。AgentBench 提供 8 个环境（OS/DB/Web 等），WebArena 提供网站操作，SWE-bench 提供代码任务。

**第四层：不同方案的权衡。**

| 维度 | 方案 A | 方案 B | 方案 C |
|------|-------|-------|-------|
| **工具标准** | MCP（开放协议） | Function Calling（OpenAI 原生） | 自定义实现 |
| 优点 | 通用、解耦、开源 | 简单、零额外依赖 | 完全控制 |
| 代价 | 需部署 MCP Server | 绑定 OpenAI API | 维护成本高 |
| **Agent 框架** | AutoGen | LangChain | CrewAI |
| 风格 | 对话式，多 Agent 原生 | 链式调用，生态最大 | 基于角色，易用性好 |
| 适用 | 研究探索、多 Agent | 生产流水线、RAG 系统 | 快速原型、中小项目 |
| **评估基准** | AgentBench | WebArena | SWE-bench |
| 评估能力 | 多环境通用能力 | 网站操作能力 | 代码工程能力 |
| 局限 | 任务偏短、环境固定 | 浏览器自动化场景 | 仅限 Python 项目 |

**一个贯穿所有维度的设计轴：标准化程度 vs. 灵活性。** 标准化越高组合越容易，但处理边缘场景的灵活性越差。

**第五层：总结升华。** Agent 工程化是 Agent 从"学术 demo"走向"产品级系统"的必经之路。MCP 协议的提出是 2024 年最重要的工程进展——它第一次明确定义了 Agent 和工具的边界，让"工具生态"成为可能。评估基准的完善则让"更好的 Agent"有了客观标准。这两条线合在一起，正在把 Agent 开发从"手工作坊"推向"工业化"——就像 Web 协议标准化催生了互联网时代。

---

## 学习目标

读完你能：

- **用一句话说清 MCP 解决了什么问题**：让 Agent 和工具之间有一个通用标准接口，破除「每种工具都要写适配代码」的碎片化
- **面对一个 Agent 项目，能做出合理的框架选型**：多 Agent 实验？→ AutoGen。生产级 RAG？→ LangChain。快速原型？→ CrewAI
- **理解 Agent 评估和 LLM 评估的核心区别**：Agent 评估需要"环境 × 任务 × 交互"的完整闭环，不是单次问答
- **能画出一个完整的 MCP 架构图**：Agent (MCP Client) ↔ MCP 协议 ↔ Tool Server (工具实现)
- **为自定义工具设计 MCP 兼容的接口**：定义 name、description、input_schema、execute 四要素

---

## 精选论文

**Xi et al. (2023) "The Rise and Potential of Large Language Model Based Agents: A Survey"**

- **一句话定位**：最全面的 Agent 综述，覆盖 500+ 文献，建了 Agent 领域的分类体系
- **阅读重点**：第 3 节（Agent 架构分类：脑/感知/行动）、第 6 节（挑战与开放问题）。图 2 是全景图
- **时间分配建议**：先读摘要 + 第 3 节建立框架，再根据兴趣深入其他章节。这是"地图"型论文不需要从头精读
- **与本模块的关系**：提供了 Agent 领域的完整分类体系，是理解工程框架全景的基础

**Qin et al. (2023) "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs"**

- **一句话定位**：大规模工具调用的全链路工作——从数据构建到模型微调再到评估
- **阅读重点**：第 3 节（ToolBench 数据构建 + ToolEval 评估）。理解"大规模 API 如何标准化描述"
- **时间分配建议**：关注数据构建和评估方法，模型训练部分可略读。核心工程创新在数据 pipeline
- **与本模块的关系**：展示了大规模 API 标准化描述和评估的工程范式

**Liu et al. (2023) "AgentBench: Evaluating LLMs as Agents"**

- **一句话定位**：第一个系统的 Agent 评估基准——8 个环境、多维度能力评估
- **阅读重点**：第 2 节（AgentBench 设计原则和环境列表）。Table 1 列出 8 个环境一目了然
- **时间分配建议**：关注评估框架设计而非排行榜结果。理解"Agent 评估比 LLM 评估复杂在哪"
- **与本模块的关系**：Agent 评估标准化的先驱工作，后续 WebArena/SWE-bench 都建立在类似思想上

**Anthropic (2024) "Model Context Protocol (MCP)" (Specification / Blog)**

- **一句话定位**：Anysphere/Claude 团队提出的开放工具标准协议，Agent 集成的 USB 接口
- **阅读重点**：协议规范中的核心概念（Resources、Tools、Prompts）和通信流程
- **时间分配建议**：阅读官方规范的前三节 + 运行 `mcp_demo.py`。协议本身很简单，理解设计意图更重要
- **与本模块的关系**：这是目前 Agent 工具集成的事实标准，理解 MCP 就是理解 Agent 工程的现在和未来

---

## 拓展阅读

- **Google (2024) "Agent-to-Agent (A2A) Protocol"** — Google 提出的 Agent 间通信协议（对标 MCP 的 Agent ↔ Agent 通信）。如果你对"Agent 与 Agent 怎么对话"的标准化感兴趣可以看看。
- **CrewAI (2024) "CrewAI Framework Documentation"** — 基于角色的多 Agent 框架。如果你需要快速搭建一个多 Agent 原型的工程方案可以看这个。

> 拓展论文不移除，放在 `05-工程框架与协议/拓展/` 文件夹下。

---

## 模块间连接

- **前置依赖**：先理解 01-Agent 基础范式（工程框架是在这个基础上的基础设施）。05 是"元知识"——关于 Agent 的知识，而非 Agent 本身的知识
- **后续衔接**：读完后建议进入 **06-安全评估与前沿挑战**（讲到这里，评估基准和协议的安全性自然引出安全问题）
- **本模块与哪些模块正交**：02-推理与规划、03-多智能体系统、04-Agent 记忆与知识——工程框架不依赖于具体的算法选择

---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| AgentBench: Evaluating LLMs as Agents | AgentBench (2023) | [arXiv](https://arxiv.org/abs/2308.03688) |
| The Rise and Potential of Large Language Model Based Agents: A Survey | Agent (2023) | [arXiv](https://arxiv.org/abs/2306.06094) |
| ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs | ToolLLM (2023) | [arXiv](https://arxiv.org/abs/2307.16789) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| AgentBench: Evaluating LLMs as Agents | [arXiv](https://arxiv.org/abs/2308.03688) |
| Agent Evaluation Survey 2025 | [arXiv](https://arxiv.org/abs/2501.03610) |
| The Rise and Potential of Large Language Model Based Agents: A Survey | [arXiv](https://arxiv.org/abs/2306.06094) |
| ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs | [arXiv](https://arxiv.org/abs/2307.16789) |
