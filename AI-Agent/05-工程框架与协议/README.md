# 05 — 工程框架与协议

## 一句话开场

> 你实现了 ReAct 循环、接上了记忆系统、添加了多 Agent 协作——但你的工具接口和别人的不通用，评估指标是你自己定义的，日志基本靠 print——怎么让 Agent 系统可维护、可扩展、可评估？这就是工程框架与协议要解决的标准化问题。

## 正文：渐进式理解

**第一层：问题定义。** Agent 的研究进展很快，但工程落地一直在追赶。每个团队都在造自己的轮子：定义工具的方式不同、通信格式不同、评估标准不同。工程框架与协议解决的根本问题是：**怎么让 Agent 系统的各组件有标准接口，能够像拼积木一样组合？**

**第二层：核心直觉。** 想想 USB 标准出现之前的外设：每个设备都要专用接口和驱动。MCP（Model Context Protocol）就是 Agent 领域的 USB 标准——它定义了"工具"的通用接口，让任何 Agent 能使用任何工具，只要双方都支持 MCP。同样，评估标准（AgentBench/WebArena）就像标准化试卷——没有它，没法比较谁好谁差。

**第三层：方案细节。** Agent 工程化的三个核心层次：① **工具标准化（MCP）**——工具端实现 MCP Server，Agent 端通过 MCP Client 发现和调用；② **Agent 框架（Orchestration）**——AutoGen/LangChain/CrewAI 提供生命周期管理、对话流程、工具注册等基础设施；③ **评估基准（Benchmark）**——AgentBench 提供 8 个环境，WebArena 提供网站操作，SWE-bench 提供代码任务。

**第四层：不同方案的权衡。**

| 维度 | 方案 A | 方案 B | 方案 C |
|------|-------|-------|-------|
| **工具标准** | MCP（开放协议） | Function Calling（OpenAI） | 自定义实现 |
| 优点 | 通用、解耦、开源 | 简单、零额外依赖 | 完全控制 |
| 代价 | 需部署 Server | 绑定 OpenAI API | 维护成本高 |
| **Agent 框架** | AutoGen | LangChain | CrewAI |
| 风格 | 对话式，多 Agent 原生 | 链式调用，生态最大 | 基于角色，易用 |
| **评估基准** | AgentBench | WebArena | SWE-bench |
| 评估能力 | 多环境通用 | 网站操作 | 代码工程 |

**第五层：总结升华。** Agent 工程化是 Agent 从"学术 demo"走向"产品级系统"的必经之路。MCP 协议的提出是 2024 年最重要的工程进展——它第一次明确定义了 Agent 和工具的边界。评估基准的完善则让"更好的 Agent"有了客观标准。

---

## 学习目标

读完你能：

- 用一句话说清 MCP 解决了什么问题：让 Agent 和工具之间有一个通用标准接口
- 面对一个 Agent 项目，能做出合理的框架选型
- 理解 Agent 评估和 LLM 评估的核心区别
- 画出一个完整的 MCP 架构图：Agent (MCP Client) ↔ MCP 协议 ↔ Tool Server
- 为自定义工具设计 MCP 兼容的接口

---

## 精选论文

**Xi et al. (2023) "The Rise and Potential of Large Language Model Based Agents: A Survey"** — [arXiv](https://arxiv.org/abs/2309.07864)

- **一句话定位**：最全面的 Agent 综述，覆盖 500+ 文献
- **阅读重点**：第 3 节（Agent 架构分类）、第 6 节（挑战与开放问题）
- **时间分配建议**：先读摘要 + 第 3 节建立框架，再深入其他章节
- **与本模块的关系**：Agent 领域完整分类体系

**Qin et al. (2023) "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs"** — [arXiv](https://arxiv.org/abs/2307.16789)

- **一句话定位**：大规模工具调用的全链路工作
- **阅读重点**：第 3 节（ToolBench 数据构建 + ToolEval 评估）
- **时间分配建议**：关注数据构建和评估方法
- **与本模块的关系**：大规模 API 标准化描述的工程范式

**Liu et al. (2023) "AgentBench: Evaluating LLMs as Agents"** — [arXiv](https://arxiv.org/abs/2308.03688)

- **一句话定位**：第一个系统的 Agent 评估基准——8 个环境
- **阅读重点**：第 2 节（AgentBench 设计原则和环境列表）
- **时间分配建议**：关注评估框架设计而非排行榜
- **与本模块的关系**：Agent 评估标准化的先驱工作

> 📌 MCP 协议是 Anthropic 2024 年提出的开放标准（非学术论文），核心思想：工具是独立服务，通过标准协议注册和调用。

---

## 拓展阅读

- **Agent Evaluation Survey (2025)** [arXiv](https://arxiv.org/abs/2501.09434) —— Agent 评估方法综述

> 拓展论文放在各模块的 拓展/ 文件夹下。

---

## 模块间连接

- **前置依赖**：先理解 01-Agent 基础范式——工程框架是在这个基础上的基础设施
- **后续衔接**：读完后建议进入 06-安全评估与前沿挑战
- **本模块与哪些模块正交**：02-推理与规划、03-多智能体系统、04-Agent 记忆与知识