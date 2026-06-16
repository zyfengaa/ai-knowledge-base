# 02 — 推理与规划

## 一句话开场

> 你让 Agent 订一张三天后从北京到上海的机票，它需要：查日历 → 确定日期 → 搜航班 → 比价格 → 下单 → 确认——这不是"调用一个 API"能解决的。当任务需要多步推理和规划时，Agent 的大脑怎么工作？

## 正文：渐进式理解

**第一层：问题定义。** 基础 Agent 能调用工具，但只能处理单步任务（"查天气"、"算个数学"）。现实任务往往是多步的：需要计划、需要分支、需要从错误中恢复。推理与规划模块解决的问题是：**Agent 怎么把复杂目标拆解成可执行的步骤序列，并在执行过程中自适应调整？**

**第二层：核心直觉。** 想象你要从北京去一个陌生城市。你可以：① 直接走（CoT：一条路走到黑）；② 在岔路口同时试几条路（ToT：多路径搜索）；③ 走错路后记下来"此路不通"再换（Reflexion：反思纠错）。三种策略对应三种推理深度，越往后越可靠但越费时。

**第三层：方案细节。** 推理增强有三个关键方向：① **多路径搜索（ToT）**——把 CoT 的单线推理扩展为树搜索，通过 BFS/DFS 选择最有希望的路径；② **自我反思（Reflexion）**——Agent 执行失败后先"反思"失败原因，把反思结果写入记忆再重试；③ **先规划再执行（Plan-and-Solve）**——执行前让 LLM 生成分步计划再按计划执行。

**第四层：不同方案的权衡。**

| 方法 | 代表工作 | 适用场景 | 成本 | 可靠性 |
|------|---------|---------|------|--------|
| **CoT**（思维链） | Wang 2022 | 数学、常识推理 | 低 | 中 |
| **CoT-SC**（多链采样） | Wang 2022 | 高可靠性单步推理 | 中（N 次采样） | 中高 |
| **ToT**（树搜索） | Yao 2023 | 搜索、规划博弈 | 高（分支×深度） | 高 |
| **Reflexion**（反思循环） | Shinn 2023 | 编码、决策、对话 | 中（额外调用） | 高 |
| **Plan-and-Solve** | Wang 2023 | 多步骤流程化任务 | 低（1 次规划） | 中 |
| **LLM+P**（外部规划器） | Liu 2023 | 确定性规划任务 | 低（调用求解器） | 高 |

**第五层：总结升华。** 推理与规划是 Agent 从"玩具"到"工具"的关键跨越。没有推理增强的 Agent 只能执行"单步指令"。值得注意的是——大多数生产 Agent 至今仍只用最基础的 CoT。理解这些高级方法的价值更多在于：① 知道"天花板在哪"；② 在需要高可靠性的场景能用出来。

---

## 学习目标

读完你能：

- 用一句话说清 CoT、CoT-SC、ToT 三者的递进关系：CoT 是单线推理，CoT-SC 是多次采样投票，ToT 是树搜索 + 节点评估
- 面对一个新任务，能判断该用哪种推理策略：有明确评估标准？→ ToT。允许试错？→ Reflexion。流程化任务？→ Plan-and-Solve
- 画出一个 Reflexion 的数据流图：Task → Attempt → Error → Reflection → Memory → Improved Attempt
- 理解为什么大多数生产 Agent 只用 CoT 而非 ToT：成本 vs. 收益的权衡
- 实现一个简化版的 ToT 搜索：生成候选 → 评估 → 选择 → 扩展 → 回溯

---

## 精选论文

**Wang et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"** — [arXiv](https://arxiv.org/abs/2201.11903)

- **一句话定位**：CoT 的提出——让 LLM 输出推理步骤而非直接给出答案
- **阅读重点**：第 3 节（CoT 提示设计 + 示例）
- **时间分配建议**：关注 CoT 的设计模式而非实验数据
- **与本模块的关系**：CoT 是所有后续推理增强工作的基础

**Wang et al. (2022) "Self-Consistency Improves Chain of Thought Reasoning in Language Models"** — [arXiv](https://arxiv.org/abs/2203.11171)

- **一句话定位**：CoT-SC——多次采样 + 投票，用计算量换取可靠性
- **阅读重点**：第 2 节（Self-Consistency 方法）
- **时间分配建议**：10 分钟即可读完。核心贡献在直觉
- **与本模块的关系**：第一个"用多次调用换可靠性"的方法

**Yao et al. (2023) "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"** — [arXiv](https://arxiv.org/abs/2305.10601)

- **一句话定位**：把推理从"单线"扩展为"树搜索"，加入评估和回溯
- **阅读重点**：第 2 节（ToT 框架）、第 4 节（24 点游戏案例）
- **时间分配建议**：建议精读。附录搜索过程可视化很有帮助
- **与本模块的关系**：推理增强的代表性工作

**Shinn et al. (2023) "Reflexion: An Autonomous Agent with Dynamic Memory and Self-Reflection"** — [arXiv](https://arxiv.org/abs/2303.11366)

- **一句话定位**：Agent 从失败中反思并自我改进的闭环框架
- **阅读重点**：第 3 节（Reflexion 框架：Actor → Evaluator → Self-Reflection → Memory）
- **时间分配建议**：第 3 节精读，实验部分略读
- **与本模块的关系**：推理增强 + 记忆系统的交叉点

---

## 拓展阅读

- **Wang et al. (2023) "Plan-and-Solve Prompting"** [arXiv](https://arxiv.org/abs/2305.04091) —— 如果你需要在生产中用最简单的推理增强方案，可以看这篇
- **Liu et al. (2023) "LLM+P: Empowering Large Language Models with Optimal Planning"** [arXiv](https://arxiv.org/abs/2304.11477) —— 用传统 AI 规划器补充 LLM 的规划能力

> 拓展论文放在各模块的 拓展/ 文件夹下。

---

## 模块间连接

- **前置依赖**：先理解 01-Agent 基础范式的 ReAct 循环——推理增强是在 ReAct 基础上添加更聪明的"思考"机制
- **后续衔接**：读完后建议进入 04-Agent 记忆与知识（Reflexion 中的记忆系统在这里深入展开）
- **本模块与哪些模块正交**：03-多智能体系统、05-工程框架与协议