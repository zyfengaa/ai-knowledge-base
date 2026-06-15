# 06 — 安全评估与前沿挑战

## 一句话开场

> 你的 Agent 能查数据库、发邮件、操作文件——如果被注入了恶意指令，它能把自己"越狱"然后删掉你所有文件。Agent 的能力越强，造成破坏的潜力越大。这个模块讨论：怎么安全地使用 Agent？以及 Agent 领域还有什么大问题没解决？

## 正文：渐进式理解

**第一层：问题定义。** Agent 的安全问题比 LLM 更严峻。LLM 的安全风险是"输出有害文本"，而 Agent 的安全风险是**"执行有害操作"**——区别就像"说脏话"和"动手打人"。当一个 Agent 有权限调用 API、写文件、操作数据库时，Prompt 注入的后果从"输出不良内容"升级为"造成实质性破坏"。安全评估与前沿挑战模块解决的问题是：**怎么在释放 Agent 能力的同时，把风险控制在可接受范围内？以及当前还没有答案的开放问题有哪些？**

**第二层：核心直觉。** 想象你雇了一个超级能干的实习生。你给了他公司系统的所有权限——他能做的事很多，但他也可能：① 被钓鱼邮件骗（Prompt 注入）；② 自己编造数据（幻觉在工具调用中被放大）；③ 过度执行（你让他"检查一下"他删掉了整张表）。你会怎么做？限制权限、设置审批流程、审计日志、定期抽查——Agent 安全也是同样的思路。

**第三层：方案细节。** Agent 安全的四层防线：

1. **权限最小化（Least Privilege）**：Agent 只获得完成任务所需的最小权限。不读的文件不给路径，不用的 API 不给 key。关键工具需要"人类审批"的确认步骤。
2. **输入过滤与沙箱（Sandboxing）**：对用户输入进行注入检测（像防范 SQL 注入一样防范 Prompt 注入）。Agent 执行环境放在沙箱中（Docker/虚拟机），限制对主机系统的访问。
3. **行为审计（Audit Trail）**：记录 Agent 的每一步操作（决策日志 + 工具调用 + 参数 + 结果），支持事后回溯。异常行为检测（如短时间内大量文件删除）。
4. **人类在环中（Human-in-the-Loop）**：高风险操作（发送邮件、删除文件、支付）必须经人类确认。设置"安全停止"机制——当 Agent 检测到不确定性时主动征求人类意见。

**第四层：不同方案的风险权衡。**

| 风险类型 | 威胁描述 | 影响级别 | 缓解措施 |
|---------|---------|---------|---------|
| **直接 Prompt 注入** | 用户输入覆盖系统指令 | 严重（可导致任意工具调用） | 输入消毒 + 系统指令强化 + 参数化工具调用 |
| **间接 Prompt 注入** | Agent 读取的网页/文档含恶意指令 | 严重（无需直接交互即被攻击） | 内容隔离 + 上下文过滤 + 敏感操作审批 |
| **工具误调用** | Agent 在不确定时错误调用工具 | 中高（取决于工具有多"强"） | 调用确认 + 工具权限分级 + 参数校验 |
| **幻觉传播** | Agent 把幻觉结果当真并在此基础上操作 | 中（错误累积） | 事实核查 + 检索增强 + 结果验证 |
| **过度自主性** | Agent 执行了用户没想让它做的事 | 中（但责任归属困难） | 审批环 + 人工确认 + 操作范围限制 |

**一个贯穿所有安全方案的设计轴：Agent 自主性 vs. 安全约束。** 你管得越严，Agent 越安全，但它越难发挥自主解决问题的能力。找到这个平衡点是 Agent 工程中最难的非技术问题。

**第五层：总结升华。** Agent 安全是"能力"的逆向面——能力越强，安全越重要。当前领域的一个残酷现实是：Agent 安全的进展严重落后于 Agent 能力的进展。大多数研究聚焦"怎么让 Agent 更强"，很少聚焦"怎么确保 Agent 安全地强"。这个模块放在最后，既是提醒也是警告：构建 Agent 系统时，安全不是可以"以后再加"的功能——它应该从第一天就融入系统设计。而当前尚未解决的问题（长期自主评估、安全基准的缺失、幻觉在工具链中的传播）正是 Agent 领域最有价值的研究方向。

---

## 学习目标

读完你能：

- **用一句话说清 Agent 安全和 LLM 安全的区别**：LLM 安全是"说坏话"，Agent 安全是"干坏事"
- **面对一个 Agent 系统，能设计其安全防护的四层防线**：权限控制 → 输入过滤 → 行为审计 → 人类审批
- **理解 Prompt 注入在 Agent 场景中为什么比在 LLM 场景中更危险**：因为注入的后续操作可以调用工具造成实际损害
- **能指出当前 Agent 领域 3 个具体的开放问题并给出自己的见解**：长期自主性、评估体系、安全对齐
- **给出"Agent 自主度 vs. 人类可控性"的权衡建议**：什么操作可以自主，什么必须审批

---

## 精选论文

**Zhou et al. (2023) "WebArena: A Realistic Web Environment for Building Autonomous Agents"**

- **一句话定位**：当前最接近真实场景的 Agent 评估环境——在真实网站上操作评估
- **阅读重点**：第 3 节（环境设计和任务构成）。理解"评估 Agent 需要什么级别的环境模拟"
- **时间分配建议**：关注评估设计思想而非具体任务。WebArena 的贡献是"提供一个可重复的真实的 Agent 考场"
- **与本模块的关系**：Agent 评估基准的代表——回答"Agent 做得好不好"的判断标准

**Liu et al. (2024) "Prompt Injection Attacks and Defenses in LLM-Integrated Applications" (or similar survey)**

- **一句话定位**：Prompt 注入攻击与防御的综述——Agent 安全最核心的威胁模型
- **阅读重点**：攻击分类（直接/间接/多轮注入）+ 防御方案（输入过滤/权限分离/代理架构）
- **时间分配建议**：关注攻击类型分类和防御框架，具体案例可快速浏览
- **与本模块的关系**：Prompt 注入是 Agent 安全的首要威胁，理解它就是理解 Agent 安全的核心

**Wang et al. (2024) "A Survey on Large Language Model based Autonomous Agents" (Frontiers / Review)**

- **一句话定位**：2024 年的 Agent 全景综述，相比 Xi 2023 增加了安全、评估、应用等更新章节
- **阅读重点**：安全与伦理章节（第 6-7 节）、挑战与未来方向章节（第 8 节）
- **时间分配建议**：聚焦安全+开放问题部分，基础架构部分可跳过（已在 01-05 模块覆盖）
- **与本模块的关系**：帮助定位当前 Agent 领域的"已知 vs. 未知"边界

---

## 拓展阅读

- **Yang et al. (2024) "The Dawn of LLM Agents: A Comprehensive Survey"** — 另一份综合综述，安全部分的表格很有参考价值。如果前面的综述不够，可以补充看这篇的 safety 章节。
- **Ouyang et al. (2023) "The False Promise of Imitating Proprietary LLMs"** — 讨论了开源 Agent 的安全风险。虽然不是 Agent 专有论文，但对"Agent 能力的双刃剑"有深刻洞察。

> 拓展论文不移除，放在 `06-安全评估与前沿挑战/拓展/` 文件夹下。

---

## 模块间连接

- **前置依赖**：先读完 01-05 全部模块再读本模块效果最好。本模块是"收尾"——把前 5 个模块中涉及的隐患和开放问题汇总并升华
- **后续衔接**：❗ 本模块是整个 AI-Agent 体系的"终点"——学习路径到这里完成。读完后的行动计划应该是：① 回到根目录的"范式跃迁表"重新遍历一遍；② 选中一个开放问题作为研究/实践方向
- **本模块与哪些模块正交**：所有模块都和安全相关——安全不是独立模块，而是贯穿整体的横切面。本模块是"安全意识的集合和升级"

---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| A Survey of GUI Agents | GUI (2024) | [arXiv](https://arxiv.org/abs/2412.04789) |
| Safety of Large Language Models: A Survey | LLM (2024) | [arXiv](https://arxiv.org/abs/2402.04249) |
| WebArena: A Realistic Web Environment for Building Autonomous Agents | WebArena (2023) | [arXiv](https://arxiv.org/abs/2307.13854) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| The Dawn of LLM Agents | [arXiv](https://arxiv.org/abs/2403.03693) |
| A Survey of GUI Agents | [arXiv](https://arxiv.org/abs/2412.04789) |
| LLM Robotics Security Survey | [arXiv](https://arxiv.org/abs/2502.09152) |
| Safety of Large Language Models: A Survey | [arXiv](https://arxiv.org/abs/2402.04249) |
| LLM Security Survey | [arXiv](https://arxiv.org/abs/2404.01355) |
| A Survey on the Planning Capabilities of Large Language Models | [arXiv](https://arxiv.org/abs/2402.01680) |
| WebArena: A Realistic Web Environment for Building Autonomous Agents | [arXiv](https://arxiv.org/abs/2307.13854) |
