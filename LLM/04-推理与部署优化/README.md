﻿﻿# 04 — 推理与部署优化

> 一个 70B 参数的模型占用 140GB 显存——你怎么在单张 A100（80GB）上让它跑起来？

## 正文：渐进式理解

**第一层：问题定义。** LLM 推理有三大瓶颈：① **显存**——模型参数（14B≈28GB, 70B≈140GB）+ KV Cache 放不下单卡；② **延迟**——自回归解码一次只生成一个 token，长序列依次生成等不完；③ **吞吐**——服务多个用户同时推理时显存竞争严重。核心矛盾是：模型越来越大，硬件带宽跟不上。

**第二层：核心直觉。** 优化策略可以分成三类，每类对应不同的杠杆：

`
Space for Time（空间换时间）— KV Cache
   — 代价是多用显存，收益是避免重复计算
Accuracy for Speed（精度换速度）— 量化
   — 代价是少量质量损失，收益是显存减半/速度翻倍
Memory Management（内存管理）— PagedAttention
   — 效仿操作系统虚拟内存，解决显存碎片化和共享
`

**第三层：方案细节。**

**KV Cache**：生成每个 token 时只算新 Q 的注意力，之前所有 K/V 都缓存。不加 Cache 要重新算所有位置的注意力（O(n²)），加 Cache 只需一次前向（O(n) 每步）。代价：Cache 随序列长度线性增长——100K 长度的对话 ≈ 12GB K/V Cache。

**PagedAttention / vLLM**：KV Cache 不是连续存储的，而是分成固定大小的「块」像操作系统分页一样管理。好处：① 解决了显存碎片问题（不需要找连续大块内存了）；② 支持共享显存（CoT 的多条推理路径共享同一个 Prefix 的 K/V）。这是 vLLM 的核心技术。

**GPTQ**：把模型权重从 16-bit 压缩到 4-bit，本质是逐层做「量化的误差补偿」——不是简单地四舍五入，而是对每一层的量化误差做二次优化（Optimal Brain Quantization 的近似）。结果：70B 模型从 140GB 降到 35GB，一张 A100 就能跑。

**第四层：不同方案的权衡。**

量化方法对比：

| 方法 | 位数 | 质量损失 | 是否需要数据 | 速度提升 | 代表框架 |
|------|------|---------|------------|---------|---------|
| GPTQ | 4-bit / 3-bit | ⚠️ 极小 | 需校准集 | 2-3x | vLLM, AutoGPTQ |
| AWQ | 4-bit | ⚠️ 极小 | 需校准集 | 2-3x | TensorRT-LLM |
| SmoothQuant | 8-bit W + 8-bit A | ✅ 几乎无损 | 需校准集 | 推理快 1.5x | PyTorch |
| GGUF 量化 | 2-8 bit 可选 | ⚠️ 可控 | 不需数据 | 灵活 | llama.cpp |

推理引擎对比：

| 引擎 | 核心优势 | 适用场景 | 支持的量化 |
|------|---------|---------|-----------|
| **vLLM** | PagedAttention + 连续批处理，吞吐最高 | 在线服务（高并发） | GPTQ / AWQ |
| **TensorRT-LLM** | NVIDIA 官方优化，FP8 原生支持 | 生产部署（性能极致） | FP8 / INT4 / INT8 |
| **llama.cpp** | CPU + Apple Silicon 友好 | 本地运行 / 边缘设备 | GGUF |
| **MLC-LLM** | 跨平台（Web / 手机 / GPU） | 多端部署 | 自定义 |

**第五层：总结升华。** 推理优化是「性价比游戏」——用最小的质量损失换取最大的效率提升。当前推理的终极瓶颈不是计算而是**显存带宽**（Memory-bound），所有优化策略（量化 / KV Cache 管理 / 分页）都在最小化「从显存读数据」的成本。理解了这个根本约束，就能理解为什么 GPTQ 选 4-bit 而不是 3-bit——因为再往下质量损失曲线陡峭上升。

---

## 学习目标

读完你能：

- 用自己的话解释 KV Cache 为什么能省计算（不用重复算之前的注意力）以及它消耗多少显存
- 用一句话说清 PagedAttention 和操作系统虚拟内存的类比（分页管理 KV Cache，解决碎片 + 共享）
- 理解 GPTQ 4-bit 量化在质量几乎不变的前提下将显存降低到 1/4 的原理
- 面对部署场景能给出引擎选择建议（高并发→vLLM，极致性能→TensorRT-LLM，本地运行→llama.cpp）
- 解释为什么 LLM 推理是「带宽瓶颈」而非「计算瓶颈」——访存算比（Byte/FLOP）是关键指标

---

## 精选论文

**Kwon et al. (2023) "Efficient Memory Management for Large Language Model Serving with PagedAttention" [[arXiv](https://arxiv.org/abs/2309.06180)]**

- **一句话定位**：vLLM 的核心论文，PagedAttention 类虚拟内存管理，LLM 推理的事实标准
- **阅读重点**：第 3 节（PagedAttention 的分块机制 + Figure 2 的调度对比）和第 5 节（实验——吞吐提升 2-4x）
- **时间分配建议**：时间紧只读第 3 节理解「怎么分块管理 KV Cache」+ 看 Figure 2 的对比图；时间充裕精读第 4 节（与操作系统的类比）

**Frantar et al. (2023) "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" [[arXiv](https://arxiv.org/abs/2210.17323)]**

- **一句话定位**：权重量化的主流方法，4-bit 量化几乎不损失质量
- **阅读重点**：第 3 节（OBC → GPTQ 的扩展——逐层量化 + Hessian 矩阵做误差补偿）
- **时间分配建议**：公式推导（Section 3.1-3.2）可以跳读，核心是理解「量化不是直接舍入而是做误差补偿」

---

## 模块间连接

- **前置依赖**：建议先读 **01-Transformer 起源**（理解注意力计算和 O(n²) 复杂度来源）。本模块的 KV Cache 和 PagedAttention 都依赖对注意力的理解
- **后续衔接**：读完本模块后推荐进入 **05-应用技术**——理解部署好之后怎么让模型在真实任务中工作
- **本模块与哪些模块正交**：与 03-训练与对齐范式完全正交——训练和部署是两个独立环节


---

## 论文参考

| 论文 | 作者(年份) | 链接 |
|---|---|---|
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | GPTQ () | [arXiv](https://arxiv.org/abs/2210.17323) |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | PagedAttention () | [arXiv](https://arxiv.org/abs/2309.06180) |

---

## 论文参考

| 论文 | 链接 |
|---|---|
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | [arXiv](https://arxiv.org/abs/2210.17323) |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | [arXiv](https://arxiv.org/abs/2309.06180) |
