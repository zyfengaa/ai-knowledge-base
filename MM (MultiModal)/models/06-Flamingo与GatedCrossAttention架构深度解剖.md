# Flamingo 架构深度解剖

> DeepMind (NeurIPS 2022) | "Flamingo: a Visual Language Model for Few-Shot Learning" —— 连接器式 VLM 的鼻祖——80B 参数证明了"冻结 LLM + 轻量桥接"路线的可行性，Gated Cross-Attention 的设计影响深远

---

## 写在前面：为什么 Flamingo 如此关键

2022 年的多模态领域格局与 2023-2024 年完全不同：

```
2022 年背景:
  - CLIP 刚发布（2021）—— 提供了可用的视觉编码器
  - GPT-3（175B）刚出不久 —— LLM 的巨大潜力已显现
  - 但没人知道"怎么把两者接起来"
  
当时的"常识":
  - 微调 80B LLM 太贵（成本 ~$10M+）
  - 必须想办法用"冻结模型"做多模态
  
Flamingo 的思路:
  "LLM 很强了，不要碰它。ViT 也很好了，也不要碰它。
  在它们之间加一个'可学习的桥'，让桥学会'翻译'视觉信号给 LLM。"
```

**Flamingo 的核心突破：**

| 维度 | 意义 |
|------|------|
| **第一个 Frozen LLM + Frozen ViT 方案** | 证明了"冻结一切，只训桥接层"的可行性 |
| **Gated Cross-Attention** | 在 LLM 层之间插入可训练的 Cross-Attention，用 tanh gating 实现"渐进激活" |
| **Perceiver Resampler** | 变长视觉 token → 固定数量（64 token），简化 LLM 的处理 |
| **交错图文序列** | 支持多图、多轮、图文混合输入（后续 VLM 都学了这个范式） |
| **Few-shot 能力强** | 通过上下文学习（In-Context Learning）实现零/少样本任务迁移 |

**Flamingo 的影响：** BLIP-2 的 Q-Former 可以看作"Perceiver Resampler + Frozen LLM"路线的简化版。ControlNet 的 Zero Convolution 和 Flamingo 的 Gating 机制有异曲同工之妙——都用了"从零开始"的渐进式注入。没有 Flamingo，2023 年的 VLM 爆发可能要晚半年。

---

## 一、整体设计理念

### 1.1 核心问题与方案

```
问题: 如何让一个 Frozen LLM 理解图像，而不改变它的任何参数？

Flamingo 的答案:
  ① 不把视觉信息"提前压缩"再输入 LLM
  ② 而是在 LLM 的每一层 Transformer block 中"注入"视觉信息
  ③ 用可训练的 Cross-Attention 层让 LLM 层"看"到图像
  
  视觉信息注入的位置:
    原始的 LLM block: [ Self-Attn → FFN ]
    加了 Flamingo 后的 block: [ Self-Attn → **GATED XATTN-DENSE** → FFN ]
                                          ↑
                              LLM 的每一层都能看到图像
```

### 1.2 Flamingo 家族

```
Flamingo (2022.11, DeepMind):
  参数量: 80B（ViT + Frozen LLM + 桥接层）
  视觉编码器: Frozen NFNet-F6（DeepMind 自己的 ViT 变体）
  LLM: Frozen Chinchilla 70B（DeepMind 自家 LLM）
  训练数据: 2.1B 图文对（M3W 数据集）
  核心创新: Gated Cross-Attention + Perceiver Resampler

Flamingo-9B (2022.11, DeepMind):
  参数量: 9B（同架构的缩小版）
  用途: 研究 scaled 规律、开源基线
```

---

## 二、Flamingo 架构解剖

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Flamingo 整体架构                            │
│                                                                  │
│  图像输入                                                         │
│    │                                                             │
│    ▼                                                             │
│  ┌──────────────────────┐                                        │
│  │  Vision Encoder      │ ← Frozen (NFNet-F6 / ViT)               │
│  │  (Normalizer-Free    │                                          │
│  │   ResNet / ViT)      │                                          │
│  └──────────┬───────────┘                                        │
│             │ 变长视觉特征（~289 token/图）                        │
│             ▼                                                     │
│  ┌──────────────────────┐                                        │
│  │  Perceiver Resampler │ ← 可训练                                │
│  │  (变长 → 固定 64     │                                          │
│  │  个视觉 token)       │                                          │
│  └──────────┬───────────┘                                        │
│             │ 64 个"精炼"的视觉 token                              │
│             │                                                       │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │              Frozen LLM (Chinchilla 70B)                 │     │
│  │                                                          │     │
│  │  Layer 1:                                                │     │
│  │    Self-Attention → GATED XATTN-DENSE → FFN             │     │
│  │                          ↑                               │     │
│  │              64 视觉 token 通过 Cross-Attn               │     │
│  │              注入到 LLM 的 hidden states                 │     │
│  │                                                          │     │
│  │  Layer 2:                                                │     │
│  │    Self-Attention → GATED XATTN-DENSE → FFN             │     │
│  │                          ↑                               │     │
│  │  ... 每层都重复                                          │     │
│  │                                                          │     │
│  │  Layer L: 同样的结构                                     │     │
│  │                                                          │     │
│  └─────────────────────────────────────────────────────────┘     │
│             │                                                     │
│             ▼                                                     │
│  文本输出（Frozen LLM 的原本输出头）                                │
└─────────────────────────────────────────────────────────────────┘
```

| 组件 | 功能 | 训练状态 |
|------|------|---------|
| **Vision Encoder** | NFNet-F6，提取图像特征 | ❌ Frozen |
| **Perceiver Resampler** | 变长视觉特征 → 固定 64 token | ✅ 可训练 |
| **Gated Cross-Attention** | 在 LLM 每层注入视觉信息 | ✅ 可训练（关键） |
| **LLM Backbone** | Chinchilla 70B | ❌ Frozen |

### 2.2 Perceiver Resampler

Flamingo 需要解决一个问题：视觉特征的数量是**变的**（不同图像、不同分辨率输出的 token 数不同），但 LLM 需要**固定的**输入格式。

```
Perceiver Resampler 的设计:

  输入: N 个视觉特征（N ≈ 289，根据图像分辨率变化）
         来自 ViT 的 patch token
         
  可学习的 latent query: 64 个 query token（可训练）
         每个 query 维度 = LLM 的 hidden dim（如 8192）
         
  处理流程:
    query ∈ ℝ^{64×d}    ← 可学习的 latent array
    visual ∈ ℝ^{N×d}    ← Vision Encoder 输出
    
    for _ in range(num_layers):
        query = query + CrossAttention(Q=query, K=visual, V=visual)
        query = query + MLP(query)
        query = query + SelfAttention(Q=query, K=query, V=query)
    
    输出: 64 个"精炼"的视觉 token
    
  作用:
    - 将变长输入 → 固定长度（简化 LLM 处理）
    - 通过 Cross-Attention 从视觉特征中"提取"最相关信息
    - 通过 Self-Attention 让 64 个 query 之间交互（避免冗余）
```

**与 BLIP-2 Q-Former 的对比：**

```
Perceiver Resampler (Flamingo):
  输入: visual features → Cross-Attn → 64 query
  训练: 和 Gated Cross-Attention 一起端到端
  位置: 在视觉编码器之后、LLM 注入之前

Q-Former (BLIP-2):
  输入: visual features → Cross-Attn → 32 query
  训练: 两阶段（先表示学习，再生成学习）
  位置: 在视觉编码器之后、LLM 之前
  
核心差异:
  - Flamingo: query 输出注入到 LLM 每层的 Cross-Attn
  - BLIP-2: query 输出作为 LLM 的输入 embedding（一次性）
  
  所以 Flamingo 的"桥接"更深入——视觉信息在 LLM 的每一层都重新注入
```

### 2.3 Gated Cross-Attention（核心创新）

Flamingo 最重要的设计——如何在 Frozen LLM 中插入视觉信息。

```
标准 Cross-Attention（如果直接插入 Frozen LLM）:
  x' = x + CrossAttn(Q=x, K=visual, V=visual)
  
  问题: Cross-Attn 初始权重是随机的
        → 刚插入时输出完全随机
        → 打破了 Frozen LLM 的原有分布
        → 训练初期模型输出混乱，难以收敛

Flamingo 的 Gated Cross-Attention:
  x' = x + tanh(α) × CrossAttn(Q=x, K=visual, V=visual)
  
  其中 α 是可训练参数，初始化为 0
  → tanh(0) = 0 → 初始时 Cross-Attn 的输出被完全屏蔽
  → 训练过程中 α 逐渐增大 → Cross-Attn 的效果"渐进"注入
  → LLM 的原始分布从未被"打断"
```

**Gating 的工作可视化：**

```
训练开始时（α=0）:
  x' = x + tanh(0) × CrossAttn(...) = x + 0 × (...) = x
  → LLM 的输出和没有 Flamingo 时完全一样
  → 视觉信息完全没有注入
  → LLM 保持原有行为

训练中期（α 逐渐增大）:
  x' = x + tanh(α) × CrossAttn(...)
  → Cross-Attn 的输出"慢慢"影响 LLM
  → LLM 有时间"适应"视觉信息的存在

训练完成（α 稳定在某值）:
  x' = x + tanh(α_final) × CrossAttn(...)
  → α_final 通常 ~1-3
  → tanh(α_final) ≈ 0.76-0.995
  → Cross-Attn 完全激活

类比:
  这就像"渐入"的音频效果——不是"啪"地一下打开
  而是从静音慢慢推到预期音量
```

**与 ControlNet Zero Convolution 的对比：**

```
ControlNet (2023.02):
  在 U-Net 的 decoder block 中注入 ControlNet 条件
  使用 Zero Convolution（初始化为 0 的卷积层）
  y = x + ZeroConv(control_feature)
  → 训练初期控制信号为 0 → U-Net 保持原样

Flamingo (2022.11):
  在 LLM block 中注入视觉特征
  使用 tanh gating（tanh(α)，α 初始化为 0）
  x' = x + tanh(α) × CrossAttn(...)
  → 训练初期视觉注入为 0 → LLM 保持原样

异曲同工:
  两者都解决了"如何在不破坏预训练模型的前提下注入新信息"
  都是把新增模块的初始输出"归零"，让主模型"渐进适应"
  
  区别:
    ControlNet: 用零初始化的卷积层
    Flamingo: 用可学习标量 × tanh 门控
```

### 2.4 交错图文序列（Interleaved Text-Image）

Flamingo 提出了**交错图文序列**的训练范式——文本和图像在序列中交替出现：

```
训练序列格式:

[BOS] text_1 [IMG] vision_64_tokens [SEP] text_2 [IMG] vision_64_tokens [SEP] text_3 [EOS]

每个 [IMG] 标记后面跟着 64 个视觉 token（Perceiver Resampler 的输出）

损失计算:
  ❌ 不计算视觉 token 的损失
  ✅ 只计算文本 token 的损失（语言建模）
  → 视觉 token 作为"上下文条件"存在

推理时:
  输入: [IMG] vision_64_tokens [SEP] "What is in this image?"
  输出: "A cat wearing a suit."
  → 模型在文本部分的 next-token prediction
  → 视觉信息通过 Gated Cross-Attention 注入
```

**这个设计使 Flamingo 天然支持多图输入：**

```
单图问答:
  [IMG] img_A [SEP] "Describe this image." → "A cat..."

多图比较:
  [IMG] img_A [SEP] [IMG] img_B [SEP] 
  "What's the difference between these two images?"
  → 模型通过注意力机制同时"看"两张图

图文交错:
  [IMG] img_A [SEP] "This is a cat." [IMG] img_B [SEP] "This is a dog."
  "Which image contains an animal?" → "Both."
  
In-Context Few-shot:
  [IMG] img_1 [SEP] "A photo of a cat." 
  [IMG] img_2 [SEP] "A photo of a dog."
  [IMG] test_img [SEP] "A photo of a ___"
  → 模型根据前两个示例"学会"了少样本图文匹配！
```

---

## 三、训练策略

### 3.1 训练数据：M3W

Flamingo 的训练数据是 DeepMind 从网页中爬取的：

```
M3W (Multi-Modal Massive Web) 数据集:
  规模: 2.1B 图文对（当时最大的多模态数据集之一）
  来源: 网页爬取（HTML 页面中的图文对 + 页面内的文本）
  格式: 交错图文序列（保留原始页面中的图文排列顺序）
  
  关键: 
    - 不是"图像"和"文本"独立存在
    - 而是保留"文本 → 图像 → 更多文本"的交错结构
    - 让模型学到的是"真实网页中的图文关系"
  
  对比:
    LAION (CLIP 用): 纯图文对，图像和文本是一一对应的
    M3W (Flamingo 用): 交错图文序列，更接近真实阅读体验
```

### 3.2 训练阶段：两阶段

```
Phase 1: Vision-Language Alignment
  目标: 训练 Perceiver Resampler 学会抽取视觉特征
  
  冻结: Vision Encoder, LLM
  训练: Perceiver Resampler + Gated Cross-Attention
  
  数据: M3W 子集（2.1B 中的一部分）
  
  注意: 
    Gated Cross-Attention 的 gating 参数 α 从 0 开始
    → 训练初期视觉信息几乎不注入
    → 训练过程中逐渐激活

Phase 2: End-to-End Fine-tuning
  目标: 全模型（除 Frozen 组件外）精调
  
  冻结: Vision Encoder, LLM
  训练: Perceiver Resampler + Gated Cross-Attention
  
  数据: M3W 全量 + 下游任务数据
  
  包括:
    - VQA（视觉问答）
    - Captioning（图像描述）
    - Few-shot 评估
```

### 3.3 训练配置

```
Flamingo 80B 的训练规模:

  Vision Encoder: NFNet-F6（~0.6B 参数，Frozen）
  LLM: Chinchilla 70B（Frozen）
  可训练参数: Perceiver Resampler + Gated Cross-Attention
              总计 ~10B 参数（主要是 Gated Cross-Attention 的投影矩阵）
  
  GPU: 512 块 TPUv4
  训练时间: ~15 天
  数据: 2.1B 图文对
  Batch size: 1,024（图文对角度）| 序列角度：每个序列 256 个 token
  
  Optimizer: AdamW
  Learning rate: 3e-4（可训练部分）
  Warmup: 5,000 steps
  Precision: bfloat16
```

---

## 四、Flamingo 的关键能力

### 4.1 Few-Shot 多模态学习

Flamingo 最令人印象深刻的能力是**通过上下文学习做多模态任务**：

```
零样本（Zero-shot）:
  [IMG] img_test [SEP] "What is this?"
  → 模型输出: "A cat"
  （依靠 LLM 原有的语言知识 + 视觉信息的 Cross-Attn）

单样本（One-shot）:
  [IMG] img_1 [SEP] "What is this?" [SEP] "A dog"
  [IMG] img_2 [SEP] "What is this?" [SEP] "A cat"
  [IMG] img_test [SEP] "What is this?"
  → 模型输出: "A bird"
  （通过 2 个示例，模型学会了"回答图像内容"的任务格式）

少样本（Few-shot, 4 shot）:
  [IMG] img_1 [SEP] "Q: What is this? A: Dog" 
  [IMG] img_2 [SEP] "Q: ... A: Cat"
  [IMG] img_3 [SEP] "Q: ... A: Bird"
  [IMG] img_4 [SEP] "Q: ... A: Fish"
  [IMG] test [SEP] "Q: What is this? A:"
  → 模型输出: "Monkey"
```

**效果（图像描述任务，COCO Caption）：**

| 方法 | CIDEr | 说明 |
|------|-------|------|
| Flamingo-80B (4-shot) | 61.5 | 4 个图文示例就能达到接近 SOTA |
| Flamingo-80B (zero-shot) | 58.4 | 零样本也很有竞争力 |
| 当时 SOTA（SimVLM） | 58.5 | 全参数微调的监督模型 |
| BLIP-2 (FLAN-T5, 微调后) | 62.1 | 需要下游微调 |

→ Flamingo 80B **4-shot 就超过了一众全参数微调的模型**。

### 4.2 视觉问答（VQA）

```
Flamingo 在 VQAv2 上的表现:

  Flamingo-80B (4-shot): 82.1
  Flamingo-80B (zero-shot): 77.3
  BLIP-2 (FLAN-T5, 微调后): 82.2
  PaLI-17B (微调后): 85.6
  GPT-4V (2023): 未公开 VQA 指标
  
  结论:
    - Flamingo 80B 4-shot 几乎追平了 BLIP-2 的微调效果
    - 说明"Frozen LLM + 少样本"路线很有竞争力
    - 但离全参数微调（PaLI）还有差距 → 微调路线不会消失
```

---

## 五、Flamingo 与后续模型的对比

### 5.1 Flamingo vs BLIP-2 vs LLaVA

```
Flamingo (2022.11):
  架构: Frozen ViT → Perceiver Resampler (64 token) 
        → 在 Frozen LLM 每层插入 Gated Cross-Attention
  可训练: ~10B（桥接层 + Cross-Attn）
  LLM: Frozen ✅
  效果: 强（Few-shot 接近微调）

BLIP-2 (2023.01):
  架构: Frozen ViT → Q-Former (32 token) 
        → 作为 LLM 的输入 embedding（一次性）
  可训练: ~188M（Q-Former 全部）
  LLM: Frozen ✅
  效果: 中等（强于简单映射，弱于全参数微调）

LLaVA (2023.04):
  架构: Frozen ViT → MLP Projector (576 token)
        → 拼入 LLM 的输入序列
  可训练: ~7B（MLP + LLM 全参数微调）
  LLM: 全参数微调 ❌
  效果: 强（微调 LLM 就是更好）
```

| 维度 | Flamingo | BLIP-2 | LLaVA |
|------|---------|--------|-------|
| **视觉 token 数** | 64（Perceiver） | 32（Q-Former） | 576（全量） |
| **LLM 状态** | Frozen | Frozen | 全参数微调 |
| **桥接方式** | 层间 Cross-Attn | 输入层 embedding | 输入层 token 拼接 |
| **训练成本** | 极高（80B 规模） | 极低（188M） | 中等（7B 微调） |
| **灵活换 LLM** | ❌（不能换） | ✅（重训 Phase 2） | ❌（重训全部） |
| **Few-shot 能力** | ✅ 强 | ⚠️ 中等 | ❌ 弱 |

**一个关键洞察：**

```
Flamingo 的"层间注入" vs LLaVA 的"输入层拼接"

Flamingo:
  视觉信息在每一层都注入 → 每层都能"看"到图像
  但代价: 需要修改 LLM 架构（加 Cross-Attn 层）
  → 换 LLM 必须重新设计 Cross-Attn 插入

LLaVA:
  视觉信息只在输入层拼接一次 → LLM 自己决定在哪层关注
  优势: 不需要修改 LLM 架构
  → 换 LLM 只需要改 MLP 的维度
  → 这也是 LLaVA 路线最终胜出的关键原因之一
```

### 5.2 为什么 Flamingo 没有成为主流

```
Flamingo 80B 在 2022 年底让人惊艳，但 2023 年被快速超越：

原因 1: 成本太高
  - 80B 模型，512 TPUv4 训 15 天
  - 只有 DeepMind/Google 级的大厂能训
  - 社区无法复现

原因 2: LLaVA 更简单
  - LLaVA 证明"不需要层间 Cross-Attn"
  - 输入层拼接就够了，前提是微调 LLM
  - 微调 LLM 虽然贵，但 7B/13B 的开源 LLM 让这变得可行

原因 3: Frozen LLM 的上限
  - Flamingo 和 BLIP-2 都发现 Frozen LLM 有上限
  - 微调 LLM（LLaVA）的效果始终更好
  - 2023 年后趋势: 微调 LLM > 冻结 LLM

Flamingo 的历史地位:
  - 是"Frozen LLM 路线"的巅峰之作
  - 证明了 Gated Cross-Attention 的设计有效
  - 为后续模型（BLIP-2、LLaVA）提供了重要参考
  - Gating 思想和 ControlNet 的 Zero Conv 异曲同工
```

---

## 六、Flamingo 的局限

| 局限 | 表现 | 原因 |
|------|------|------|
| **架构非通用** | 只能与特定 LLM 搭配 | Gated Cross-Attn 需要修改 LLM 架构 |
| **训练成本高** | 512 TPUv4 × 15 天 | 80B + ~10B 可训练参数 |
| **Frozen LLM 上限** | 达不到微调 LLM 的效果 | 语言理解能力受限于 LLM 原始能力 |
| **视觉 token 瓶颈** | 64 个 token 编码信息有限 | Perceiver Resampler 的压缩比高 |
| **分辨率限制** | NFNet 固定分辨率 | 不支持高分辨率图像 |
| **中文支持弱** | 主要在英文上训练 | 数据来源（M3W）以英文为主 |
| **未开源训练代码** | 只有推理代码 | 社区无法复现训练过程 |

---

## 七、总结

> **Flamingo 是连接器式 VLM 的开创者——它用 Gated Cross-Attention + Perceiver Resampler 证明了"冻结 LLM + 桥接层"这条路是可行的。虽然这条路线在 2024 年被 LLaVA 的"微调 LLM"路线取代，但 Flamingo 的许多设计（Gating、Perceiver、交错图文序列）已经融入了后续所有 VLM 的血脉。**

| 维度 | Flamingo 的定位 |
|------|---------------|
| **历史位置** | 2022 年—连接器式 VLM 的鼻祖，DeepMind 的巅峰之作 |
| **技术贡献** | Gated Cross-Attention（渐进激活）、Perceiver Resampler（变长→固定）、交错图文序列 |
| **与后续关系** | BLIP-2 简化了 Flamingo 的"层间注入"为"输入层 Q-Former"；LLaVA 进一步简化为 MLP + 微调 LLM |
| **跨领域影响** | Gating 思想影响了 ControlNet 的 Zero Convolution；Perceiver Resampler 影响了后续的 QR-Former、C-Abstractor |
| **未竟之业** | 太贵 + 架构不通用 + Frozen LLM 上限，促使开源社区走向 LLaVA 路线 |

> 一句话：**Flamingo 是 VLM 领域的"拓荒者"——它证明了方向可行，但价格贵到只有 DeepMind 用得起。LLaVA 是"普惠者"——用 Flamingo 的终点作为起点，把成本降了几个数量级。**

---

**Sources:**
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) — Alayrac et al., DeepMind 2022
- [Chinchilla: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Hoffmann et al., DeepMind 2022
- [Scaling Vision Transformers](https://arxiv.org/abs/2106.04560) — NFNet / ViT scaling, DeepMind 2021
- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) — Li et al. 2023
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) — Liu et al. 2023
- [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — Zhang et al. 2023
- [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03806) — Jaegle et al., DeepMind 2021
