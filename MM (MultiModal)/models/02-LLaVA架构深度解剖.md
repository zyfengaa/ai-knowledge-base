# LLaVA 架构深度解剖

> UW-Madison / Microsoft (NeurIPS 2023) | "Visual Instruction Tuning" —— 开源 VLM 的事实标准，用最简架构实现了最强性价比

---

## 写在前面：为什么 LLaVA 如此重要

2023 年初，多模态领域的路线是这样的：

| 模型 | 架构复杂度 | 训练成本 | 效果 |
|------|-----------|---------|------|
| Flamingo (DeepMind) | GATED Cross-Attn + Frozen LLM | 极高（80B 参数）| ✅ 好 |
| BLIP-2 (Salesforce) | ViT + Q-Former + Frozen LLM | 中等（只训 Q-Former）| ✅ 好 |
| GPT-4 (OpenAI) | 未知（推测极大）| 极高 | ✅ 极好 |
| LLaVA (UW-Madison) | **ViT + MLP + LLM** | **极低** | ✅ 接近 GPT-4V |

**LLaVA 的核心洞察：不需要复杂的 Q-Former、不需要 Cross-Attention、不需要门控机制。一个两层 MLP 把 CLIP ViT 的输出映射到 LLM 的输入空间就够了。**

| 维度 | LLaVA v1 | LLaVA-1.5 |
|------|---------|-----------|
| ViT | CLIP ViT-L/14（224²）| CLIP ViT-L/14（336²）|
| Projector | Linear（单层）| **MLP（2层）** |
| LLM | Vicuna 7B/13B | Vicuna/LLaMA 7B/13B |
| 预训练数据 | 558K（CC-595K）| 558K |
| 指令数据 | 150K（LLaVA-Instruct）| 665K（150K + 学术 VQA）|
| 训练硬件 | 8×A100 | 8×A100 |
| 总训练时间 | ~24h | ~48h |
| 效果 | GPT-4 的 85.1% | 开源 SOTA |

---

## 一、整体设计理念

### 1.1 和 BLIP-2 的根本差异

```
BLIP-2（Q-Former 路线）:
  图像 → Frozen ViT → Q-Former（32 query token）→ Frozen LLM
                          ↑
               query 从 ViT 中"提取"信息
               → 信息经过压缩（32 token bottleck）
               
LLaVA（MLP Projector 路线）:
  图像 → Frozen ViT → MLP（576 token 全保留）→ LLM
                          ↑
               ViT 输出的所有 token 都送进 LLM
               → 没有压缩、没有瓶颈
               → LLM 自己处理所有视觉信息
```

**为什么 LLaVA 的极简方案能工作？**

```
Q-Former 的设计假设: LLM 需要"精选"的视觉信息
  → 32 个 query token 已经足够

LLaVA 的反直觉假设: LLM 可以自己处理原始视觉 token
  → 576 个 token 全送进去，LLM 自己决定关注什么
  → LLM 足够大（7B+ params）→ 不需要预处理

结果发现: LLaVA 是对的。LLM 的自注意力机制天然可以处理
          576 个额外 token，不需要专门的"信息筛选器"。

为什么要等 LLaVA 才发现这个？
  因为 BLIP-2 和 Flamingo 的 LLM 是冻结的
  → 冻结时确实需要 Q-Former 做信息提取
  → LLaVA 微调了整个 LLM → LLM 自己学会了处理视觉 token
```

### 1.2 LLaVA 的设计哲学

```
LLaVA 的三句话设计哲学:

① "Visual tokens are just additional language tokens"
   视觉 token 不需要特殊对待——它们就是 LLM 输入序列的一部分
    
② "A single linear/MLP layer is sufficient for alignment"
   对齐器不需要很复杂——ViT 和 LLM 的表示空间已经接近
    
③ "Instruction tuning works for multimodal too"
   GPT-4 能生成高质量指令数据 → 用 GPT-4 教 LLaVA 看图文
```

---

## 二、LLaVA 架构解剖

### 2.1 整体架构

```
┌──────────────────────────────────────────────────────────┐
│                       LLaVA                               │
│                                                           │
│  图像                                                    │
│    │                                                     │
│    ▼                                                     │
│  ┌─────────────────┐                                     │
│  │ CLIP ViT-L/14   │  ← Frozen（权重固定）                  │
│  │ (224² or 336²)  │                                      │
│  └────────┬────────┘                                     │
│           │ 256/576 个 patch token + 1 [CLS]              │
│           ▼                                               │
│  ┌─────────────────┐                                     │
│  │ MLP Projector   │  ← 可训练（线性层或 2 层 MLP）          │
│  │ (Linear / MLP)  │                                      │
│  └────────┬────────┘                                     │
│           │ 256/576 个"视觉 token"，映射到 LLM 的 emb dim    │
│           ▼                                               │
│  ┌────────────────────────────────────────────┐           │
│  │               LLM (Vicuna/LLaMA)          │           │
│  │                                           │           │
│  │  视觉 token ──┐                          │           │
│  │               │                          │           │
│  │  System: "A chat between a curious ..."  │           │
│  │  User: "What is in this image?"          │           │
│  │  V token: [visual token 1..N]           │           │
│  │                                          │           │
│  │  Self-Attention: 所有 token 互相关注       │           │
│  │  → LLM 在理解文本序列的同时"看到"图像      │           │
│  └──────────────────────────────────────────┘           │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### 2.2 MLP Projector（核心组件）

LLaVA 的视觉-语言桥接器是一个极简单的 MLP：

```
LLaVA v1（单层 Linear）:
  输入视觉 token: 1024 维（CLIP ViT-L 的输出维度）
  输出视觉 token: 4096 维（LLaMA/Vicuna 的 embedding 维度）

  Projector: Linear(1024 → 4096)
  参数量: 1024 × 4096 = ~4.2M

LLaVA-1.5（2 层 MLP）:
  输入视觉 token: 1024 维
  中间层: 4096 维（GELU 激活）
  输出视觉 token: 4096 维

  Projector: Linear(1024→4096) → GELU → Linear(4096→4096)
  参数量: 1024×4096 + 4096×4096 = ~21M
```

**为什么 LLaVA-1.5 从 Linear 升级到 MLP？**

```
Linear: y = W · x
  → 假设 ViT 和 LLM 的表示空间是"线性可映射的"
  → 经验发现这个假设部分成立但不够

MLP: h = GELU(W₁·x + b₁), y = W₂·h + b₂
  → 允许更复杂的非线性映射
  → LLaVA-1.5 实验中 MLP 比 Linear 高了 ~2-3% 在不同 benchmark 上
```

### 2.3 图像分辨率的影响

LLaVA 系列经历了分辨率提升的演进：

```
LLaVA v1（224²）:
  CLIP ViT-L/14 → 16×16=256 patches + 1 CLS = 257 token
  实际效果: 粗粒度理解可以，OCR 失败

LLaVA-1.5（336²）:
  CLIP ViT-L/14（支持 336²）→ 24×24=576 patches + 1 CLS = 577 token
  实际效果: OCR 有改善但仍有限

LLaVA-NeXT（动态分辨率）:
  更大的图像切成 336×336 的 patch，每个独立编码再合并
  支持最高 4K 分辨率（多张 336² 拼接）

  问题: token 数量大增（4K 图像 → ~4000 个 token）
        LLM 的 context window 压力增大
```

---

## 三、训练策略：两阶段训练

### 3.1 Stage 1: 特征对齐预训练

```
目标: 让 MLP Projector 学会将 ViT 的视觉特征映射到 LLM 的文本空间

数据: CC-595K（Conceptual Captions，595K 图文对）
      使用 BLIP 过滤后的较高质量子集

训练配置:
  冻结: CLIP ViT ✅（不更新）
  冻结: LLM ✅（不更新）
  训练: MLP Projector 只 ✅
  损失: 语言建模损失（Causal LM）
  batch: 128
  lr: 1e-3
  步数: ~6K steps

训练示例:
  User: "What does this image describe?"
  Assistant: "A cat wearing a suit sitting on a chair."

  → 模型需要根据图像 token + 文本 prompt，生成描述文字
  → LLM 冻结 → 只能通过调整 MLP 来改变预测
  → MLP 学会将视觉特征"翻译"成 LLM 能理解的表示
```

### 3.2 Stage 2: 视觉指令微调

```
目标: 让 LLM 学会"对话式"的视觉问答

数据: LLaVA-Instruct-150K
      - 由 GPT-4 基于 COCO 图像生成
      - 三种对话类型:
        (a) 对话（Conversation）: 92K 多轮对话
        (b) 详细描述（Detailed Description）: 23K 详细描述
        (c) 复杂推理（Complex Reasoning）: 35K 需要推理的问题

训练配置:
  冻结: CLIP ViT ✅（冻结）
  训练: MLP + LLM ✅（全参数微调）
  损失: 语言建模损失
  batch: 128
  lr: 2e-5
  步数: ~3K steps（1 epoch）
```

### 3.3 LLaVA-1.5 的数据升级

```
LLaVA-1.5 将指令数据从 150K 扩充到 665K：

新增数据:
  - VQA v2（视觉问答）: ~443K
  - GQA（组合式问答）: ~165K
  - OCR-VQA（文字识别问答）: ~170K
  - TextCaps（文字描述）: ~120K
  - RefCOCO（引用分割理解）: ~100K

效果: 在 VQA 上有显著提升（+8%），尤其是 OCR 相关任务

关键发现: 学术 VQA 数据的"质量"比 GPT-4 生成的"质量"高
          → 因为学术数据是人工标注的、准确的
          → GPT-4 生成的指令可能有幻觉
```

### 3.4 训练效率

LLaVA 的训练效率在当时是开创性的：

```
LLaVA v1（8×A100 80GB）:
  Stage 1: ~1h（558K 数据，1 epoch）
  Stage 2: ~20h（150K 数据，1 epoch）
  总时间: ~24h
  → 总成本: ~500 美元（云 GPU 价格）

对比其他 VLM:
  BLIP-2: Q-Former 训练需要 ~12h + LLM 适配（需要更多数据）
  Flamingo: 80B 模型，数千 GPU·天的级别
  GPT-4V: 未公开，但估计是百万美元级

LLaVA 证明: 高质量 VLM 不一定要烧大钱
```

---

## 四、指令数据构建

### 4.1 LLaVA-Instruct-150K 生成流程

LLaVA 开源了数据生成流程，这是对社区最大的贡献之一：

```
步骤 1: 获取图像描述
  COCO 图像 → BLIP Caption → 短描述
  (cat wearing suit on chair)

步骤 2: 构造 GPT-4 的 Prompt
  System:
    You are an AI visual assistant. You will be provided 
    with a caption of an image: "A cat wearing a suit 
    sitting on a chair."
    
    Generate 3 types of conversations:
    (a) Conversation: natural QA about the image
    (b) Detail description: comprehensive description
    (c) Complex reasoning: questions requiring reasoning

步骤 3: GPT-4 生成输出

  Conversation:
    User: What is in this image?
    Assistant: There is a cat wearing a suit sitting on a chair.
    
  Detail Description:
    The image shows a cat dressed in a suit, sitting on a 
    wooden chair. The cat appears calm and the suit fits 
    snugly...
    
  Complex Reasoning:
    User: Why might someone dress a cat in a suit?
    Assistant: This could be for a costume event or photoshoot,
    or the owner's sense of humor...
```

### 4.2 数据格式

```
LLaVA 的多轮对话格式:

[
  {
    "id": "0000001",
    "image": "COCO_val2014_000000000139.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "What is in this image?"
      },
      {
        "from": "gpt",
        "value": "A cat wearing a suit on a chair."
      },
      {
        "from": "human", 
        "value": "What color is the suit?"
      },
      {
        "from": "gpt",
        "value": "The cat is wearing a black suit."
      }
    ]
  }
]
```

### 4.3 训练时的 Token 拼接

```
LLaVA 将图像 token 和文本 token 拼接为统一序列:

[SYSTEM TOKEN]  ← 系统 prompt
  |  "A chat between a curious user and an assistant..."

[USER TOKEN]    ← 用户问题
  |  "What is in this image?"

[VISUAL TOKEN]  ← 576 个图像 token（特殊占位符）
  |  <image_1>, <image_2>, ..., <image_576>

[ASSISTANT TOKEN] ← 模型需要生成的回答
  |  "A cat wearing a suit."

损失只计算 Assistant token 部分
→ 不计算 System/User/Visual token 的损失
```

---

## 五、LLaVA-NeXT（LLaVA-1.6）的改进

### 5.1 动态分辨率

```
LLaVA-1.5 的局限: 固定 336² 分辨率
  → 处理 1280×720 图像时会缩放到 336×336
  → 大量细节丢失

LLaVA-NeXT 的动态分辨率:
  ① 将大图分割成 336×336 的 tile
  ② 每个 tile 独立经过 CLIP ViT 编码
  ③ 所有 tile 的 token 合并 + thumbnail（全局缩略图）
  
  例: 1280×720 图像
    → 分割为 4×3 = 12 个 tile（336² 每个）
    → 12 × 576 = 6912 个 visual token + 1 thumbnail 576 = 7488 token
    → 相比 LLaVA-1.5 的 576 token → 13× 更多视觉信息

可选的 tile 排列: AnyRes
  最大 tile 数: 6（LLaVA-NeXT 限制）
  支持多种宽高比: 1×2, 2×1, 1×3, 3×1, 2×2
```

### 5.2 LLaVA-OneVision（2025）

```
进一步升级到 LLaVA-OneVision:
  - 视觉处理器: Qwen2VL 的动态分辨率方案
  - LLM: Qwen3（从 LLaMA 迁移到 Qwen 系列）
  - Mid-training: 在 85M 多模态样本上继续训练
  - 视频支持: 多帧输入（8-32 帧）
  
  参数量: 8B（相比 LLaVA-1.5 的 13B 更小，但效果更好）
```

---

## 六、LLaVA 的局限与后续方向

| 局限 | 表现 | 后续改进 |
|------|------|---------|
| **ViT 分辨率有限** | 336² 上限，OCR 不够好 | 动态分辨率（LLaVA-NeXT）|
| **Frozen ViT** | 无法端到端优化视觉编码器 | InternVL 路线（端到端训练 ViT）|
| **单图为主** | 多图理解弱 | LLaVA-OneVision（多帧支持）|
| **纯文本输出** | 不能生成图像 | 后续统一模型（Emu3、BLIP3-o）|
| **英文为主** | 中文 VLM 需要专门训练 | Qwen-VL 补充中文能力 |

---

## 七、总结

> **LLaVA 的核心贡献不是发明了什么新技术——它把 VLM 的架构简化到了极致：ViT + MLP + LLM。这种"降维式"的简化在其他领域可能被批评为缺乏创新，但在多模态领域，它恰恰证明了"复杂架构（Q-Former、Cross-Attention）不是必须的"。**

| 维度 | LLaVA 的方案 | 与 BLIP-2 对比 |
|------|-----------|--------------|
| **桥接层** | MLP（2 层非线性映射）| Q-Former（可学习 query）|
| **视觉 token 数** | 576（全部保留）| 32（压缩）|
| **LLM 训练** | **全参数微调** | 冻结 + 适配 |
| **训练成本** | ~500 USD | ~更高（Q-Former 训练耗时）|
| **相同 LLM 下的效果** | **更好** | 稍弱 |

> LLaVA 的意义：**它证明了开源社区不需要烧几百万美元也能做出高质量的 VLM。** 550K 预训练数据 + 150K 指令数据 + 8×A100 一天 = 接近 GPT-4V 的视觉理解能力。这也解释了为什么 LLaVA-1.5 成为了 2024 年开源 VLM 中被引用最多的基线模型。

---

**Sources:**
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) — Liu et al. 2023
- [LLaVA-1.5: Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744) — Liu et al. 2024
- [LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-next/) — 2024
- [Visual Instruction Tuning (LLaVA-Instruct Dataset)](https://github.com/haotian-liu/LLaVA)
- [Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality](https://lmsys.org/blog/2023-03-30-vicuna/) — 2023
