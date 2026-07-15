# FLUX + DiT 架构深度解剖

> Black Forest Labs (2024.08) & Stability AI (2024.02-10) | "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" —— 从 U-Net 到 Transformer 的范式革命，开源文生图质量天花板

---

## 写在前面

2023 年底，行业发现 U-Net 架构的 Scaling Law 到达瓶颈——SDXL 已经把参数堆到 3.5B，但继续增加参数带来的收益越来越小。卷积架构的归纳偏置（局部感受野 + 平移不变性）在扩散任务中成了天花板。

2024 年两条新方向的交汇彻底改变了格局：

1. **MM-DiT（多模态扩散 Transformer）**——文本和图像 token 在同一个 Transformer 空间中双向交互
2. **Rectified Flow（整流流）**——用直线路径替代 DDPM 的随机漫步，推理效率大幅提升

**注意：本文涉及的三个模型（SD3、SD3.5、FLUX.1）共享同一个 MM-DiT 架构底座。** 它们的关系不是"代际差异"，而是同一技术的不同实现和规模：

| 模型 | 时间 | 参数量 | VAE | 团队 | 核心差异 |
|------|------|--------|-----|------|---------|
| **SD3 Medium** | 2024.06 | 2.5B | 16ch | Stability AI | 最早的 MM-DiT 开源实践，但质量问题多 |
| **FLUX.1** | 2024.08 | **12B** | 16ch | Black Forest Labs | SD 原班人马，当前质量天花板 |
| **SD3.5 Large** | 2024.10 | 8.1B | 16ch | Stability AI | 修复 SD3 问题，接近 FLUX 质量 |

---

## 一、DiT：从 U-Net 到 Transformer 的架构变革

### 1.1 U-Net 的局限

要理解为什么 DiT 取代 U-Net，需要先看 U-Net 在 2023 年底遇到的问题：

```
U-Net 架构的核心假设:
  ① 图像有"空间结构"——需要卷积的局部感受野
  ② 下采样上采样可以高效表示多尺度特征
  ③ 跳跃连接帮助信息流动

但这些假设在足够大的算力和数据面前成了限制:
  ① 卷积的感受野有限——即使堆了很多层，要看到 1024² 的全局信息仍然困难
  ② 下采样必然丢失细节——更多参数只能缓解不能解决
  ③ 跳跃连接是固定路径——无法像 Transformer 那样动态选择信息流向
```

### 1.2 DiT 的核心设计

DiT（Peebles & Xie, 2022）的论文首先提出了"用 Transformer 替代 U-Net 做扩散"的想法：

```
U-Net 的视角:
  图像 → 卷积 → 空间特征图（保留空间结构 + 局部归纳偏置）

DiT 的视角:
  图像 → 切 patch → token 序列 → Transformer（彻底抛弃空间归纳偏置）

关键转变:
  U-Net: "图像是一个 2D 网格，我用 2D 卷积处理它"
  DiT: "图像是一串 token，我用 Transformer 处理它们"
```

整体流程：

```
原始图像 (1024×1024×3)
    │
    ├── VAE Encoder → 潜变量 (128×128×16)  ← 注意: 16ch VAE
    │
    ├── Patchify: 将潜变量切为 patch
    │   patch_size = 2 → 64×64 = 4096 个 patch
    │   每个 patch 展平为: 2×2×16 = 64 维向量
    │
    ├── Linear 投影: 64 → 4096   ← 每个 patch 映射到 Transformer 维度
    │
    ├── Positional Encoding (RoPE)  ← 位置信息通过旋转位置编码注入
    │
    ├── N × Transformer Block (MM-DiT)
    │   ├── Self-Attention (QK-Normalized)
    │   ├── Dual-Stream Cross-Attention (文本-图像双向)
    │   ├── MLP (GELU)
    │   └── LayerNorm (Pre-LN)
    │
    ├── Linear 投影回 patch 维度
    │
    ├── Depatchify: 恢复为 128×128×16 潜变量
    │
    └── VAE Decoder → 生成图像
```

| 对比维度 | U-Net (SD 1.5 / SDXL) | DiT (SD3 / FLUX) |
|---------|----------------------|-----------------|
| **输入形式** | 2D 特征图（保留空间结构）| 1D token 序列（压平所有空间信息）|
| **核心操作** | Conv + Self-Attn + Cross-Attn | Self-Attention + Cross-Attention |
| **下采样** | stride=2 conv / 池化 | 下采样通过 patch_size 控制 |
| **长程依赖** | 受限于深度的感受野 | 从第一层就是全局的 |
| **条件注入** | Cross-Attention（单向）| MM-DiT 双流注意力（双向）|
| **参数量缩放** | 收益递减（SDXL 已见天花板）| 持续有效（12B 还在 Scaling）|

---

## 二、MM-DiT（多模态扩散 Transformer）——核心架构创新

### 2.1 从"文本单向注入"到"双向圆桌会议"

先回顾 U-Net 时代的条件注入方式：

```
U-Net (LDM / SDXL):
  Cross-Attention: Q = 图像特征, K,V = 文本嵌入

  方向: 文本 → 图像（单向）
  图像能看文本: ✅    文本能看图像: ❌
  
  类似: "老师在台上讲课，学生只能听不能问"
```

MM-DiT 的核心变化是 **文本 token 和图像 token 在同一个 Transformer 空间中"坐在一起"**，彼此关注：

```
MM-DiT (SD3 / FLUX):
  双流 Cross-Attention: 文本 token 和图像 token 互为 Q 和 KV
  
  第一流: Q_img = 图像特征, K,V_img = 文本嵌入 + 图像嵌入（正文互关注）
  第二流: Q_txt = 文本嵌入, K,V_txt = 图像嵌入 + 文本嵌入（文关注图）

  方向: 文本 ↔ 图像（双向）
  图像能看文本: ✅    文本能看图像: ✅
  
  类似: "圆桌会议——每个人都能看到所有人的发言"
```

### 2.2 MM-DiT 的具体架构

```
MM-DiT Block（单层）:
                    ┌──────────────────────┐
                    │   文本 token 序列      │
                    │   (512 × 4096)        │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   LayerNorm (文本)    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
               ┌────│   Self-Attention     │
               │    │   (文本内部, RoPE)    │
               │    └──────────┬───────────┘
               │               │
               │    ┌──────────▼───────────┐
               │    │   LayerNorm (文本)    │
               │    └──────────┬───────────┘
               │               │
               │    ┌──────────▼───────────┐
               │    │   Cross-Attention    │
               │    │   Q_txt: 文本        │
               │    │   K,V: 图像嵌入      │
               │    └──────────┬───────────┘
               │               │
               │    ┌──────────▼───────────┐
               │    │     MLP (GELU)       │
               │    └──────────┬───────────┘
               │               │
               │    ┌──────────▼───────────┐
               │    │   文本 token 输出     │
               │    └──────────────────────┘
               │
               │    ┌──────────────────────┐
               │    │   图像 token 序列      │
               │    │   (4096 × 4096)       │
               │    └──────────┬───────────┘
               │               │
               │    ┌──────────▼───────────┐
               │    │   LayerNorm (图像)    │
               │    └──────────┬───────────┘
               │               │
               │    ┌──────────▼───────────┐
               └────│   Self-Attention     │
               │    │   (图像内部, RoPE)    │
               │    └──────────┬───────────┘
               │               │
               │    ┌──────────▼───────────┐
               │    │   LayerNorm (图像)    │
               │    └──────────┬───────────┘
               │               │
               │    ┌──────────▼───────────┐
               │    │   Cross-Attention    │
               │    │   Q_img: 图像        │
               │    │   K,V: 文本嵌入      │
               │    └──────────┬───────────┘
               │               │
               │    ┌──────────▼───────────┐
               │    │     MLP (GELU)       │
               │    └──────────┬───────────┘
               │               │
               │    ┌──────────▼───────────┐
               │    │   图像 token 输出     │
               │    └──────────────────────┘
```

**关键设计细节：**

| 设计 | 说明 | 与 U-Net 对比 |
|------|------|--------------|
| **双流独立 LayerNorm** | 文本和图像各自有独立的 LayerNorm，权重不同 | U-Net 用 GroupNorm |
| **双流 Self-Attention** | 文本内部、图像内部各自做自注意力 | U-Net 也有 Self-Attn |
| **交叉 Cross-Attention** | 文本看图像 + 图像看文本（双向） | U-Net 只有图像看文本 |
| **RoPE 位置编码** | 旋转位置编码（Extend Context） | U-Net 用 Sinusoidal + 可学习 |
| **QK-Normalization** | SD3.5/FLUX 在 Attention 中对 Q 和 K 做 LayerNorm | SD3 Medium 没有，导致训练不稳定 |

### 2.3 QK-Normalization —— 稳定训练的关键

为什么 SD3 Medium 的人体结构扭曲？一个关键原因是缺少 QK-Normalization：

```
问题: 大模型 Attention 中的 Q·Kᵀ 内积值会随参数量增大而爆炸

数学:
  Q, K 的维度 = d_model
  Q·Kᵀ 的方差 ≈ d_model (假设 Q, K 各元素独立)
  
  d_model = 4096 (SD3 Medium 2.5B) → QK 内积方差 4096
  d_model = 8192 (SD3.5 Large 8.1B) → QK 内积方差 8192
  更大的 d_model → 更大的内积 → softmax 趋向 one-hot → 梯度消失

解决方案 — QK-Normalization:
  Q' = LayerNorm(Q) · √d                  ← 对 Q 做 LN 再缩放
  K' = LayerNorm(K) · √d                  ← 对 K 做 LN 再缩放
  Attention = softmax(Q' · K'ᵀ / √d) · V
  内积范围不再随 d_model 增长
```

**FLUX 和 SD3.5 都使用了 QK-Normalization，而 SD3 Medium 没有——这是 SD3 Medium 质量不稳定的主要原因之一。**

---

## 三、Rectified Flow + Flow Matching

### 3.1 DDPM 的"随机漫步"问题

```
DDPM 的前向/反向过程（T=1000 步）:
  前向: x₀ → 加噪声(x₁) → 加更多噪声(x₂) → ... → 纯噪声(x_T)
  反向: 纯噪声(x_T) → 去噪(x_{T-1}) → 去更多噪声 → 图像(x₀)
  
  每次去噪都是"猜测"——没有直接路径，全靠模型慢慢摸索

问题:
  - 路径是弯曲的: 每一步去噪的方向不确定
  - 重建质量受限于步数: 步数越少 → 路径越弯曲 → 质量越差
  - 1000 步 → 50 步（DDIM）是"切割"路径，不是"缩短"路径
```

### 3.2 Rectified Flow —— 直线路径

Rectified Flow 的核心思想：**不随机漫步，沿着一条直线从噪声走到数据。**

```
DDPM:             x_T ──x_{T-1}──x_{T-2}── ... ──x₀（弯曲路径）
                   ↑     ↑     ↑              ↑
                   每步都弯曲，方向不确定

Rectified Flow:   x₁ ─────────────── x₀（直线路径）
                   ↑  
                   一步到位，不需要中间步

数学表达:
  DDPM:  dx = -β_t/2 · x · dt + √β_t · dw  ← 随机微分方程
  Rectified Flow:  dx = (x₀ - x₁) · dt       ← 常微分方程（ODE）
                   朴素表达: "从 x₁ 沿着直线走向 x₀"
```

Rectified Flow 的微分方程与 DDPM 的对比：

```
DDPM 的 SDE（随机）:
  dx = f(x, t) · dt + g(t) · dw    
       ↑                 ↑
  漂移项（确定的）  扩散项（随机噪声）

Rectified Flow 的 ODE（确定性的）:
  dx = (x₀ - x₁) · dt
       ↑
  直接指向目标的向量
  没有随机项！

核心优势:
  1. 路径是直的 → 大步长采样也能保持方向
  2. 采样步数极大减少 → 4-8 步就能生成
  3. 确定性 → 同 seed 同结果，可复现
```

### 3.3 Flow Matching —— 如何训练 Rectified Flow

Flow Matching 是训练 Rectified Flow 的训练目标：

```
训练目标（简单版本）:
  给定:
    - x₀: 真实图像（VAE 编码后的潜变量）
    - x₁: 噪声（~N(0, I)）
    - t: 时间步 ∈ [0, 1]

  线性插值:
    x_t = (1-t) · x₁ + t · x₀    ← 在噪声和图像之间做线性插值
  
  目标向量:
    v = x₀ - x₁                   ← "从噪声指向图像的方向"
  
  模型预测:
    v_pred = v_θ(x_t, t, text_emb)
  
  损失:
    L_flow = ||v - v_pred||²      ← 预测"方向向量"而非"噪声"
```

**Flow Matching vs DDPM 的损失对比：**

```
DDPM: L = ||ε - ε_θ(√ᾱ_t·x₀ + √(1-ᾱ_t)·ε, t)||²
      预测: 噪声 ε
      变量: x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε （非线性混合）

Flow Matching: L = ||v - v_θ(x_t, t, c)||²
      预测: 方向 (x₀ - x₁)
      变量: x_t = (1-t)·x₁ + t·x₀ （线性插值）

差异:
  - DDPM: 噪声和图像按特定比例混合，路径弯曲
  - Flow Matching: 噪声和图像线性混合，路径笔直
  - Flow 的训练目标更简单——回归一个常向量
```

### 3.4 采样过程

```
算法: Rectified Flow 采样（4-20 步）

输入: 训练好的 v_θ，文本嵌入 c，步数 N
输出: 生成图像潜变量 x₀

1. x₁ ~ N(0, I)                    ← 从噪声开始
2. Δt = 1/N                        ← 每步步长
3. for step = 1 to N:              ← N 可以是 4, 8, 20
4.     t = 1 - (step - 1) · Δt     ← 当前时间步
5.     v_pred = v_θ(x_t, t, c)     ← 预测方向
6.     x_{t-Δt} = x_t + v_pred · Δt ← 沿直线走一步
7. return x₀

注意: N=4 时也能得到不错的结果（schnell 模式）
      N=20-50 时质量最好（dev/pro 模式）
      相比之下，SDXL 需要 20-50 步 DDIM
```

---

## 四、三编码器融合

FLUX 和 SD3 使用了三个文本编码器的融合输出。这比 SDXL 的双编码器又进了一步：

```
文本 prompt: "一只穿西装的猫在巴黎铁塔前，油画风格，细节丰富"
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌────────┐ ┌──────────┐ ┌──────────┐
   │CLIP-L  │ │OpenCLIP-G│ │ T5-XXL   │
   │ 77 ×768│ │77 ×1280  │ │512 ×4096 │
   └────┬───┘ └─────┬────┘ └────┬─────┘
        │           │           │
        └───────────┼───────────┘
                    │
            ┌───────▼───────┐
            │   融合策略     │
            │  CLIP: 短语义  │
            │  T5: 长语义    │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │  MM-DiT 输入   │
            │  文本 token    │
            └───────────────┘
```

| 编码器 | 维度 | token 数 | 参数量 | 角色 |
|--------|------|---------|--------|------|
| **CLIP ViT-L** | 768 | 77 | ~340M | 短语义、概念对齐（"猫"、"油画"）|
| **OpenCLIP ViT-G** | 1280 | 77 | ~1.2B | 更丰富的视觉概念表示 |
| **T5-XXL** | 4096 | 512 | **~11B** | 理解长难句、复杂逻辑、否定表达 |

**为什么 T5-XXL 是关键？**

```
CLIP 的局限:
  prompt: "一只黑色的猫蹲在红色的椅子上，旁边有一杯冒着热气的咖啡"
  CLIP 理解: 猫(黑色) + 椅子(红色) + 咖啡 → 基本概念
  CLIP 丢失: "蹲在……上"的空间关系、"冒着热气"的状态描述
  
T5-XXL 的能力:
  prompt: 同上
  T5-XXL 理解: 
    - "黑色的猫"作主语，"蹲"是动作
    - "红色的椅子上"是位置
    - "一杯……咖啡"是描述对象
    - "冒着热气"是咖啡的状态
    - 所有关系被正确解析

一句话: CLIP 是"关键词匹配"，T5 是"真正理解语言"
```

---

## 五、FLUX 的三个规格对比

FLUX 发布了三个规格，覆盖从本地运行到商业 API 的不同场景：

```
FLUX.1 [schnell]      ← 快速模式（开源，Apache 2.0）
  步数: 4-8 步
  显存: ~12GB（消费级 GPU 可跑）
  质量: 接近 SDXL，但速度快 5×
  许可证: Apache 2.0（最宽松的开源协议）
  技术: 使用 Guidance Distillation（蒸馏训练）
  
FLUX.1 [dev]          ← 开发者模式（开源，非商用）
  步数: 20-50 步  
  显存: ~24GB
  质量: 全面超越 SD3.5，接近 Midjourney
  许可证: 非商用（Black Forest Labs 自有协议）
  技术: 完整 12B 模型，无蒸馏

FLUX.1 [pro]          ← 专业模式（闭源 API）
  步数: 自动（API 决定）
  质量: 最高的
  访问: API 调用
  特点: 额外的安全过滤、内容审核、企业级可用性
```

### 蒸馏技术（Guidance Distillation）

schnell 版本能将步数压缩到 4-8 步而不大幅损失质量，核心是两种蒸馏技术的结合：

```
蒸馏训练流程:

  第 1 阶段: Step Distillation（步数压缩）
    Teacher: FLUX.1 [dev]（20 步）
    Student: FLUX.1 [schnell]（4 步）
    
    训练: Student 学习再现 Teacher 的"4 步后的 output"
    目标: 在节省 5× 步数的前提下保持输出相似
    损失: ||Teacher_output - Student_output||²

  第 2 阶段: Adversarial Distillation（对抗蒸馏）
    Student + PatchGAN 判别器联合训练
    Student 生成的图像尽量逼真
    Discriminator 区分 Student 的图和真实图
    → 补偿"少步数"带来的信息损失
```

---

## 六、SD3 Medium vs SD3.5 Large vs FLUX.1

这三个模型共享 MM-DiT 架构，但实现上的差异决定了质量的不同：

### 6.1 架构参数对比

| 维度 | SD3 Medium | SD3.5 Large | FLUX.1 [dev] |
|------|-----------|------------|-------------|
| **总参数量** | 2.5B | **8.1B** | **12B** |
| **Transformer 层数** | 24 | 42 | 48 |
| **Attention 头数** | 24 | 48 | 64 |
| **d_model** | 3072 | **8192** | **8192** |
| **QK-Normalization** | ❌ | ✅ | ✅ |
| **训练数据** | ~3B | ~5B | ~5B+ |
| **DDIM 兼容** | ✅ | ❌（只用 Rectified Flow）| ❌（只用 Rectified Flow）|
| **人体结构评价** | ❌ 经常扭曲 | ✅ 大幅改善 | ✅ 接近完美 |

### 6.2 SD3 Medium 质量问题根因分析

```
SD3 Medium 的核心问题: 
  人体结构扭曲 → 手指多一根、腿扭曲

Why?
  ├── ① 缺少 QK-Normalization
  │     → Attention 训练不稳定 → 某些 attention head "死掉"
  │     → 特定语义（人手、脚踝）的注意力被丢失
  │
  ├── ② 参数量不够（2.5B vs 8.1B/12B）
  │     → 缺乏足够 capacity 建模"人体关节"这种精细结构
  │     → 尤其是在步数少的情况下
  │
  └── ③ 训练数据量不足
        → 5B 图文对 vs 2.5B 参数的模型需要更多
        → 高质量图（人体结构精准的）占比不够

SD3.5 Large 的修复:
  ① + ② + ③ 全部改善
  → QK-Norm 稳定了训练
  → 8.1B 参数提供了足够容量
  → 更多/更好的训练数据
```

---

## 七、总结

> **2024 年是视觉生成领域的"范式革命之年"——MM-DiT 取代 U-Net，Rectified Flow 取代 DDPM，T5-XXL 取代 CLIP。SD3 率先验证路径，FLUX.1 将这条路走到极致，SD3.5 完成了 Stability AI 的救赎。**

| 技术转变 | 旧路线 | 新路线 | 效果 |
|---------|-------|-------|------|
| **骨干网络** | U-Net（卷积） | DiT / MM-DiT（Transformer） | 更好的 Scaling Law |
| **文本-图像交互** | Cross-Attention 单向 | 双流注意力双向 | 更精准的语义对齐 |
| **推理路径** | DDPM 随机漫步 | Rectified Flow 直线 | 更少步数、更高质量 |
| **文本编码** | CLIP 关键词匹配 | T5-XXL 语言理解 | 复杂 prompt 不再失败 |
| **潜空间** | 4ch VAE | 16ch VAE | 细节保留更好 |

**FLUX 的故事对行业的意义：**
- SD 的核心作者（Rombach, Blattmann, Lorenz）从 Stability AI 出走
- 创立 Black Forest Labs，用 8 个月做出 12B 参数的 FLUX.1
- 证明了一件事：**在 AI 领域，人才是核心竞争力，不是公司品牌**

---

**Sources:**
- [DiT: Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) — Peebles & Xie 2022
- [SD3: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stability AI 2024
- [FLUX.1: Rectified Flow Transformer](https://blackforestlabs.ai/) — Black Forest Labs 2024
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al. 2022
- [Rectified Flow: A Marginal Preserving Approach to Optimal Transport](https://arxiv.org/abs/2209.14577) — Liu et al. 2022
- [QK-Normalization: Training Stability in Large Transformers](https://arxiv.org/abs/2501.04215) — SD3.5 Technical Report 2024
- [Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — Sauer et al. 2023
