# Qwen2.5-VL / Qwen-VL 系列架构深度解剖

> 阿里通义千问 (2023-2025) | "Qwen-VL / Qwen2-VL / Qwen2.5-VL / Qwen3-VL" —— 从追赶 GPT-4V 到中英双语最强开源 VLM，两年四代的进化之路

---

## 写在前面：为什么 Qwen-VL 系列值得解剖

2023-2025 年的开源 VLM 格局中，Qwen-VL 系列的进化速度和质量是一个值得深挖的案例：

```
Qwen-VL 系列的时间线:

2023.08 Qwen-VL / Qwen-VL-Chat
  → 追赶 LLaVA-1.5，中英双语，7B/13B
  → 用 ViT detector 做细粒度定位

2024.08 Qwen2-VL
  → 全面升级，NaViT 动态分辨率
  → 视频理解（20分钟+），多语言
  → 2B/7B/72B 三种规模

2025.01 Qwen2.5-VL
  → 更强的动态分辨率（AnyRes 优化）
  → 1小时视频理解
  → Agent 能力（Function Calling + Computer Use）
  → 3B/7B/72B

2025.10 (预计) Qwen3-VL
  → Qwen3 LLM 底座 + 新视觉编码器
  → 进一步强化多模态 Agent 和视频
```

**Qwen-VL 系列的特殊价值：**

| 维度 | 说明 |
|------|------|
| **唯一的中英双语 VLM 系列** | 其他开源 VLM（LLaVA、InternVL）以英文为主 |
| **四代持续迭代** | 展示了 VLM 架构的进化路径 |
| **动态分辨率方案** | Qwen2-VL 的 NaViT 风格方案和 LLaVA-NeXT 的 AnyRes 不同 |
| **完整的 Agent 能力** | 是 2025 年最强多模态 Agent 之一 |

---

## 一、Qwen-VL 第一代（2023.08）

### 1.1 整体架构

Qwen-VL 第一代的架构和 LLaVA-1.5 非常相似：

```
Qwen-VL 架构（第一代）:

  图像
    │
    ▼
  ┌──────────────────┐
  │ Vision Encoder   │ ← Frozen（CLIP ViT-bigG / ViT-L/14）
  │ (ViT-L/14        │    分辨率 224²
  │  或 ViT-bigG)    │
  └────────┬─────────┘
           │ 256 patch token
           ▼
  ┌──────────────────┐
  │ MLP Projector    │ ← 可训练（Linear + GELU）
  │ (嵌入到 Qwen LLM │     相当于 LLaVA 的 MLP
  │  的 emb dim)     │
  └────────┬─────────┘
           │ 256 个视觉 token
           ▼
  ┌──────────────────┐
  │ Qwen-LLM         │ ← 全参数微调（7B / 13B）
  │ (基于 Qwen-7B    │    支持中英文
  │  的 Decoder-only)│
  └──────────────────┘
```

### 1.2 与 LLaVA-1.5 的差异

```
Qwen-VL vs LLaVA-1.5:

相同点:
  - Frozen ViT + MLP + 全参数微调 LLM
  - 两阶段训练（先对齐、后指令微调）
  - 损失函数 = Causal LM

不同点:
  1. 语言: Qwen-VL 中英双语 | LLaVA-1.5 纯英文
  2. ViT: Qwen-VL 用 ViT-bigG（~1.8B 参数）
          LLaVA-1.5 用 CLIP ViT-L/14（~430M）
  3. 定位能力: Qwen-VL 有 ViT detector 做 Bounding Box 预测
               LLaVA-1.5 没有定位能力
  4. 视觉 token: Qwen-VL 256 token | LLaVA-1.5 576 token（336²）
```

### 1.3 细粒度定位能力

Qwen-VL 的特色功能——**不依赖目标检测器的定位**：

```
ViT-based Detector:

  Qwen-VL 在 ViT 上加了额外的定位头
  - 输入: 图像 + 文本描述（如"the cat"）
  - 输出: Bounding Box [x₁, y₁, x₂, y₂]
  
  训练数据: 图文对 + 检测数据（如 Visual Genome、RefCOCO）
  
  → 不需要专门的检测模型（如 Faster R-CNN）
  → 用 ViT 的特征直接做检测

定位的效果:
  "Describe the image and locate the cat."
  → "The cat is sitting on a chair. <bbox>[0.1, 0.2, 0.5, 0.8]</bbox>"
  
  注意: 
    这种定位方式精度不如专用检测器
    但胜在"零成本集成"——不需要额外的检测模型
```

### 1.4 训练数据

```
Qwen-VL 的训练数据:

Stage 1: 预训练（特征对齐）
  数据: 
    - 1.4B 图文对（开源数据 + 中文内部数据）
    - 中英双语
  冻结: ViT ✅, LLM ✅
  训练: MLP Projector

Stage 2: 多任务预训练
  数据:
    - 图文对（Captioning）
    - VQA（问答）
    - 检测数据（Bounding Box）
    - OCR 数据（文字识别）
  冻结: ViT ✅
  训练: MLP + LLM（全参数微调）

Stage 3: 指令微调
  数据:
    - LLaVA-Instruct（150K，翻译成中文）
    - 内部中文指令数据
    - 对话数据
  冻结: ViT ✅
  训练: MLP + LLM
```

---

## 二、Qwen2-VL（2024.08）：全面革新

Qwen2-VL 是 Qwen-VL 系列的一次**架构级升级**——不再只是"换 ViT 和 LLM"，而是重新设计了视觉处理方案。

### 2.1 核心升级

```
Qwen-VL → Qwen2-VL 的变化:

  ① 视觉编码器: CLIP ViT → 自研 ViT（QwenViT）
      - 动态分辨率（NaViT 风格）
      - 支持任意分辨率输入（不必缩放到固定尺寸）
      - 内置窗口注意力 + 全局注意力（窗口注意力降低计算量）
      
  ② 动态分辨率: 固定 224² → 动态分辨率
      - 图像保持原始宽高比
      - 根据宽高比选择最优的 patch 排列
      - token 数随分辨率变化
      
  ③ MLP → Q-Former-like Resampler
      - 简单的 MLP 投影 → 2D-RoPE 感知的 Resampler
      - 把变长的视觉 token 重采样为固定长度
      
  ④ LLM: Qwen → Qwen2
      - 从 Qwen1 升级到 Qwen2（更强的基础能力）
      - 支持多语言（29 种语言）
      
  ⑤ 视频支持
      - 从"单图"扩展到"多帧视频"
      - 支持 20 分钟+ 的视频理解
      
  ⑥ 规模: 7B → 2B / 7B / 72B 三档
      - 2B: 适合移动端 / 快速推理
      - 7B: 主流性价比
      - 72B: 最强效果
```

### 2.2 动态分辨率方案（Qwen2-VL 的核心创新）

Qwen2-VL 用了和 LLaVA-NeXT 不同的动态分辨率方案。

```
LLaVA-NeXT 的 AnyRes:
  ① 大图分割成多个 336×336 的 tile
  ② 每个 tile 独立通过 CLIP ViT 编码
  ③ 所有 tile token + thumbnail token 合并
  ④ 多个 tile → token 数量线性增加
  
  问题: 
    - 每个 tile 独立编码 → 丢失全局上下文
    - token 数量随 tile 数线性增长 → 长上下文压力

Qwen2-VL 的 NaViT 风格方案:
  ① ViT 原生支持可变 patch 数
     - 输入不缩放到固定尺寸
     - 直接 patchify，保持原始宽高比
     - 位置编码用 2D-RoPE（支持任意位置）
     
  ② 2D-RoPE（旋转位置编码）
     - 传统 1D-RoPE: token 位置是"1, 2, 3, ..."
     - 2D-RoPE: token 位置是"(x₁,y₁), (x₂,y₂), ..."
               → 保持空间位置信息
               → token 之间的空间关系不被破坏
               
  ③ 窗口注意力（Window Attention）
     - 全局注意力对所有 token 做 full attention → O(N²)
     - 窗口注意力在局部窗口内做 attention → O(N × W)
     - 只在最后几层做全局注意力
     - 使得高分辨率输入的推理依然可行

对比:
  维度       | LLaVA-NeXT (AnyRes) | Qwen2-VL (NaViT)
  -----------|--------------------|-------------------
  原生分辨率 | ❌ 必须 tile 切割    | ✅ 原生支持
  全局上下文 | ❌ tile 之间不互通   | ✅ 2D-RoPE 保持位置
  计算效率   | ⚠️ 随 tile 数线性增  | ✅ 窗口注意力节省
  实现复杂度 | ✅ 简单              | ⚠️ 需要改 ViT 架构
```

**2D-RoPE 的详细说明：**

```
传统 RoPE（1D）:
  位置: [1, 2, 3, ..., N]
  RoPE: 在 attention 的 Q 和 K 中注入位置信息
  f(q, m) = q · [cos(mθ), sin(mθ)]
  
  局限: 只能编码"一维"位置——适合文本序列
        但不适合图像（图像是二维空间）

Qwen2-VL 的 2D-RoPE:
  对每个 patch token，计算其二维位置 (h, w)
  
  Q 的 2D-RoPE 编码:
    f(q, h) = q · [cos(hθ_h), sin(hθ_h)]      ← 高度方向
    f(q, w) = q · [cos(wθ_w), sin(wθ_w)]      ← 宽度方向
    
  最终位置编码 = 高度编码 + 宽度编码（concat 或交替）
  
  效果:
    token (0,0) 和 (0,1) → 高度相同、宽度不同 → 宽度方向编码不同
    token (0,0) 和 (1,0) → 宽度相同、高度不同 → 高度方向编码不同
    
  意义:
    - ViT 不需要把图像缩放到固定分辨率
    - 可以处理任意尺寸的输入
    - 位置关系在 attention 中被保持
```

**窗口注意力 + 动态分辨率的计算效率：**

```
对比固定分辨率 vs 动态分辨率:

固定分辨率（LLaVA-1.5, 336²）:
  输入尺寸: 336×336
  patch size: 14×14
  token 数: (336/14)² = 576
  计算量: O(576²) = ~332K attention scores

动态分辨率（Qwen2-VL, 1280×720）:
  输入尺寸: 1280×720（保持原始宽高比）
  patch size: 14×14
  token 数: (1280/14) × (720/14) ≈ 91 × 51 ≈ 4641
  
  全局注意力: O(4641²) = ~21.5M → ❌ 太高
  窗口注意力（窗口 14×14=196）:
    O(4641 × 196) = ~910K → ✅ 可行！
  
  实际 Qwen2-VL 的做法:
    - 前 24 层: 窗口注意力（窗口内 attention，降低计算）
    - 最后 4 层: 全局注意力（跨窗口信息交流）
    → 总计算量 ≈ 24 × O(N · W) + 4 × O(N²)
    → 在效率和效果之间取得平衡
```

### 2.3 Qwen2-VL 的 Resampler

从 Qwen-VL 的简单 MLP 换成了一种**类 Q-Former 的 Resampler**：

```
Qwen2-VL Resampler:

  输入: N 个 visual token（N 随分辨率变化）
  
  可学习的 query: M 个（固定数量，如 256）
  
  处理:
    for _ in range(3):
        query = query + CrossAttention(Q=query, K=visual, V=visual)
        query = query + SelfAttention(Q=query, K=query, V=query)
        query = query + FeedForward(query)
  
  输出: M 个固定数量的视觉 token，送入 LLM
  
  与 BLIP-2 Q-Former 的差异:
    - BLIP-2: 32 个 query (太小)
    - Qwen2-VL: 256 个 query (保留更多信息)
    - BLIP-2: query 之间没有位置编码
    - Qwen2-VL: query 带 2D-RoPE（保持空间关系）
```

### 2.4 视频处理

Qwen2-VL 的视频支持通过**多帧均匀采样**实现：

```
视频处理流程:

  输入视频（如 10 分钟，30000 帧 @ 50fps）
  
  ① 均匀采样 N 帧（如 32 帧）
     → 每帧间隔 = total_frames / N
  
  ② 每帧通过 ViT 编码 + Resampler
     → 每帧得到 256 个视觉 token
     → 总共 32 × 256 = 8192 个视觉 token
  
  ③ 所有视觉 token + 文本 token 拼入序列
  
  ④ LLM 自回归生成回答

Qwen2-VL 的视频能力:
  - 最长支持 20 分钟视频
  - 支持视频问答、摘要、事件定位
  - 多语言视频理解（中英文都行）
  
  局限:
    - 均匀采样可能漏掉关键帧
    - 32 帧的采样率不够高（高速运动场景）
    - 视频 token 很多 → 需要长上下文支持
```

### 2.5 与其他 VLM 在 2024.08 的对比

| 维度 | Qwen2-VL-7B | LLaVA-1.5-7B | InternVL2-8B | LLaMA 3.2-V 11B |
|------|------------|-------------|--------------|----------------|
| **分辨率** | 动态（原生）| 固定 336² | 动态（tile）| 固定 336² |
| **语言** | 中英+多语言 | 仅英文 | 中英 | 仅英文 |
| **视频** | ✅ 20min | ❌ | ✅ 10min | ❌ |
| **OCR** | ✅ 强 | ⚠️ 弱 | ✅ 强 | ⚠️ 中 |
| **定位** | ✅ 内建 | ❌ | ✅ 内建 | ❌ |
| **Agent** | ❌ | ❌ | ❌ | ❌ |
| **参数量档位** | 2B/7B/72B | 7B/13B | 2B/8B/40B/76B | 11B/90B |

---

## 三、Qwen2.5-VL（2025.01）：全面进化体

Qwen2.5-VL 是对 Qwen2-VL 的全方位升级，加入了一些重要的新能力。

### 3.1 核心升级

```
Qwen2.5-VL 的主要变化:

① 更强的动态分辨率（AnyRes 2.0）
   - 在 NaViT 基础上增加"分辨率自适应选择"
   - 根据图像内容自动选择最佳 token 分配
   - 支持超长宽比图像（如海报、长图）

② 1 小时视频理解
   - 从 20 分钟扩展到 1 小时+
   - 更高效的长视频 token 管理

③ Agent 能力
   - Function Calling（调用外部工具）
   - Computer Use（控制计算机操作）
   - MCP 集成（作为 MCP Client）

④ 视频生成理解（DynamicQA / Video Captioning）
   - 不仅理解静态帧，还能理解动态变化
   - 如"视频中物体是如何运动的"

⑤ 新 LLM 底座: Qwen2.5
   - 更强的推理能力（数学、代码、逻辑）
   - 更好的指令跟随

⑥ 更多多语言支持
   - 从 29 语言扩展到 100+ 语言
```

### 3.2 AnyRes 2.0 动态分辨率

```
Qwen2.5-VL 的分辨率方案进一步优化:

Step 1: 对图像进行"复杂度分析"
  简单图像（纯色背景、单一物体）→ 少 token
  复杂图像（人群、文字、细节丰富）→ 多 token
  → 智能分配计算资源

Step 2: 选择最优分辨率
  预定义分辨率集合:
    - 336²（基本）
    - 672²（中等）
    - 1008²（高）
    - 1344²（超高）
  根据图像内容复杂度自动选择

Step 3: 2D-RoPE 编码
  不管选哪个分辨率，都用 2D-RoPE 保持空间位置
  → 模型看到的是"位置敏感"的 token 序列

效果:
  - 简单图像: 用 336²，只生成约 600 token → 快
  - 复杂图像: 用 1344²，生成约 10000 token → 细节丰富
  - 平均: 比固定高分辨率方案节省 ~50% 计算量
```

### 3.3 Agent 能力详解

Qwen2.5-VL 加入了多模态 Agent 能力，这是其他开源 VLM 在 2025 年初不具备的：

```
Function Calling:

  输入: 图像 + "What's the weather in this city?"
  模型内部推理: 
    1. 图像理解 → 识别城市名（如"Shanghai"）
    2. 选择工具 → get_weather(city="Shanghai")
    3. 生成函数调用 → <function_call>get_weather(city="Shanghai")</function_call>
  外部执行:
    调用 API → 返回天气数据
  模型最终输出:
    "The weather in Shanghai is 25°C and sunny."

Computer Use（计算机操作）:

  输入: 屏幕截图 + "帮我打开 Chrome 浏览器"
  模型输出: 
    1. 分析屏幕截图 → 找到 Chrome 图标位置
    2. 生成操作序列 →
       <action type="mouse_move" x="1420" y="300">
       <action type="click">
    3. 等待下一张屏幕截图 → 检查是否打开成功

MCP 集成:
  
  Qwen2.5-VL 可以接入 MCP Server
  → 通过视觉理解 + Function Calling 调用 MCP 工具
  → 这是"多模态 Agent"的一种实现方式
  → 但注意: MCP 是 2025 年的新概念，Qwen2.5-VL 是适配 MCP 的模型之一
```

### 3.4 长视频理解（1h+）

```
Qwen2.5-VL 的视频理解升级:

视频采样策略优化:
  ① 场景检测（Scene Detection）
     → 不是均匀采样，而是检测场景切换
     → 每个场景取 2-3 帧关键帧
     → 避免"长时间相同场景"的冗余 token
     
  ② 动态帧数
     → 简单视频（演讲）：~64 帧
     → 复杂视频（动作片）：~192 帧
     → 根据需要动态调整

  ③ 长上下文支持
     → Qwen2.5 LLM 的上下文窗口 128K
     → 支持 ~192 帧 × 256 token/帧 ≈ 49K 视觉 token
     → 再加上文本，不超过 128K
     
  效果:
    - 1 小时视频理解：能回答"视频中说了什么"
    - 时间定位："在第 35 分钟出现了什么"
    - 多人对话：区分说话人
```

---

## 四、Qwen3-VL（2025）：最新一代

截至 2025 年 7 月，Qwen3-VL 是 Qwen-VL 系列的最新版本。

### 4.1 主要升级

```
Qwen3-VL 的全新设计:

① Qwen3 LLM 底座
   - Qwen3-7B/14B/32B/72B/235B
   - 更强的多语言能力（支持 100+ 语言）
   - 更强的推理（数学、代码）
   - 更长的上下文（128K → 256K）

② 新的视觉编码器
   - 基于 Qwen2-VL 的自研 ViT 进一步升级
   - 更大的视觉编码器（~2B 参数）
   - 更好的多分辨率支持
   - 更强的细粒度视觉理解

③ 统一的视觉定位
   - 之前的 Qwen2-VL 需要用特殊的 <bbox> 标记
   - Qwen3-VL 支持自然的"点击坐标"输出
   
④ 视频生成理解
   - 不仅是理解视频内容
   - 还能生成视频时间轴 / 事件序列
```

### 4.2 Qwen2.5-VL vs Qwen3-VL 对比

| 维度 | Qwen2.5-VL | Qwen3-VL |
|------|-----------|---------|
| **LLM 底座** | Qwen2.5（128K ctx） | Qwen3（128K-256K ctx）|
| **视觉编码器** | 自研 ViT（~900M） | 自研 ViT v2（~2B） |
| **分辨率方案** | AnyRes 2.0 | AnyRes 2.5（进一步优化）|
| **视频支持** | 1h+ | 1h+（更强的事件定位）|
| **Agent 能力** | Function Calling + Computer Use | 同上 + MCP Client |
| **多语言** | 100+ 语言 | 100+ 语言（更强）|
| **参数量** | 3B / 7B / 72B | 7B / 14B / 32B / 72B / 235B |

---

## 五、Qwen-VL 系列的技术贡献总结

```
Qwen-VL 系列的四代演进:

Qwen-VL (2023.08)
  || 关键突破: 中英双语 VLM + 内建定位
  || 架构类型: LLaVA 式（Frozen ViT + MLP + 全参数微调 LLM）
  || 不足: 固定分辨率、视觉编码器不够强
  
  ▼ 全面革新

Qwen2-VL (2024.08)
  || 关键突破: 动态分辨率（NaViT + 2D-RoPE + 窗口注意力）
  || 架构类型: LLaVA 式但 ViT 自研、加 Resampler
  || 新增: 视频理解、多语言
  || 意义: 从"跟随 LLaVA"变成"自研架构"
  
  ▼ 全能进化

Qwen2.5-VL (2025.01)
  || 关键突破: AnyRes 2.0 + Agent 能力
  || 架构类型: 和 Qwen2-VL 一致但所有组件升级
  || 新增: 1h 视频、Function Calling、Computer Use、MCP
  || 意义: 从"视觉理解"到"多模态 Agent"
  
  ▼ 进一步 Scale

Qwen3-VL (2025)
  || 关键突破: 更大更强的 ViT 和 LLM
  || 架构类型: 延续 Qwen2-VL 的架构路线
  || 新增: 更强推理、更长上下文
  || 意义: Scaling Law 在 VLM 上继续生效
```

### 5.1 架构设计的关键经验

```
从 Qwen-VL 系列的进化中可以总结的经验:

1. 动态分辨率是 VLM 的必备能力
   - LLaVA-1.5 的固定 336² 到 2024 年已经不够
   - 所有领先 VLM（Qwen2-VL、InternVL2、LLaVA-NeXT）都支持动态分辨率
   - 区别只在具体实现（NaViT vs AnyRes tile）

2. 视觉编码器不能太小
   - Qwen-VL: CLIP ViT（300M-1.8B）
   - Qwen2-VL: 自研 ViT（~900M）
   - Qwen3-VL: ViT v2（~2B）
   - 趋势: 视觉编码器在变大（但不如 InternVL 的 6B 那么极端）

3. Resampler 优于简单 MLP
   - Qwen-VL: 简单 MLP（256 token，信息无损但缺少精炼）
   - Qwen2-VL: Resampler（256 token，精炼后送入 LLM）
   - 特别是动态分辨率场景：变长 → 固定长度的处理需要 Resampler

4. Agent 能力成为差异化方向
   - 2024 年 Qwen2-VL 和 LLaVA 在视觉理解上差距不大
   - 2025 年 Qwen2.5-VL 通过 Agent 能力拉开差距
   - "看"是基础，"看 + 做"才是进阶
```

### 5.2 Qwen-VL 系列 vs 其他开源 VLM

| 维度 | Qwen2.5-VL | InternVL2 | LLaVA-OneVision | LLaMA 3.2-V |
|------|-----------|-----------|----------------|------------|
| **中英双语** | ✅ **强** | ✅ 好 | ⚠️ 英文为主 | ✅ 英文为主 |
| **动态分辨率** | ✅ NaViT 风格 | ✅ Tile 风格 | ✅ Tile 风格 | ❌ 固定 |
| **视频理解** | ✅ 1h+ | ✅ 10min | ✅ 多帧 | ❌ |
| **Agent** | ✅ **Function Calling** | ❌ | ❌ | ❌ |
| **OCR** | ✅ **强** | ✅ 强 | ⚠️ 中 | ⚠️ 中 |
| **定位** | ✅ 内建 | ✅ 内建 | ❌ | ❌ |
| **模型规模** | 3B/7B/72B | 2B/8B/40B/76B | 8B | 11B/90B |

**Qwen2.5-VL 在 2025 年初的综合定位：**

```
如果你需要:
  - 中英双语 VLM → Qwen2.5-VL ✅ 第一选择
  - 多模态 Agent → Qwen2.5-VL ✅ 几乎唯一选择
  - 视频理解 → Qwen2.5-VL ✅ 时长最长
  - 最强 OCR → Qwen2.5-VL / InternVL2 ✅ 两者都好
  - 最简部署 → LLaVA-OneVision ✅ 架构最简单
  - 跟随 Meta 生态 → LLaMA 3.2-V ✅

Qwen2.5-VL 就是"六边形战士"——不是每个维度最强
但综合能力最强
```

---

## 六、Qwen-VL 系列的局限

| 局限 | 表现 | 原因 |
|------|------|------|
| **中文偏见** | 中文描述比英文更详细、更准确 | 训练数据中中文比例更高 |
| **图像生成** | 不能生成图像 | 纯 VLM（理解型），不是全模态 |
| **端到端音频** | 不支持音频输入输出 | 不是 Omni-Modal（Qwen-Omni 负责这块）|
| **极端分辨率** | 超大图（8K+）仍然吃力 | 窗口注意力在超大分辨率下效率下降 |
| **Agent 稳定** | 复杂多步操作成功率不够高 | Agent 能力较新，还在进化 |
| **参数量档位** | 3B/7B 之间、7B/72B 之间跳跃大 | 缺少 12B 和 30B 这种中间档 |

---

## 七、总结

> **Qwen-VL 系列是中国开源 VLM 的一面旗帜——从 2023 年追赶 LLaVA，到 2025 年成为中英双语最强、功能最全面的开源 VLM。它的四代进化本身就是一部 VLM 架构演进的微缩史：固定分辨率 → 动态分辨率；纯视觉理解 → 多模态 Agent；英文为主 → 100+ 语言。**

| 代次 | 发布时间 | 核心突破 | 意义 |
|------|---------|---------|------|
| **Qwen-VL** | 2023.08 | 中英双语 VLM + 内建定位 | 追上 LLaVA-1.5 |
| **Qwen2-VL** | 2024.08 | 动态分辨率 + 视频理解 | 架构自研、超越 LLaVA |
| **Qwen2.5-VL** | 2025.01 | Agent 能力 + 1h 视频 | 从"看"到"看 + 做" |
| **Qwen3-VL** | 2025 | 更大更强 + 长上下文 | Scaling 继续生效 |

> 一句话：**如果你需要一个"什么都能做"的开源 VLM（中文、英文、OCR、视频、操作计算机），Qwen2.5-VL 是 2025 年上半年的最佳选择。Qwen3-VL 在 2025 年下半年继续延续这个路线，但架构层面没有 Qwen-VL → Qwen2-VL 那种级别的革命性变化。**

---

**Sources:**
- [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966) — Bai et al., Alibaba 2023
- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191) — Wang et al., Alibaba 2024
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) — Alibaba 2025
- [Qwen-VL Official GitHub](https://github.com/QwenLM/Qwen-VL) — Alibaba
- [Qwen2-VL Official GitHub](https://github.com/QwenLM/Qwen2-VL) — Alibaba
- [Qwen2.5-VL Official GitHub](https://github.com/QwenLM/Qwen2.5-VL) — Alibaba
- [NaViT: Vision Transformers for Mobile Real-Time Applications](https://arxiv.org/abs/2307.06304) — 2023
- [2D RoPE: Position Encoding for Vision Transformers](https://arxiv.org/abs/2403.12345) — 2024 (Qwen2-VL 引用)
