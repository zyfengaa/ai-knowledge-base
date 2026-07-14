# CosyVoice 系列网络结构深度解剖（1.0 / 2.0 / 3.0）

> 阿里通义实验室 (FunAudioLLM) 出品 | 多语言零样本语音合成
> CosyVoice 1.0 (2024.07) → CosyVoice 2.0 (2024.12) → CosyVoice 3.0 (2025.12)

---

## 写在前面：三部曲的演进

CosyVoice 系列由阿里通义实验室开发，与 SenseVoice（ASR）组成 **FunAudioLLM** 语音大模型框架。三个版本的演进清晰反映了开源 TTS 的技术方向变化：

| 版本 | 时间 | 核心架构 | 参数量 | 关键改进 |
|------|------|---------|--------|---------|
| **1.0** | 2024.07 | LLM + Flow Matching（Conformer） | ~300M | 零样本克隆 + 18 方言 |
| **2.0** | 2024.12 | Qwen2.5 + 因果 Flow Matching | ~500M | 流式合成 <150ms + FSQ 码本 |
| **3.0** | 2025.12 | DiT + RLHF | ~500M | RL 对齐 + 方言扩展 |

> 本文以 **2.0** 为主版本分析，在结尾标注 1.0→2.0→3.0 的关键改进。

---

## 一、整体架构设计哲学

### 核心思想

> **"语义-声学分离：LLM 决定'说什么'，Flow Matching 决定'怎么说'。"**

CosyVoice 将 TTS 分解为三个阶段：
1. **Text → Token**：LLM 将文本条件转化为离散语音 token（语义层）
2. **Token → Mel**：Flow Matching 将语音 token 解码为连续 Mel 频谱（声学层）
3. **Mel → Waveform**：HiFi-GAN 声码器还原为波形（物理层）

这种分离的设计哲学——**LLM 负责内容理解，Flow matching 负责语音生成**——是 CosyVoice 区别于 VITS（联合训练）和 Bark（纯 LM 生成）的关键。

### 架构总览（2.0）

```
文本: "今天天气真不错"
    │
    ├── ① FSQ 语义 Tokenizer
    │   └── 文本 → 监督语义 token (25Hz)
    │
    ├── ② LLM (Qwen2.5-0.5B)
    │   └── 语义 token 自回归生成
    │
    ├── ③ Chunk-Aware Causal Flow Matching
    │   └── 语音 token → Mel 频谱 (50Hz ↑)
    │
    └── ④ HiFi-GAN Vocoder
        └── Mel 频谱 → 24kHz 波形
```

---

## 二、各模块深度解剖

### 2.1 Supervised Semantic Speech Tokenizer（S³ Tokenizer）

**定位**：将语音编码为离散的"监督语义 token"，作为 LLM 和 Flow Matching 之间的接口。

#### v1.0: 标准 VQ

```
音频 → SenseVoice Encoder (6 层 Trans.) → VQ(码本=4096)
    → 码本利用率仅 23%
```

#### v2.0: FSQ（Finite Scalar Quantization）

```
语音 → 编码器₁ (6 层 Transformer + RoPE)
    │
    ├── FSQ 量化:
    │   ├── D 维向量 → 每维量化到 [-K, K] (K=3, D=7)
    │   ├── 总码本: (2K+1)^D = 7^7 = 823,543 种组合
    │   └── 实际有效: 通过投影控制, 约 6561
    │
    ├── 编码器₂ + ASR Decoder (SenseVoice-Large)
    │
    └── 输出: 离散 token ID (25Hz 帧率)
```

| 参数 | VQ (v1.0) | FSQ (v2.0) | 含义 |
|------|-----------|-----------|------|
| 码本利用率 | **~23%** | **~100%** | VQ 大量码本死掉（从未被选中） |
| 有效码本数 | 约 940 | **约 6561** | FSQ 完全利用每维的组合空间 |
| 训练稳定性 | ⚠️ 码本坍塌 | ✅ 无需 EMA/替代梯度 | FSQ 的 ROUND 操作直通估计梯度 |

**FSQ 的核心优势**：标准 VQ 在训练中大量码本向量因为初始化差或训练不足而"死掉"（从未被匹配）。FSQ 直接对每个维度独立量化（round 到最近整数），不需要维护码本表——所有"码本组合"自动生成且天然可用。

---

### 2.2 LLM（语言模型）

**定位**：将文本条件转换为离散语音 token 序列。这是 CosyVoice 中"理解能力"的来源。

#### v1.0: 专用 Transformer-LM

```
文本编码器 (Conformer) + Speaker Embedding (CAM++, 192-D)
    → 拼接输入 → Transformer-LM (约 300M)
    → 预测语音 token
```

#### v2.0: Qwen2.5 骨干

```
输入: [文本 token + 语音 token + 说话人信息]
    │
    └── Qwen2.5-0.5B (500M)
        ├── 32 层 Transformer
        ├── RoPE 位置编码
        ├── SwiGLU FFN
        └── GQA (8Q / 4KV)
    │
    └── 输出: 语音 token 序列 (25Hz)
```

| 架构差异 | v1.0 | v2.0 |
|---------|------|------|
| 文本编码器 | 单独的 Conformer | **直接使用 Qwen2.5 内置能力** |
| 说话人嵌入 | CAM++ 192-D | **移除**（发现会泄露语种信息） |
| 训练方式 | 标准 LM 预训练 | **添加 DPO 训练**提升发音准确度 |
| 参数量 | ~300M | ~500M |

**v2.0 删除说话人嵌入的理由**：实验发现，显式的说话人嵌入（如 CAM++ 向量）会"泄漏"与语种相关的副语言信息——中文说话人嵌入和英文说话人嵌入在向量空间中天然聚类。LLM 学到这个模式后，跨语言克隆时音色不一致。去掉说话人嵌入后，让 LLM 通过上下文（prompt）隐式学习音色。

---

### 2.3 Conditional Flow Matching Model（条件流匹配模型）

**定位**：CosyVoice 的声学模型——将离散语音 token 解码为连续 Mel 频谱。这是 CosyVoice 三个版本中变化最大的模块。

#### Flow Matching 原理回顾

```
Flow Matching 是一个生成框架（非扩散模型）:

传统扩散模型:
  数据 → 加噪 N 步（正向）→ 去噪 N 步（逆向）
  每一步都是独立的去噪预测

Flow Matching:
  数据 → 定义一条连续路径 φ_t (从噪声到数据的直线)
  学习路径上的"速度场" ν_t(φ_t, t)
  推理时: 从噪声出发，沿着 ν_t 走少量几步 ODE 求解
```

**Flow Matching vs 扩散模型的关键差异**：

| 维度 | 扩散模型 (DDPM) | Flow Matching |
|------|----------------|--------------|
| 生成路径 | 多步离散去噪 | **连续 ODE 路径** |
| 步数 | 50-1000 步 | **5-20 步** |
| 训练目标 | 预测噪声 ε | **预测速度场 ν** |
| 条件控制 | Classifier-Free Guidance | 条件 CFG |

#### v1.0 Flow: 非因果 Conformer

```
输入: 语音 token (25Hz) + Speaker Embedding
    │
    ├── Masked Diffusion with Xvec
    │   ├── Conformer Encoder (非因果)
    │   ├── 全局注意力
    │   └── 输出: Mel (50Hz, 上采样 2×)
    │
    └── 限制: 必须等整句生成完毕→非流式
```

#### v2.0 Flow: Chunk-Aware Causal Flow Matching（核心创新）

```
输入: 语音 token [batch, T, d]
    │
    ├── Chunk-Aware Causal Transformer-UNet:
    │
    │   每帧可看到的上下文:
    │   ├── Non-causal (Full): 全部上下文——离线高精度
    │   ├── Full-causal: 仅历史帧——低延迟流式
    │   ├── Chunk-M: 历史 + 未来 M 帧——首包快速输出
    │   └── Chunk-2M: 历史 + 未来 2M 帧——后续高质量输出
    │
    │   Look-ahead 卷积:
    │   │   右填充 1D 卷积, pad=P, kernel=P+1
    │   │   → 为因果模块暴露有限未来信息
    │   │
    │   └── 训练策略: 4 种 mask 类型均匀采样 (各 25%)
    │       → 单一模型同时支持流式 + 非流式
    │
    └── 输出: Mel 频谱 [batch, T_mel=2×T, 80]
```

**v2.0 Flow 网络细节**：

| 参数 | 值 | 含义 |
|------|-----|------|
| Mel 帧率 | **50Hz** | 语音 token 25Hz 经 2× 上采样对齐 |
| Chunk size | 25 | 每个 chunk 包含 25 帧 (~0.5s) |
| ODE Solver | Euler | 固定步长欧拉法 |
| NFE (Function Evaluations) | **10** | 仅需 10 步即可生成 |
| 时间调度器 | Cosinet: t := 1 - cos(½tπ) | 在生成初期分配更多步数 |
| CFG β | 0.7 | Classifier-Free Guidance 强度 |
| 训练损失 | L1 loss | 匹配预测和真实速度场 |

#### v3.0 Flow: DiT

```
v3.0 将 Flow 的 Backbone 从 Conformer/CNN 换成 DiT (Diffusion Transformer):
  - 每个空间位置是独立 token
  - Self-Attention 捕捉全局依赖
  - 更灵活的条件注入
  
改动理由: Conformer 的局部卷积在流式场景中效果有限,
DiT 的自注意力在"有限未来上下文"下对齐更准
```

---

### 2.4 HiFi-GAN Vocoder（声码器）

**定位**：Mel 频谱 → 24kHz 波形。所有版本使用相同结构。

```
Mel [batch, T_mel, 80]
    │
    ├── 转置卷积上采样 (×2, ×4, ×8)
    │
    ├── MRF (Multi-Receptive Field Fusion)
    │   ├── ResBlock(k=3) + ResBlock(k=7) + ResBlock(k=11)
    │   └── 求和
    │
    └── 输出: [batch, 1, T_audio]
```

| 参数 | 值 |
|------|-----|
| 采样率 | 24000 Hz |
| hop_length | 480 (50Hz 帧率 × 480 = 24000) |
| Mel 带数 | 80 |

---

## 三、流式合成机制详解（2.0）

这是 CosyVoice 2.0 与 1.0 最本质的区别——**从冷启动到说话的延迟从 >300ms 降到 ~150ms**。

### 文本-语音 Token 交错

```
非流式 (1.0):
  [S] 全部文本 token [T] 全部语音 token [E]
  → 必须等整句文本 token 全部生成 → 再全部语音 token → 开始播放

流式 (2.0):
  [S] 文本5 [T] 语音15 [FILL] 文本5 [T] 语音15 [FILL] ... [E]
  → 只需 5 个文字就能开始合成语音 → 实时播放

其中:
  [FILL] token: LLM 预测到 FILL 时, 自动补充下 N 个文本 token
  文本/语音比例 (N:M) = (5:15) @ 25Hz
```

### 首包延迟分解

```
用户输入: "今天天气真不错" (7 个字)
    │
    │── 用户输入前 5 个字 "今天天气真" → 约 0.3s
    │
    ├── LLM 预测首批语音 token: ~20ms
    │
    ├── Flow Matching (10步 ODE): ~80ms
    │
    ├── HiFi-GAN 解码: ~30ms
    │
    └── 开始播放: ~20ms (填充 DAC 缓冲区)
        └── 总延迟: ~150ms
        ← 在人耳无感知范围 (<200ms)
```

---

## 四、CosyVoice 1.0 → 2.0 → 3.0 全景对比

| 维度 | CosyVoice 1.0 (2024.07) | CosyVoice 2.0 (2024.12) | CosyVoice 3.0 (2025.12) |
|------|------------------------|------------------------|------------------------|
| **LLM Backbone** | 专用 Transformer-LM | **Qwen2.5-0.5B** | 同 2.0 + RL 训练 |
| **量化方式** | VQ (4096, 利用率 23%) | **FSQ** (利用率 100%) | FSQ |
| **Flow 架构** | Conformer (非因果) | **因果 Transformer-UNet** | **DiT (因果)** |
| **流式能力** | ❌ 不支持 | ✅ **<150ms** | ✅ 同 2.0 |
| **方言/语言** | 9 语言 + 18 方言 | 同 1.0 | **方言扩展 + 低资源语言** |
| **说话人相似度** | SIM ~0.78 | SIM ~0.87 | **SIM > 0.92 (RL 对齐)** |
| **参数量** | ~300M | ~500M | ~500M |
| **开源许可** | Apache 2.0 | Apache 2.0 | Apache 2.0 |

### "删除"了什么——CosyVoice 2.0 的精简哲学

2.0 删除了 1.0 中的两个模块：

```
v1.0: 文本编码器 (Conformer) → 拼接 Speaker Embedding → LLM → Flow
v2.0: [删除 文本编码器] [删除 Speaker Embedding] → LLM (Qwen2.5) → Flow
```

**删除文本编码器**：Qwen2.5 本身已经是一个经过预训练的强大语言模型——它自己就能理解文本，不需要独立的 Conformer 编码器。这是将"理解"职责从专用模块转移到 LLM 的范例。

**删除 Speaker Embedding**：CAM++ 嵌入会泄漏语种信息，导致跨语言克隆时音色不一致。让 Flow 直接从 prompt 音频中隐式学习说话人特征。

---

## 五、实际部署效果

### 评测指标

| 指标 | CosyVoice 1.0 | CosyVoice 2.0 | 意义 |
|------|:---:|:---:|------|
| MOS (中文) | ~4.2 | **~4.4** | 主观音质评分 |
| 说话人相似度 SIM | ~0.78 | **~0.87** | 克隆音色与目标音色的接近程度 |
| 发音错误率 (WER) | baseline | **降低 30-50%** | 绕口令/多音字/生僻字 |
| 首包延迟 | >300ms (非流式) | **~150ms** | 从输入到首次听到声音 |
| 流式 vs 非流式 WER 差距 | N/A (无流式) | **<0.1%** | 流式几乎无损 |

### 生产部署

- **阿里云 ModelScope**：提供 CosyVoice 系列在线推理 API
- **DashScope API**：阿里云商业版语音合成接口
- **HuggingFace**：FunAudioLLM 官方仓库，社区下载量百万级

---

## 六、总结：一张图看穿 CosyVoice

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CosyVoice 2.0 架构全景                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  文本: "今天天气真不错"                                              │
│       │                                                             │
│  ┌────┴───────────────┐                                             │
│  │  Qwen2.5-0.5B LLM  │  32层 SwiGLU + GQA + RoPE                  │
│  │  └─ 自回归预测       │  文本 token → 语音 token (25Hz)              │
│  └────┬───────────────┘                                             │
│       │  离散语音 token [25Hz, 6561 vocab]                          │
│       │                                                             │
│  ┌────┴──────────────────────────────────────────────────────┐      │
│  │  Chunk-Aware Causal Flow Matching                         │      │
│  │                                                             │      │
│  │  ├── 4 种 Chunk Masks: Non-causal / Causal / M / 2M       │      │
│  │  ├── Look-ahead Conv (kernel=P+1)                          │      │
│  │  ├── OT-Flow ODE: Ẋₜ = ωₜ(Xₜ|X₁), NFE=10                 │      │
│  │  └── 输出: Mel 频谱 [50Hz, 80 bands]                       │      │
│  └────┬──────────────────────────────────────────────────────┘      │
│       │                                                             │
│  ┌────┴────────────┐                                                │
│  │  HiFi-GAN       │  MRF + 转置卷积 → 波形                         │
│  └────┬────────────┘                                                │
│       │                                                             │
│  波形: 24kHz                                                        │
│                                                                     │
│  一句话总结 CosyVoice：                                             │
│  "Qwen2.5 理解文本生成语音 token，Causal Flow Matching(10步)         │
│   实时解码 Mel，HFG HiFi-GAN 快速还原波形——150ms 流式首包。"          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 附录：关键配置速查

### CosyVoice 2.0 完整配置

```python
# FSQ Tokenizer
tokenizer = {
    "input_frame_rate": 50,      # 原始帧率
    "output_frame_rate": 25,     # 量化后帧率
    "quantizer": "FSQ",
    "fsq_params": {
        "dimensions": [7, 7, 7, 7, 7, 7, 7],  # 7 维，每维范围 [-3,3]
        "flatten": True,
    }
}

# LLM (Qwen2.5-0.5B)
llm = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,  # GQA
    "intermediate_size": 4864,
    "vocab_size": 151936,
    "rope_theta": 1000000,
}

# Flow Matching
flow = {
    "feature_dim": 80,           # Mel 带数
    "hidden_dim": 512,
    "num_layers": 12,
    "chunk_size": 25,
    "lookahead_pad": 2,
    "num_timesteps": 10,         # NFE
    "scheduler": "cosinet",
    "cfg_beta": 0.7,
    "token_mel_ratio": 2,        # 25Hz token → 50Hz Mel
}

# HiFi-GAN
vocoder = {
    "sample_rate": 24000,
    "hop_length": 480,
    "n_mels": 80,
    "upsample_rates": [8, 6, 5, 2],
    "upsample_kernel_sizes": [16, 12, 10, 4],
    "resblock_kernel_sizes": [3, 7, 11],
}

# 训练
training = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "lr_scheduler": "warmup_cosine",
    "warmup_steps": 10000,
    "num_epochs": 100,
    "mask_sampling": ["full", "causal", "chunk_m", "chunk_2m"],
    "mask_weights": [0.25, 0.25, 0.25, 0.25],
}
```

---

*本文基于 CosyVoice 论文 (arXiv:2407.05407)、CosyVoice 2 论文 (arXiv:2412.10117)、FunAudioLLM GitHub 仓库 (Apache 2.0) 及 HuggingFace 模型文档整理分析。*

**Sources:**
- [CosyVoice: A Scalable Multilingual Zero-shot TTS based on Supervised Semantic Tokens - 阿里 (arXiv 2024)](https://arxiv.org/abs/2407.05407)
- [CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models - 阿里 (arXiv 2024)](https://arxiv.org/abs/2412.10117)
- [FunAudioLLM: Voice Understanding and Generation Foundation Models - GitHub (Apache 2.0)](https://github.com/FunAudioLLM)
- [CosyVoice on HuggingFace - FunAudioLLM](https://huggingface.co/FunAudioLLM)
- [CosyVoice Demo](https://funaudiollm.github.io)
