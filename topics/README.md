# LLM 核心组件学习指南

本目录包含大语言模型（LLM）核心组件的学习资料，涵盖从基础到进阶的完整知识体系。

## 模块结构

```
topics/
├── tokenization/      # 分词器 (NEW)
├── embedding/         # 嵌入层
│   ├── word-embedding/
│   └── position-embedding/
├── normalization/     # 归一化
├── activation/        # 激活函数
├── attention/         # 注意力机制
├── encoder/           # 编码器
├── decoder/           # 解码器
├── generation/        # 文本生成 (NEW)
├── ffn/               # 前馈网络层
├── moe/               # 混合专家模型
├── attention-variants/ # 注意力变体 (NEW)
├── pre-training/      # 预训练 (NEW)
├── fine-tuning/       # 微调技术
│   ├── lora-finetune/
│   └── full-finetune/
├── alignment/         # 对齐技术 (NEW)
├── long-context/      # 长上下文 (NEW)
├── inference/         # 推理优化
│   ├── quantization/
│   └── inference-acceleration/
├── distributed-training/  # 分布式训练 (NEW)
├── compression/       # 模型压缩 (NEW)
│   ├── distillation/
│   ├── pruning/
│   └── sparse-attention/
├── agents/            # 智能体 (NEW)
│   ├── tool-use/
│   ├── planning/
│   ├── memory/
│   └── multi-agent/
└── multimodal/        # 多模态 (NEW)
    ├── vision/
    ├── vision-language/
    └── audio/
```

## 推荐学习路径

### 第一阶段：输入与输出

1. **[分词器（Tokenization）](tokenization/beginner-guide.md)** ⭐ NEW
   - 理解文本如何变成数字
   - BPE、tiktoken、SentencePiece
   - 所有 LLM 的入口

2. **[Embedding（嵌入）](embedding/README.md)**
   - 理解如何将离散符号转换为连续向量
   - 词嵌入和位置嵌入是所有 LLM 的输入层

3. **[归一化（Normalization）](normalization/beginner-guide.md)**
   - LayerNorm、RMSNorm 的原理
   - 理解为什么 Transformer 需要 Pre-LN

4. **[激活函数（Activation）](activation/beginner-guide.md)**
   - ReLU、GELU、Swish 的特点
   - 为什么 Transformer 使用 GELU

### 第二阶段：核心机制

5. **[注意力机制（Attention）](attention/beginner-guide.md)**
   - Self-Attention 的核心思想
   - Multi-Head Attention 的设计动机
   - 这是 Transformer 的"心脏"

6. **[前馈网络层（FFN）](ffn/beginner-guide.md)**
   - FFN 的"升维-激活-降维"结构
   - SwiGLU 等现代变体
   - 知识存储的载体

### 第三阶段：架构理解

7. **[编码器（Encoder）](encoder/beginner-guide.md)**
   - 双向上下文理解
   - BERT 架构详解
   - 适合理解任务

8. **[解码器（Decoder）](decoder/beginner-guide.md)**
   - 自回归生成原理
   - GPT 架构详解
   - KV Cache 优化

9. **[文本生成（Generation）](generation/beginner-guide.md)** ⭐ NEW
   - Temperature、Top-K、Top-P 采样
   - Beam Search
   - 控制模型输出的"创意度"

### 第四阶段：训练方法

10. **[混合专家模型（MoE）](moe/beginner-guide.md)**
    - 稀疏激活原理
    - 路由策略设计

11. **[注意力变体（Attention Variants）](attention-variants/beginner-guide.md)** ⭐ NEW
    - GQA、MQA 高效注意力
    - Flash Attention 原理
    - 减少 KV Cache，加速推理

12. **[预训练（Pre-training）](pre-training/beginner-guide.md)** ⭐ NEW
    - 下一个词预测
    - 缩放定律
    - 模型如何获得基础能力

### 第五阶段：对齐与优化

13. **[微调技术（Fine-tuning）](fine-tuning/README.md)**
    - LoRA/QLoRA 参数高效微调
    - 全参数微调
    - 模型适配特定任务

14. **[对齐（Alignment）](alignment/beginner-guide.md)** ⭐ NEW
    - SFT、RLHF、DPO
    - 如何让模型变得有帮助、诚实、无害
    - ChatGPT 的关键训练步骤

15. **[长上下文（Long Context）](long-context/beginner-guide.md)** ⭐ NEW
    - RoPE 缩放
    - YaRN、ALiBi
    - 扩展模型的"记忆容量"

16. **[推理优化（Inference）](inference/README.md)**
    - 模型量化（GPTQ、AWQ）
    - 推理加速（vLLM、KV Cache）
    - 高效部署大模型

### 第六阶段：高级应用

17. **[分布式训练（Distributed Training）](distributed-training/beginner-guide.md)** ⭐ NEW
    - 数据并行、张量并行、流水线并行
    - ZeRO、FSDP
    - 训练大模型的"工程基础"

18. **[模型压缩（Compression）](compression/)** ⭐ NEW
    - 知识蒸馏（Distillation）
    - 剪枝（Pruning）
    - 稀疏注意力（Sparse Attention）

19. **[智能体（Agents）](agents/)** ⭐ NEW
    - [工具调用（Tool Use）](agents/tool-use/beginner-guide.md)
    - [规划与推理（Planning）](agents/planning/beginner-guide.md)
    - [记忆系统（Memory）](agents/memory/beginner-guide.md)
    - [多智能体协作（Multi-Agent）](agents/multi-agent/beginner-guide.md)

20. **[多模态（Multimodal）](multimodal/)** ⭐ NEW
    - [视觉编码器（Vision）](multimodal/vision/beginner-guide.md)
    - [视觉语言模型（Vision-Language）](multimodal/vision-language/beginner-guide.md)
    - [音频处理（Audio）](multimodal/audio/beginner-guide.md)

## 每个模块的内容

每个主题目录包含：

| 文件 | 内容 | 适合读者 |
|------|------|----------|
| `beginner-guide.md` | 科普版，使用类比和直观解释 | 零基础读者 |
| `advanced-guide.md` | 深入版，包含数学推导和论文引用 | 有 ML 基础的读者 |
| `diagram.md` | 流程图解，Mermaid 可视化 | 所有读者 |
| `examples/` | Python 代码示例，可运行 | 所有读者 |

## 快速索引

### 按组件类型

| 组件 | 作用 | 代表模型 |
|------|------|----------|
| [Tokenization](tokenization/) | 文本→数字 | 所有 LLM |
| [Embedding](embedding/) | 输入表示 | 所有 LLM |
| [Normalization](normalization/) | 稳定训练 | 所有 LLM |
| [Attention](attention/) | 信息聚合 | 所有 Transformer |
| [FFN](ffn/) | 特征变换 | 所有 Transformer |
| [Encoder](encoder/) | 双向理解 | BERT |
| [Decoder](decoder/) | 自回归生成 | GPT |
| [Generation](generation/) | 输出控制 | GPT, Claude |
| [MoE](moe/) | 高效扩展 | Mixtral, DeepSeek |
| [Attention Variants](attention-variants/) | 高效注意力 | LLaMA 2/3 |
| [Pre-training](pre-training/) | 基础能力 | 所有 LLM |
| [Fine-tuning](fine-tuning/) | 任务适配 | 领域定制 |
| [Alignment](alignment/) | 安全对齐 | ChatGPT, Claude |
| [Long Context](long-context/) | 长文本处理 | GPT-4, Claude |
| [Inference](inference/) | 高效部署 | 生产环境 |

### 按模型架构

| 模型类型 | 核心组件 | 学习重点 |
|----------|----------|----------|
| BERT | Encoder | 双向注意力、MLM 预训练 |
| GPT | Decoder | 因果掩码、自回归生成、KV Cache |
| GPT-4 | Decoder + Alignment | RLHF、长上下文 |
| Claude | Decoder + Alignment | RLHF、Constitutional AI |
| LLaMA | Decoder (w/ GQA + RMSNorm + SwiGLU) | GQA、现代优化 |
| Mixtral | MoE Decoder | 稀疏路由、滑动窗口 |

## 学习建议

1. **循序渐进**：按照推荐路径学习，前面的知识是后面的基础
2. **动手实践**：运行 `examples/` 中的代码，加深理解
3. **对比学习**：比较 Encoder 和 Decoder 的区别
4. **关注细节**：如 Pre-LN vs Post-LN、不同激活函数的选择
5. **理解全貌**：从 Tokenization 到 Inference，理解完整的 LLM 流程

## 前置知识

- **科普版**：无需任何基础
- **深入版**：线性代数、概率论、机器学习基础
- **代码示例**：Python、PyTorch 基础

## 参考资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原始论文
- [BERT](https://arxiv.org/abs/1810.04805) - 双向编码器
- [GPT-3](https://arxiv.org/abs/2005.14165) - 自回归解码器
- [LLaMA](https://arxiv.org/abs/2302.13971) - 现代 LLM 架构
- [Mixtral](https://arxiv.org/abs/2401.04088) - MoE 架构
- [InstructGPT](https://arxiv.org/abs/2203.02155) - RLHF
- [DPO](https://arxiv.org/abs/2305.18290) - 直接偏好优化
