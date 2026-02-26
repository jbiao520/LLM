# LLM 核心组件学习指南

本目录包含大语言模型（LLM）核心组件的学习资料，涵盖从基础到进阶的完整知识体系。

## 模块结构

```
topics/
├── embedding/        # 嵌入层
│   ├── word-embedding/
│   └── position-embedding/
├── normalization/    # 归一化
├── activation/       # 激活函数
├── attention/        # 注意力机制
├── encoder/          # 编码器
├── decoder/          # 解码器
├── ffn/              # 前馈网络层
├── moe/              # 混合专家模型
├── fine-tuning/      # 微调技术
│   ��── lora-finetune/
│   └── full-finetune/
└── inference/        # 推理优化
    ├── quantization/
    └── inference-acceleration/
```

## 推荐学习路径

### 第一阶段：基础知识

按照以下顺序学习基础组件：

1. **[Embedding（嵌入）](embedding/README.md)**
   - 理解如何将离散符号转换为连续向量
   - 词嵌入和位置嵌入是所有 LLM 的输入层

2. **[归一化（Normalization）](normalization/beginner-guide.md)**
   - LayerNorm、RMSNorm 的原理
   - 理解为什么 Transformer 需要 Pre-LN

3. **[激活函数（Activation）](activation/beginner-guide.md)**
   - ReLU、GELU、Swish 的特点
   - 为什么 Transformer 使用 GELU

### 第二阶段：核心机制

4. **[注意力机制（Attention）](attention/beginner-guide.md)**
   - Self-Attention 的核心思想
   - Multi-Head Attention 的设计动机
   - 这是 Transformer 的"心脏"

5. **[前馈网络层（FFN）](ffn/beginner-guide.md)**
   - FFN 的"升维-激活-降维"结构
   - SwiGLU 等现代变体
   - 知识存储的载体

### 第三阶段：架构理解

6. **[编码器（Encoder）](encoder/beginner-guide.md)**
   - 双向上下文理解
   - BERT 架构详解
   - 适合理解任务

7. **[解码器（Decoder）](decoder/beginner-guide.md)**
   - 自回归生成原理
   - GPT 架构详解
   - KV Cache 优化

### 第四阶段：进阶主��

8. **[混合专家模型（MoE）](moe/beginner-guide.md)**
   - 稀疏激活原理
   - 路由策略设计
   - 现代大模型的核心技术

### 第五阶段：微调与部署

9. **[微调技术（Fine-tuning）](fine-tuning/README.md)**
   - LoRA/QLoRA 参数高效微调
   - 全参数微调
   - 模��适配特定任务

10. **[推理优化（Inference）](inference/README.md)**
    - 模型量化（GPTQ、AWQ）
    - 推理加速（vLLM、KV Cache）
    - 高效部署大模型

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
| [Embedding](embedding/) | 输入表示 | 所有 LLM |
| [Normalization](normalization/) | 稳定训练 | 所有 LLM |
| [Attention](attention/) | 信息聚合 | 所有 Transformer |
| [FFN](ffn/) | 特征变换 | 所有 Transformer |
| [Encoder](encoder/) | 双向理解 | BERT |
| [Decoder](decoder/) | 自回归生成 | GPT |
| [MoE](moe/) | 高效扩展 | Mixtral, DeepSeek |
| [Fine-tuning](fine-tuning/) | 模型适配 | 领域定制 |
| [Inference](inference/) | 高效部署 | 生产环境 |

### 按模型架构

| 模型类型 | 核心组件 | 学习重点 |
|----------|----------|----------|
| BERT | Encoder | 双向注意力、MLM 预训练 |
| GPT | Decoder | 因果掩码、自回归生成、KV Cache |
| T5 | Encoder + Decoder | Cross-Attention、条件生成 |
| LLaMA | Decoder (w/ RMSNorm + SwiGLU) | 现代优化技术 |
| Mixtral | MoE Decoder | 稀疏路由、负载均衡 |

## 学习建议

1. **循序渐进**：按照推荐路径学习，前面的知识是后面的基础
2. **动手实践**：运行 `examples/` 中的代码，加深理解
3. **对比学习**：比较 Encoder 和 Decoder 的区别
4. **关注细节**：如 Pre-LN vs Post-LN、不同激活函数的选择

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
