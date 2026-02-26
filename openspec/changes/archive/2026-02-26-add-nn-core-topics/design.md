## Context

当前 `topics/embedding/` 目录已经建立了良好的文档结构模式：
- 每个主题有独立的目录
- 包含 `beginner-guide.md`（科普版，面向零基础读者）
- 包含 `advanced-guide.md`（深入版，面向有 ML 基础的读者）
- 包含 `examples/` 目录，提供 Python 代码示例

本变更将此模式扩展到 7 个新的神经网络核心主题。

## Goals / Non-Goals

**Goals:**
- 创建 7 个新主题目录，每个包含科普版、深入版文档和代码示例
- 保持与现有 embedding 模块一致的文档风格和结构
- 每个主题的示例代码可独立运行，包含必要的注释
- 更新主 README.md 添加新主题的索引

**Non-Goals:**
- 不创建交互式教程或 Jupyter Notebook
- 不添加英文版本的文档（保持中文为主）
- 不涉及实际的模型训练代码，仅提供概念演示

## Decisions

### 1. 目录结构
采用与 embedding 模块相同的结构：
```
topics/
├── normalization/
│   ├── beginner-guide.md
│   ├── advanced-guide.md
│   └── examples/
├── activation/
│   └── ...
├── attention/
│   └── ...
├── moe/
│   └── ...
├── encoder/
│   └── ...
├── decoder/
│   └── ...
└── ffn/
    └── ...
```

### 2. 文档内容组织
- **科普版**: 使用类比、图表说明、避免数学公式，面向零基础读者
- **深入版**: 包含数学公式、算法细节、论文引用，面向有 ML 基础的读者
- **示例代码**: 使用 PyTorch 作为主要框架，包含详细的中英文注释

### 3. 主题内容规划

| 主题 | 科普版重点 | 深入版重点 | 示例代码 |
|------|-----------|-----------|----------|
| 归一化 | 为什么需要归一化、直观理解 | LayerNorm/BatchNorm/RMSNorm 数学原理 | layernorm_example.py, rmsnorm_example.py |
| 激活函数 | 为什么需要非线性、各函数特点 | ReLU/GELU/Swish 数学推导和梯度 | activation_comparison.py, gelu_example.py |
| 注意力机制 | 注意力的直观理解 | Self-Attention/MHA 数学原理、Flash Attention | self_attention_example.py, mha_example.py |
| MoE | 专家混合的直观理解 | 路由策略、负载均衡、Top-K 路由 | moe_router_example.py, sparse_moe_example.py |
| 编码器 | 编码器的作用 | Transformer Encoder 结构、BERT | encoder_example.py, bert_layer_example.py |
| 解码器 | 解码器的作用、自回归生成 | Transformer Decoder 结构、GPT、KV Cache | decoder_example.py, kv_cache_example.py |
| FFN | 前馈层的作用 | FFN/SwiGLU/GLU 变体 | ffn_example.py, swiglu_example.py |

## Risks / Trade-offs

**内容深度把控** → 参考现有 embedding 文档的深度，保持一致性

**代码示例的复杂度** → 每个示例控制在 200 行以内，聚焦单一概念

**主题间的依赖关系** → 在 README 中说明推荐的学习顺序
