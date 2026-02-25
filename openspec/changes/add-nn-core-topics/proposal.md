## Why

当前 `topics/embedding/` 目录只包含嵌入相关内容，但深度学习和 LLM 的核心组件远不止于此。为了让学习路径更加完整，需要补充神经网络和 Transformer 架构的基础组件文档。这些主题是理解现代大语言模型的关键。

## What Changes

在 `topics/` 目录下新增 7 个核心主题模块，每个模块遵循现有的文档结构（科普版、深入��、示例代码）：

- **normalization**: 归一化技术（LayerNorm, BatchNorm, RMSNorm 等）
- **activation**: 激活函数（ReLU, GELU, Swish, Softmax 等）
- **attention**: 注意力机制（Self-Attention, Multi-Head Attention, Flash Attention 等）
- **moe**: 混合专家模型（MoE 架构、路由策略、负载均衡等）
- **encoder**: 编码器（Transformer Encoder, BERT-style 等）
- **decoder**: 解码器（Transformer Decoder, GPT-style, 自回归生成等）
- **ffn**: 前馈网络层（FFN, MLP, SwiGLU 等）

## Capabilities

### New Capabilities

- `normalization-topic`: 归一化技术的科普版和深入版文档，包含 LayerNorm、BatchNorm、RMSNorm 的原理和代码示例
- `activation-topic`: 激活函数的科普版和深入版文档，包含 ReLU、GELU、Swish、Softmax 的原理��代码示例
- `attention-topic`: 注意力机制的科普版和深入版文档，包含 Self-Attention、Multi-Head、Flash Attention 的原理和代码示例
- `moe-topic`: 混合专家模型的科普版和深入版文档，包含 MoE 架构、路由策略的原理和代码示例
- `encoder-topic`: 编码器的科普版和深入版文档，包含 Transformer Encoder 的原理和代码示例
- `decoder-topic`: 解码器的科普版和深入版文档，包含 Transformer Decoder、自回归生成的原理和代码示例
- `ffn-topic`: 前馈网络层的科普版和深入版文档，包含 FFN、SwiGLU 的原理和代码示例

### Modified Capabilities

- `embedding-topic`: 更新 README.md 添加新主题的索引和学习路径

## Impact

- 新增 7 个主题目录，每个包含 `beginner-guide.md`、`advanced-guide.md` 和 `examples/` 目录
- 更新 `topics/` 目录的主 README.md
- 每个主题包含 2-3 个 Python 示例文件
- 文档风格与现有 embedding 模块保持一致
