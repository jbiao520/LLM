## Why

当前 `fine-tuning` 和 `inference` 两个主题目录仅有简单的 README 占位，缺乏实际的学习内容。而其他主题（如 embedding、normalization）已按照完整的目录结构填充了内容。此外，所有主题都缺少可视化的工作流程图，不利于学习者快速理解核心概念。

## What Changes

### 1. 填充 Fine-tuning 主题内容

按照 embedding 的目录结构，创建以下子主题：
- `lora-finetune/` - LoRA/QLoRA 参数高效微调
- `full-finetune/` - 全参数微调

每个子主题包含：
- `beginner-guide.md` - 科普版入门指南
- `advanced-guide.md` - 深入版技术详解
- `diagram.md` - 工作流程图解
- `examples/` - 代码示例

### 2. 填充 Inference 主题内容

按照 embedding 的目录结构，创建以下子主题：
- `quantization/` - 模型量化技术
- `inference-acceleration/` - 推理加速技术

每个子主题包含：
- `beginner-guide.md` - 科普版入门指南
- `advanced-guide.md` - 深入版技术详解
- `diagram.md` - 工作流程图解
- `examples/` - 代码示例

### 3. 为现有主题添加 diagram.md

为以下主题添加 `diagram.md`（放在主题根目录）：
- `embedding/` - Embedding 工作流程图
- `normalization/` - 归一化工作流程图
- `activation/` - 激活函数工作流程图
- `attention/` - 注意力机制工作流程图
- `moe/` - MoE 工作流程图
- `encoder/` - Encoder 工作流程图
- `decoder/` - Decoder 工作流程图
- `ffn/` - FFN 工作流程图

## Capabilities

### New Capabilities

- `fine-tuning-lora-topic`: LoRA/QLoRA 参数高效微调主题，包括原理科普、技术深入、代码示例和流程图解
- `fine-tuning-full-topic`: 全参数微调主题，包括原理科普、技术深入、代码示例和流程图解
- `inference-quantization-topic`: 模型量化主题，包括原理科普、技术深入、代码示例和流程图解
- `inference-acceleration-topic`: 推理加速主题，包括原理科普、技术深入、代码示例和流程图解
- `topic-diagrams`: 为所有主题添加 Mermaid 流程图，形象展示各主题的工作流程

### Modified Capabilities

- `embedding-topic`: 添加 diagram.md 流程图
- `normalization-topic`: 添加 diagram.md 流程图
- `activation-topic`: 添加 diagram.md 流程图
- `attention-topic`: 添加 diagram.md 流程图
- `moe-topic`: 添加 diagram.md 流程图
- `encoder-topic`: 添加 diagram.md 流程图
- `decoder-topic`: 添加 diagram.md 流程图
- `ffn-topic`: 添加 diagram.md 流程图

## Impact

- 新增约 16 个 markdown 文档（4 个子主题 × 4 个文件）
- 新增约 8 个 Python 示例代码文件
- 修改 8 个现有主题目录，各添加 1 个 diagram.md
- 更新 `topics/fine-tuning/README.md` 和 `topics/inference/README.md`
- 更新 `topics/README.md` 主索引
