## Context

本项目是一个 LLM 学习资源库，采用层次化的主题结构。当前已有 embedding、normalization、attention 等 8 个主题已填充完整内容，每个子主题包含：
- `beginner-guide.md` - 科普版（面向零基础）
- `advanced-guide.md` - 深入版（面向有 ML 基础）
- `examples/` - Python 代码示例

fine-tuning 和 inference 两个主题仅有 README 占位符，需要按照相同模式填充内容。

本次新增要求：为所有主题添加 `diagram.md`，使用 Mermaid 图表形象展示工作流程。

## Goals / Non-Goals

**Goals:**
- 填充 fine-tuning 主题的 2 个子主题（lora-finetune、full-finetune）
- 填充 inference 主题的 2 个子主题（quantization、inference-acceleration）
- 为所有 10 个主题添加 diagram.md 流程图
- 保持与现有主题一致的文档风格和质量

**Non-Goals:**
- 不创建新的主题目录结构（fine-tuning 和 inference 目录已存在）
- 不修改现有 beginner-guide.md 和 advanced-guide.md 的内容
- 不涉及代码执行或测试（纯文档内容）

## Decisions

### 1. 子主题划分

**Fine-tuning:**
- `lora-finetune/` - LoRA 和 QLoRA（主流 PEFT 方法）
- `full-finetune/` - 全参数微调（传统方法，作为对比）

**Inference:**
- `quantization/` - 量化技术（GPTQ、AWQ、INT8/INT4）
- `inference-acceleration/` - 推理加速（vLLM、TensorRT-LLM、KV Cache 优化）

**理由**: 按照技术相关性分组，每个子主题足够聚焦，便于深入学习。

### 2. Diagram 设计模式

使用 Mermaid 流程图，包含以下元素：
- **输入/输出**：用圆角矩形表示
- **处理步骤**：用矩形表示
- **决策点**：用菱形表示
- **数据流**：用箭头连接

每个 diagram.md 包含：
- 核心流程图（Mermaid 代码）
- 图解说明（配合图的文字解释）
- 关键概念标注

### 3. 代码示例选择

| 子主题 | 示例代码 |
|--------|----------|
| lora-finetune | lora_example.py, qlora_example.py |
| full-finetune | full_finetune_example.py |
| quantization | gptq_example.py, awq_example.py |
| inference-acceleration | vllm_example.py, kv_cache_optimization.py |

## Risks / Trade-offs

**风险 1：内容深度不均**
- → 每个文档参考现有 embedding 主题的篇幅和深度

**风险 2：Mermaid 图表在某些 Markdown 渲染器中不支持**
- → 在 diagram.md 中同时提供 Mermaid 代码和文字描述

**风险 3：技术更新导致内容过时**
- → 聚焦基础原理，版本相关内容标注日期和版本号
