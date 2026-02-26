## ADDED Requirements

### Requirement: 编码器主题科普版文档
系统 SHALL 提供编码器的科普版文档，面向零基础读者。

#### Scenario: 读者理解编码器的作用
- **WHEN** 读者阅读科普版文档
- **THEN** 读者能够理解编码器如何将输入序列转换为上下文表示

#### Scenario: 读者理解编码器的应用场景
- **WHEN** 读者阅读科普版文档
- **THEN** 读者能够了解 BERT 等编码器模型的应用（文本分类、NER 等）

### Requirement: 编码器主题深入版文档
系统 SHALL 提供编码器的深入版文档，面向有机器学习基础的读者。
文档 SHALL 对每个数学公式提供详细中文解释，包含变量含义与直觉说明，以支持数学基础薄弱读者理解。

#### Scenario: 读者掌握 Transformer Encoder 结构
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够理解 Encoder 层的完整结构（Self-Attention + FFN + Residual）

#### Scenario: 读者理解双向注意力
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够理解编码器如何实现双向上下文建模

#### Scenario: 读者了解 BERT 架构
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够了解 BERT 的预训练任务（MLM、NSP）和架构细节

#### Scenario: 读者理解公式含义
- **WHEN** 读者阅读深入版文档
- **THEN** 文档对每个数学公式提供中文解释，包含变量含义与直觉说明

### Requirement: 编码器主题代码示例
系统 SHALL 提供可运行的编码器代码示例。

#### Scenario: 代码示例展示 Encoder Layer 实现
- **WHEN** 用户运行 encoder_example.py
- **THEN** 代码展示 Transformer Encoder Layer 的实现

#### Scenario: 代码示例展示 BERT Layer 实现
- **WHEN** 用户运行 bert_layer_example.py
- **THEN** 代码展示 BERT 风格的编码层实现

### Requirement: Encoder Workflow Diagram
The encoder topic SHALL include a diagram.md file showing the Transformer Encoder workflow.

#### Scenario: Reader visualizes encoder architecture
- **WHEN** reader views topics/encoder/diagram.md
- **THEN** they see a Mermaid flowchart showing multi-layer encoder stack with self-attention and FFN
