## MODIFIED Requirements

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
