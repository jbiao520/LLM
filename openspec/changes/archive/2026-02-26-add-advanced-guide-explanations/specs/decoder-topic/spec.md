## MODIFIED Requirements

### Requirement: 解码器主题深入版文档
系统 SHALL 提供解码器的深入版文档，面向有机器学习基础的读者。
文档 SHALL 对每个数学公式提供详细中文解释，包含变量含义与直觉说明，以支持数学基础薄弱读者理解。

#### Scenario: 读者掌握 Transformer Decoder 结构
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够理解 Decoder 层的完整结构（Masked Self-Attention + Cross-Attention + FFN）

#### Scenario: 读者理解因果掩码
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够理解因果掩码如何防止信息泄露

#### Scenario: 读者理解 KV Cache
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够理解 KV Cache 如何加速推理

#### Scenario: 读者了解 GPT 架构
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够了解 GPT 系列模型的架构演进

#### Scenario: 读者理解公式含义
- **WHEN** 读者阅读深入版文档
- **THEN** 文档对每个数学公式提供中文解释，包含变量含义与直觉说明
