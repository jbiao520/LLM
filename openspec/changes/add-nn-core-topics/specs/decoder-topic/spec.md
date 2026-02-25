## ADDED Requirements

### Requirement: 解码器主题科普版文档
系统 SHALL 提供解码器的科普版文档，面向零基础读者。

#### Scenario: 读者理解解码器的作用
- **WHEN** 读者阅读科普版文档
- **THEN** 读者能够理解解码器如何自回归地生成输出序列

#### Scenario: 读者理解自回归生成
- **WHEN** 读者阅读科普版文档
- **THEN** 读者能够理解"逐词生成"的过程

### Requirement: 解码器主题深入版文档
系统 SHALL 提供解码器的深入版文档，面向有机器学习基础的读者。

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

### Requirement: 解码器主题代码示例
系统 SHALL 提供可运行的解码器代码示例。

#### Scenario: 代码示例展示 Decoder Layer 实现
- **WHEN** 用户运行 decoder_example.py
- **THEN** 代码展示 Transformer Decoder Layer 的实现和因果掩码

#### Scenario: 代码示例展示 KV Cache 实现
- **WHEN** 用户运行 kv_cache_example.py
- **THEN** 代码展示 KV Cache 的实现和推理加速效果
