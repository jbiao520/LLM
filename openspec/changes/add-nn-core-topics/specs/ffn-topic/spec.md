## ADDED Requirements

### Requirement: 前馈网络层主题科普版文档
系统 SHALL 提供前馈网络层的科普版文档，面向零基础读者。

#### Scenario: 读者理解 FFN 的作用
- **WHEN** 读者阅读科普版文档
- **THEN** 读者能够理解 FFN 在 Transformer 中的作用（特征变换）

#### Scenario: 读者理解 FFN 的结构
- **WHEN** 读者阅读科普版文档
- **THEN** 读者能够直观理解"升维-激活-降维"的过程

### Requirement: 前馈网络层主题深入版文档
系统 SHALL 提供前馈网络层的深入版文档，面向有机器学习基础的读者。

#### Scenario: 读者掌握标准 FFN 结构
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够理解标准 FFN 的数学公式和参数量计算

#### Scenario: 读者理解 SwiGLU
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够理解 SwiGLU 的设计动机和数学定义

#### Scenario: 读者了解 FFN 变体
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够了解 GLU、GEGLU 等变体的设计

### Requirement: 前馈网络层主题代码示例
系统 SHALL 提供可运行的 FFN 代码示例。

#### Scenario: 代码示例展示标准 FFN 实现
- **WHEN** 用户运行 ffn_example.py
- **THEN** 代码展示标准 Transformer FFN 的实现

#### Scenario: 代码示例展示 SwiGLU 实现
- **WHEN** 用户运行 swiglu_example.py
- **THEN** 代码展示 SwiGLU 的实现和与标准 FFN 的对比
