## ADDED Requirements

### Requirement: 注意力机制主题科普版文档
系统 SHALL 提供注意力机制的科普版文档，面向零基础读者。

#### Scenario: 读者理解注意力的直观含义
- **WHEN** 读者阅读科普版文档
- **THEN** 读者能够通过类比理解"注意力"的概念（如阅读时关注重点词）

#### Scenario: 读者理解 Self-Attention 的核心思想
- **WHEN** 读者阅读科普版文档
- **THEN** 读者能够理解 Query、Key、Value 的直观含义

### Requirement: 注意力机制主题深入版文档
系统 SHALL 提供注意力机制的深入版文档，面向有机器学习基础的读者。

#### Scenario: 读者掌握 Self-Attention 数学原理
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够理解 Self-Attention 的完整数学公式、缩放因子的作用

#### Scenario: 读者理解 Multi-Head Attention
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够理解多头注意力的设计动机和计算过程

#### Scenario: 读者了解 Flash Attention
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够理解 Flash Attention 的优化原理

### Requirement: 注意力机制主题代码示例
系统 SHALL 提供可运行的注意力机制代码示例。

#### Scenario: 代码示例展示 Self-Attention 实现
- **WHEN** 用户运行 self_attention_example.py
- **THEN** 代码展示 Self-Attention 的从头实现和 PyTorch 使用

#### Scenario: 代码示例展示 Multi-Head Attention 实现
- **WHEN** 用户运行 mha_example.py
- **THEN** 代码展示 Multi-Head Attention 的实现细节
