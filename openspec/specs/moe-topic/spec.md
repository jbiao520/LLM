## ADDED Requirements

### Requirement: 混合专家模型主题科普版文档
系统 SHALL 提供混合专家模型(MoE)的科普版文档，面向零基础读者。

#### Scenario: 读者理解 MoE 的直观含义
- **WHEN** 读者阅读科普版文档
- **THEN** 读者能够通过类比理解"专家混合"的概念（如会诊时不同专家各司其职）

#### Scenario: 读者理解 MoE 的核心优势
- **WHEN** 读者阅读科普版文档
- **THEN** 读者能够理解 MoE 如何实现"参数量大但计算量小"

### Requirement: 混合专家模型主题深入版文档
系统 SHALL 提供混合专家模型的深入版文档，面向有���器学习基础的读者。
文档 SHALL 对每个数学公式提供详细中文解释，包含变量含义与直觉说明，以支持数学基础薄弱读者理解。

#### Scenario: 读者掌握路由策略
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够理解 Top-K 路由、Softmax 路由的数学原理

#### Scenario: 读者理解负载均衡问题
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够理解辅助损失函数如何解决专家负载不均衡问题

#### Scenario: 读者了解主流 MoE 架构
- **WHEN** 读者阅读深入版文档
- **THEN** 读者能够了解 Mixtral、DeepSeek-MoE 等架构的设计

#### Scenario: 读者理解公式含义
- **WHEN** 读者阅读深入版文档
- **THEN** 文档对每个数学公式提供中文解释，包含变量含义与直觉说明

### Requirement: 混合专家模型主题代码示例
系统 SHALL 提供可运行的 MoE 代码示例。

#### Scenario: 代码示例展示 MoE 路由实现
- **WHEN** 用户运行 moe_router_example.py
- **THEN** 代码展示 Top-K 路由的实现和负载均衡损失

#### Scenario: 代码示例展示稀疏 MoE 层实现
- **WHEN** 用户运行 sparse_moe_example.py
- **THEN** 代码展示完整的稀疏 MoE 层实现

### Requirement: MoE Workflow Diagram
The MoE topic SHALL include a diagram.md file showing the Mixture of Experts workflow.

#### Scenario: Reader visualizes MoE routing
- **WHEN** reader views topics/moe/diagram.md
- **THEN** they see a Mermaid flowchart showing router -> expert selection -> weighted combination flow
