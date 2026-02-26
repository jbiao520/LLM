## MODIFIED Requirements

### Requirement: 混合专家模型主题深入版文档
系统 SHALL 提供混合专家模型的深入版文档，面向有机器学习基础的读者。
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
