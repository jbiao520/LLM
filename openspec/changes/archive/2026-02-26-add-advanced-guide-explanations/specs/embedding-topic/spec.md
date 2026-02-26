## MODIFIED Requirements

### Requirement: Advanced guide content style

深入版本 SHALL 包含数学原理、公式推导、论文引用和技术实现细节。
深入版本 SHALL 对每个数学公式提供详细中文解释，包含变量含义与直觉说明，以支持数学基础薄弱读者理解。

#### Scenario: Advanced guide has depth
- **WHEN** 阅读 `advanced-guide.md`
- **THEN** 内容包含数学公式、原理推导和相关论文引用

#### Scenario: 读者理解公式含义
- **WHEN** 阅读 `advanced-guide.md`
- **THEN** 文档对每个数学公式提供中文解释，包含变量含义与直觉说明
