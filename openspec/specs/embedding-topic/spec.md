## ADDED Requirements

### Requirement: Embedding topic directory structure

项目 SHALL 在 `topics/` 下提供 `embedding/` 目录，包含词嵌入和位置嵌入两个子主题。

#### Scenario: Embedding directory exists
- **WHEN** 查看 `topics/` 目录
- **THEN** 存在 `embedding/` 子目录

#### Scenario: Subtopic directories exist
- **WHEN** 查看 `topics/embedding/` 目录
- **THEN** 存在 `word-embedding/` 和 `position-embedding/` 子目录

### Requirement: Dual-version documentation

每个子主题 SHALL 提供两个版本的讲解文件：科普版（beginner-guide.md）和深入版（advanced-guide.md）。

#### Scenario: Word embedding has dual versions
- **WHEN** 查看 `topics/embedding/word-embedding/` 目录
- **THEN** 存在 `beginner-guide.md` 和 `advanced-guide.md` 文件

#### Scenario: Position embedding has dual versions
- **WHEN** 查看 `topics/embedding/position-embedding/` 目录
- **THEN** 存在 `beginner-guide.md` 和 `advanced-guide.md` 文件

### Requirement: Beginner guide content style

科普版本 SHALL 使用通俗易懂的语言，避免复杂数学公式，使用类比和图解辅助理解。

#### Scenario: Beginner guide is accessible
- **WHEN** 阅读 `beginner-guide.md`
- **THEN** 内容不包含复杂数学推导，使用日常类比解释概念

### Requirement: Advanced guide content style

深入版本 SHALL 包含数学原理、公式推导、论文引用和技术实现细节。
深入版本 SHALL 对每个数学公式提供详细中文解释，包含变量含义与直觉说明，以支持数学基础薄弱读者理解。

#### Scenario: Advanced guide has depth
- **WHEN** 阅读 `advanced-guide.md`
- **THEN** 内容包含数学公式、原理推导和相关论文引用

#### Scenario: 读者理解公式含义
- **WHEN** 阅读 `advanced-guide.md`
- **THEN** 文档对每个数学公式提供中文解释，包含变量含义与直觉说明

### Requirement: Topic overview documentation

`topics/embedding/` 目录 SHALL 包含 `README.md` 概��文件，介绍模块内容和结构。

#### Scenario: README provides overview
- **WHEN** 查看 `topics/embedding/README.md`
- **THEN** 文件包含模块简介、目录结构和学习建议

### Requirement: Embedding Workflow Diagram
The embedding topic SHALL include a diagram.md file showing the embedding workflow.

#### Scenario: Reader visualizes embedding process
- **WHEN** reader views topics/embedding/diagram.md
- **THEN** they see a Mermaid flowchart showing token -> embedding lookup -> position addition -> output flow
