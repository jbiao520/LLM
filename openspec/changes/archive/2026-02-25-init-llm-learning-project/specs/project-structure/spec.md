## ADDED Requirements

### Requirement: Knowledge domain directories

项目 SHALL 提供按知识领域分类的目录结构，每个领域作为 `topics/` 下的独立子目录。

#### Scenario: Initial topic directories created
- **WHEN** 项目初���化完成
- **THEN** `topics/` 目录下存在以下子目录：
  - `prompt-engineering/`
  - `rag/`
  - `fine-tuning/`
  - `inference/`

### Requirement: Topic directory structure

每个知识领域目录 SHALL 包含 `README.md` 说明文件和 `examples/` 代码示例目录。

#### Scenario: Topic directory contains README
- **WHEN** 查看任意知识领域目录
- **THEN** 目录中存在 `README.md` 文件，包含该领域的简介

#### Scenario: Topic directory contains examples
- **WHEN** 查看任意知识领域目录
- **THEN** 目录中存在 `examples/` 子目录用于存放代码示例

### Requirement: Project root documentation

项目根目录 SHALL 包含 `README.md` 文件，说明项目目的、结构和如何开始学习。

#### Scenario: README contains project overview
- **WHEN** 查看 `README.md`
- **THEN** 文件包含项目目的说明和目录结构概览
