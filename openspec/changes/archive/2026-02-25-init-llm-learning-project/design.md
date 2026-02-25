## Context

这是一个用于学习大语言模型相关知识的 Python 项目。项目需要支持：
- 多个知识领域的分类管理（如 prompt-engineering、rag、fine-tuning 等）
- 每个领域下包含���体的知识点和代码示例
- 良好的 Python 开发体验（依赖管理、代码检查等）

当前项目目录为空，需要从零开始搭建。

## Goals / Non-Goals

**Goals:**
- 创建清晰的知识领域目录结构
- 配置现代 Python 项目工具链（uv/poetry、ruff、mypy）
- 支持不同知识点的代码隔离和依赖管理
- 易于扩展新的学习领域

**Non-Goals:**
- 不包含具体的 LLM 知识内容（后续逐步添加）
- 不配置 CI/CD 流程（学习项目暂不需要）
- 不创建复杂的多包结构（monorepo）

## Decisions

### 1. 目录结构设计

采用按知识领域分类的扁平结构：

```
LLM/
├── pyproject.toml          # 项目配置和依赖
├── .python-version         # Python 版本锁定
├── README.md               # 项目说明
├── src/                    # 共享代码（如有）
└── topics/                 # 知识领域目录
    ├── prompt-engineering/ # 提示工程
    ├── rag/                # 检索增强生成
    ├── fine-tuning/        # 微调
    ├── inference/          # 推理优化
    └── ...
```

**理由**：扁平结构简单直观，每个知识领域独立，便于管理和查找。

### 2. 依赖管理工具

使用 `uv` 作为依赖管理工具，备选 `poetry`。

**理由**：uv 速度快、兼容 pip/poetry 生态，适合学习项目。

### 3. 代码质量工具

- **Ruff**：代码格式化和 lint（替代 black、flake8、isort）
- **mypy**：类型检查

**理由**：Ruff 是现代 Python 工具链的标准选择，速度快、配置简单。

### 4. Python 版本

锁定 Python 3.11+。

**理由**：LLM 相关库（如 transformers、langchain）对 3.11+ 支持良好。

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|----------|
| 不同知识点可能需要不同版本的依赖 | 使用可选依赖组（extras）分离 |
| 知识领域划分可能需要调整 | 保持目录结构灵活，易于重命名和合并 |
