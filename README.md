# LLM Learning Project

大语言模型学习项目 - 系统学习 LLM 相关知识点，包含理论笔记和代码示例。

## 项目结构

```
LLM/
├── pyproject.toml          # 项目配置和依赖
├── .python-version         # Python 版本 (3.11+)
├── src/                    # 共享代码模块
└── topics/                 # 知识领域
    ├── prompt-engineering/ # 提示工程
    ├── rag/                # 检索增强生成
    ├── fine-tuning/        # 微调
    └── inference/          # 推理优化
```

## 知识领域

| 领域 | 描述 |
|------|------|
| [prompt-engineering](topics/prompt-engineering/) | 提示词设计与优化技术 |
| [rag](topics/rag/) | 检索增强生成系统构建 |
| [fine-tuning](topics/fine-tuning/) | 模型微调方法与实践 |
| [inference](topics/inference/) | 推理优化与部署 |

## 快速开始

```bash
# 安装依赖（需要 uv 或 pip）
uv sync --extra dev

# 或使用 pip
pip install -e ".[dev]"
```

## 开发工具

- **Ruff**: 代码格式化和 lint
- **mypy**: 类型检查

```bash
# 运行代码检查
ruff check .

# 运行类型检查
mypy src/
```

## 学习资源

- [Hugging Face Documentation](https://huggingface.co/docs)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
