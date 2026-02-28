# LLM Learning Project

大语言模型学习项目 - 系统学习 LLM 相关知识点，包含理论笔记和代码示例。

## 项目结构

```
LLM/
├── pyproject.toml          # 项目配置和依赖
├── package.json            # Node.js 依赖（PDF 生成）
├── .python-version         # Python 版本 (3.11+)
├── src/                    # 共享代码模块
├── scripts/                # 构建脚本
│   ├── build-pdf.mjs       # PDF 构建脚本
│   ├── pdf-config.yaml     # 章节配置
│   └── templates/          # 样式模板
├── output/                 # 输出文件
│   └── LLM学习指南.pdf
└── topics/                 # 知识领域
    ├── embedding/          # 嵌入层
    ├── normalization/      # 归一化
    ├── activation/         # 激活函数
    ├── attention/          # 注意力机制
    ├── ffn/                # 前馈网络
    ├── encoder/            # 编码器
    ├── decoder/            # 解码器
    ├── moe/                # 混合专家模型
    ├── fine-tuning/        # 微调
    └── inference/          # 推理优化（量化、加速）
```

## 知识领域

| 领域 | 描述 |
|------|------|
| [embedding](topics/embedding/) | 词嵌入与位置嵌入 |
| [normalization](topics/normalization/) | LayerNorm, BatchNorm 等 |
| [activation](topics/activation/) | ReLU, GELU, SwiGLU 等 |
| [attention](topics/attention/) | 自注意力机制 |
| [ffn](topics/ffn/) | 前馈网络层 |
| [encoder](topics/encoder/) | Transformer 编码器 |
| [decoder](topics/decoder/) | Transformer 解码器 |
| [moe](topics/moe/) | 混合专家模型 |
| [fine-tuning](topics/fine-tuning/) | 全量微调与 LoRA |
| [inference](topics/inference/) | 模型量化与推理加速 |

## 快速开始

### Python 环境

```bash
# 安装依赖（需要 uv 或 pip）
uv sync --extra dev

# 或使用 pip
pip install -e ".[dev]"
```

### PDF 电子书生成

将所有学习笔记合并为 PDF 电子书：

```bash
# 安装 Node.js 依赖
npm install

# 生成 PDF
npm run build:pdf

# 输出: output/LLM学习指南.pdf
```

PDF 特性：
- Mermaid 图表渲染
- LaTeX 数学公式
- 代码语法高亮
- 中文支持
- 代码示例附录

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
