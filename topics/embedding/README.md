# Embedding（嵌入）

Embedding 是将离散的符号（如单词、位置）映射到连续向量空间的技术。它是现代深度学习和自然语言处理的基石。

## 模块结构

```
embedding/
├── word-embedding/      # 词嵌入
│   ├── beginner-guide.md
│   ├── advanced-guide.md
│   └── examples/
└── position-embedding/  # 位置嵌入
    ├── beginner-guide.md
    ├── advanced-guide.md
    └── examples/
```

## 学习路径

### 1. Word Embedding（词嵌入）

将词语转换为稠密向量表示，使语义相近的词在向量空间中距离相近。

- **科普版**: [beginner-guide.md](word-embedding/beginner-guide.md) - 适合零基础读者
- **深入版**: [advanced-guide.md](word-embedding/advanced-guide.md) - 适合有机器学习基础的读者

### 2. Position Embedding（位置嵌入）

为序列中的每个位置生成唯一的向量表示，让模型能够理解词序信息。

- **科普版**: [beginner-guide.md](position-embedding/beginner-guide.md) - 适合零基础读者
- **深入版**: [advanced-guide.md](position-embedding/advanced-guide.md) - 适合有机器学习基础的读者

## 学习建议

1. 先阅读 Word Embedding，理解"将词变成向量"的基本概念
2. 再学习 Position Embedding，理解"如何表示顺序信息"
3. 两者��合，理解 Transformer 的输入表示

## 前置知识

- 科普版：无需基础
- 深入版：线性代数、概率论基础
