## Context

为 Embedding 学习模块添加 Python 代码示例。示例代码需要：
- 展示 LLM 中真实使用的 embedding 技术
- 包含完整的中英文注释
- 代码片段清晰易懂，不要求完整运行

## Goals / Non-Goals

**Goals:**
- Word Embedding 示例：Word2Vec、GloVe、BERT embedding 调用
- Position Embedding 示例：Sinusoidal encoding、RoPE
- 代码注释详细，中英文对照
- 展示实际 LLM 应用中的用法

**Non-Goals:**
- 不提供可独立运行的完整程序
- 不包含训练 embedding 的代码
- 不涉及 embedding 可视化

## Decisions

### 1. Word Embedding 示例内容

| 文件 | 内容 |
|------|------|
| `word2vec_example.py` | Gensim Word2Vec 加载和使用 |
| `glove_example.py` | GloVe 向量加载和相似度计算 |
| `bert_embedding_example.py` | HuggingFace Transformers 获取 BERT 词向量 |

**理由**：覆盖从传统方法到现代预训练模型的演进。

### 2. Position Embedding 示例内容

| 文件 | 内容 |
|------|------|
| `sinusoidal_pe_example.py` | 正弦位置编码的 PyTorch 实现 |
| `rope_example.py` | RoPE 旋转位置编码的实现 |

**理由**：Sinusoidal 是 Transformer 原始方法，RoPE 是现代 LLM 主流方法。

### 3. 注释风格

- 每个代码块前有中文说明
- 关键行有行内中英文注释
- 函数/类有 docstring

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|----------|
| 代码可能因库版本更新而过时 | 注明依赖版本，使用稳定 API |
| 部分示例需要预训练模型 | 提供模型加载的替代方案说明 |
