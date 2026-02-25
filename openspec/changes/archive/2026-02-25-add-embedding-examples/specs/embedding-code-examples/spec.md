## ADDED Requirements

### Requirement: Word Embedding code examples

`topics/embedding/word-embedding/examples/` 目录 SHALL 包含展示常见词嵌入技术的 Python 代码示例。

#### Scenario: Word2Vec example exists
- **WHEN** 查看 `word-embedding/examples/` 目录
- **THEN** 存在 `word2vec_example.py` 文件，展示 Gensim Word2Vec 的加载和使用

#### Scenario: GloVe example exists
- **WHEN** 查看 `word-embedding/examples/` 目录
- **THEN** 存在 `glove_example.py` 文件，展示 GloVe 向量的加载和相似度计算

#### Scenario: BERT embedding example exists
- **WHEN** 查看 `word-embedding/examples/` 目录
- **THEN** 存在 `bert_embedding_example.py` 文件，展示使用 Transformers 获取 BERT ���向量

### Requirement: Position Embedding code examples

`topics/embedding/position-embedding/examples/` 目录 SHALL 包含展示常见位置编码技术的 Python 代码示例。

#### Scenario: Sinusoidal PE example exists
- **WHEN** 查看 `position-embedding/examples/` 目录
- **THEN** 存在 `sinusoidal_pe_example.py` 文件，展示正弦位置编码的实现

#### Scenario: RoPE example exists
- **WHEN** 查看 `position-embedding/examples/` 目录
- **THEN** 存在 `rope_example.py` 文件，展示旋转位置编码的实现

### Requirement: Bilingual comments

所有代码示例 SHALL 包含完整的中英文注释。

#### Scenario: Comments are bilingual
- **WHEN** 查看任意示例代码文件
- **THEN** 关键代码行包含中英文注释，函数/类包含中英文 docstring

### Requirement: LLM-relevant examples

代码示例 SHALL 展示 LLM 应用中真实使用的技术和 API 调用。

#### Scenario: Examples use production-relevant APIs
- **WHEN** 查看示例代码
- **THEN** 使用 Gensim、PyTorch、Transformers 等主流库的常用 API
