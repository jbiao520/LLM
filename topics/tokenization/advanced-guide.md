# 分词器（Tokenization）深入版

> 面向有机器学习基础读者的技术详解

## 概述

分词器是 LLM 流水线的第一步，直接影响模型的词汇覆盖率、序列长度和下游任务性能。本文深入分析 BPE、WordPiece、Unigram 等主流分词算法的数学原理。

## 分词问题的形式化

给定文本语料库 $\mathcal{C}$，目标是学习一个分词函数：

$$f: \text{Text} \rightarrow (t_1, t_2, ..., t_n)$$

其中 $t_i \in \mathcal{V}$，$\mathcal{V}$ 是词汇表。

### 优化目标

$$\min_{\mathcal{V}, f} \sum_{x \in \mathcal{C}} |f(x)|$$

<a id="formula-tokenization-1"></a>

**公式解释**
- **公式含义**：找到最优的词汇表和分词函数，使所有文本的 token 数量最少。
- **变量说明**：$\mathcal{V}$ 是词汇表；$f(x)$ 返回文本 $x$ 的 token 序列；$|f(x)|$ 是 token 数量。
- **直觉/作用**：更少的 token 意味着更高的效率和更好的压缩率。

## BPE（Byte Pair Encoding）

### 算法原理

BPE 是一种迭代合并算法，从字符级别开始，逐步合并最常见的相邻对。

### 训练过程

1. **初始化**：将所有单词拆分为字符序列

```
语料：["low", "lower", "newest"]
初始：{"l o w": 5, "l o w e r": 2, "n e w e s t": 6}
```

2. **统计频率**：计算所有相邻字符对的出现频率

3. **合并最高频对**：将最常见的对合并为新 token

4. **重复**：直到达到目标词汇表大小

### 伪代码

```python
def train_bpe(corpus, vocab_size):
    # 初始化：将所有词拆分为字符
    word_freqs = tokenize_to_chars(corpus)

    while len(vocab) < vocab_size:
        # 统计所有相邻对的频率
        pairs = count_pairs(word_freqs)

        # 找到最高频的对
        best_pair = max(pairs, key=pairs.get)

        # 合并该对
        word_freqs = merge_pair(word_freqs, best_pair)
        vocab.append(best_pair)

    return vocab
```

### 编码过程

给定训练好的词汇表，编码新文本：

```python
def encode(text, vocab):
    # 拆分为字符
    tokens = list(text)

    while len(tokens) > 1:
        # 找到词汇表中存在的最高优先级对
        pairs = get_pairs(tokens)
        best_pair = find_highest_priority_pair(pairs, vocab)

        if best_pair is None:
            break

        # 合并该对
        tokens = merge(tokens, best_pair)

    return tokens
```

### 复杂度分析

- **训练复杂度**：$O(V \cdot N)$，其中 $V$ 是目标词汇表大小，$N$ 是语料大小
- **编码复杂度**：$O(n \cdot m)$，其中 $n$ 是文本长度，$m$ 是最大 token 长度

## BBPE（Byte-level BPE）

### 动机

标准 BPE 在字符级别操作，可能遇到：
- 未知字符
- 不同编码问题
- 大小写不一致

BBPE 在**字节级别**操作，解决这些问题。

### 工作原理

1. 将文本转换为 UTF-8 字节序列
2. 在字节级别执行 BPE
3. 基础词汇表大小固定为 256（所有可能的字节）

### 优势

$$|\mathcal{V}_{base}| = 256$$

<a id="formula-tokenization-2"></a>

**公式解释**
- **公式含义**：BBPE 的基础词汇表包含 256 个字节值。
- **变量说明**：$\mathcal{V}_{base}$ 是基础词汇��。
- **直觉/作用**：任何文本都能表示为字节，保证 100% 覆盖率。

### GPT-2/3/4 的实现

```python
def bytes_to_unicode():
    """
    创建字节到 unicode 字符的映射
    避免控制字符，使用可打印字符
    """
    bs = list(range(ord("!"), ord("~")+1))  # 可打印 ASCII
    bs += list(range(ord("¡"), ord("¬")+1))
    bs += list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
```

## WordPiece

### 算法差异

WordPiece 与 BPE 的主要区别在于**合并标准**：

- **BPE**：基于频率合并
- **WordPiece**：基于似然提升合并

### 似然最大化

选择合并 $A, B$ 的标准：

$$\text{score}(A, B) = \frac{P(AB)}{P(A) \cdot P(B)}$$

<a id="formula-tokenization-3"></a>

**公式解释**
- **公式含义**：计算合并后 token 的概率与单独概率乘积的比值。
- **变量说明**：$P(\cdot)$ 是 token 在语料中的频率。
- **直觉/作用**：比值越高，说明 A 和 B 结合得越紧密。

### 标记方式

WordPiece 使用 `##` 标记非词首 token：

```
"unaffable" → ["un", "##aff", "##able"]
```

## Unigram Language Model

### 概率模型

Unigram 假设每个 token 独立生成：

$$P(x) = \prod_{t_i \in f(x)} P(t_i)$$

<a id="formula-tokenization-4"></a>

**公式解释**
- **公式含义**：文本的概率是其所有 token 概率的乘积。
- **变量说明**：$f(x)$ 是文本的分词结果；$P(t_i)$ 是 token 的概率。
- **直觉/作用**：用于评估不同分词方案的概率。

### 训练过程

1. **初始化**：使用所有可能的子词作为初始词汇表
2. **EM 迭代**：
   - E 步：计算每个 token 的期望计数
   - M 步：更新每个 token 的概率
3. **剪枝**：移除对似然贡献最小的 token
4. **重复**：直到达到目标词汇表大小

### 损失函数

$$\mathcal{L} = -\sum_{x \in \mathcal{C}} \log P(x)$$

<a id="formula-tokenization-5"></a>

**公式解释**
- **公式含义**：最小化所有训练文本的负对数似然。
- **变量说明**：$\mathcal{C}$ 是语料库；$P(x)$ 是文本概率。
- **直觉/作用**：让模型更好地拟合训练数据。

## SentencePiece

### 端到端设计

SentencePiece 的关键创新：
- **语言无关**：不依赖预分词（如空格分割）
- **纯数据驱动**：直接从原始文本学习

### 空格处理

SentencePiece 将空格视为特殊字符 `_`：

```
"Hello world" → ["▁Hello", "▁world"]
```

### 支持的算法

- BPE
- Unigram
- Character
- Word

## 词汇表大小的影响

| 大小 | 优点 | 缺点 |
|------|------|------|
| 小 (10K) | 更快的训练，更小的模型 | 更多 OOV，更长序列 |
| 中 (30K-50K) | 平衡性能和效率 | 需要调优 |
| 大 (100K+) | 更好的覆盖率，更短序列 | 更大的嵌入层，更慢 |

### 最优大小的选择

经验法则：

$$|\mathcal{V}| \approx \sqrt{|\mathcal{C}|}$$

<a id="formula-tokenization-6"></a>

**公式解释**
- **公式含义**：词汇表大小约为语料库大小（词数）的平方根。
- **变量说明**：$|\mathcal{C}|$ 是语料库中的总词数。
- **直觉/作用**：提供压缩效率和覆盖率的平衡起点。

## 特殊 Token 的设计

### 常见特殊 Token

| Token | 用途 | 示例 |
|-------|------|------|
| `<s>` / `<\|begin\|>` | 序列开始 | 用于标记生成开始 |
| `</s>` / `<\|end\|>` | 序列结束 | 用于标记生成结束 |
| `<pad>` | 填充 | 批处理时填充短序列 |
| `<unk>` | 未知词 | 处理 OOV |
| `<mask>` | 掩码 | MLM 预训练 |
| `<sep>` | 分隔 | 分隔不同部分 |
| `<cls>` | 分类 | 句子级别表示 |

### 位置预留

训练时为特殊 token 预留位置：

```python
# 典型的词汇表结构
special_tokens = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>"]
vocab = special_tokens + learned_tokens
```

## 编码解码的一致性

### 可逆性要求

$$\text{decode}(\text{encode}(x)) = x$$

### 常见问题

1. **空格丢失**：某些 tokenizer 不保留精确的空格
2. **大小写**：某些 tokenizer 统一转换为小写
3. **Unicode 规范化**：可能改变字符表示

### 最佳实践

```python
# 始终验证编解码一致性
original = "Hello, 世界!"
tokens = tokenizer.encode(original)
decoded = tokenizer.decode(tokens)
assert decoded == original, f"Mismatch: {original} != {decoded}"
```

## 多语言分词

### 挑战

- 不同语言的字符集差异大
- 某些语言没有空格分隔（中文、日文）
- 脚本混合（如中英混合文本）

### 解决方案

1. **字节级别**：BBPE 天然支持多语言
2. **语言特定预处理**：先分词再训练
3. **混合词汇表**：为不同语言分配词汇表比例

## 参考文献

1. Sennrich et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*
2. Wu et al. (2016). *Google's Neural Machine Translation System*
3. Kudo (2018). *Subword Regularization: Improving Neural Network Translation Models*
4. Kudo & Richardson (2018). *SentencePiece: A simple and language independent subword tokenizer*
5. Radford et al. (2019). *Language Models are Unsupervised Multitask Learners*
