# Word Embedding 深入版

> 面向有机器学习基础读者的技术详解

## 概述

词嵌入（Word Embedding）是将离散的词符号映射到连续稠密向量空间的技术。形式化地，对于词汇表 $V$，词嵌入是一个映射函数：

$$f: V \rightarrow \mathbb{R}^d$$

其中 $d$ 是嵌入维度（通常 100-300）。

## 分布式假设

> "You shall know a word by the company it keeps" — J.R. Firth (1957)

词嵌入的理论基础是**分布式假设**：语义相似的词出现在相似的上下文中。

## Word2Vec

### 1. Skip-gram 模型

给定词序列 $w_1, w_2, ..., w_T$，Skip-gram 的目标是最大化平均对数概率：

$$\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t)$$

其中 $c$ 是上下文窗口大小。

使用 softmax 定义条件概率：

$$p(w_o | w_i) = \frac{\exp(v_{w_o}^T v_{w_i})}{\sum_{w=1}^{W} \exp(v_w^T v_{w_i})}$$

其中 $v_w$ 是词 $w$ 的嵌入向量。

### 2. 负采样 (Negative Sampling)

原始 softmax 计算量太大（需要遍历整个词表）。负采样将其转化为二分类问题：

$$\log p(w_o | w_i) = \log \sigma(v_{w_o}^T v_{w_i}) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)} [\log \sigma(-v_{w_k}^T v_{w_i})]$$

其中：
- $\sigma(x) = \frac{1}{1 + e^{-x}}$ 是 sigmoid 函数
- $K$ 是负样本数量（通常 5-20）
- $P_n(w)$ 是噪声分布，通常取 $P_n(w) \propto f(w)^{0.75}$

### 3. CBOW 模型

CBOW (Continuous Bag of Words) 与 Skip-gram 相反：用上下文预测中心词。

$$p(w_t | \text{context}) = \frac{\exp(v_{w_t}^T \cdot \bar{v}_{\text{context}})}{\sum_{w=1}^{W} \exp(v_w^T \cdot \bar{v}_{\text{context}})}$$

其中 $\bar{v}_{\text{context}} = \frac{1}{2c} \sum_{-c \leq j \leq c, j \neq 0} v_{w_{t+j}}$

## GloVe

GloVe (Global Vectors) 结合了全局矩阵分解和局部上下文窗口方法。

### 共现矩阵

定义共现矩阵 $X$，其中 $X_{ij}$ 表示词 $i$ 和词 $j$ 在上下文中共现的次数。

### 目标函数

$$J = \sum_{i,j=1}^{V} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

其中：
- $w_i$ 是中心词向量
- $\tilde{w}_j$ 是上下文词向量
- $b_i, \tilde{b}_j$ 是偏置项
- $f(x)$ 是权重函数，用于减少高频词的影响：

$$f(x) = \begin{cases} (x/x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$$

## 语义属性

### 1. 余弦相似度

$$\text{sim}(w_1, w_2) = \frac{v_{w_1} \cdot v_{w_2}}{\|v_{w_1}\| \|v_{w_2}\|}$$

### 2. 类比关系

词嵌入可以捕捉语义类比关系：

$$\vec{v}_{king} - \vec{v}_{man} + \vec{v}_{woman} \approx \vec{v}_{queen}$$

这表明词向量空间中存在有意义的线性子结构。

## 局限性

### 1. 多义词问题

一个词可能有多个含义，但只有一个向量表示：

- "bank" 可以是河岸，也可以是银行
- 传统词嵌入无法区分

### 2. OOV 问题

未登录词（Out-of-Vocabulary）无法获得向量表示。

### 3. 静态表示

词向量是固定的，无法根据上下文动态变化。

## 从静态到动态

这些局限性催生了上下文相关的词表示方法（如 ELMo、BERT），我们将在后续章节讨论。

## 参考文献

1. Mikolov et al. (2013). *Efficient Estimation of Word Representations in Vector Space*
2. Mikolov et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality*
3. Pennington et al. (2014). *GloVe: Global Vectors for Word Representation*
4. Goldberg & Levy (2014). *word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method*
