"""
GloVe 示例代码 / GloVe Example Code
====================================

本示例展示如何加载和使用 GloVe (Global Vectors for Word Representation) 词向量。
This example demonstrates how to load and use GloVe word vectors.

GloVe 是 Stanford 提出的词嵌入方法，结合了全局矩阵分解和局部上下文窗口的优点。
GloVe is a word embedding method proposed by Stanford, combining global matrix
factorization with local context window advantages.

论文 / Paper: GloVe: Global Vectors for Word Representation (Pennington et al., 2014)

依赖安装 / Dependencies:
    pip install numpy scipy

GloVe 预训练向量下载 / Pre-trained GloVe vectors download:
    https://nlp.stanford.edu/projects/glove/
"""

import numpy as np
from pathlib import Path

# =============================================================================
# 1. 加载 GloVe 向量 / Loading GloVe Vectors
# =============================================================================

def load_glove_vectors(glove_file):
    """
    从文件加载 GloVe 词向量
    Load GloVe word vectors from file

    参数 / Args:
        glove_file: GloVe 文件路径 / Path to GloVe file
                   格式: 每行一个词，空格分隔，第一个是词，后面是向量值
                   Format: Each line contains a word followed by vector values

    返回 / Returns:
        word_to_vec: 词到向量的字典 / Dictionary mapping words to vectors
        words: 词列表 / List of words
        vectors: 向量矩阵 / Matrix of vectors
    """
    print(f"正在加载 GloVe 向量... / Loading GloVe vectors from {glove_file}")

    word_to_vec = {}
    words = []
    vectors = []

    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)

            word_to_vec[word] = vector
            words.append(word)
            vectors.append(vector)

    vectors = np.array(vectors)
    print(f"加载完成！词表大小 / Loaded! Vocabulary size: {len(words)}")
    print(f"向量维度 / Vector dimension: {vectors.shape[1]}")

    return word_to_vec, words, vectors


# 示例: 加载 GloVe 文件
# Example: Loading GloVe file
# glove_path = "glove.6B.100d.txt"  # 100维版本 / 100-dimension version
# word_to_vec, words, vectors = load_glove_vectors(glove_path)

# =============================================================================
# 2. 使用 Gensim 加载 GloVe (推荐方式) / Using Gensim to Load GloVe (Recommended)
# =============================================================================

import gensim.downloader as api
from gensim.models import KeyedVectors

# Gensim 提供了预加载的 GloVe 模型
# Gensim provides pre-loaded GloVe models
# 可用模型 / Available models:
#   - 'glove-wiki-gigaword-50': 50维 / 50 dimensions (65MB)
#   - 'glove-wiki-gigaword-100': 100维 / 100 dimensions (128MB)
#   - 'glove-wiki-gigaword-200': 200维 / 200 dimensions (252MB)
#   - 'glove-wiki-gigaword-300': 300维 / 300 dimensions (376MB)
#   - 'glove-twitter-25': 25维，Twitter 数据 / 25 dimensions, Twitter data
#   - 'glove-twitter-50': 50维，Twitter 数据 / 50 dimensions, Twitter data

print("正在加载 GloVe 模型... / Loading GloVe model...")
model = api.load("glove-wiki-gigaword-100")
print(f"模型加载完成！词表大小 / Model loaded! Vocabulary size: {len(model)}")

# =============================================================================
# 3. 获取词向量 / Getting Word Vectors
# =============================================================================

# 获取单个词的向量
# Get vector for a single word
word = "computer"
vector = model[word]

print(f"\n词 / Word: '{word}'")
print(f"向量维度 / Vector dimension: {len(vector)}")
print(f"向量前10维 / First 10 dimensions: {vector[:10]}")

# 获取多个词的向量（用于批量处理）
# Get vectors for multiple words (for batch processing)
words = ["computer", "software", "hardware", "programming"]
vectors = [model[w] for w in words]
print(f"\n批量获取向量 / Batch vector retrieval: {len(vectors)} 个词 / words")

# =============================================================================
# 4. 语义相似度计算 / Semantic Similarity Computation
# =============================================================================

def cosine_similarity(v1, v2):
    """
    计算两个向量的余弦相似度
    Compute cosine similarity between two vectors

    公式 / Formula: cos(θ) = (A · B) / (||A|| * ||B||)

    参数 / Args:
        v1, v2: 向量 / Vectors

    返回 / Returns:
        余弦相似度，范围 [-1, 1] / Cosine similarity, range [-1, 1]
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return dot_product / (norm_v1 * norm_v2)


def euclidean_distance(v1, v2):
    """
    计算两个向量的欧几里得距离
    Compute Euclidean distance between two vectors

    参数 / Args:
        v1, v2: 向量 / Vectors

    返回 / Returns:
        欧几里得距离 / Euclidean distance
    """
    return np.linalg.norm(v1 - v2)


# 计算不同概念之间的相似度
# Compute similarity between different concepts
word_pairs = [
    ("computer", "software"),    # 相关 / Related
    ("computer", "hardware"),    # 相关 / Related
    ("computer", "banana"),      # 不相关 / Unrelated
    ("king", "queen"),           # 语义相关 / Semantically related
    ("good", "bad"),             # 反义词但相关 / Antonyms but related
]

print("\n词对相似度 / Word Pair Similarities:")
print("-" * 60)
for w1, w2 in word_pairs:
    cos_sim = model.similarity(w1, w2)  # Gensim 内置方法 / Gensim built-in method
    euc_dist = euclidean_distance(model[w1], model[w2])
    print(f"  {w1:15} vs {w2:15}: cos={cos_sim:.4f}, euclidean={euc_dist:.4f}")

# =============================================================================
# 5. 语义搜索 / Semantic Search
# =============================================================================

def semantic_search(query, model, top_k=5):
    """
    语义搜索：找出与查询词最相似的词
    Semantic search: Find words most similar to the query

    参数 / Args:
        query: 查询词 / Query word
        model: 词向量模型 / Word vector model
        top_k: 返回的最相似词数量 / Number of similar words to return

    返回 / Returns:
        相似词列表，每项为 (词, 相似度) 元组
        List of similar words as (word, similarity) tuples
    """
    if query not in model:
        print(f"词 '{query}' 不在词表中 / Word '{query}' not in vocabulary")
        return []

    return model.most_similar(query, topn=top_k)


# 搜索与 "computer" 语义相关的词
# Search for words semantically related to "computer"
print("\n语义搜索结果 / Semantic Search Results:")
print("-" * 60)
query_word = "computer"
similar = semantic_search(query_word, model, top_k=10)
print(f"与 '{query_word}' 最相似的词 / Words most similar to '{query_word}':")
for word, score in similar:
    print(f"  {word:20}: {score:.4f}")

# =============================================================================
# 6. 词类比 (Word Analogy)
# =============================================================================

def word_analogy(model, a, b, c, top_k=3):
    """
    词类比任务：a 之于 b 就像 ? 之于 c
    Word analogy task: a is to b as ? is to c

    数学形式 / Mathematical form: result = b - a + c
    例如 / Example: king - man + woman = queen

    参数 / Args:
        model: 词向量模型 / Word vector model
        a, b, c: 类比词 / Analogy words
        top_k: 返回的结果数量 / Number of results to return

    返回 / Returns:
        类比结果列表 / List of analogy results
    """
    # 使用向量运算
    # Use vector arithmetic
    result = model.most_similar(
        positive=[b, c],
        negative=[a],
        topn=top_k
    )
    return result


# 经典词类比示例
# Classic word analogy examples
analogies = [
    ("man", "king", "woman"),        # king - man + woman = queen
    ("france", "paris", "germany"),  # paris - france + germany = berlin
    ("good", "better", "bad"),       # better - good + bad = worse
    ("walk", "walked", "run"),       # walked - walk + run = ran
]

print("\n词类比结果 / Word Analogy Results:")
print("-" * 60)
for a, b, c in analogies:
    results = word_analogy(model, a, b, c)
    print(f"  {a} : {b} :: {c} : ?")
    for word, score in results:
        print(f"    → {word} (相似度 / similarity: {score:.4f})")

# =============================================================================
# 7. 句子/文档嵌入 / Sentence/Document Embedding
# =============================================================================

def sentence_embedding_glove(model, sentence, method='mean'):
    """
    使用 GloVe 生成句子嵌入
    Generate sentence embedding using GloVe

    参数 / Args:
        model: GloVe 模型 / GloVe model
        sentence: 输入句子 / Input sentence
        method: 聚合方法 ('mean', 'max', 'weighted')
                Aggregation method ('mean', 'max', 'weighted')

    返回 / Returns:
        句子嵌入向量 / Sentence embedding vector
    """
    # 简单分词（实际应用使用更复杂的分词器）
    # Simple tokenization (use sophisticated tokenizer in practice)
    words = sentence.lower().split()

    # 过滤不在词表中的词
    # Filter words not in vocabulary
    valid_words = [w for w in words if w in model]

    if not valid_words:
        return np.zeros(model.vector_size)

    # 获取词向量
    # Get word vectors
    vectors = np.array([model[w] for w in valid_words])

    if method == 'mean':
        # 平均池化 / Mean pooling
        return np.mean(vectors, axis=0)
    elif method == 'max':
        # 最大池化 / Max pooling
        return np.max(vectors, axis=0)
    elif method == 'weighted':
        # TF-IDF 加权（简化版，实际需要 IDF 值）
        # TF-IDF weighted (simplified, needs IDF values in practice)
        weights = np.ones(len(vectors)) / len(vectors)
        return np.average(vectors, axis=0, weights=weights)

    return np.mean(vectors, axis=0)


# 句子相似度示例
# Sentence similarity example
sentences = [
    "the king sat on the throne",
    "the queen ruled the kingdom",
    "the computer runs software programs",
    "i like to eat pizza for dinner"
]

print("\n句子嵌入与相似度 / Sentence Embeddings and Similarity:")
print("-" * 60)

# 计算句子嵌入
# Compute sentence embeddings
sentence_vectors = [sentence_embedding_glove(model, s) for s in sentences]

# 计算句子对相似度
# Compute pairwise sentence similarity
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine_similarity(sentence_vectors[i], sentence_vectors[j])
        print(f"  句子 {i+1} vs 句子 {j+1}: {sim:.4f}")
        if i < 2:  # 只显示第一组比较的句子 / Only show first comparison group
            print(f"    '{sentences[i]}'")
            print(f"    '{sentences[j]}'")

# =============================================================================
# 8. 最近邻搜索 / Nearest Neighbor Search
# =============================================================================

def find_nearest_neighbors(query_vector, model, k=5, exclude_words=None):
    """
    在词向量空间中找最近邻
    Find nearest neighbors in word vector space

    参数 / Args:
        query_vector: 查询向量 / Query vector
        model: 词向量模型 / Word vector model
        k: 返回的邻居数量 / Number of neighbors to return
        exclude_words: 要排除的词列表 / Words to exclude

    返回 / Returns:
        最近邻词和距离列表 / List of nearest neighbor words and distances
    """
    exclude_words = exclude_words or []
    similarities = []

    for word in model.index_to_key:
        if word in exclude_words:
            continue
        sim = cosine_similarity(query_vector, model[word])
        similarities.append((word, sim))

    # 按相似度降序排序
    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]


# =============================================================================
# 9. 词向量可视化准备 / Word Vector Visualization Preparation
# =============================================================================

def prepare_for_visualization(model, words):
    """
    准备词向量的 2D 可视化数据（PCA 降维）
    Prepare word vectors for 2D visualization (PCA dimensionality reduction)

    参数 / Args:
        model: 词向量模型 / Word vector model
        words: 要可视化的词列表 / Words to visualize

    返回 / Returns:
        2D 坐标数组 / Array of 2D coordinates
    """
    from sklearn.decomposition import PCA

    # 获取词向量
    # Get word vectors
    vectors = np.array([model[w] for w in words if w in model])

    # PCA 降维到 2D
    # PCA reduction to 2D
    pca = PCA(n_components=2)
    coordinates_2d = pca.fit_transform(vectors)

    print(f"PCA 解释方差比 / PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"总解释方差 / Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

    return coordinates_2d


# 可视化示例词
# Example words for visualization
visualization_words = [
    "king", "queen", "prince", "princess",  # 皇室 / Royalty
    "computer", "software", "hardware", "program",  # 科技 / Technology
    "apple", "banana", "orange", "fruit",  # 水果 / Fruits
]

print("\n可视化准备 / Visualization Preparation:")
print("-" * 60)
# coords = prepare_for_visualization(model, visualization_words)
# 然后使用 matplotlib 绘制 / Then plot with matplotlib

# =============================================================================
# 总结 / Summary
# =============================================================================
"""
本示例展示了 GloVe 的常用操作:
This example demonstrates common GloVe operations:

1. 加载 GloVe 向量（文件方式 & Gensim）/ Loading GloVe vectors (file & Gensim)
2. 获取词向量 / Getting word vectors
3. 语义相似度计算 / Semantic similarity computation
4. 语义搜索 / Semantic search
5. 词类比任务 / Word analogy tasks
6. 句子嵌入 / Sentence embeddings
7. 最近邻搜索 / Nearest neighbor search
8. 可视化准备 / Visualization preparation

GloVe vs Word2Vec 主要区别 / Main differences between GloVe and Word2Vec:

| 特性 | GloVe | Word2Vec |
|------|-------|----------|
| 训练方法 | 全局共现矩阵 | 局部上下文窗口 |
| 计算复杂度 | 较高 | 较低 |
| 语义捕捉 | 全局统计 | 局部模式 |
| 类比性能 | 略优 | 优秀 |

在 LLM 应用中的使用场景 / Use cases in LLM applications:
- 文本相似度计算 / Text similarity computation
- 语义搜索 / Semantic search
- 推荐系统 / Recommendation systems
- 文本分类特征 / Text classification features
- 知识图谱构建 / Knowledge graph construction
"""
