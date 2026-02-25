"""
Word2Vec 示例代码 / Word2Vec Example Code
==========================================

本示例展示如何使用 Gensim 库加载和使用预训练的 Word2Vec 模型。
This example demonstrates how to load and use pre-trained Word2Vec models with Gensim.

Word2Vec 是 Google 提出的词嵌入方法，通过预测上下文来学习词的向量表示。
Word2Vec is a word embedding method proposed by Google, learning vector representations
by predicting context words.

依赖安装 / Dependencies:
    pip install gensim
"""

# =============================================================================
# 1. 加载预训练模型 / Loading Pre-trained Model
# =============================================================================

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

# 加载 Google News 预训练模型 (约 3.4GB)
# Load Google News pre-trained model (~3.4GB)
# 注意: 需要先下载 GoogleNews-vectors-negative300.bin
# Note: Need to download GoogleNews-vectors-negative300.bin first
#
# model = KeyedVectors.load_word2vec_format(
#     datapath('/path/to/GoogleNews-vectors-negative300.bin'),
#     binary=True
# )

# 为了演示，我们使用 Gensim 内置的小型测试模型
# For demonstration, we use Gensim's built-in small test model
import gensim.downloader as api

# 加载小型预训练模型 (GloVe 格式，Gensim 可以兼容)
# Load small pre-trained model (GloVe format, compatible with Gensim)
# 可用模型列表 / Available models:
#   - 'word2vec-ruscorpora-300': 俄语语料库 / Russian corpus
#   - 'word2vec-google-news-300': Google News (1.5GB)
#   - 'glove-wiki-gigaword-100': 维基百科 + Gigaword (128MB)

print("正在加载模型... / Loading model...")
model = api.load("glove-wiki-gigaword-100")
print(f"模型加载完成！词表大小 / Model loaded! Vocabulary size: {len(model)}")

# =============================================================================
# 2. 获取词向量 / Getting Word Vectors
# =============================================================================

# 获取单个词的向量表示
# Get the vector representation of a single word
word = "king"
vector = model[word]

print(f"\n词 / Word: '{word}'")
print(f"向量维度 / Vector dimension: {len(vector)}")
print(f"向量前10维 / First 10 dimensions: {vector[:10]}")

# =============================================================================
# 3. 计算词相似度 / Computing Word Similarity
# =============================================================================

# 计算两个词之间的余弦相似度
# Compute cosine similarity between two words
similarity = model.similarity("king", "queen")
print(f"\n'king' 和 'queen' 的相似度 / Similarity: {similarity:.4f}")

similarity = model.similarity("king", "car")
print(f"'king' 和 'car' 的相似度 / Similarity: {similarity:.4f}")

# =============================================================================
# 4. 查找相似词 / Finding Similar Words
# =============================================================================

# 找出与给定词最相似的词
# Find the most similar words to a given word
similar_words = model.most_similar("king", topn=5)
print(f"\n与 'king' 最相似的词 / Words most similar to 'king':")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

# 多个词的相似词（取平均）
# Similar words to multiple words (averaged)
similar_words = model.most_similar(
    positive=["king", "woman"],  # 正向词 / Positive words
    topn=5
)
print(f"\n与 'king' + 'woman' 最相似的词 / Words similar to 'king' + 'woman':")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

# =============================================================================
# 5. 词类比任务 / Word Analogy Task
# =============================================================================

# 经典的词类比: king - man + woman = queen
# Classic word analogy: king - man + woman = queen
# 语义: "国王"之于"男人"就像"？"之于"女人"
# Semantics: "king" is to "man" as "?" is to "woman"

result = model.most_similar(
    positive=["king", "woman"],  # 正向词 / Positive words
    negative=["man"],            # 负向词 / Negative words
    topn=3
)
print(f"\n词类比: king - man + woman = ?")
print(f"Word analogy: king - man + woman = ?")
for word, score in result:
    print(f"  {word}: {score:.4f}")

# 另一个例子: 巴黎之于法国，就像柏林之于？
# Another example: Paris is to France as Berlin is to ?
result = model.most_similar(
    positive=["berlin", "france"],
    negative=["paris"],
    topn=3
)
print(f"\n词类比: berlin - paris + france = ?")
print(f"Word analogy: berlin - paris + france = ?")
for word, score in result:
    print(f"  {word}: {score:.4f}")

# =============================================================================
# 6. 找出不匹配的词 / Finding the Odd One Out
# =============================================================================

# 从一组词中找出语义上不相关的词
# Find the semantically unrelated word from a group
odd_one = model.doesnt_match(["breakfast", "lunch", "dinner", "car"])
print(f"\n不匹配的词 / Odd one out in [breakfast, lunch, dinner, car]: {odd_one}")

# =============================================================================
# 7. 句子相似度（简单平均法）/ Sentence Similarity (Simple Averaging)
# =============================================================================

import numpy as np

def sentence_vector(model, sentence):
    """
    计算句子的向量表示（词向量平均）
    Compute sentence vector representation (average of word vectors)

    参数 / Args:
        model: Word2Vec 模型 / Word2Vec model
        sentence: 句子字符串 / Sentence string

    返回 / Returns:
        句子的向量表示 / Vector representation of the sentence
    """
    # 简单分词（实际应用中应使用更复杂的分词器）
    # Simple tokenization (use more sophisticated tokenizer in practice)
    words = sentence.lower().split()

    # 获取所有词的向量（忽略不在词表中的词）
    # Get vectors for all words (ignore words not in vocabulary)
    vectors = [model[word] for word in words if word in model]

    if not vectors:
        return np.zeros(model.vector_size)

    # 返回平均向量 / Return averaged vector
    return np.mean(vectors, axis=0)

def cosine_similarity(v1, v2):
    """
    计算两个向量的余弦相似度
    Compute cosine similarity between two vectors
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 计算两个句子的相似度
# Compute similarity between two sentences
sent1 = "the king sits on the throne"
sent2 = "the queen sits on the throne"
sent3 = "the car drives on the road"

vec1 = sentence_vector(model, sent1)
vec2 = sentence_vector(model, sent2)
vec3 = sentence_vector(model, sent3)

print(f"\n句子相似度 / Sentence Similarity:")
print(f"  '{sent1}' vs '{sent2}': {cosine_similarity(vec1, vec2):.4f}")
print(f"  '{sent1}' vs '{sent3}': {cosine_similarity(vec1, vec3):.4f}")

# =============================================================================
# 8. 训练自己的 Word2Vec 模型 / Training Your Own Word2Vec Model
# =============================================================================

from gensim.models import Word2Vec

# 示例语料（实际应用中需要大量文本）
# Sample corpus (need large amount of text in practice)
corpus = [
    ["the", "king", "sat", "on", "the", "throne"],
    ["the", "queen", "sat", "next", "to", "the", "king"],
    ["the", "prince", "is", "the", "son", "of", "the", "king"],
    ["the", "princess", "is", "the", "daughter", "of", "the", "queen"],
]

# 训练 Word2Vec 模型
# Train Word2Vec model
# 参数说明 / Parameter explanation:
#   vector_size: 向量维度 / Vector dimension
#   window: 上下文窗口大小 / Context window size
#   min_count: 忽略出现次数少于此值的词 / Ignore words with frequency less than this
#   workers: 并行线程数 / Number of parallel threads
custom_model = Word2Vec(
    sentences=corpus,
    vector_size=100,  # 向量维度 / Vector dimension
    window=5,         # 上下文窗口 / Context window
    min_count=1,      # 最小词频 / Minimum word frequency
    workers=4,        # 并行线程 / Parallel threads
    sg=0              # 0=CBOW, 1=Skip-gram
)

print(f"\n自定义模型词表大小 / Custom model vocabulary size: {len(custom_model.wv)}")
print(f"与 'king' 最相似的词 / Words similar to 'king':")
for word, score in custom_model.wv.most_similar("king", topn=3):
    print(f"  {word}: {score:.4f}")

# 保存和加载模型
# Save and load model
# custom_model.save("my_word2vec.model")
# loaded_model = Word2Vec.load("my_word2vec.model")

# =============================================================================
# 总结 / Summary
# =============================================================================
"""
本示例展示了 Word2Vec 的常用操作:
This example demonstrates common Word2Vec operations:

1. 加载预训练模型 / Loading pre-trained models
2. 获取词向量 / Getting word vectors
3. 计算词相似度 / Computing word similarity
4. 查找相似词 / Finding similar words
5. 词类比任务 / Word analogy tasks
6. 找出不匹配词 / Finding odd one out
7. 句子相似度（词向量平均）/ Sentence similarity (averaging)
8. 训练自定义模型 / Training custom models

在 LLM 应用中，Word2Vec 常用于:
In LLM applications, Word2Vec is commonly used for:
- 文本检索 / Text retrieval
- 推荐系统 / Recommendation systems
- 文本分类特征 / Text classification features
- 语义搜索 / Semantic search
"""
