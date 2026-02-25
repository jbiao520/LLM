"""
BERT Embedding 示例代码 / BERT Embedding Example Code
======================================================

本示例展示如何使用 HuggingFace Transformers 获取 BERT 的词/句嵌入。
This example demonstrates how to get BERT word/sentence embeddings using
HuggingFace Transformers.

BERT (Bidirectional Encoder Representations from Transformers) 是 Google 提出的
预训练语言模型，可以生成上下文相关的词嵌入。
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained
language model proposed by Google, capable of generating context-aware word embeddings.

与 Word2Vec/GloVe 不同，BERT 的词向量是动态的，同一个词在不同上下文中有不同的向量表示。
Unlike Word2Vec/GloVe, BERT word vectors are dynamic - the same word has different
vector representations in different contexts.

依赖安装 / Dependencies:
    pip install transformers torch
"""

import torch
from transformers import BertTokenizer, BertModel

# =============================================================================
# 1. 加载 BERT 模型 / Loading BERT Model
# =============================================================================

# 使用 BERT-base 中文模型
# Use BERT-base Chinese model
# 其他可用模型 / Other available models:
#   - 'bert-base-uncased': 英文小写模型 / English lowercase model
#   - 'bert-base-cased': 英文大小写敏感 / English case-sensitive model
#   - 'bert-large-uncased': 英文大模型 / English large model
#   - 'bert-base-multilingual-cased': 多语言模型 / Multilingual model

print("正在加载 BERT 模型... / Loading BERT model...")
MODEL_NAME = "bert-base-chinese"  # 中文 BERT / Chinese BERT

# 加载分词器 / Load tokenizer
# 分词器负责将文本转换为模型可理解的 token 序列
# Tokenizer converts text to token sequences that the model can understand
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 加载模型 / Load model
# BertModel 只包含编码器部分，不包含下游任务头
# BertModel only contains the encoder, without downstream task heads
model = BertModel.from_pretrained(MODEL_NAME)

# 设置为评估模式（不计算梯度）
# Set to evaluation mode (no gradient computation)
model.eval()

print(f"模型加载完成！ / Model loaded!")
print(f"词表大小 / Vocabulary size: {tokenizer.vocab_size}")
print(f"隐藏层维度 / Hidden size: {model.config.hidden_size}")
print(f"层数 / Number of layers: {model.config.num_hidden_layers}")

# =============================================================================
# 2. 文本编码 / Text Encoding
# =============================================================================

def encode_text(text, tokenizer, max_length=128):
    """
    将文本编码为 BERT 输入格式
    Encode text to BERT input format

    参数 / Args:
        text: 输入文本 / Input text
        tokenizer: BERT 分词器 / BERT tokenizer
        max_length: 最大序列长度 / Maximum sequence length

    返回 / Returns:
        input_ids: token ID 序列 / Token ID sequence
        attention_mask: 注意力掩码 / Attention mask
    """
    # tokenizer 会自动添加 [CLS] 和 [SEP] token
    # tokenizer automatically adds [CLS] and [SEP] tokens
    # [CLS]: 句子开头，用于分类任务 / Start of sentence, used for classification
    # [SEP]: 句子分隔符 / Sentence separator
    encoded = tokenizer(
        text,
        padding='max_length',      # 填充到最大长度 / Pad to max length
        truncation=True,           # 超长截断 / Truncate if too long
        max_length=max_length,
        return_tensors='pt'        # 返回 PyTorch 张量 / Return PyTorch tensors
    )

    return encoded['input_ids'], encoded['attention_mask']


# 编码示例文本
# Encode example text
text = "自然语言处理是人工智能的重要分支。"
print(f"\n原始文本 / Original text: {text}")

input_ids, attention_mask = encode_text(text, tokenizer)

# 显示编码结果
# Show encoding result
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print(f"Token 序列 / Token sequence: {tokens[:15]}...")
print(f"Input IDs 形状 / Input IDs shape: {input_ids.shape}")
print(f"Attention Mask 形状 / Attention mask shape: {attention_mask.shape}")

# =============================================================================
# 3. 获取 BERT 嵌入 / Getting BERT Embeddings
# =============================================================================

def get_bert_embeddings(model, input_ids, attention_mask):
    """
    获取 BERT 的各种嵌入表示
    Get various BERT embedding representations

    参数 / Args:
        model: BERT 模型 / BERT model
        input_ids: token ID 序列 / Token ID sequence
        attention_mask: 注意力掩码 / Attention mask

    返回 / Returns:
        last_hidden_state: 最后一层的隐藏状态 (batch, seq_len, hidden_size)
                          Last layer hidden states
        pooler_output: [CLS] token 经过池化层的输出 (batch, hidden_size)
                      Pooled output of [CLS] token
        hidden_states: 所有层的隐藏状态（可选）
                      Hidden states from all layers (optional)
    """
    # 不计算梯度（推理模式）
    # No gradient computation (inference mode)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True  # 输出所有层 / Output all layers
        )

    return outputs


# 获取嵌入
# Get embeddings
outputs = get_bert_embeddings(model, input_ids, attention_mask)

print(f"\nBERT 输出 / BERT Outputs:")
print(f"  last_hidden_state 形状 / shape: {outputs.last_hidden_state.shape}")
print(f"  pooler_output 形状 / shape: {outputs.pooler_output.shape}")
print(f"  hidden_states 数量 / count: {len(outputs.hidden_states)}")

# =============================================================================
# 4. 获取词级嵌入 / Getting Word-Level Embeddings
# =============================================================================

def get_word_embedding(outputs, token_index):
    """
    获取特定位置 token 的嵌入向量
    Get embedding vector for a token at specific position

    参数 / Args:
        outputs: BERT 模型输出 / BERT model outputs
        token_index: token 位置索引 / Token position index

    返回 / Returns:
        词嵌入向量 (hidden_size,) / Word embedding vector
    """
    # last_hidden_state: (batch_size, seq_len, hidden_size)
    return outputs.last_hidden_state[0, token_index, :].numpy()


# 获取 [CLS] token 的嵌入（常用于句子级表示）
# Get [CLS] token embedding (commonly used for sentence-level representation)
cls_embedding = get_word_embedding(outputs, 0)
print(f"\n[CLS] token 嵌入前10维 / First 10 dimensions: {cls_embedding[:10]}")

# 获取实际词的嵌入
# Get embeddings for actual words
# 注意：中文 BERT 使用字符级分词，每个汉字是一个 token
# Note: Chinese BERT uses character-level tokenization, each character is a token
for i, token in enumerate(tokens[:10]):
    if token not in ['[CLS]', '[SEP]', '[PAD]']:
        embedding = get_word_embedding(outputs, i)
        print(f"Token '{token}' 嵌入范数 / embedding norm: {np.linalg.norm(embedding):.4f}")

# =============================================================================
# 5. 获取句子嵌入 / Getting Sentence Embeddings
# =============================================================================

import numpy as np

def get_sentence_embedding(outputs, method='cls'):
    """
    从 BERT 输出获取句子嵌入
    Get sentence embedding from BERT outputs

    参数 / Args:
        outputs: BERT 模型输出 / BERT model outputs
        method: 池化方法 / Pooling method
            - 'cls': 使用 [CLS] token 的表示 / Use [CLS] token representation
            - 'mean': 对所有 token 取平均 / Average all tokens
            - 'max': 对所有 token 取最大值 / Max pooling all tokens
            - 'pooler': 使用 pooler_output / Use pooler_output

    返回 / Returns:
        句子嵌入向量 / Sentence embedding vector
    """
    if method == 'cls':
        # [CLS] token 的嵌入
        # Embedding of [CLS] token
        return outputs.last_hidden_state[0, 0, :].numpy()

    elif method == 'mean':
        # 对所有 token 的嵌入取平均
        # Average embeddings of all tokens
        # 注意：应该排除 padding token
        # Note: Should exclude padding tokens
        return outputs.last_hidden_state[0, :, :].mean(dim=0).numpy()

    elif method == 'max':
        # 对所有 token 的嵌入取最大值
        # Max pooling of all token embeddings
        return outputs.last_hidden_state[0, :, :].max(dim=0)[0].numpy()

    elif method == 'pooler':
        # 使用 BERT 的 pooler 输出
        # Use BERT's pooler output
        # pooler_output 是 [CLS] 经过 tanh 激活后的结果
        # pooler_output is [CLS] after tanh activation
        return outputs.pooler_output[0].numpy()

    return outputs.last_hidden_state[0, 0, :].numpy()


# 比较不同池化方法
# Compare different pooling methods
print("\n句子嵌入比较 / Sentence Embedding Comparison:")
for method in ['cls', 'mean', 'max', 'pooler']:
    embedding = get_sentence_embedding(outputs, method)
    print(f"  {method:8}: 范数/norm={np.linalg.norm(embedding):.4f}, 前5维/first 5: {embedding[:5]}")

# =============================================================================
# 6. 上下文相关的词嵌入 / Context-Aware Word Embeddings
# =============================================================================

def compare_contextual_embeddings(model, tokenizer, word, contexts):
    """
    比较同一个词在不同上下文中的嵌入
    Compare embeddings of the same word in different contexts

    这是 BERT 与静态词嵌入（Word2Vec/GloVe）的关键区别
    This is the key difference between BERT and static embeddings (Word2Vec/GloVe)

    参数 / Args:
        model: BERT 模型 / BERT model
        tokenizer: 分词器 / Tokenizer
        word: 目标词 / Target word
        contexts: 上下文句子列表 / List of context sentences
    """
    embeddings = []

    for context in contexts:
        # 找到目标词在句子中的位置
        # Find position of target word in sentence
        encoded = tokenizer(context, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**encoded)

        # 获取 token 列表
        # Get token list
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

        # 找到目标词的索引（简化版，实际需要更精确的匹配）
        # Find index of target word (simplified, needs more precise matching)
        target_idx = None
        for i, token in enumerate(tokens):
            if word in token:
                target_idx = i
                break

        if target_idx:
            emb = outputs.last_hidden_state[0, target_idx, :].numpy()
            embeddings.append((context, emb))

    return embeddings


# 比较"苹果"在不同上下文中的嵌入
# Compare embeddings of "苹果" (apple/Apple) in different contexts
word = "果"
contexts = [
    "我喜欢吃苹果，非常甜。",        # 水果 / Fruit
    "苹果公司发布了新手机。",        # 公司 / Company
    "这个苹果很红很好看。",          # 水果 / Fruit
]

print("\n上下文相关嵌入比较 / Context-Aware Embedding Comparison:")
print(f"目标词 / Target word: '{word}'")
print("-" * 60)

context_embeddings = compare_contextual_embeddings(model, tokenizer, word, contexts)
for context, emb in context_embeddings:
    print(f"上下文 / Context: {context}")
    print(f"  嵌入范数 / Embedding norm: {np.linalg.norm(emb):.4f}")
    print()

# 计算不同上下文中同一词的相似度
# Compute similarity of same word in different contexts
if len(context_embeddings) >= 2:
    emb1, emb2 = context_embeddings[0][1], context_embeddings[1][1]
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"上下文1和上下文2中 '{word}' 的相似度: {similarity:.4f}")
    print(f"Similarity of '{word}' in context 1 and context 2: {similarity:.4f}")

# =============================================================================
# 7. 句子相似度计算 / Sentence Similarity Computation
# =============================================================================

def compute_sentence_similarity(model, tokenizer, sent1, sent2, method='cls'):
    """
    计算两个句子的语义相似度
    Compute semantic similarity between two sentences

    参数 / Args:
        model: BERT 模型 / BERT model
        tokenizer: 分词器 / Tokenizer
        sent1, sent2: 句子 / Sentences
        method: 池化方法 / Pooling method

    返回 / Returns:
        余弦相似度 / Cosine similarity
    """
    # 编码句子 / Encode sentences
    ids1, mask1 = encode_text(sent1, tokenizer)
    ids2, mask2 = encode_text(sent2, tokenizer)

    # 获取嵌入 / Get embeddings
    with torch.no_grad():
        outputs1 = model(ids1, attention_mask=mask1, output_hidden_states=True)
        outputs2 = model(ids2, attention_mask=mask2, output_hidden_states=True)

    # 获取句子嵌入 / Get sentence embeddings
    emb1 = get_sentence_embedding(outputs1, method)
    emb2 = get_sentence_embedding(outputs2, method)

    # 计算余弦相似度 / Compute cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    return similarity


# 句子相似度示例
# Sentence similarity examples
sentence_pairs = [
    ("我喜欢吃苹果。", "我爱吃水果。"),           # 语义相近 / Semantically similar
    ("我喜欢吃苹果。", "今天天气很好。"),         # 语义不相关 / Semantically unrelated
    ("机器学习很有趣。", "深度学习是AI的分支。"), # 相关话题 / Related topics
    ("他在北京工作。", "他在上海工作。"),         # 句式相同 / Same structure
]

print("\n句子相似度 / Sentence Similarity:")
print("-" * 60)
for sent1, sent2 in sentence_pairs:
    sim = compute_sentence_similarity(model, tokenizer, sent1, sent2)
    print(f"句子1 / Sentence 1: {sent1}")
    print(f"句子2 / Sentence 2: {sent2}")
    print(f"相似度 / Similarity: {sim:.4f}")
    print()

# =============================================================================
# 8. 批量处理 / Batch Processing
# =============================================================================

def batch_encode(texts, tokenizer, max_length=128):
    """
    批量编码文本
    Batch encode texts

    参数 / Args:
        texts: 文本列表 / List of texts
        tokenizer: 分词器 / Tokenizer
        max_length: 最大长度 / Maximum length

    返回 / Returns:
        input_ids, attention_mask 张量 / Tensors
    """
    encoded = tokenizer(
        texts,
        padding=True,               # 填充到批次中最长 / Pad to longest in batch
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']


def batch_get_embeddings(model, input_ids, attention_mask):
    """
    批量获取嵌入
    Batch get embeddings

    参数 / Args:
        model: BERT 模型 / BERT model
        input_ids: token ID 张量 / Token ID tensor
        attention_mask: 注意力掩码 / Attention mask

    返回 / Returns:
        句子嵌入矩阵 (batch_size, hidden_size) / Sentence embedding matrix
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # 使用 [CLS] token 作为句子表示
    # Use [CLS] token as sentence representation
    return outputs.last_hidden_state[:, 0, :].numpy()


# 批量处理示例
# Batch processing example
texts = [
    "自然语言处理很有趣。",
    "深度学习改变了人工智能领域。",
    "今天天气晴朗。"
]

print("\n批量处理 / Batch Processing:")
batch_ids, batch_mask = batch_encode(texts, tokenizer)
print(f"批次大小 / Batch size: {batch_ids.shape[0]}")
print(f"序列长度 / Sequence length: {batch_ids.shape[1]}")

batch_embeddings = batch_get_embeddings(model, batch_ids, batch_mask)
print(f"嵌入矩阵形状 / Embedding matrix shape: {batch_embeddings.shape}")

# =============================================================================
# 9. 使用不同层的嵌入 / Using Embeddings from Different Layers
# =============================================================================

def get_layer_embedding(outputs, layer_idx, token_idx=0):
    """
    获取特定层的 token 嵌入
    Get token embedding from a specific layer

    BERT 的不同层捕捉不同层次的信息：
    - 浅层：语法信息（词性、句法结构）
    - 深层：语义信息（含义、推理）

    Different BERT layers capture different levels of information:
    - Lower layers: syntactic info (POS, syntax structure)
    - Higher layers: semantic info (meaning, reasoning)

    参数 / Args:
        outputs: BERT 输出 / BERT outputs
        layer_idx: 层索引 (0=embedding层, 1-12=transformer层)
                  Layer index (0=embedding layer, 1-12=transformer layers)
        token_idx: token 索引 / Token index

    返回 / Returns:
        嵌入向量 / Embedding vector
    """
    # hidden_states 是一个元组，包含 embedding 层 + 12 个 transformer 层
    # hidden_states is a tuple containing embedding layer + 12 transformer layers
    return outputs.hidden_states[layer_idx][0, token_idx, :].numpy()


# 比较不同层的嵌入
# Compare embeddings from different layers
print("\n不同层的嵌入比较 / Embeddings from Different Layers:")
print("-" * 60)

layer_indices = [0, 4, 8, 12]  # embedding层, 中间层, 高层 / embedding, middle, high layers
for layer_idx in layer_indices:
    emb = get_layer_embedding(outputs, layer_idx, token_idx=1)
    print(f"层 {layer_idx:2d}: 范数/norm={np.linalg.norm(emb):.4f}")

# =============================================================================
# 10. 其他预训练模型 / Other Pre-trained Models
# =============================================================================

# 除了 BERT，还有许多其他预训练模型可用于获取嵌入
# Besides BERT, many other pre-trained models can be used for embeddings

OTHER_MODELS = """
其他常用模型 / Other Common Models:

1. RoBERTa (robustly optimized BERT approach)
   - BERT 的改进版本，更好的预训练策略
   - Improved BERT with better pre-training strategies
   - 模型名: 'roberta-base', 'roberta-large'

2. DistilBERT
   - BERT 的轻量化版本，速度更快
   - Lightweight BERT, faster inference
   - 模型名: 'distilbert-base-uncased'

3. ALBERT (A Lite BERT)
   - 参数高效的 BERT 变体
   - Parameter-efficient BERT variant
   - 模型名: 'albert-base-v2'

4. DeBERTa
   - 解耦注意力机制，性能更强
   - Disentangled attention, better performance
   - 模型名: 'microsoft/deberta-base'

5. Sentence-BERT (SBERT)
   - 专门为句子相似度优化的 BERT
   - BERT optimized for sentence similarity
   - 需要安装: pip install sentence-transformers

使用示例 / Usage Example:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(["句子1", "句子2"])
"""
print(OTHER_MODELS)

# =============================================================================
# 总结 / Summary
# =============================================================================
"""
本示例展示了 BERT Embedding 的常用操作:
This example demonstrates common BERT embedding operations:

1. 加载 BERT 模型和分词器 / Loading BERT model and tokenizer
2. 文本编码 / Text encoding
3. 获取词级嵌入 / Getting word-level embeddings
4. 获取句子嵌入（多种池化方法）/ Getting sentence embeddings (various pooling methods)
5. 上下文相关的词嵌入 / Context-aware word embeddings
6. 句子相似度计算 / Sentence similarity computation
7. 批量处理 / Batch processing
8. 使用不同层的嵌入 / Using embeddings from different layers
9. 其他预训练模型 / Other pre-trained models

BERT vs 静态词嵌入的主要优势 / Main advantages of BERT over static embeddings:
1. 上下文相关 / Context-aware: 同一个词在不同上下文中有不同表示
2. 处理多义词 / Handle polysemy: 能区分一词多义
3. 无需额外训练 / No additional training: 可直接用于各种任务

在 LLM 应用中的使用场景 / Use cases in LLM applications:
- 语义搜索 / Semantic search
- 文本相似度 / Text similarity
- 文本分类 / Text classification
- 命名实体识别 / Named entity recognition
- 问答系统 / Question answering
- 情感分析 / Sentiment analysis
"""
