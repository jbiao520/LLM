"""
正弦位置编码示例 / Sinusoidal Position Encoding Example
======================================================

本示例展示 Transformer 原始论文中提出的正弦位置编码实现。
This example demonstrates the Sinusoidal Position Encoding proposed in the
original Transformer paper.

��文 / Paper: Attention Is All You Need (Vaswani et al., 2017)

正弦位置编码使用不同频率的正弦和余弦函数来表示位置信息。
Sinusoidal position encoding uses sine and cosine functions of different
frequencies to represent positional information.

依赖安装 / Dependencies:
    pip install torch numpy matplotlib
"""

import torch
import torch.nn as nn
import numpy as np
import math

# =============================================================================
# 1. 正弦位置编码实现 / Sinusoidal Position Encoding Implementation
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码模块 / Sinusoidal Positional Encoding Module

    使用正弦和余弦���数生成位置编码，公式如下：
    Uses sine and cosine functions to generate position encodings:

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    其中 / Where:
        pos: 位置索引 / Position index
        i: 维度索引 / Dimension index
        d_model: 模型维度 / Model dimension

    这种编码方式的关键特性 / Key properties of this encoding:
    1. 每个位置都有唯一的编码表示 / Each position has a unique encoding
    2. 编码值有界 [-1, 1] / Bounded values in [-1, 1]
    3. 可以处理任意长度的序列 / Can handle sequences of any length
    4. 相邻位置的编码相似 / Similar encodings for adjacent positions
    5. 相对位置可以通过线性变换获得 / Relative position can be obtained via linear transformation
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        初始化位置编码 / Initialize Positional Encoding

        参数 / Args:
            d_model: 模型的嵌入维度 / Embedding dimension of the model
            max_len: 最大序列长度 / Maximum sequence length
            dropout: Dropout 概率 / Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵 / Create positional encoding matrix
        # 形状: (max_len, d_model) / Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 位置索引 / Position indices: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算分母中的指数项 / Compute the exponential term in denominator
        # div_term = 10000^(2i/d_model) for i in [0, d_model/2)
        # 使用指数函数计算更稳定 / Using exp function for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数维度使用 sin，奇数维度使用 cos
        # Even dimensions use sin, odd dimensions use cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度 / Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度 / Odd dimensions

        # 添加 batch 维度 / Add batch dimension
        # 形状: (1, max_len, d_model) / Shape: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # 注册为 buffer（不参与训练）/ Register as buffer (not trained)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播 / Forward pass

        参数 / Args:
            x: 输入嵌入张量，形状 (batch_size, seq_len, d_model)
               Input embedding tensor, shape (batch_size, seq_len, d_model)

        返回 / Returns:
            添加位置编码后的张量 / Tensor with positional encoding added
        """
        seq_len = x.size(1)
        # 将位置编码加到输入上 / Add positional encoding to input
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# =============================================================================
# 2. 基本使用示例 / Basic Usage Example
# =============================================================================

# 模型参数 / Model parameters
d_model = 512      # 嵌入维度 / Embedding dimension
max_len = 100      # 最大序列长度 / Maximum sequence length
batch_size = 2     # 批次大小 / Batch size
seq_len = 10       # 序列长度 / Sequence length

# 创建位置编码模块 / Create positional encoding module
pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

# 模拟输入嵌入 / Simulated input embeddings
# 在实际应用中，这是词嵌入层的输出
# In practice, this is the output of the word embedding layer
x = torch.randn(batch_size, seq_len, d_model)

# 添加位置编码 / Add positional encoding
output = pos_encoder(x)

print(f"输入形状 / Input shape: {x.shape}")
print(f"输出形状 / Output shape: {output.shape}")
print(f"位置编码矩阵形状 / PE matrix shape: {pos_encoder.pe.shape}")

# =============================================================================
# 3. 可视化位置编码 / Visualizing Positional Encoding
# =============================================================================

def visualize_positional_encoding(pe_matrix, num_positions=50, num_dimensions=100):
    """
    可视化位置编码矩阵 / Visualize positional encoding matrix

    参数 / Args:
        pe_matrix: 位置编码矩阵 (1, max_len, d_model) / PE matrix
        num_positions: 要可视化的位置数量 / Number of positions to visualize
        num_dimensions: 要可视化的维度数量 / Number of dimensions to visualize
    """
    import matplotlib.pyplot as plt

    # 提取编码矩阵 / Extract encoding matrix
    pe = pe_matrix[0, :num_positions, :num_dimensions].numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(pe, aspect='auto', cmap='RdBu')
    plt.colorbar(label='Encoding Value / 编码值')
    plt.xlabel('Dimension / 维度')
    plt.ylabel('Position / 位置')
    plt.title('Sinusoidal Positional Encoding / 正弦位置编码')
    plt.tight_layout()
    plt.savefig('positional_encoding.png', dpi=150)
    plt.close()
    print("可视化已保存到 / Visualization saved to: positional_encoding.png")


# 可视化（取消注释以运行）/ Visualize (uncomment to run)
# visualize_positional_encoding(pos_encoder.pe)

# =============================================================================
# 4. 分析位置编码的特性 / Analyzing Properties of Positional Encoding
# =============================================================================

def analyze_pe_properties(pe_matrix):
    """
    分析位置编码的数学特性 / Analyze mathematical properties of positional encoding

    参数 / Args:
        pe_matrix: 位置编码矩阵 (1, max_len, d_model) / PE matrix
    """
    pe = pe_matrix[0]  # 移除 batch 维度 / Remove batch dimension

    print("\n" + "="*60)
    print("位置编码特性分析 / Positional Encoding Properties Analysis")
    print("="*60)

    # 1. 编码范围 / Encoding range
    print(f"\n1. 编码值范围 / Encoding value range:")
    print(f"   最小值 / Min: {pe.min().item():.6f}")
    print(f"   最大值 / Max: {pe.max().item():.6f}")

    # 2. 相邻位置的相似度 / Similarity between adjacent positions
    print(f"\n2. 相邻位置的余弦相似度 / Cosine similarity of adjacent positions:")
    for pos in [0, 10, 50, 100]:
        if pos + 1 < pe.size(0):
            v1, v2 = pe[pos], pe[pos+1]
            sim = torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
            print(f"   位置 {pos} vs {pos+1}: {sim.item():.6f}")

    # 3. 不同位置之间的相似度 / Similarity between different positions
    print(f"\n3. 不同位置之间的余弦相似度 / Cosine similarity between different positions:")
    pos_pairs = [(0, 1), (0, 10), (0, 50), (10, 20), (10, 100)]
    for p1, p2 in pos_pairs:
        if p2 < pe.size(0):
            v1, v2 = pe[p1], pe[p2]
            sim = torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
            print(f"   位置 {p1} vs {p2}: {sim.item():.6f}")

    # 4. 特定维度的波形 / Wave patterns in specific dimensions
    print(f"\n4. 不同维度的波形分析 / Wave pattern analysis for different dimensions:")
    dims_to_check = [0, 10, 100, 256, 511]
    for dim in dims_to_check:
        if dim < pe.size(1):
            values = pe[:20, dim]
            print(f"   维度 {dim}: 前5个值 = {values[:5].tolist()}")


analyze_pe_properties(pos_encoder.pe)

# =============================================================================
# 5. 相对位置关系 / Relative Position Relationships
# =============================================================================

def demonstrate_relative_position(pe_matrix):
    """
    演示正弦编码的相对位置特性 / Demonstrate relative position property

    正弦编码的一个重要特性是：
    An important property of sinusoidal encoding:
    PE(pos+k) 可以表示为 PE(pos) 的线性函数
    PE(pos+k) can be expressed as a linear function of PE(pos)

    这是因为:
    This is because:
    sin(x+k) = sin(x)cos(k) + cos(x)sin(k)
    cos(x+k) = cos(x)cos(k) - sin(x)sin(k)
    """
    pe = pe_matrix[0]

    print("\n" + "="*60)
    print("相对位置关系演示 / Relative Position Relationship Demo")
    print("="*60)

    # 选择两个位置，检查它们的相对关系
    # Choose two positions and check their relative relationship
    pos1, pos2 = 10, 20
    k = pos2 - pos1  # 相对位移 / Relative offset

    v1 = pe[pos1]
    v2 = pe[pos2]

    # 计算相似度 / Compute similarity
    sim = torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
    print(f"\n位置 {pos1} 和 {pos2} (偏移 k={k}) 的余弦相似度: {sim.item():.6f}")

    # 检查其他具有相同相对偏移的位置对
    # Check other position pairs with same relative offset
    print(f"\n其他偏移 k={k} 的位置对相似度:")
    for start in [0, 30, 50, 100]:
        if start + k < pe.size(0):
            v_start = pe[start]
            v_end = pe[start + k]
            sim = torch.cosine_similarity(v_start.unsqueeze(0), v_end.unsqueeze(0))
            print(f"   位置 {start} vs {start+k}: {sim.item():.6f}")


demonstrate_relative_position(pos_encoder.pe)

# =============================================================================
# 6. 与词嵌入结合 / Combining with Word Embeddings
# =============================================================================

class EmbeddingWithPositionalEncoding(nn.Module):
    """
    词嵌入 + 位置编码的组合模块 / Word Embedding + Positional Encoding Module

    这是 Transformer 编码器输入层的典型实现
    This is a typical implementation of Transformer encoder input layer
    """

    def __init__(self, vocab_size, d_model, max_len=5000, dropout=0.1):
        """
        参数 / Args:
            vocab_size: 词表大小 / Vocabulary size
            d_model: 模型维度 / Model dimension
            max_len: 最大序列长度 / Maximum sequence length
            dropout: Dropout 概率 / Dropout probability
        """
        super().__init__()

        # 词嵌入层 / Word embedding layer
        # padding_idx=0 表示 padding token 的嵌入固定为 0
        # padding_idx=0 means padding token embedding is fixed to 0
        self.word_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 位置编码 / Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # 缩放因子 / Scaling factor
        # 原始论文建议对词嵌入乘以 sqrt(d_model)
        # Original paper suggests scaling word embeddings by sqrt(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        """
        前向传播 / Forward pass

        参数 / Args:
            x: token ID 张量 (batch_size, seq_len)
               Token ID tensor

        返回 / Returns:
            带位置编码的嵌入 / Embeddings with positional encoding
        """
        seq_len = x.size(1)

        # 获取词嵌入并缩放 / Get word embeddings and scale
        word_emb = self.word_embedding(x) * self.scale

        # 添加位置编码 / Add positional encoding
        return self.pos_encoding(word_emb)


# 示例使用 / Example usage
vocab_size = 10000
d_model = 512

model = EmbeddingWithPositionalEncoding(vocab_size, d_model)

# 模拟输入 token IDs / Simulated input token IDs
# 形状: (batch_size, seq_len) / Shape: (batch_size, seq_len)
input_ids = torch.randint(0, vocab_size, (2, 20))

# 获取带位置编码的嵌入 / Get embeddings with positional encoding
embeddings = model(input_ids)

print(f"\n组合模型输出 / Combined Model Output:")
print(f"输入 IDs 形状 / Input IDs shape: {input_ids.shape}")
print(f"输出嵌入形状 / Output embeddings shape: {embeddings.shape}")

# =============================================================================
# 7. NumPy 实现（用于理解）/ NumPy Implementation (for Understanding)
# =============================================================================

def get_positional_encoding_numpy(seq_len, d_model):
    """
    使用 NumPy 实现正弦位置编码 / Sinusoidal PE using NumPy

    这个实现更清晰地展示了计算过程
    This implementation more clearly shows the computation process

    参数 / Args:
        seq_len: 序列长度 / Sequence length
        d_model: 模型维度 / Model dimension

    返回 / Returns:
        位置编码矩阵 (seq_len, d_model) / Positional encoding matrix
    """
    # 初始化编码矩阵 / Initialize encoding matrix
    pe = np.zeros((seq_len, d_model))

    # 对每个位置 / For each position
    for pos in range(seq_len):
        # 对每个维度 / For each dimension
        for i in range(d_model // 2):
            # 计算频率项 / Compute frequency term
            # freq = 1 / (10000^(2i/d_model))
            freq = 1.0 / (10000 ** (2 * i / d_model))

            # 偶数维度使用 sin / Even dimensions use sin
            pe[pos, 2*i] = np.sin(pos * freq)

            # 奇数维度使用 cos / Odd dimensions use cos
            pe[pos, 2*i + 1] = np.cos(pos * freq)

    return pe


# 使用 NumPy 实现 / Using NumPy implementation
pe_numpy = get_positional_encoding_numpy(10, 512)
print(f"\nNumPy 实现的编码矩阵形状 / NumPy PE matrix shape: {pe_numpy.shape}")

# 验证 NumPy 和 PyTorch 实现的一致性
# Verify consistency between NumPy and PyTorch implementations
pe_torch = pos_encoder.pe[0, :10, :].numpy()
difference = np.abs(pe_numpy - pe_torch).max()
print(f"NumPy 和 PyTorch 实现的最大差异 / Max difference: {difference:.10f}")

# =============================================================================
# 8. 长度外推能力 / Length Extrapolation Capability
# =============================================================================

def test_extrapolation(d_model=512, train_len=100, test_lengths=[150, 200, 500]):
    """
    测试正弦编码的长度外推能力 / Test extrapolation capability

    正弦编码理论上可以处理任意长度的序列
    Sinusoidal encoding can theoretically handle sequences of any length

    参数 / Args:
        d_model: 模型维度 / Model dimension
        train_len: 训练时的最大长度 / Maximum length during training
        test_lengths: 测试长度列表 / List of test lengths
    """
    print("\n" + "="*60)
    print("长度外推测试 / Length Extrapolation Test")
    print("="*60)

    # 创建一个支持长序列的编码器 / Create encoder supporting long sequences
    pe = SinusoidalPositionalEncoding(d_model, max_len=max(test_lengths) + 100)

    # 测试不同长度 / Test different lengths
    for length in [train_len] + test_lengths:
        x = torch.randn(1, length, d_model)
        output = pe(x)
        print(f"  长度 {length}: 编码成功 / Encoding successful, 输出形状 / output shape: {output.shape}")


test_extrapolation()

# =============================================================================
# 总结 / Summary
# =============================================================================
"""
本示例展示了正弦位置编码的完整实现和分析:
This example demonstrates complete implementation and analysis of sinusoidal PE:

1. 基本实现 / Basic Implementation
   - 使用 sin/cos 函数生成编码 / Generate encoding using sin/cos functions
   - 不同维度使用不同频率 / Different dimensions use different frequencies

2. 关键特性 / Key Properties
   - 唯一性: 每个位置编码唯一 / Uniqueness: each position has unique encoding
   - 有界性: 值在 [-1, 1] 范围内 / Bounded: values in [-1, 1]
   - 连续性: 相邻位置编码相似 / Continuity: adjacent positions have similar encodings
   - 外推性: 可处理任意长度序列 / Extrapolation: can handle any sequence length

3. 与词嵌入结合 / Combining with Word Embeddings
   - 输入 = 词嵌入 * sqrt(d_model) + 位置编码
   - Input = Word Embedding * sqrt(d_model) + Positional Encoding

4. 应用场景 / Use Cases
   - Transformer 编码器输入 / Transformer encoder input
   - 任何需要位置信息的序列模型 / Any sequence model needing position info

与可学习位置编码的比较 / Comparison with Learnable Positional Encoding:
| 特性 | 正弦编码 | 可学习编码 |
|------|----------|------------|
| 参数量 | 0 | max_len * d_model |
| 外推能力 | 强 | 弱（受限于训练长度）|
| 灵活性 | 固定 | 可适应数据 |
"""
