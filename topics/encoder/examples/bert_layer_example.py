"""
BERT 风格编码层示例 / BERT-Style Encoder Layer Example
======================================================

本示例展示 BERT 风格的编码器实现。
This example demonstrates BERT-style encoder implementation.

依赖安装 / Dependencies:
    pip install torch transformers matplotlib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# 1. BERT Embeddings
# =============================================================================

class BertEmbeddings(nn.Module):
    """
    BERT 的输入嵌入层 / BERT Input Embedding Layer

    包含三部分 / Contains three parts:
    1. Token Embeddings
    2. Segment (Token Type) Embeddings
    3. Position Embeddings
    """
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout=0.1):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        """
        参数 / Args:
            input_ids: [batch, seq_len] - token indices
            token_type_ids: [batch, seq_len] - segment indices (0 or 1)

        返回 / Returns:
            embeddings: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)

        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)

        # Token type (segment) embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # 合并所有 embeddings
        embeddings = token_embeds + position_embeds + token_type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

# =============================================================================
# 2. BERT Self-Attention
# =============================================================================

class BertSelfAttention(nn.Module):
    """BERT 的自注意力层 / BERT Self-Attention Layer"""
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super().__init__()

        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        # 计算 Q, K, V
        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 应用 attention mask
        if attention_mask is not None:
            # attention_mask: [batch, 1, 1, seq_len] (扩展维度以匹配 scores)
            scores = scores + attention_mask

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return output, attn_weights

# =============================================================================
# 3. BERT Layer
# =============================================================================

class BertLayer(nn.Module):
    """
    完整的 BERT 层 / Complete BERT Layer

    包含 / Contains:
    - Self-Attention
    - Intermediate (FFN 第一层)
    - Output (FFN 第二层)
    """
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout=0.1):
        super().__init__()

        # Self-Attention
        self.attention = BertSelfAttention(hidden_size, num_attention_heads, dropout)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.attention_dropout = nn.Dropout(dropout)

        # FFN (Intermediate + Output)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        # Self-Attention
        attention_output, attn_weights = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)

        # Add & Norm (Post-LN style)
        hidden_states = self.attention_norm(hidden_states + attention_output)

        # FFN
        intermediate_output = F.gelu(self.intermediate(hidden_states))
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)

        # Add & Norm
        layer_output = self.output_norm(hidden_states + layer_output)

        return layer_output, attn_weights

# =============================================================================
# 4. 简化的 BERT 模型 / Simplified BERT Model
# =============================================================================

class SimpleBERT(nn.Module):
    """
    简化的 BERT 模型 / Simplified BERT Model

    用于理解 BERT 的核心架构
    For understanding BERT's core architecture
    """
    def __init__(self, vocab_size=30522, hidden_size=768, num_attention_heads=12,
                 intermediate_size=3072, num_hidden_layers=12,
                 max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
        super().__init__()

        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout
        )

        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])

        self.pooler = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        参数 / Args:
            input_ids: [batch, seq_len]
            token_type_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (1 for actual tokens, 0 for padding)

        返回 / Returns:
            sequence_output: [batch, seq_len, hidden_size]
            pooled_output: [batch, hidden_size] - [CLS] token 的池化表示
        """
        # Embeddings
        hidden_states = self.embeddings(input_ids, token_type_ids)

        # 准备 attention mask
        if attention_mask is not None:
            # 转换为 additive mask: 0 -> 0, 1 -> -inf
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        # 通过所有层
        all_attentions = []
        for layer in self.layers:
            hidden_states, attn_weights = layer(hidden_states, extended_attention_mask)
            all_attentions.append(attn_weights)

        # Pooler ([CLS] token)
        pooled_output = torch.tanh(self.pooler(hidden_states[:, 0]))

        return hidden_states, pooled_output, all_attentions

# =============================================================================
# 5. 测试 BERT 模型 / Test BERT Model
# =============================================================================

print("=" * 60)
print("BERT 模型测试 / BERT Model Test")
print("=" * 60)

torch.manual_seed(42)

# 创建小规模 BERT 用于测试 / Create small BERT for testing
bert = SimpleBERT(
    vocab_size=1000,
    hidden_size=256,
    num_attention_heads=8,
    intermediate_size=1024,
    num_hidden_layers=4,
    max_position_embeddings=128
)

# 创建输入 / Create input
batch_size, seq_len = 2, 32
input_ids = torch.randint(0, 1000, (batch_size, seq_len))
token_type_ids = torch.zeros_like(input_ids)
attention_mask = torch.ones_like(input_ids)

# 前向传播 / Forward pass
sequence_output, pooled_output, attentions = bert(input_ids, token_type_ids, attention_mask)

print(f"\n输入形状 / Input shape: {input_ids.shape}")
print(f"序列输出形状 / Sequence output shape: {sequence_output.shape}")
print(f"池化输出形状 / Pooled output shape: {pooled_output.shape}")
print(f"注意力层数 / Number of attention layers: {len(attentions)}")

# 参数统计 / Parameter statistics
total_params = sum(p.numel() for p in bert.parameters())
print(f"\n总参数量 / Total parameters: {total_params:,}")

# =============================================================================
# 6. MLM 任务演示 / MLM Task Demonstration
# =============================================================================

print("\n" + "=" * 60)
print("MLM (Masked Language Model) 任务演示 / MLM Task Demo")
print("=" * 60)

class BertMLMHead(nn.Module):
    """BERT 的 MLM 预测头 / BERT MLM Prediction Head"""
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):
        # Transform
        hidden_states = F.gelu(self.transform(hidden_states))
        hidden_states = self.layer_norm(hidden_states)
        # Decode to vocab
        logits = self.decoder(hidden_states)
        return logits

# 创建 MLM 头 / Create MLM head
mlm_head = BertMLMHead(256, 1000)

# 获取 MLM 预测 / Get MLM predictions
mlm_logits = mlm_head(sequence_output)

print(f"\nMLM logits 形状 / MLM logits shape: {mlm_logits.shape}")
print(f"  [batch, seq_len, vocab_size]")

# 对 [MASK] 位置进行预测 / Predict at [MASK] position
# 假设位置 5 是 [MASK]
mask_position = 5
mask_logits = mlm_logits[0, mask_position]
predicted_token = mask_logits.argmax().item()

print(f"\n位置 {mask_position} 的预测 token ID / Predicted token at position {mask_position}: {predicted_token}")
print(f"Top-5 预测 / Top-5 predictions:")
top5 = torch.topk(mask_logits, 5)
for i, (score, idx) in enumerate(zip(top5.values, top5.indices)):
    print(f"  {i+1}. Token {idx.item()}: {score.item():.4f}")

# =============================================================================
# 7. 分类任务演示 / Classification Task Demonstration
# =============================================================================

print("\n" + "=" * 60)
print("分类任务演示 / Classification Task Demo")
print("=" * 60)

class BertClassificationHead(nn.Module):
    """BERT 的分类头 / BERT Classification Head"""
    def __init__(self, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pooled_output):
        return self.classifier(self.dropout(pooled_output))

# 创建分类头 / Create classification head
num_classes = 3  # 例如：负面、中性、正面
classification_head = BertClassificationHead(256, num_classes)

# 获取分类预测 / Get classification predictions
class_logits = classification_head(pooled_output)
predicted_class = class_logits.argmax(dim=-1)

print(f"\n分类 logits 形状 / Classification logits shape: {class_logits.shape}")
print(f"预测类别 / Predicted classes: {predicted_class.tolist()}")

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了 BERT 的核心架构:
This example demonstrates BERT's core architecture:

1. Embeddings:
   - Token + Position + Segment (Token Type)
   - LayerNorm + Dropout

2. Encoder Layer (Post-LN):
   - Multi-Head Self-Attention
   - Feed-Forward Network (GELU activation)

3. 特殊设计:
   - [CLS] token 用于分类任务
   - [SEP] token 分隔句子
   - [MASK] token 用于 MLM 预训练

4. 下游任务:
   - MLM: Masked Language Model
   - Classification: 使用 [CLS] 的池化输出
   - NER: 每个位置的分类
   - QA: Span 预测

5. BERT-Base 配置:
   - 12 layers, 768 hidden, 12 heads, 110M params
"""
