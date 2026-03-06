# 稀疏注意力（Sparse Attention）深入版

> 面向有机器学习基础读者的稀疏注意力深度指南

## 1. 标准注意力复杂度分析

### 1.1 计算复杂度

标准自注意力：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

对于序列长度 $n$：
- $QK^T$: $O(n^2 \cdot d)$
- Softmax: $O(n^2)$
- 乘以 $V$: $O(n^2 \cdot d)$

**总复杂度: $O(n^2 \cdot d)$**

### 1.2 内存复杂度

需要存储 $n \times n$ 的注意力矩阵：
$$\text{Memory} = O(n^2)$$

对于 $n = 16K$，仅注意力矩阵就需要 $16K \times 16K \times 4 \text{ bytes} = 1\text{GB}$

## 2. 稀疏注意力模式

### 2.1 数学定义

稀疏注意力用掩码矩阵 $M$ 定义：
$$\text{Attention}_{sparse} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \odot M\right) V$$

其中 $M_{ij} \in \{0, 1\}$，0 表示屏蔽。

**稀疏度：**
$$\text{Sparsity} = 1 - \frac{\sum_{i,j} M_{ij}}{n^2}$$

### 2.2 局部注意力

**滑动窗口：**
$$M_{ij} = \begin{cases} 1 & \text{if } |i - j| \leq w \\ 0 & \text{otherwise} \end{cases}$$

复杂度：$O(n \cdot w \cdot d)$，其中 $w$ 是窗口大小。

**扩张滑动窗口：**
$$M_{ij} = \begin{cases} 1 & \text{if } |i - j| \mod d = 0 \\ 0 & \text{otherwise} \end{cases}$$

可以扩大感受野而不增加计算。

### 2.3 全局注意力

特定位置 $G$ 关注所有位置：
$$M_{ij} = 1 \quad \text{if } i \in G \text{ or } j \in G$$

复杂度增加：$O(|G| \cdot n)$

### 2.4 随机注意力

每个位置随机选择 $r$ 个位置：
$$M_{ij} = \begin{cases} 1 & \text{with probability } r/n \\ 0 & \text{otherwise} \end{cases}$$

期望复杂度：$O(n \cdot r)$

## 3. 代表性算法

### 3.1 Longformer

**注意力模式：**

```
┌─────────────────────────────────────────────────────────────────┐
│                     Longformer 注意力                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   M = M_local + M_global + M_dilated                           │
│                                                                 │
│   M_local: 滑动窗口 (w=512)                                     │
│   M_global: 特定 token 关注全部                                  │
│   M_dilated: 扩张窗口                                           │
│                                                                 │
│   每层复杂度: O(n × (w + g + d))                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**实现技巧：**
- 局部注意力用 `im2col` 或 `unfold` 高效实现
- 全局注意力单独计算后合并

### 3.2 BigBird

**注意力模式：**
$$M = M_{random} + M_{window} + M_{global}$$

**理论保证：**

BigBird 证明了这种组合是完整注意力的良好近似：

$$\mathbb{E}[\text{Attention}_{sparse}] \approx \text{Attention}_{full}$$

**关键参数：**
- 随机注意力：$r = \sqrt{n}$
- 窗口大小：$w = 3\sqrt{n}$
- 全局位置：$g = \sqrt{n}$

总复杂度：$O(n\sqrt{n})$

### 3.3 LongNet (Dilated Attention)

**扩张注意力：**

按距离分组：
$$A_i^{(d)} = \text{Attention}(Q_i, K_{i+d\mathbb{Z}}, V_{i+d\mathbb{Z}})$$

不同扩张率 $d$ 捕获不同距离的信息。

**多尺度融合：**
$$A = \sum_d w_d \cdot A^{(d)}$$

复杂度：$O(n)$（当扩张率随距离增加）

### 3.4 Sparse Transformer (OpenAI)

**固定稀疏模式：**

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sparse Transformer                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Strided pattern:              Fixed pattern:                  │
│   ┌─────────────────┐           ┌─────────────────┐            │
│   │ █ █ █ █ █ █ █ █ │           │ █     █     █   │            │
│   │ █ █ █ █ █ █ █ █ │           │ █ █ █ █ █ █ █ █ │            │
│   │   █ █ █ █ █ █ █ │           │ █     █     █   │            │
│   │   █ █ █ █ █ █ █ │           │   █ █ █ █ █ █   │            │
│   │     █ █ █ █ █ █ │           │ █     █     █   │            │
│   │     █ █ █ █ █ █ │           │   █ █ █ █ █ █   │            │
│   │       █ █ █ █ █ │           │ █     █     █   │            │
│   │       █ █ █ █ █ │           │     █ █ █ █     │            │
│   └─────────────────┘           └─────────────────┘            │
│   等步长采样                    固定位置（如列头）               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4. 理论分析

### 4.1 近似误差

稀疏注意力与完整注意力的误差：

$$\|A_{sparse} - A_{full}\| \leq \epsilon$$

**BigBird 的理论结果：**
- 使用 $r = \Theta(\sqrt{n})$ 随机连接
- 以高概率达到 $\epsilon$ 近似

### 4.2 感受野分析

**问题：** 局部注意力的感受野有限

**解决方案：**
1. **堆叠层数：** $L$ 层窗口 $w$ 的感受野 = $L \times w$
2. **扩张窗口：** 类似空洞卷积
3. **全局位置：** 信息可以通过全局位置传递

```
层 1: 局部感受野 w
层 2: 感受野扩大到 2w
层 3: 感受野扩大到 3w
...
层 L: 感受野 = L × w
```

## 5. 实现细节

### 5.1 高效局部注意力

**im2col 方法：**
```python
def local_attention(q, k, v, window_size):
    n, d = q.shape
    # 使用 im2col 提取局部窗口
    k_local = im2col(k, window_size)  # [n, window_size, d]
    v_local = im2col(v, window_size)  # [n, window_size, d]

    # 计算局部注意力
    scores = torch.einsum('nd,nwd->nw', q, k_local) / sqrt(d)
    attn = softmax(scores, dim=-1)
    return torch.einsum('nw,nwd->nd', attn, v_local)
```

### 5.2 块稀疏注意力

将注意力矩阵分块，整块置零：

```python
def block_sparse_attention(q, k, v, block_size, sparsity):
    n = q.shape[0]
    num_blocks = n // block_size

    # 计算块级重要性
    block_importance = compute_block_importance(q, k, block_size)

    # 选择保留的块
    keep_mask = select_top_k_blocks(block_importance, 1 - sparsity)

    # 只计算保留块的注意力
    return sparse_attention_with_mask(q, k, v, keep_mask)
```

## 6. LLM 应用

### 6.1 长文本建模

| 任务 | 序列长度 | 推荐方法 |
|------|----------|----------|
| 文档分类 | 4K-16K | Longformer |
| 长文档问答 | 16K-64K | BigBird |
| 代码理解 | 32K-128K | LongNet |
| 全书处理 | 100K+ | 线性注意力 |

### 6.2 与 Flash Attention 结合

稀疏注意力可以和 Flash Attention 结合：

1. 将稀疏模式转换为块掩码
2. 使用 Flash Attention 的块稀疏变体
3. 只计算非零块

## 参考文献

- [Longformer](https://arxiv.org/abs/2004.05150) - Beltagy et al., 2020
- [BigBird](https://arxiv.org/abs/2007.14062) - Zaheer et al., 2020
- [Sparse Transformer](https://arxiv.org/abs/1904.10509) - Child et al., 2019
- [LongNet](https://arxiv.org/abs/2307.02486) - Ding et al., 2023
- [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)
