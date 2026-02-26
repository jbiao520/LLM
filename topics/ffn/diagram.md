# FFN (Feed-Forward Network) 流程图解

> 通过可视化图表理解前馈网络的工作流程

## FFN 基本结构

```mermaid
flowchart LR
    A["输入<br/>[batch, seq, d]"] --> B["线性层 Up<br/>W1: [d, 4d]"]
    B --> C["激活函数<br/>GELU/ReLU"]
    C --> D["线性层 Down<br/>W2: [4d, d]"]
    D --> E["输出<br/>[batch, seq, d]"]

    style E fill:#c8e6c9
```

## FFN 在 Transformer 中的位置

```mermaid
flowchart TB
    A["Attention 输出"] --> B["残差连接"]
    B --> C["Layer Norm"]
    C --> D["FFN"]
    D --> E["残差连接"]
    E --> F["Layer Norm"]
    F --> G["输出到下一层"]

    style D fill:#fff9c4
```

## 维度变化

```mermaid
flowchart TB
    A["输入: 768"] --> B["Up投影: 3072<br/>(4x expansion)"]
    B --> C["激活: 3072"]
    C --> D["Down投影: 768"]

    E["中间层通常是输入的 4 倍"]

    style D fill:#c8e6c9
```

## GELU vs ReLU

```mermaid
flowchart TB
    subgraph ReLU
        R1["x > 0 → x"]
        R2["x ≤ 0 → 0"]
        R3["硬边界"]
    end

    subgraph GELU
        G1["平滑过渡"]
        G2["处处可微"]
        G3["Transformer 常用"]
    end

    style G3 fill:#c8e6c9
```

## SwiGLU 结构

```mermaid
flowchart TB
    A["输入 x"] --> B["W_gate * x"]
    A --> C["W_up * x"]

    B --> D["SiLU 激活<br/>x * sigmoid(x)"]
    D --> E["逐元素相乘 ⊙"]
    C --> E

    E --> F["W_down"]
    F --> G["输出"]

    style G fill:#c8e6c9
```

## 标准 FFN vs SwiGLU FFN

```mermaid
flowchart TB
    subgraph 标准FFN
        S1["x → W1 → GELU → W2 → out"]
        S2["2 个权重矩阵"]
    end

    subgraph SwiGLU
        G1["x → W_g → SiLU"] --> G2["⊙ W_u(x)"]
        G2 --> G3["→ W_d → out"]
        G4["3 个权重矩阵"]
    end

    style S2 fill:#bbdefb
    style G4 fill:#c8e6c9
```

## FFN 的作用

```mermaid
flowchart LR
    A["Attention"] --> B["聚合信息"]
    C["FFN"] --> D["非线性变换<br/>记忆/知识存储"]

    B --> E["Transformer 层"]
    D --> E

    style D fill:#fff9c4
```

## 图解说明

### 关键参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| 扩展比 | 中间层/输入维度 | 4x |
| 激活函数 | 非线性 | GELU/SwiGLU |
| Dropout | 正则化 | 0.1 |

### 参数量估算

对于隐藏维度 $d$：
- 标准 FFN: $2 \times d \times 4d = 8d^2$
- SwiGLU: $3 \times d \times \frac{4d}{3} \times 2 = 8d^2$

### 激活函数选择

| 激活函数 | 模型 | 特点 |
|----------|------|------|
| ReLU | 早期 | 简单 |
| GELU | BERT, GPT | 平滑 |
| SwiGLU | LLaMA | 性能最好 |
