# Normalization 流程图解

> 通过可视化图表理解归一化的工作流程

## 归一化位置

```mermaid
flowchart LR
    subgraph Transformer层
        A["输入"] --> B["Layer Norm"]
        B --> C["Self-Attention"]
        C --> D["残差连接"]
        D --> E["Layer Norm"]
        E --> F["FFN"]
        F --> G["残差连接"]
        G --> H["输出"]
    end

    style B fill:#fff9c4
    style E fill:#fff9c4
```

## Layer Normalization 流程

```mermaid
flowchart TB
    A["输入向量 x<br/>[x1, x2, ..., xd]"] --> B["计算均值<br/>μ = mean(x)"]
    A --> C["计算方差<br/>σ² = var(x)"]

    B --> D["标准化<br/>(x - μ) / √(σ² + ε)"]
    C --> D

    D --> E["缩放和平移<br/>γ * x + β"]
    E --> F["输出<br/>归一化后的向量"]

    style F fill:#c8e6c9
```

## Batch Norm vs Layer Norm

```mermaid
flowchart TB
    subgraph BatchNorm
        BN1["沿批次维度归一化"]
        BN2["每特征一个统计量"]
        BN3["适合 CV 任务"]
        BN1 --> BN2 --> BN3
    end

    subgraph LayerNorm
        LN1["沿特征维度归一化"]
        LN2["每样本独立计算"]
        LN3["适合 NLP 任务"]
        LN1 --> LN2 --> LN3
    end

    style BN3 fill:#bbdefb
    style LN3 fill:#c8e6c9
```

## RMS Norm 简化流程

```mermaid
flowchart LR
    A["输入 x"] --> B["计算 RMS<br/>√(mean(x²))"]
    B --> C["归一化<br/>x / RMS"]
    C --> D["缩放<br/>γ * x"]

    style D fill:#c8e6c9
```

## 归一化对比

```mermaid
flowchart TB
    subgraph LayerNorm
        L1["计算均值和方差"]
        L2["减均值除标准差"]
        L3["可学习的 γ, β"]
    end

    subgraph RMSNorm
        R1["只计算 RMS"]
        R2["除以 RMS"]
        R3["只有 γ，无 β"]
    end

    subgraph 效率对比
        E1["RMSNorm 更快<br/>减少约 25% 计算"]
    end

    style E1 fill:#c8e6c9
```

## 训练稳定性

```mermaid
flowchart LR
    subgraph 无归一化
        A1["梯度爆炸/消失"]
        A2["训练不稳定"]
    end

    subgraph 有归一化
        B1["梯度稳定"]
        B2["收敛更快"]
        B3["性能更好"]
    end

    style A2 fill:#ffcdd2
    style B3 fill:#c8e6c9
```

## 图解说明

### 关键概念

| 方法 | 公式 | 特点 |
|------|------|------|
| Layer Norm | (x-μ)/σ * γ + β | NLP 常用 |
| RMS Norm | x/RMS * γ | LLaMA 使用 |
| Batch Norm | 沿批次归一化 | CV 常用 |

### 选择建议

- **Transformer NLP**: Layer Norm 或 RMS Norm
- **LLaMA 类模型**: RMS Norm（更快）
- **CNN 视觉**: Batch Norm
