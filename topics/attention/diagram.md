# Attention 流程图解

> 通过可视化图表理解注意力机制的工作流程

## 自注意力核心流程

```mermaid
flowchart TB
    A["输入 X"] --> B["线性投影"]
    B --> C["Q (Query)"]
    B --> D["K (Key)"]
    B --> E["V (Value)"]

    C --> F["Q × K^T<br/>注意力分数"]
    D --> F

    F --> G["Scale<br/>除以 √d"]
    G --> H["Softmax<br/>归一化"]
    H --> I["× V<br/>加权求和"]
    E --> I

    I --> J["输出"]

    style J fill:#c8e6c9
```

## 注意力分数计算

```mermaid
flowchart LR
    subgraph QK矩阵乘法
        A["Q: [seq, d]"] --> C["分数: [seq, seq]"]
        B["K: [seq, d]"] --> C
    end

    C --> D["Softmax"]
    D --> E["注意力权重"]

    style E fill:#c8e6c9
```

## 多头注意力

```mermaid
flowchart TB
    A["输入"] --> B["分割为 h 个头"]

    B --> C1["头 1<br/>Attention(Q1,K1,V1)"]
    B --> C2["头 2<br/>Attention(Q2,K2,V2)"]
    B --> C3["..."]
    B --> C4["头 h<br/>Attention(Qh,Kh,Vh)"]

    C1 --> D["Concat"]
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E["线性投影"]
    E --> F["输出"]

    style F fill:#c8e6c9
```

## 因果掩码

```mermaid
flowchart TB
    subgraph 掩码矩阵
        A["位置 1"] --> B["✓ ✗ ✗ ✗"]
        C["位置 2"] --> D["✓ ✓ ✗ ✗"]
        E["位置 3"] --> F["✓ ✓ ✓ ✗"]
        G["位置 4"] --> H["✓ ✓ ✓ ✓"]
    end

    I["只能看到之前的位置"]

    style B fill:#c8e6c9
    style D fill:#c8e6c9
    style F fill:#c8e6c9
    style H fill:#c8e6c9
```

## 注意力可视化

```mermaid
flowchart LR
    subgraph "The cat sat on mat"
        A["The"] -->|"0.1"| B["cat"]
        B -->|"0.8"| C["sat"]
        C -->|"0.3"| D["on"]
        D -->|"0.5"| E["mat"]
    end

    F["线越粗 = 注意力越高"]

    style B fill:#fff9c4
```

## Flash Attention 优化

```mermaid
flowchart TB
    subgraph 标准实现
        S1["计算完整注意力矩阵<br/>O(N²) 内存"]
        S2["存储到 HBM"]
        S3["应用 Softmax"]
    end

    subgraph Flash Attention
        F1["分块计算<br/>O(N) 内存"]
        F2["在 SRAM 中完成"]
        F3["只写最终结果"]
    end

    style S1 fill:#ffcdd2
    style F1 fill:#c8e6c9
```

## 图解说明

### 注意力公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 关键概念

| 概念 | 说明 |
|------|------|
| Query | 查询向量，"我在找什么" |
| Key | 键向量，"我是什么" |
| Value | 值向量，"我的内容" |
| 头数 | 并行注意力的数量 |

### 计算复杂度

- 时间: O(N² × d)
- 空间: O(N²) (标准) / O(N) (Flash)
