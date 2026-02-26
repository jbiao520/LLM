# Activation Function 流程图解

> 通过可视化图表理解激活函数的工作原理

## 激活函数在神经网络中的位置

```mermaid
flowchart LR
    A["线性变换<br/>z = Wx + b"] --> B["激活函数<br/>a = f(z)"]
    B --> C["输出<br/>传递到下一层"]

    style B fill:#fff9c4
```

## 常见激活函数对比

```mermaid
flowchart TB
    subgraph ReLU
        R1["f(x) = max(0, x)"]
        R2["简单快速"]
        R3["可能有死亡神经元"]
    end

    subgraph GELU
        G1["f(x) = x * Φ(x)"]
        G2["平滑可微"]
        G3["Transformer 常用"]
    end

    subgraph Softmax
        S1["f(x) = exp(x) / Σexp"]
        S2["输出为概率分布"]
        S3["用于输出层"]
    end

    style R1 fill:#bbdefb
    style G1 fill:#c8e6c9
    style S1 fill:#fff9c4
```

## ReLU 工作流程

```mermaid
flowchart TB
    A["输入 z"] --> B{"z > 0?"}
    B -->|"是"| C["输出 = z"]
    B -->|"否"| D["输出 = 0"]

    C --> E["神经元激活"]
    D --> F["神经元静息"]

    style E fill:#c8e6c9
    style F fill:#ffcdd2
```

## GELU vs ReLU

```mermaid
flowchart LR
    subgraph ReLU
        R["硬切换<br/>在 0 处不可微"]
    end

    subgraph GELU
        G["平滑过渡<br/>处处可微"]
    end

    G --> A["更适合深度网络"]
    R --> B["计算更快"]

    style A fill:#c8e6c9
```

## Softmax 计算流程

```mermaid
flowchart TB
    A["logits<br/>[2.0, 1.0, 0.1]"] --> B["exp<br/>[7.39, 2.72, 1.11]"]
    B --> C["求和<br/>11.22"]
    C --> D["归一化<br/>[0.66, 0.24, 0.10]"]

    D --> E["概率分布<br/>和为 1"]

    style E fill:#c8e6c9
```

## FFN 中的激活函数

```mermaid
flowchart LR
    A["输入 x"] --> B["线性层 Up<br/>W1 * x"]
    B --> C["激活函数<br/>GELU/ReLU"]
    C --> D["线性层 Down<br/>W2 * x"]
    D --> E["输出"]

    style C fill:#fff9c4
```

## SwiGLU 结构

```mermaid
flowchart TB
    A["输入 x"] --> B["W_gate * x"]
    A --> C["W_up * x"]

    B --> D["SiLU 激活"]
    D --> E["逐元素相乘"]
    C --> E

    E --> F["W_down"]
    F --> G["输出"]

    style G fill:#c8e6c9
```

## 图解说明

### 激活函数选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| FFN (经典) | ReLU | 简单快速 |
| FFN (现代) | GELU | 更平滑 |
| FFN (LLaMA) | SwiGLU | 性能最好 |
| 输出层 | Softmax | 概率分布 |

### 关键特性

- **非线性**：使网络能学习复杂模式
- **梯度流**：影响训练稳定性
- **计算效率**：影响推理速度
