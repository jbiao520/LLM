# MoE (Mixture of Experts) 流程图解

> 通过可视化图表理解混合专家模型的工作流程

## MoE 核心架构

```mermaid
flowchart TB
    A["输入 token"] --> B["Router<br/>路由网络"]
    B --> C{"选择 Top-k<br/>专家"}

    C -->|"Expert 1"| D1["专家 1<br/>FFN"]
    C -->|"Expert 2"| D2["专家 2<br/>FFN"]
    C -->|"Expert 3"| D3["专家 3<br/>FFN"]
    C -->|"..."| D4["专家 N<br/>FFN"]

    D1 --> E["加权组合<br/>Σ w_i * output_i"]
    D2 --> E
    D3 --> E
    D4 --> E

    E --> F["输出"]

    style B fill:#fff9c4
    style F fill:#c8e6c9
```

## Router 工作流程

```mermaid
flowchart TB
    A["输入 x"] --> B["Router<br/>W_router * x"]
    B --> C["Softmax<br/>得到专家概率"]
    C --> D["选择 Top-k<br/>概率最高的 k 个"]

    D --> E["归一化权重<br/>w_i = p_i / Σ p_topk"]

    style E fill:#c8e6c9
```

## 稀疏激活

```mermaid
flowchart LR
    subgraph 稠密模型
        D1["所有参数激活"]
        D2["计算量: 100%"]
    end

    subgraph MoE模型
        M1["只激活 Top-k 专家"]
        M2["计算量: ~10%"]
    end

    style D2 fill:#ffcdd2
    style M2 fill:#c8e6c9
```

## 专家并行

```mermaid
flowchart TB
    A["输入批次"] --> B["分发到各 GPU"]

    B --> C1["GPU 1<br/>专家 1-8"]
    B --> C2["GPU 2<br/>专家 9-16"]
    B --> C3["GPU 3<br/>专家 17-24"]
    B --> C4["GPU 4<br/>专家 25-32"]

    C1 --> D["All-to-All<br/>通信"]
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E["聚合结果"]

    style E fill:#c8e6c9
```

## 负载均衡

```mermaid
flowchart TB
    subgraph 问题
        A["某些专家被过度使用"]
        B["其他专家闲置"]
        C["训练不均衡"]
    end

    subgraph 解决方案
        D["辅助损失函数"]
        E["专家容量限制"]
        F["噪声 Top-k"]
    end

    A --> D
    B --> E
    C --> F

    style D fill:#c8e6c9
    style E fill:#c8e6c9
    style F fill:#c8e6c9
```

## MoE vs 稠密模型对比

```mermaid
flowchart LR
    subgraph 稠密模型
        D1["参数: 7B"]
        D2["激活: 7B"]
        D3["推理成本: 100%"]
    end

    subgraph MoE模型
        M1["参数: 47B"]
        M2["激活: 8B"]
        M3["推理成本: ~15%"]
    end

    style D3 fill:#ffcdd2
    style M3 fill:#c8e6c9
```

## 图解说明

### 关键概念

| 概念 | 说明 | 典型值 |
|------|------|--------|
| 专家数 | FFN 的数量 | 8-128 |
| Top-k | 每次激活的专家数 | 1-2 |
| 容量因子 | 每专家最大处理量 | 1.25-2.0 |

### 著名 MoE 模型

| 模型 | 总参数 | 激活参数 | 专家数 |
|------|--------|----------|--------|
| Mixtral 8x7B | 47B | 13B | 8 |
| Switch Transformer | 1.6T | - | 2048 |
| GPT-4 | ? | ? | ? |

### 优势

- 更少的计算量
- 更大的知识容量
- 更好的专业化
