# 量化流程图解

> 通过可视化图表理解模型量化的完整工作流程

## 量化核心原理

```mermaid
flowchart LR
    subgraph 原始精度
        A["FP16 权重<br/>0.123456789<br/>2 字节/参数"]
    end

    subgraph 量化过程
        B["缩放 Scale<br/>s = max|W| / 2^(b-1)"]
        C["舍入 Round<br/>q = round(w/s)"]
        D["钳位 Clamp<br/>限制在 [-2^(b-1), 2^(b-1)-1]"]
    end

    subgraph 量化结果
        E["INT4 权重<br/>5<br/>0.5 字节/参数"]
    end

    A --> B --> C --> D --> E

    style A fill:#bbdefb
    style E fill:#c8e6c9
```

## 量化方法对比

```mermaid
flowchart TB
    subgraph GPTQ
        G1["逐列量化"] --> G2["Hessian 逆更新"]
        G2 --> G3["精度高，量化慢"]
    end

    subgraph AWQ
        A1["激活感知"] --> A2["保护重要权重"]
        A2 --> A3["速度快，精度好"]
    end

    subgraph GGUF
        GG1["CPU 优化"] --> GG2["多种量化级别"]
        GG2 --> GG3["适合边缘设备"]
    end

    style G3 fill:#fff9c4
    style A3 fill:#c8e6c9
    style GG3 fill:#bbdefb
```

## 精度与大小关系

```mermaid
flowchart LR
    subgraph 模型大小对比["70B 模型"]
        FP16["FP16<br/>140 GB"] --> INT8["INT8<br/>70 GB"]
        INT8 --> INT4["INT4<br/>35 GB"]
    end

    subgraph 质量影响
        Q1["FP16: 基准"]
        Q2["INT8: ~0% 损失"]
        Q3["INT4: ~1% 损失"]
    end

    FP16 -.-> Q1
    INT8 -.-> Q2
    INT4 -.-> Q3

    style FP16 fill:#ffcdd2
    style INT4 fill:#c8e6c9
```

## GPTQ 量化流程

```mermaid
flowchart TB
    A[加载预训练模型] --> B[准备校准数据<br/>~512 样本]
    B --> C[逐层处理]

    subgraph 每层处理
        C --> D[量化当前列<br/>w_i → q_i]
        D --> E[计算误差<br/>δ = w - q]
        E --> F[更新后续列<br/>补偿误差]
        F --> G{还有列?}
        G -->|是| D
        G -->|否| H[保存量化模型]
    end

    style H fill:#c8e6c9
```

## AWQ 量化流程

```mermaid
flowchart TB
    A[加载预训练模型] --> B[运行校准数据<br/>获取激活值]
    B --> C[分析权重重要性<br/>基于激活值分布]
    C --> D[计算缩放因子<br/>保护重要权重]
    D --> E[执行量化<br/>缩放→量化→反缩放]
    E --> F[保存量化模型]

    style F fill:#c8e6c9
```

## 选择量化方法决策树

```mermaid
flowchart TD
    Start["需要量化大模型"] --> Q1{"推理设备?"}

    Q1 -->|"CPU/边缘设备"| GGUF["使用 GGUF<br/>llama.cpp"]
    Q1 -->|"GPU"| Q2{"优先级?"}

    Q2 -->|"速度优先"| AWQ["AWQ<br/>快速量化"]
    Q2 -->|"精度优先"| GPTQ["GPTQ<br/>高精度"]
    Q2 -->|"平衡"| Q3{"显存大小?"}

    Q3 -->|"< 24GB"| Q4["INT4 量化"]
    Q3 -->|">= 24GB"| Q5["INT8 或混合精度"]

    GGUF --> Done["开始量化"]
    AWQ --> Done
    GPTQ --> Done
    Q4 --> Done
    Q5 --> Done

    style Done fill:#c8e6c9
```

## 量化粒度对比

```mermaid
flowchart TB
    subgraph Per-Tensor["Per-Tensor (张量级)"]
        T1["整个矩阵一个 scale"]
        T2["参数少，速度快"]
        T3["精度损失较大"]
    end

    subgraph Per-Channel["Per-Channel (通道级)"]
        C1["每行/列一个 scale"]
        C2["参数适中"]
        C3["精度较好"]
    end

    subgraph Per-Group["Per-Group (分组级)"]
        G1["每组 128 个元素"]
        G2["参数较多"]
        G3["精度最好"]
    end

    style T3 fill:#ffcdd2
    style C3 fill:#fff9c4
    style G3 fill:#c8e6c9
```

## 图解说明

### 关键概念

| 概念 | 说明 | 推荐值 |
|------|------|--------|
| 位宽 (bits) | 每个参数的位数 | 4 或 8 |
| 分组大小 | Per-group 的组大小 | 128 |
| 校准数据 | 量化时的参考数据 | 512 样本 |

### 方法选择

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| GPTQ | 精度高 | 量化慢 | 追求精度 |
| AWQ | 速度快 | 略低于 GPTQ | 生产环境 |
| GGUF | CPU 友好 | GPU 不最优 | 边缘设备 |

### 显存节省

```
70B 模型:
FP16:  140 GB  ████████████████████████████████████
INT8:   70 GB  ████████████████████
INT4:   35 GB  ██████████

节省 75% 显存！
```
