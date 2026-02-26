# 推理加速流程图解

> 通过可视化图表理解推理加速的完整工作流程

## 自回归生成过程

```mermaid
flowchart LR
    subgraph 生成过程
        A["输入: 你好"] --> B["生成: 你"]
        B --> C["生成: 好"]
        C --> D["生成: !"]
        D --> E["输出: 你好!"]
    end

    subgraph 每步计算
        F["完整前向传播<br/>计算所有层"]
    end

    B -.-> F
    C -.-> F
    D -.-> F

    style F fill:#ffcdd2
```

## KV Cache 原理

```mermaid
flowchart TB
    subgraph 无缓存
        A1["生成 token 1<br/>计算 K1,V1"]
        A2["生成 token 2<br/>计算 K1,V1,K2,V2"]
        A3["生成 token 3<br/>计算 K1,V1,K2,V2,K3,V3"]
        A1 --> A2 --> A3
    end

    subgraph 有缓存
        B1["生成 token 1<br/>计算 K1,V1<br/>缓存"]
        B2["生成 token 2<br/>复用 K1,V1<br/>只算 K2,V2"]
        B3["生成 token 3<br/>复用 K1-K2,V1-V2<br/>只算 K3,V3"]
        B1 --> B2 --> B3
    end

    style A3 fill:#ffcdd2
    style B3 fill:#c8e6c9
```

## PagedAttention 内存管理

```mermaid
flowchart TB
    subgraph 传统预分配
        T1["请求1: 预留 4096 slots"]
        T2["请求2: 预留 4096 slots"]
        T3["请求3: 预留 4096 slots"]
        T4["实际使用 ~50%<br/>大量浪费"]
    end

    subgraph PagedAttention
        P1["请求1: 页1, 页3, 页5"]
        P2["请求2: 页2, 页4"]
        P3["请求3: 页6, 页7, 页8"]
        P4["按需分配<br/>利用率 ~95%"]
    end

    style T4 fill:#ffcdd2
    style P4 fill:#c8e6c9
```

## 连续批处理流程

```mermaid
flowchart LR
    subgraph 静态批处理
        S1["Batch: [R1, R2, R3]"]
        S2["等待所有完成"]
        S3["R1✓ R2✓ R3✓<br/>同时结束"]
        S1 --> S2 --> S3
    end

    subgraph 连续批处理
        C1["T1: [R1, R2, R3]"]
        C2["T2: [R1, R2] R3✓"]
        C3["T3: [R1, R4] R2✓"]
        C1 --> C2 --> C3
    end

    style S3 fill:#fff9c4
    style C3 fill:#c8e6c9
```

## Flash Attention 优化

```mermaid
flowchart TB
    subgraph 标准注意力
        A1["计算 N×N 注意力矩阵"]
        A2["存储到 HBM"]
        A3["应用 Softmax"]
        A4["再次读写 HBM"]
        A1 --> A2 --> A3 --> A4
    end

    subgraph Flash Attention
        B1["分块读取到 SRAM"]
        B2["SRAM 中完成<br/>Softmax + 缩放"]
        B3["只写回最终结果"]
        B1 --> B2 --> B3
    end

    style A4 fill:#ffcdd2
    style B3 fill:#c8e6c9
```

## vLLM 推理流程

```mermaid
flowchart TB
    A[接收请求] --> B[加入调度队列]
    B --> C{调度决策}

    C -->|有GPU资源| D[分配 KV Cache 页]
    C -->|资源不足| E[等待或抢占]

    D --> F[执行前向传播]
    F --> G[生成 token]
    G --> H{生成完成?}

    H -->|否| F
    H -->|是| I[释放 KV Cache]
    I --> J[返回结果]

    style D fill:#c8e6c9
    style I fill:#c8e6c9
```

## 框架选择决策树

```mermaid
flowchart TD
    Start["需要部署 LLM 推理"] --> Q1{"主要目标?"}

    Q1 -->|"最高吞吐量"| Q2{"硬件?"}
    Q1 -->|"最低延迟"| Q3{"GPU 类型?"}
    Q1 -->|"易于部署"| vLLM["vLLM<br/>推荐选择"]

    Q2 -->|"NVIDIA GPU"| TRT["TensorRT-LLM"]
    Q2 -->|"混合/其他"| vLLM

    Q3 -->|"NVIDIA"| TRT
    Q3 -->|"其他"| vLLM

    Start -->|"CPU only"| llama["llama.cpp"]

    style vLLM fill:#c8e6c9
    style TRT fill:#bbdefb
    style llama fill:#fff9c4
```

## 性能优化层次

```mermaid
flowchart TB
    subgraph 应用层
        A1["批处理策略"]
        A2["请求调度"]
    end

    subgraph 算法层
        B1["KV Cache"]
        B2["投机解码"]
        B3["注意力优化"]
    end

    subgraph 系统层
        C1["并行策略"]
        C2["内存管理"]
    end

    subgraph 硬件层
        D1["CUDA 内核"]
        D2["Tensor Cores"]
    end

    A1 --> B1 --> C1 --> D1

    style A1 fill:#e1f5fe
    style B1 fill:#bbdefb
    style C1 fill:#c8e6c9
    style D1 fill:#fff9c4
```

## 图解说明

### 关键技术收益

| 技术 | 延迟 | 吞吐量 | 内存 |
|------|------|--------|------|
| KV Cache | ↓50% | ↑2x | ↑使用 |
| PagedAttention | - | ↑4x | ↓50% |
| Flash Attention | ↓30% | ↑2x | ↓90% |
| 连续批处理 | - | ↑4x | - |

### 框架推荐

| 场景 | 推荐框架 |
|------|----------|
| 生产服务 | vLLM |
| NVIDIA 极致性能 | TensorRT-LLM |
| CPU/边缘部署 | llama.cpp |
| 研究原型 | HuggingFace |

### 性能瓶颈分析

```
推理时间分解:
├── 计算时间: 30%
│   └── GPU 利用率影响
├── 内存传输: 50%  ← 主要瓶颈!
│   └── 模型参数 + KV Cache
└── 调度开销: 20%
    └── 批处理效率
```
