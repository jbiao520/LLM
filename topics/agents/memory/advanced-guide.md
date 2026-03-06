# 记忆系统（Memory）深入版

> 面向有 ML 基础读者的记忆系统深度指南

## 1. 记忆架构

### 1.1 分层记忆系统

```
┌─────────────────────────────────────────────────────────────────┐
│                     分层记忆架构                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    L1: 工作记忆                          │   │
│   │  容量: 4K-128K tokens                                   │   │
│   │  延迟: 0ms (直接在 Context)                             │   │
│   │  内容: 当前对话 + 最近历史                               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    L2: 会话记忆                          │   │
│   │  容量: 1M+ tokens                                       │   │
│   │  延迟: 10-100ms (摘要检索)                              │   │
│   │  内容: 当前会话的摘要和关键信息                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    L3: 长期记忆                          │     │
│   │  容量: 无限                                              │   │
│   │  延迟: 100-500ms (向量检索)                             │   │
│   │  内容: 历史会话、用户画像、知识库                        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 向量检索详解

### 2.1 相似度度量

**余弦相似度：**
$$\text{sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$$

**欧氏距离：**
$$\text{dist}(a, b) = \|a - b\|_2$$

**点积：**
$$\text{score}(a, b) = a \cdot b$$

### 2.2 检索增强

**Hybrid Search（混合检索）：**
$$\text{score} = \alpha \cdot \text{dense\_score} + (1-\alpha) \cdot \text{sparse\_score}$$

结合向量检索（语义）和关键词检索（精确）。

**Re-ranking（重排序）：**
```python
def retrieve_with_rerank(query, k=100, top_k=5):
    # 1. 初步检索
    candidates = vector_db.search(query, k=k)

    # 2. 重排序（使用更强的模型）
    reranker = CrossEncoder("model-name")
    scores = reranker.predict([(query, c.text) for c in candidates])

    # 3. 返回 top-k
    return sorted(candidates, key=scores)[:top_k]
```

### 2.3 最大边际相关性 (MMR)

平衡相关性和多样性：

$$\text{MMR} = \arg\max_{d \in R \setminus S} [\lambda \cdot \text{Sim}(q, d) - (1-\lambda) \cdot \max_{s \in S} \text{Sim}(d, s)]$$

## 3. 记忆写入策略

### 3.1 信息提取

```python
def extract_memories(message):
    """从消息中提取结构化记忆"""
    prompt = f"""
    从以下消息中提取关键信息，返回 JSON 列表：
    消息: {message}

    提取: 用户偏好、事实信息、重要事件
    """

    memories = llm.generate(prompt)
    return parse_json(memories)
```

### 3.2 去重与合并

```python
def deduplicate_memories(new_memory, existing_memories, threshold=0.9):
    """避免存储重复记忆"""
    new_emb = embed(new_memory)

    for existing in existing_memories:
        sim = cosine_similarity(new_emb, existing.embedding)
        if sim > threshold:
            # 合并或更新
            return merge_memories(new_memory, existing)

    return new_memory  # 新记忆
```

### 3.3 记忆重要性评分

$$\text{importance} = w_1 \cdot \text{novelty} + w_2 \cdot \text{relevance} + w_3 \cdot \text{emotional\_impact}$$

用于决定记忆保留优先级。

## 4. 记忆压缩

### 4.1 滑动摘要

```python
def sliding_summary(messages, window_size=10):
    """滑动窗口摘要"""
    summaries = []

    for i in range(0, len(messages), window_size):
        window = messages[i:i+window_size]
        summary = llm.generate(f"总结以下对话：{window}")
        summaries.append(summary)

    return summaries
```

### 4.2 层次化摘要

```
原始对话 (1000条)
    │
    ├── 段落摘要 (每10条 → 1条摘要) = 100条
    │       │
    │       └── 章节摘要 (每10条段落 → 1条) = 10条
    │               │
    │               └── 全局摘要 = 1条
    │
    └── 检索时: 先查全局 → 章节详情 → 段落详情
```

### 4.3 记忆遗忘

$$\text{retention} = e^{-\lambda \cdot t}$$

其中 $t$ 是时间，$\lambda$ 是遗忘率。

定期清理低重要性 + 旧记忆。

## 5. 实现示例

### 5.1 记忆管理器

```python
class MemoryManager:
    def __init__(self, vector_db, llm, embedder):
        self.db = vector_db
        self.llm = llm
        self.embedder = embedder

    def add_memory(self, content, metadata=None):
        """添加记忆"""
        embedding = self.embedder.embed(content)
        self.db.upsert(
            id=hash(content),
            embedding=embedding,
            metadata={"content": content, **metadata}
        )

    def retrieve(self, query, top_k=5):
        """检索相关记忆"""
        query_emb = self.embedder.embed(query)
        results = self.db.query(query_emb, top_k=top_k)
        return [r.metadata["content"] for r in results]

    def get_context(self, query, max_tokens=4000):
        """构建上下文"""
        # 1. 检索相关记忆
        memories = self.retrieve(query)

        # 2. 构建提示
        context = "用户背景信息：\n"
        for m in memories:
            if len(context) + len(m) < max_tokens:
                context += f"- {m}\n"

        return context
```

### 5.2 与 Agent 集成

```python
class MemoryAgent:
    def __init__(self, llm, memory_manager):
        self.llm = llm
        self.memory = memory_manager

    def chat(self, user_input):
        # 1. 检索相关记忆
        context = self.memory.get_context(user_input)

        # 2. 构建完整提示
        prompt = f"""
        {context}

        用户: {user_input}
        助手:
        """

        # 3. 生成回复
        response = self.llm.generate(prompt)

        # 4. 存储新记忆
        self.memory.add_memory(f"用户说: {user_input}")
        self.memory.add_memory(f"助手回复: {response}")

        return response
```

## 6. 评估指标

| 指标 | 定义 |
|------|------|
| 检索准确率 | 返回的相关记忆比例 |
| 检索召回率 | 相关记忆被检索到的比例 |
| 上下文利用率 | 检索记忆被 LLM 引用的比例 |
| 记忆命中率 | 回答需要记忆时的成功比例 |

## 参考文献

- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
- [Mem0: Memory Layer for AI Applications](https://github.com/mem0ai/mem0)
