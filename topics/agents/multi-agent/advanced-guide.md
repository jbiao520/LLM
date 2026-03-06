# 多智能体协作（Multi-Agent）深入版

> 面向有 ML 基础读者的多智能体深度指南

## 1. 多智能体系统架构

### 1.1 系统组件

```
┌─────────────────────────────────────────────────────────────────┐
│                     多智能体系统架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   Orchestrator                          │   │
│   │  任务分解、Agent 选择、结果整合                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   Communication Layer                   │   │
│   │  消息路由、状态同步、事件分发                            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│            ┌─────────────────┼─────────────────┐                │
│            ▼                 ▼                 ▼                │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│   │   Agent 1   │   │   Agent 2   │   │   Agent N   │          │
│   │ ┌─────────┐ │   │ ┌─────────┐ │   │ ┌─────────┐ │          │
│   │ │  LLM    │ │   │ │  LLM    │ │   │ │  LLM    │ │          │
│   │ ├─────────┤ │   │ ├─────────┤ │   │ ├─────────┤ │          │
│   │ │ Memory  │ │   │ │ Memory  │ │   │ │ Memory  │ │          │
│   │ ├─────────┤ │   │ ├─────────┤ │   │ ├─────────┤ │          │
│   │ │ Tools   │ │   │ │ Tools   │ │   │ │ Tools   │ │          │
│   │ └─────────┘ │   │ └─────────┘ │   │ └─────────┘ │          │
│   └─────────────┘   └─────────────┘   └─────────────┘          │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   Shared State                          │   │
│   │  全局状态、任务队列、结果缓存                            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 通信协议

### 2.1 消息格式

```python
@dataclass
class AgentMessage:
    sender: str           # 发送者 ID
    receiver: str         # 接收者 ID (或 "broadcast")
    content: str          # 消息内容
    message_type: str     # "task" | "result" | "query" | "control"
    metadata: dict        # 额外元数据
    timestamp: float      # 时间戳
    reply_to: Optional[str]  # 回复的消息 ID
```

### 2.2 通信模式

**同步通信：**
```python
def sync_communication(sender, receiver, message):
    """发送并等待响应"""
    response = receiver.receive(message)
    return response
```

**异步通信：**
```python
async def async_communication(sender, receiver, message):
    """发送后继续，不阻塞"""
    await receiver.send(message)
    # 继续其他工作
```

**发布-订阅：**
```python
class EventBus:
    def __init__(self):
        self.subscribers = defaultdict(list)

    def subscribe(self, event_type, agent):
        self.subscribers[event_type].append(agent)

    def publish(self, event_type, message):
        for agent in self.subscribers[event_type]:
            agent.receive(message)
```

## 3. 协调算法

### 3.1 任务分配

**能力匹配：**
$$\text{score}(a, t) = \text{capability}(a) \cdot \text{requirement}(t)$$

选择得分最高的 Agent。

**负载均衡：**
$$\text{load}(a) = \frac{\text{active\_tasks}(a)}{\text{capacity}(a)}$$

优先分配给负载最低的 Agent。

### 3.2 共识机制

**投票：**
```python
def vote(agents, question):
    responses = [agent.vote(question) for agent in agents]
    return majority_vote(responses)
```

**加权投票：**
$$\text{result} = \arg\max_r \sum_{i=1}^{n} w_i \cdot \mathbb{1}[v_i = r]$$

权重 $w_i$ 基于 Agent 的历史准确率。

**辩论收敛：**
```python
def debate(agents, question, max_rounds=3):
    opinions = [agent.opinion(question) for agent in agents]

    for round in range(max_rounds):
        # 每个 Agent 看到其他人的观点后更新
        new_opinions = []
        for agent in agents:
            context = f"其他人观点: {opinions}\n你的观点:"
            new_opinions.append(agent.update(context))

        # 检查是否收敛
        if unanimous(new_opinions):
            return new_opinions[0]

        opinions = new_opinions

    return majority_vote(opinions)
```

## 4. 状态管理

### 4.1 共享状态

```python
class SharedState:
    def __init__(self):
        self.state = {}
        self.lock = asyncio.Lock()

    async def update(self, key, value):
        async with self.lock:
            self.state[key] = value

    async def get(self, key):
        async with self.lock:
            return self.state.get(key)
```

### 4.2 状态同步策略

| 策略 | 特点 | 适用场景 |
|------|------|----------|
| 即时同步 | 每次更新广播 | 小规模、强一致性 |
| 定期同步 | 定时批量同步 | 大规模、弱一致性 |
| 按需同步 | 需要时才同步 | 稀疏交互 |

## 5. 高级协作模式

### 5.1 层级化协作 (Society of Mind)

```
┌─────────────────────────────────────────────────────────────────┐
│                     层级化协作                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                     ┌───────────┐                               │
│                     │   CEO     │                               │
│                     │  Agent    │                               │
│                     └─────┬─────┘                               │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                  │
│         ▼                 ▼                 ▼                  │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐             │
│   │   CTO     │    │   CFO     │    │   COO     │             │
│   │  Agent    │    │  Agent    │    │  Agent    │             │
│   └─────┬─────┘    └─────┬─────┘    └─────┬─────┘             │
│         │                │                │                     │
│    ┌────┴────┐          │           ┌────┴────┐               │
│    ▼         ▼          │           ▼         ▼               │
│ ┌─────┐  ┌─────┐       │        ┌─────┐  ┌─────┐              │
│ │ Dev │  │ QA  │       │        │ Ops │  │ DBA │              │
│ │Team │  │Team │       │        │Team │  │Team │              │
│ └─────┘  └─────┘       │        └─────┘  └─────┘              │
│                                                                 │
│   每个"Team"本身也是一个多智能体系统                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 动态角色分配

```python
class DynamicAgent:
    def __init__(self, base_llm):
        self.llm = base_llm
        self.current_role = None

    def assign_role(self, task_requirements):
        """根据任务需求动态分配角色"""
        self.current_role = self._select_role(task_requirements)
        self.system_prompt = self._build_prompt(self.current_role)

    def _select_role(self, requirements):
        # 分析任务需求，选择最适合的角色
        role_scores = {}
        for role in self.available_roles:
            score = self._calculate_fit(role, requirements)
            role_scores[role] = score

        return max(role_scores, key=role_scores.get)
```

### 5.3 自我修复

```python
class SelfHealingTeam:
    def __init__(self, agents):
        self.agents = agents
        self.failure_count = defaultdict(int)

    def execute(self, task):
        for attempt in range(MAX_RETRIES):
            try:
                agent = self.select_agent(task)
                result = agent.execute(task)
                return result
            except AgentFailure as e:
                self.failure_count[agent.id] += 1

                # 自我修复
                if self.failure_count[agent.id] > THRESHOLD:
                    self.replace_agent(agent)

                # 重试
                continue
```

## 6. 评估指标

| 指标 | 定义 |
|------|------|
| 任务成功率 | 成功完成的任务比例 |
| 平均完成时间 | 完成任务的平均耗时 |
| 通信效率 | 有用消息 / 总消息数 |
| 协作质量 | Agent 间协作的顺畅程度 |
| 可扩展性 | Agent 数量增加时的性能变化 |

## 7. 框架对比

| 特性 | AutoGen | CrewAI | LangGraph |
|------|---------|--------|-----------|
| 通信模式 | 对话 | 任务流 | 图结构 |
| 状态管理 | 上下文 | 共享 | 状态机 |
| 并行执行 | ✓ | ✓ | ✓ |
| 人机协作 | ✓ | 部分 | ✓ |
| 学习曲线 | 中 | 低 | 高 |

## 参考文献

- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155)
- [CrewAI Documentation](https://docs.crewai.com/)
- [LangGraph: Multi-Agent Workflows](https://langchain-ai.github.io/langgraph/)
- [Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924)
