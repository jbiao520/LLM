# 工具调用（Tool Use）深入版

> 面向有 ML 基础读者的工具调用深度指南

## 1. Function Calling 技术细节

### 1.1 工具描述注入

工具定义被注入到系统提示中：

```
System: You have access to the following tools:

get_weather:
  description: 获取城市天气
  parameters: {"city": {"type": "string"}}

search_web:
  description: 搜索互联网
  parameters: {"query": {"type": "string"}}

To use a tool, respond with:
<tool_call={"name": "...", "arguments": {...}}>
```

### 1.2 结构化输出

LLM 输出被约束为 JSON 格式：

```json
{
  "thought": "需要查询北京天气",
  "action": "get_weather",
  "arguments": {"city": "北京"}
}
```

**实现方式：**
- Constrained decoding（约束解码）
- Grammar-based sampling
- JSON mode（OpenAI）

### 1.3 多轮对话状态

```
┌─────────────────────────────────────────────────────────────────┐
│                     对话状态管理                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Messages:                                                     │
│   [                                                             │
│     {"role": "system", "content": "...工具定义..."},            │
│     {"role": "user", "content": "北京天气"},                    │
│     {"role": "assistant", "tool_calls": [...]},                 │
│     {"role": "tool", "content": "{\"temp\": 25}"},              │
│     {"role": "assistant", "content": "北京今天25°C"}            │
│   ]                                                             │
│                                                                 │
│   tool role: 记录工具返回结果                                    │
│   tool_calls: 记录 LLM 的工具调用请求                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2. ReAct 算法详解

### 2.1 Prompt 模板

```
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
```

### 2.2 解析与执行循环

```python
def react_loop(llm, tools, question):
    while True:
        # 1. LLM 生成
        response = llm.generate(context)

        # 2. 解析响应
        if "Final Answer:" in response:
            return parse_final_answer(response)

        action, action_input = parse_action(response)

        # 3. 执行工具
        tool = tools[action]
        observation = tool.execute(action_input)

        # 4. 更新上下文
        context += f"\nObservation: {observation}"
```

### 2.3 停止条件

- 到达 `Final Answer`
- 超过最大步数（防止无限循环）
- 工具执行失败超过阈值

## 3. 工具选择优化

### 3.1 工具检索

当工具数量很多时（如 100+），需要检索相关工具：

```python
def retrieve_tools(query, tool_embeddings, top_k=5):
    query_emb = embed(query)
    scores = cosine_similarity(query_emb, tool_embeddings)
    return top_k_tools(scores)
```

### 3.2 工具排序

基于历史使用频率和成功率：

$$\text{score}(t) = \alpha \cdot \text{relevance}(t) + \beta \cdot \text{success\_rate}(t)$$

## 4. 并行工具调用

### 4.1 独立调用检测

```python
def detect_parallel_calls(planned_actions):
    """检测可并行执行的调用"""
    independent_groups = []
    for action in planned_actions:
        # 检查是否有依赖关系
        deps = get_dependencies(action, planned_actions)
        if not deps:
            independent_groups.append(action)
    return independent_groups
```

### 4.2 示例

```
用户: "比较北京和上海的天气"

并行调用:
├── get_weather(city="北京")
└── get_weather(city="上海")

合并结果后回答
```

## 5. 错误处理

### 5.1 重试机制

```python
def execute_with_retry(tool, args, max_retries=3):
    for attempt in range(max_retries):
        try:
            return tool.execute(args)
        except ToolError as e:
            # 让 LLM 修正参数
            correction = llm.generate(
                f"Tool call failed: {e}. Suggest corrected arguments."
            )
            args = parse_correction(correction)
    raise ToolExecutionError("Max retries exceeded")
```

### 5.2 Fallback 策略

| 错误类型 | Fallback |
|----------|----------|
| 工具超时 | 使用缓存或默认值 |
| 参数无效 | 要求用户澄清 |
| 权限不足 | 跳过并说明限制 |
| 网络错误 | 重试或使用备用工具 |

## 6. 安全考虑

### 6.1 工具权限控制

```python
TOOL_PERMISSION = {
    "read_file": {"paths": ["/data/*"]},
    "write_file": {"paths": [], "deny": True},  # 禁止写入
    "execute_code": {"sandbox": True},
}
```

### 6.2 输入验证

```python
def validate_tool_input(tool_name, arguments):
    schema = TOOL_SCHEMAS[tool_name]
    validate(instance=arguments, schema=schema)

    # 额外检查
    if tool_name == "execute_code":
        check_dangerous_operations(arguments["code"])
```

### 6.3 沙箱执行

代码执行工具应在沙箱中运行：

- Docker 容器隔离
- 资源限制（CPU、内存、时间）
- 网络限制
- 文件系统隔离

## 7. 评估指标

| 指标 | 定义 |
|------|------|
| 工具选择准确率 | 正确选择工具的比例 |
| 参数填充准确率 | 参数正确的比例 |
| 任务完成率 | 成功完成任务的比例 |
| 平均步数 | 完成任务的平均工具调用次数 |

## 参考文献

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.02525)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
