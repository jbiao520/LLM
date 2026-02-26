"""
vLLM 推理加速示例代码

演示如何使用 vLLM 进行高性能 LLM 推理。
vLLM 是目前最流行的开源推理框架。
"""

import time
from vllm import LLM, SamplingParams

# ============================================
# 1. 基本使用
# ============================================

print("加载模型...")

# vLLM 会自动处理:
# - PagedAttention 内存管理
# - 连续批处理
# - CUDA 内核优化

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",

    # 并行配置
    tensor_parallel_size=1,  # 单 GPU
    # tensor_parallel_size=2,  # 双 GPU 并行

    # 内存配置
    gpu_memory_utilization=0.9,  # 使用 90% GPU 显存

    # 上下文长度
    max_model_len=4096,

    # 其他选项
    trust_remote_code=True,
    dtype="float16",
)

# ============================================
# 2. 配置采样参数
# ============================================

sampling_params = SamplingParams(
    # 生成长度
    max_tokens=100,

    # 温度 (控制随机性)
    temperature=0.7,

    # Top-p 采样
    top_p=0.9,

    # Top-k 采样
    top_k=50,

    # 重复惩罚
    repetition_penalty=1.1,

    # 停止词
    stop=["###", "\n\n"],
)

# ============================================
# 3. 单请求推理
# ============================================

print("\n单请求推理测试...")

prompt = "请解释什么是机器学习？"
start_time = time.time()

outputs = llm.generate([prompt], sampling_params)

elapsed = time.time() - start_time
print(f"延迟: {elapsed:.2f} 秒")
print(f"输出: {outputs[0].outputs[0].text[:200]}...")

# ============================================
# 4. 批量推理 (展示 vLLM 的优势)
# ============================================

print("\n批量推理测试...")

prompts = [
    "什么是人工智能？",
    "机器学习和深度学习有什么区别？",
    "解释一下神经网络的工作原理。",
    "什么是自然语言处理？",
    "深度学习有哪些应用场景？",
    *["请介绍一下深度学习。"] * 10,  # 15 个请求
]

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start_time

# 计算吞吐量
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
throughput = total_tokens / elapsed

print(f"批处理大小: {len(prompts)}")
print(f"总延迟: {elapsed:.2f} 秒")
print(f"吞吐量: {throughput:.1f} tokens/秒")

# ============================================
# 5. 流式输出
# ============================================

print("\n流式输出示例...")

from vllm import SamplingParams
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 注意: vLLM 的流式输出需要使用 AsyncLLMEngine
# 这里展示基本概念

"""
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-2-7b-hf",
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

async def generate_stream(prompt):
    results_generator = engine.generate(prompt, sampling_params, request_id="1")
    async for request_output in results_generator:
        yield request_output.outputs[0].text
"""

# ============================================
# 6. 与 HuggingFace 对比
# ============================================

def compare_with_huggingface():
    """与 HuggingFace 原生推理对比"""

    print("\n" + "="*50)
    print("性能对比 (估算)")
    print("="*50)

    print("""
    | 指标           | HuggingFace | vLLM      |
    |----------------|-------------|-----------|
    | 单请求延迟     | 2.0s        | 1.5s      |
    | 10并发吞吐量   | 50 tok/s    | 500 tok/s |
    | 显存利用率     | ~60%        | ~90%      |
    | KV Cache 管理  | 静态        | 动态分页  |
    """)

compare_with_huggingface()

# ============================================
# 7. 高级配置
# ============================================

"""
# 多 GPU 张量并行
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # 4 GPU 并行
    gpu_memory_utilization=0.9,
)

# 使用量化模型
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
)

# 自定义 CUDA 图
llm = LLM(
    model="...",
    enforce_eager=False,  # 启用 CUDA 图
)
"""

# ============================================
# 8. API 服务部署
# ============================================

"""
# 启动 OpenAI 兼容的 API 服务
# 命令行:
# python -m vllm.entrypoints.openai.api_server \\
#     --model meta-llama/Llama-2-7b-hf \\
#     --host 0.0.0.0 \\
#     --port 8000

# 客户端调用:
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "你好"}],
)
"""

print("\n" + "="*50)
print("vLLM 示例完成!")
print("="*50)
