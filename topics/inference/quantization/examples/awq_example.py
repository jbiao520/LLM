"""
AWQ 量化示例代码

演示如何使用 AWQ (Activation-aware Weight Quantization) 进行模型量化。
AWQ 是一种速度快、精度高的量化方法。
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM

# ============================================
# 1. 模型配置
# ============================================

model_name = "meta-llama/Llama-2-7b-hf"
output_dir = "./quantized-awq"

# ============================================
# 2. 加载模型
# ============================================

print("加载预训练模型...")

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ============================================
# 3. 配置量化参数
# ============================================

# AWQ 量化配置
quant_config = {
    "zero_point": True,      # 使用零点（非对称量化）
    "q_group_size": 128,     # 分组大小
    "w_bit": 4,              # 权重位宽
    "version": "GEMM"        # 使用 GEMM 内核（更快）
}

# ============================================
# 4. 准备校准数据
# ============================================

print("准备校准数据...")

# AWQ 需要校准数据来确定哪些权重更重要
# 通常使用 512 个样本
calibration_data = [
    "人工智能正在改变我们的生活方式。",
    "机器学习算法可以从数据中学习模式。",
    "深度学习使用多层神经网络处理复杂任务。",
    "自然语言处理让计算机理解人类语言。",
    "计算机视觉使机器能够'看见'世界。",
    # ... 添加更多样本 (建议 512+)
]

# ============================================
# 5. 执行量化
# ============================================

print("开始 AWQ 量化...")

# AWQ 量化（比 GPTQ 更快）
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_data,
    # 可选参数
    # max_calib_seq_len=512,  # 校准序列最大长度
    # max_calib_samples=512,  # 校准样本数量
)

# ============================================
# 6. 保存量化模型
# ============================================

print(f"保存量化模型到 {output_dir}...")

model.save_quantized(output_dir)
tokenizer.save_pretrained(output_dir)

# ============================================
# 7. 加载量化模型推理
# ============================================

print("\n加载量化模型进行推理...")

# 加载量化模型
quantized_model = AutoAWQForCausalLM.from_quantized(
    output_dir,
    device_map="auto",
    fuse_layers=True  # 融合层以提高推理速度
)

# 推理测试
prompt = "请解释什么是机器学习？"
inputs = tokenizer(prompt, return_tensors="pt").to(quantized_model.device)

with torch.no_grad():
    outputs = quantized_model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print("\n生成的回答:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ============================================
# 8. 与 vLLM 集成
# ============================================

"""
AWQ 量化的模型可以与 vLLM 配合使用以获得最佳推理性能:

from vllm import LLM, SamplingParams

llm = LLM(
    model="./quantized-awq",
    quantization="awq",
    tensor_parallel_size=1
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(["什么是AI?"], sampling_params)
"""

# ============================================
# 9. 性能测试
# ============================================

def benchmark_inference(model, tokenizer, num_iterations=10):
    """简单的推理性能测试"""
    import time

    prompt = "测试性能的提示词。"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 预热
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=50)

    # 计时
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=50)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations
    print(f"\n平均推理时间: {avg_time:.2f} 秒")
    return avg_time

# benchmark_inference(quantized_model, tokenizer)

# ============================================
# 10. AWQ vs GPTQ 对比
# ============================================

"""
AWQ vs GPTQ 对比:

| 特性          | AWQ              | GPTQ             |
|--------------|------------------|------------------|
| 量化速度      | 快 (~10分钟)     | 慢 (~1小时)      |
| 推理速度      | 快 (融合内核)    | 快 (ExLlama)     |
| 精度          | 略好             | 好               |
| 显存使用      | 相同             | 相同             |
| CPU支持       | 有限             | 有限             |

推荐选择:
- 追求速度: AWQ
- 追求精度: GPTQ
- 生产环境: AWQ (与 vLLM 配合)
"""

print("\n" + "="*50)
print("AWQ 量化完成!")
print(f"量化模型保存在: {output_dir}")
print("="*50)

# ============================================
# 附录: 从 Hugging Face 下载预量化模型
# ============================================

"""
很多模型已经提供了预量化的 AWQ 版本:

from awq import AutoAWQForCausalLM

# 下载预量化模型
model = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-AWQ",
    device_map="auto"
)

推荐的 AWQ 模型源:
- TheBloke: https://huggingface.co/TheBloke
- CasperhAI: https://huggingface.co/casperhansen
"""
