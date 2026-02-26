"""
GPTQ 量化示例代码

演示如何使用 GPTQ 对大语言模型进行 4-bit 量化。
GPTQ 是最流行的训练后量化方法之一。
"""

import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# ============================================
# 1. 模型配置
# ============================================

model_name = "meta-llama/Llama-2-7b-hf"
output_dir = "./quantized-gptq"

# ============================================
# 2. 配置量化参数
# ============================================

quantize_config = BaseQuantizeConfig(
    # 目标位宽 (4-bit 是最常用的选择)
    bits=4,

    # 分组大小 - 越小精度越高，但模型越大
    # 128 是常用的平衡点
    group_size=128,

    # 是否使用激活感知量化
    # True 可以提高精度，但量化时间更长
    desc_act=True,

    # 非对称量化 (通常更好)
    sym=False,

    # 动态量化 (某些层)
    dynamic=None,

    # 是否使用 ExLlama 内核 (更快推理)
    use_exllama=True,
)

# ============================================
# 3. 加载模型
# ============================================

print("加载预训练模型...")

# 加载未量化的模型用于量化过程
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config,
    max_memory={0: "24GB"}  # 限制 GPU 显存使用
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ============================================
# 4. 准备校准数据
# ============================================

print("准备校准数据...")

# GPTQ 需要校准数据来计算最优量化参数
# 通常使用 512-1024 个样本
calibration_data = [
    "人工智能是计算机科学的一个分支，它企图了解智能的实质。",
    "机器学习是人工智能的核心，是使计算机具有智能的根本途径。",
    "深度学习是机器学习的一个子集，它使用多层神经网络来学习数据表示。",
    # ... 更多样本
]

def tokenize_for_calibration(data):
    """将校准数据分词"""
    return tokenizer(
        data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

# 量化校准数据
calibration_examples = [tokenize_for_calibration(text) for text in calibration_data]

# ============================================
# 5. 执行量化
# ============================================

print("开始 GPTQ 量化（这可能需要几分钟）...")

# 量化模型
model.quantize(calibration_examples)

# ============================================
# 6. 保存量化模型
# ============================================

print(f"保存量化模型到 {output_dir}...")

model.save_quantized(output_dir)
tokenizer.save_pretrained(output_dir)

# ============================================
# 7. 加载量化模型进行推理
# ============================================

print("\n加载量化模型进行推理...")

# 加载量化后的模型
quantized_model = AutoGPTQForCausalLM.from_quantized(
    output_dir,
    device_map="auto",
    use_exllama=True  # 使用优化的 ExLlama 内核
)

# 推理测试
prompt = "什么是深度学习？"
inputs = tokenizer(prompt, return_tensors="pt").to(quantized_model.device)

with torch.no_grad():
    outputs = quantized_model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

print("\n生成的回答:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ============================================
# 8. 性能对比
# ============================================

def compare_model_sizes():
    """对比量化前后的模型大小"""
    import os

    # 原始模型大小估算
    original_size = 7e9 * 2  # 7B 参数 * 2 bytes (FP16)

    # 量化后模型大小估算
    quantized_size = 7e9 * 0.5  # 7B 参数 * 0.5 bytes (INT4)

    print(f"\n模型大小对比:")
    print(f"原始 (FP16): {original_size / 1e9:.1f} GB")
    print(f"量化 (INT4): {quantized_size / 1e9:.1f} GB")
    print(f"压缩比: {original_size / quantized_size:.1f}x")

compare_model_sizes()

# ============================================
# 9. 与 Transformers 集成
# ============================================

"""
也可以使用 transformers 的原生接口加载 GPTQ 模型:

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto",
    torch_dtype=torch.float16,
    revision="gptq-4bit-128g"
)
"""

print("\n" + "="*50)
print("GPTQ 量化完成!")
print(f"量化模型保存在: {output_dir}")
print("="*50)
