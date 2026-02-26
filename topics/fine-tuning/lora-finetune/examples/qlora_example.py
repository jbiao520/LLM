"""
QLoRA 微调示例代码

演示如何使用 QLoRA (Quantized LoRA) 在有限显存下微调大模型。
QLoRA 结合了 4-bit 量化和 LoRA，大幅降低显存需求。
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# ============================================
# 1. 配置 4-bit 量化 (QLoRA 的核心)
# ============================================

# QLoRA 的量化配置
bnb_config = BitsAndBytesConfig(
    # 使用 4-bit 量化
    load_in_4bit=True,

    # 使用 NF4 (NormalFloat4) 数据类型
    # 这是专门为正态分布权重设计的量化格式
    bnb_4bit_quant_type="nf4",

    # 使用双重量化 (进一步压缩)
    bnb_4bit_use_double_quant=True,

    # 计算时使用 16-bit 精度
    bnb_4bit_compute_dtype=torch.float16
)

# ============================================
# 2. 加载量化模型
# ============================================

model_name = "meta-llama/Llama-2-7b-hf"

print("加载 4-bit 量化模型...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 以 4-bit 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# ============================================
# 3. 准备模型进行 k-bit 训练
# ============================================

print("准备模型进行 QLoRA 训练...")

# 这一步很重要：
# - 启用梯度检查点以节省显存
# - 设置必要的 dtype 转换
model = prepare_model_for_kbit_training(model)

# ============================================
# 4. 配置 LoRA (与标准 LoRA 类似)
# ============================================

lora_config = LoraConfig(
    r=16,                    # 低秩维度
    lora_alpha=32,           # 缩放因子
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
        "gate_proj", "up_proj", "down_proj"       # FFN 层 (可选)
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================
# 5. 显存对比
# ============================================

def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU 显存: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")

print("\n当前显存使用:")
print_memory_usage()

# 显存对比 (7B 模型):
# - 全参数微调 (FP16): ~28 GB
# - LoRA (FP16):       ~16 GB
# - QLoRA (4-bit):     ~6 GB

# ============================================
# 6. 训练配置
# ============================================

training_args = TrainingArguments(
    output_dir="./qlora-output",

    # QLoRA 可以使用更大的批次（显存占用低）
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,

    # 学习率
    learning_rate=2e-4,

    # 训练轮数
    num_train_epochs=3,

    # 使用 16-bit 混合精度
    fp16=True,

    # 梯度检查点（进一步节省显存）
    gradient_checkpointing=True,

    # 优化器设置
    optim="paged_adamw_8bit",  # 使用分页优化器，避免 OOM

    # 日志
    logging_steps=10,
    save_steps=100,
    warmup_steps=100,
)

# ============================================
# 7. 准备数据
# ============================================

# 示例：指令微调数据格式
train_data = [
    {
        "instruction": "解释什么是过拟合",
        "input": "",
        "output": "过拟合是指模型在训练数据上表现很好，但在测试数据上表现较差的现象..."
    },
    {
        "instruction": "写一首关于春天的诗",
        "input": "",
        "output": "春风拂面柳丝长，桃花盛开满园香..."
    },
    # ... 更多数据
]

def format_instruction(sample):
    """格式化为指令微调格式"""
    return f"""### 指令:
{sample['instruction']}

### 输入:
{sample['input']}

### 回答:
{sample['output']}"""

def tokenize(sample):
    text = format_instruction(sample)
    result = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    result["labels"] = result["input_ids"].copy()
    return result

dataset = Dataset.from_list(train_data)
tokenized_dataset = dataset.map(tokenize)

# ============================================
# 8. 训练
# ============================================

from transformers import Trainer

print("\n开始 QLoRA 微调...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# ============================================
# 9. 保存和加载
# ============================================

print("\n保存 QLoRA 适配器...")
model.save_pretrained("./qlora-output")

# ============================================
# 10. 加载已训练的 QLoRA 模型进行推理
# ============================================

def load_qlora_for_inference(model_name, adapter_path):
    """加载 QLoRA 模型进行推理"""
    from peft import PeftModel

    # 加载基础模型（4-bit）
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, adapter_path)

    return model

# 使用示例
# inference_model = load_qlora_for_inference(model_name, "./qlora-output")

print("\n" + "="*50)
print("QLoRA 微调完成!")
print("适配器权重保存在: ./qlora-output")
print("="*50)

# ============================================
# 附录: QLoRA vs LoRA vs 全参数微调
# ============================================

"""
显存需求对比 (7B 模型):

| 方法           | 基础显存 | 训练显存 | 模型精度 |
|---------------|---------|---------|---------|
| 全参数微调      | 14 GB   | ~28 GB  | FP16    |
| LoRA          | 14 GB   | ~16 GB  | FP16    |
| QLoRA         | 3.5 GB  | ~6 GB   | 4-bit   |

适用场景:
- QLoRA: 消费级显卡 (如 RTX 3060 12GB)
- LoRA: 中端显卡 (如 RTX 3090 24GB)
- 全参数: 专业显卡 (如 A100 40GB+)
"""
