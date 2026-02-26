"""
LoRA 微调示例代码

演示如何使用 LoRA 对大语言模型进行参数高效微调。
使用 Hugging Face PEFT 库实现。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ============================================
# 1. 加载预训练模型和分词器
# ============================================

model_name = "meta-llama/Llama-2-7b-hf"  # 可替换为其他模型

print("加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 以 FP16 加载以节省显存
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ============================================
# 2. 配置 LoRA
# ============================================

lora_config = LoraConfig(
    # 低秩维度 - 决定 LoRA 的表达能力
    r=16,

    # 缩放因子 - 控制 LoRA 更新的强度
    lora_alpha=32,

    # 应用 LoRA 的目标模块
    # q_proj, v_proj 是最小配置
    # 可以扩展到 ["q_proj", "k_proj", "v_proj", "o_proj"]
    target_modules=["q_proj", "v_proj"],

    # LoRA 层的 dropout
    lora_dropout=0.05,

    # 不训练 bias
    bias="none",

    # 任务类型
    task_type=TaskType.CAUSAL_LM
)

# ============================================
# 3. 应用 LoRA 到模型
# ============================================

print("\n应用 LoRA 配置...")
model = get_peft_model(model, lora_config)

# 打印可训练参数信息
model.print_trainable_parameters()
# 输出示例:
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622

# ============================================
# 4. 准备训练数据
# ============================================

# 示例数据（实际使用时替换为你的数据集）
train_data = [
    {"text": "用户: 什么是机器学习?\n助手: 机器学习是人工智能的一个分支..."},
    {"text": "用户: 如何学习编程?\n助手: 学习编程可以从以下步骤开始..."},
    # ... 更多数据
]

def tokenize_function(examples):
    """对文本进行分词"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

# 创建 Dataset 并分词
dataset = Dataset.from_list(train_data)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ============================================
# 5. 训练配置
# ============================================

training_args = TrainingArguments(
    output_dir="./lora-output",

    # 批次大小
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 有效批次 = 4 * 4 = 16

    # 学习率 - LoRA 通常可以用较高的学习率
    learning_rate=1e-4,

    # 训练轮数
    num_train_epochs=3,

    # 混合精度训练
    fp16=True,

    # 日志
    logging_steps=10,
    save_steps=100,

    # 其他
    warmup_steps=100,
    save_total_limit=2,
)

# ============================================
# 6. 开始训练
# ============================================

from transformers import Trainer

print("\n开始 LoRA 微调训练...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# ============================================
# 7. 保存 LoRA 权重
# ============================================

print("\n保存 LoRA 权重...")
model.save_pretrained("./lora-output")

# 只保存 LoRA 适配器权重（非常小，几 MB）
# 可以随时加载到原始模型上

# ============================================
# 8. 推理示例
# ============================================

print("\n推理测试...")
prompt = "用户: 什么是深度学习?\n助手:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ============================================
# 9. 合并 LoRA 到基础模型（可选）
# ============================================

# 如果想要独立的模型（不需要 PEFT 库）:
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained("./merged-model")

print("\n" + "="*50)
print("LoRA 微调完成!")
print("LoRA 权重保存在: ./lora-output")
print("="*50)
