"""
全参数微调示例代码

演示如何对大语言模型进行全参数微调。
适用于需要最大程度适应新任务的场景。
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# ============================================
# 1. 加载预训练模型
# ============================================

model_name = "meta-llama/Llama-2-7b-hf"

print("加载预训练模型...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 全参数微调需要加载完整模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 使用 BF16（Ampere+ GPU）
    device_map="auto"
)

# 打印参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total_params / 1e9:.2f}B")
print(f"可训练参数: {trainable_params / 1e9:.2f}B (100%)")

# ============================================
# 2. 准备训练数据
# ============================================

# 示例：指令微调数据
train_data = [
    {
        "instruction": "解释什么是神经网络",
        "input": "",
        "output": "神经网络是一种模仿生物神经系统的机器学习模型..."
    },
    {
        "instruction": "用简单的语言解释量子计算",
        "input": "",
        "output": "量子计算使用量子力学的原理来处理信息..."
    },
    # ... 更多数据
]

def format_prompt(sample):
    """格式化为训练文本"""
    return f"""### 指令:
{sample['instruction']}

### 输入:
{sample['input']}

### 回答:
{sample['output']}"""

def tokenize_function(examples):
    """分词函数"""
    prompts = [format_prompt(ex) for ex in examples]
    return tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

# 创建数据集
dataset = Dataset.from_list(train_data)
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=dataset.column_names
)

# ============================================
# 3. 训练配置（全参数微调关键参数）
# ============================================

training_args = TrainingArguments(
    output_dir="./full-finetune-output",

    # ===== 批次设置 =====
    # 全参数微调显存需求高，通常用小批次 + 梯度累积
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # 有效批次 = 16

    # ===== 学习率设置（关键！）=====
    # 全参数微调使用较低的学习率，防止灾难性遗忘
    learning_rate=2e-5,  # 比预训练低 10-20 倍

    # 学习率调度
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,  # 预热 3% 的步数

    # 权重衰减（正则化）
    weight_decay=0.01,

    # ===== 精度设置 =====
    bf16=True,          # 使用 BF16（推荐 Ampere+ GPU）
    tf32=True,          # 启用 TF32 加速

    # ===== 显存优化 =====
    gradient_checkpointing=True,  # 梯度检查点，节省显存
    optim="adamw_torch",          # 优化器

    # ===== 训练设置 =====
    num_train_epochs=3,
    max_grad_norm=1.0,  # 梯度裁剪

    # ===== 保存设置 =====
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,  # 只保留最近 2 个 checkpoint

    # ===== 日志设置 =====
    logging_steps=10,
    logging_dir="./logs",

    # ===== 其他 =====
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

# ============================================
# 4. 数据整理器
# ============================================

# 用于因果语言模型的数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 不使用掩码语言模型
)

# ============================================
# 5. 开始训练
# ============================================

print("\n开始全参数微调...")
print(f"预计显存使用: ~60GB (7B 模型)")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

# ============================================
# 6. 保存微调后的模型
# ============================================

print("\n保存微调后的模型...")
model.save_pretrained("./full-finetune-output")
tokenizer.save_pretrained("./full-finetune-output")

# 注意：全参数微调保存的是完整模型（~14GB for 7B）
# 而 LoRA 只保存适配器（~200MB）

# ============================================
# 7. 推理测试
# ============================================

print("\n推理测试...")
model.eval()

prompt = "### 指令:\n解释什么是深度学习\n\n### 输入:\n\n### 回答:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ============================================
# 8. 评估对比（可选）
# ============================================

def evaluate_on_original_task(model, tokenizer, test_data):
    """评估微调后模型在原始任务上的表现（检测灾难性遗忘）"""
    # 实现原始任务评估逻辑
    pass

# ============================================
# 附录: 显存优化技巧
# ============================================

"""
全参数微调显存优化清单:

1. 梯度检查点 (gradient_checkpointing=True)
   - 节省: 30-50% 激活显存
   - 代价: 训练速度下降 ~20%

2. 混合精度 (bf16=True / fp16=True)
   - 节省: 50% 模型显存
   - 加速: 2-3x

3. 梯度累积 (gradient_accumulation_steps)
   - 不节省显存，但允许用小批次模拟大批次

4. DeepSpeed / FSDP (多卡)
   - 将模型分片到多张 GPU
   - 适合 >13B 的模型

显存需求估算 (7B 模型):
- 模型权重 (FP16):     14 GB
- 梯度:                14 GB
- 优化器状态 (AdamW):  28 GB
- 激活值:              ~4 GB
----------------------------------
总计:                  ~60 GB
"""

print("\n" + "="*50)
print("全参数微调完成!")
print("模型保存在: ./full-finetune-output")
print("="*50)
