import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

torch.manual_seed(42)

# 模型
model_name = "Qwen-1_8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

dataset = load_dataset("text", data_files={
    "train": [
        "data/heart.csv",
        "data/ecg_classification.csv",
        "data/cardio_disease.csv",
        "data/heart_clinical_notes.txt"
    ]
})
# 分词
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# 训练参数
training_args = TrainingArguments(
    output_dir="./qwen_continued_pretrain",
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    seed=42,
    bf16=True,
    save_strategy="epoch",
    logging_steps=10,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
model.save_pretrained("./qwen_continued_pretrain_final")
tokenizer.save_pretrained("./qwen_continued_pretrain_final")