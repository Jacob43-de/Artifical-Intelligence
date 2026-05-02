import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset

# 固定随机种子
torch.manual_seed(42)

# 加载预训练后的Qwen
model = AutoModelForCausalLM.from_pretrained(
    "./qwen_continued_pretrain_final",
    trust_remote_code=True
).cuda()

tokenizer = AutoTokenizer.from_pretrained(
    "./qwen_continued_pretrain_final",
    trust_remote_code=True
)

df1 = pd.read_csv("data/heart.csv")
df2 = pd.read_csv("data/ecg_classification.csv")
df3 = pd.read_csv("data/cardio_disease.csv")
df = pd.concat([df1, df2, df3], axis=0)

def build_prompt(row):
    return f"Clinical instruction: Predict cardiovascular risk or diagnose heart disease.\nInput: {str(row.to_dict())}\nOutput:"

df["text"] = df.apply(build_prompt, axis=1)
dataset = Dataset.from_pandas(df[["text"]])

# 分词
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 训练参数
training_args = TrainingArguments(
    output_dir="./qwen_sft_final",
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
model.save_pretrained("./qwen_sft_final")
tokenizer.save_pretrained("./qwen_sft_final")