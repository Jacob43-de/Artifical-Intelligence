import os
import pandas as pd
import torch
import random  
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

LOCAL_BERT_PATH = r"E:\three step\bert_local"  
DATA_FILES = [r"E:\three step\cardio_train_organized.txt"]
TARGET_TRAIN_NUM = 50000  

def load_local_annotated_data():
    """读取本地的标注TXT文件，转为BERT训练用的语料"""
    all_corpus = []
    for file_path in DATA_FILES:
        if not os.path.exists(file_path):
            print(f" 跳过不存在的文件：{file_path}")
            continue
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk") as f:
                content = f.read().strip()
        
        samples = [s.strip() for s in content.split("\n\n") if len(s.strip()) > 20]
        all_corpus.extend(samples)
    
    all_corpus = list(set(all_corpus))
    if len(all_corpus) == 0:
        print(" 未加载到任何有效数据，请检查文件内容！")
        exit()
    print(f" 原始有效医学语料条数：{len(all_corpus)}")
    return all_corpus

# 加载数据
corpus = load_local_annotated_data()

random.seed(42)  
if len(corpus) > TARGET_TRAIN_NUM:
    corpus = random.sample(corpus, TARGET_TRAIN_NUM)
    print(f" 已随机采样{TARGET_TRAIN_NUM}条数据用于训练")
else:
    print(f" 原始数据不足{TARGET_TRAIN_NUM}条，使用全部{len(corpus)}条数据")


dataset = Dataset.from_dict({"text": corpus})

tokenizer = BertTokenizer.from_pretrained(
    LOCAL_BERT_PATH,
    local_files_only=True
)
model = BertForMaskedLM.from_pretrained(
    LOCAL_BERT_PATH,
    local_files_only=True,
    torch_dtype=torch.float32
)
print(" 本地BERT模型加载完成！")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)
print(" 数据编码完成！")

training_args = TrainingArguments(
    output_dir=r"E:\three step\cardio_bert_trained",
    per_device_train_batch_size=4,
    learning_rate=3e-6,
    num_train_epochs=2,
    logging_steps=20,
    save_steps=100,
    fp16=False,
    report_to="none",
    gradient_accumulation_steps=1,
    disable_tqdm=False,
    dataloader_pin_memory=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("\n 开始BERT模型心血管领域预训练（纯CPU）...")
trainer.train()
print(f" 训练完成！模型已保存至：{training_args.output_dir}")