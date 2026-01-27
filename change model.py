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
TARGET_TRAIN_NUM = 25000  

DATA_FILES = [
    r"E:\three step\cardio_train_organized.txt",
    r"E:\three step\mitbih_test_annotated.txt",
    r"E:\three step\mitbih_train_annotated.txt",
    r"E:\three step\ptbdb_abnormal_annotated.txt",
    r"E:\three step\ptbdb_normal_annotated.txt",
    r"E:\three step\标注好的完整数据文件.txt"
]

def load_cardio_medical_data():
    all_medical_corpus = []
    for file_path in DATA_FILES:
        if not os.path.exists(file_path):
            print(f"文件不存在，跳过：{file_path}")
            continue
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except:
            with open(file_path, "r", encoding="gbk") as f:
                content = f.read().strip()
        
        samples = [s.strip() for s in content.split("\n\n") if len(s.strip()) > 10]
        all_medical_corpus.extend(samples)
        print(f"加载完成：{os.path.basename(file_path)} → 有效样本数：{len(samples)}")
    all_medical_corpus = list(set(all_medical_corpus))
    print(f"\n 去重后原始数据量：{len(all_medical_corpus)} 条")
    return all_medical_corpus

medical_corpus = load_cardio_medical_data()
if not medical_corpus:
    print(" 未加载到任何有效数据！")
    exit()

random.seed(42)  
random.shuffle(medical_corpus)  
medical_corpus = medical_corpus[:TARGET_TRAIN_NUM]  
print(f" 强制限制后数据量：{len(medical_corpus)} 条（目标：{TARGET_TRAIN_NUM}条）")

dataset = Dataset.from_dict({"text": medical_corpus})
print(f" 最终用于训练的Dataset长度：{len(dataset)} 条")  

tokenizer = BertTokenizer.from_pretrained(
    LOCAL_BERT_PATH,
    local_files_only=True,
    do_lower_case=True
)

model = BertForMaskedLM.from_pretrained(
    LOCAL_BERT_PATH,
    local_files_only=True,
    torch_dtype=torch.float32
)
print(" 本地BERT模型加载完成")


def tokenize_medical_text(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
        return_attention_mask=True
    )

tokenized_dataset = dataset.map(
    tokenize_medical_text,
    batched=True,
    batch_size=16,
    remove_columns=["text"]
)
print(f" 编码后数据量：{len(tokenized_dataset)} 条")  

training_args = TrainingArguments(
    output_dir=r"E:\three step\cardio_bert_finetuned",
    per_device_train_batch_size=4,
    learning_rate=3e-6,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=50,
    fp16=False,
    report_to="none",
    gradient_accumulation_steps=1,
    disable_tqdm=False,
    seed=42,
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

print("\n 开始BERT模型心血管领域微调（CPU）...")
trainer.train()

# 保存模型
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
print(f"\n 微调完成！模型保存至：{training_args.output_dir}")