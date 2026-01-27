import json
import torch
import os
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    TrainingArguments,
    Trainer,  
    DataCollatorForLanguageModeling
)
from datasets import Dataset

BASE_MODEL_PATH = r"E:\three step\cardio_bert_finetuned"
DPO_DATA_PATH = r"E:\three step\偏好数据对.json"
OUTPUT_PATH = r"E:\three step\cardio_bert_preference_finetuned"
def load_and_process_data():
    # 检查数据文件是否存在
    if not os.path.exists(DPO_DATA_PATH):
        raise FileNotFoundError(f"偏好数据文件不存在：{DPO_DATA_PATH}")
    # 读取JSON数据
    with open(DPO_DATA_PATH, "r", encoding="utf-8") as f:
        preference_data = json.load(f)

    train_texts = []
    for item in preference_data:
        if "prompt" in item and "chosen" in item:
            # 拼接：问题 + 优质回答（作为BERT的训练文本）
            train_text = f"{item['prompt']} {item['chosen']}"
            train_texts.append(train_text)
    # 转换为datasets格式
    dataset = Dataset.from_dict({"text": train_texts})
    print(f"加载偏好数据完成！有效训练样本数：{len(dataset)}")
    return dataset

def load_model_tokenizer():
    """加载BERT模型和分词器，处理pad token"""
    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"基础BERT模型不存在：{BASE_MODEL_PATH}")
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        do_lower_case=True,
        padding_side="right"
    )
    # 补全pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.cls_token  # 用cls_token替代pad_token
    # 加载BERT掩码语言模型
    model = BertForMaskedLM.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float32,  
        device_map="auto"  
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print(" 加载BERT模型和分词器完成！")
    return model, tokenizer
def tokenize_dataset(dataset, tokenizer):
    """对数据集进行编码，生成模型可识别的token格式"""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,  
            max_length=256,   
            padding="max_length",  
            return_attention_mask=True  
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,  
        remove_columns=["text"]  
    )
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
    print("数据编码完成！")
    return tokenized_dataset

def run_preference_training():
    try:
        # 加载数据
        raw_dataset = load_and_process_data()
        
        # 加载模型和分词器
        model, tokenizer = load_model_tokenizer()
        
        # 编码数据（生成tokenized_dataset）
        tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer)
        
        # 创建数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,  
            mlm_probability=0.15  
        )
        
        training_args = TrainingArguments(
            output_dir=OUTPUT_PATH,
            per_device_train_batch_size=2,  
            gradient_accumulation_steps=2,  
            learning_rate=3e-6,  
            num_train_epochs=3,  
            logging_steps=5,     
            save_steps=20,       
            fp16=False,          
            report_to="none",    
            seed=42,             
            disable_tqdm=False,  
            remove_unused_columns=False,
            # 防止过拟合
            weight_decay=0.01,
            warmup_steps=50,
        )
        
        #初始化Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        print("\n开始BERT模型偏好优化训练...")
        trainer.train()
        
        trainer.save_model(OUTPUT_PATH)
        tokenizer.save_pretrained(OUTPUT_PATH)
        print(f"\n训练完成！模型已保存至：{OUTPUT_PATH}")
        
    except Exception as e:
        print(f"\n训练过程出错：{str(e)}")
        raise

if __name__ == "__main__":
    run_preference_training()