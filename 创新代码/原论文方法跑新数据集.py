import torch
from transformers import (
    BertTokenizer, 
    BertForMaskedLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import pandas as pd
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_name = r"E:\创新数据\bert-base-Chinese"  
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name).to(device)

def load_text_data(csv_files):
    all_texts = []
    
    for file in csv_files:
        df = pd.read_csv(file)
        if 'text' in df.columns:
            texts = df['text'].tolist()
            all_texts.extend(texts)
            print(f"Loaded {len(texts)} texts from {file}")
    
    return all_texts

text_files = [
    r'E:\创新数据\转换文件\heart_text.csv',           
    r'E:\创新数据\转换文件\cleveland_text.csv',        
    r'E:\创新数据\转换文件\framingham_text.csv'        
]
texts = load_text_data(text_files)
print(f"Total texts for MLM: {len(texts)}")
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128,
        return_special_tokens_mask=True
    )

dataset = Dataset.from_dict({'text': texts})
dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir='./results/stage1_mlm',
    overwrite_output_dir=True,
    num_train_epochs=3,                    
    per_device_train_batch_size=2,          
    gradient_accumulation_steps=2,           
    learning_rate=3e-6,                      
    weight_decay=0.01,                       
    warmup_steps=50,                         
    logging_dir='./logs/stage1',
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),          
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("Starting continued pre-training...")
trainer.train()

model.save_pretrained('./models/continued_pretrained')
tokenizer.save_pretrained('./models/continued_pretrained')
print("Stage 1 completed! Model saved to ./models/continued_pretrained")

def evaluate_mlm_accuracy(model, tokenizer, test_texts, num_samples=100):
    model.eval()
    correct = 0
    total = 0
    
    for text in test_texts[:num_samples]:
        inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        mask_position = torch.randint(1, input_ids.shape[1]-1, (1,)).item()
        masked_input_ids = input_ids.clone()
        original_token = input_ids[0, mask_position].item()
        masked_input_ids[0, mask_position] = tokenizer.mask_token_id
        with torch.no_grad():
            outputs = model(masked_input_ids)
            predictions = outputs.logits[0, mask_position]
            predicted_token = predictions.argmax().item()
        
        if predicted_token == original_token:
            correct += 1
        total += 1
    
    accuracy = correct / total * 100
    print(f"MLM Accuracy on {total} samples: {accuracy:.2f}%")
    return accuracy
evaluate_mlm_accuracy(model, tokenizer, texts)