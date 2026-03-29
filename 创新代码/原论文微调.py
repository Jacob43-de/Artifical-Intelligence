import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model_path = r'E:\创新数据\models\continued_pretrained'
print(f"模型路径: {os.path.abspath(model_path)}")
if not os.path.exists(model_path):
    print(f"警告：模型路径不存在，使用原始BERT")
    model_path = "bert-base-chinese"

print("\n加载tokenizer...")
tokenizer = BertTokenizer.from_pretrained(model_path)
print("加载模型...")
model = BertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2
).to(device)

print("模型加载成功！")
def load_instruction_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['input'] for item in data]
    labels = []
    for item in data:
        if 'has heart disease' in item['label'] or 'high' in item['label']:
            labels.append(1)
        else:
            labels.append(0)
    return texts, labels
train_texts = []
train_labels = []
datasets = [
    r'E:\创新数据\instruction JSON文件\heart_instructions.json',
    r'E:\创新数据\instruction JSON文件\cleveland_instructions.json', 
    r'E:\创新数据\instruction JSON文件\framingham_instructions.json'
]

for json_file in datasets:
    try:
        texts, labels = load_instruction_data(json_file)
        train_texts.extend(texts)
        train_labels.extend(labels)
        print(f"Loaded {len(texts)} samples from {json_file}")
    except FileNotFoundError as e:
        print(f"Warning: {json_file} not found: {e}")

print(f"\nTotal training samples: {len(train_texts)}")
print(f"Label distribution: {np.bincount(train_labels)}")
def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128,
    )
train_dataset = Dataset.from_dict({
    'text': train_texts,
    'label': train_labels
})
train_dataset = train_dataset.map(preprocess_function, batched=True)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1
    }
training_args = TrainingArguments(
    output_dir='./results/stage2_supervised',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=3e-6,
    weight_decay=0.01,
    warmup_steps=50,
    logging_dir='./logs/stage2',
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    fp16=False,  
    dataloader_pin_memory=False,  
    remove_unused_columns=False,
    report_to="none",  
)

print("\n训练参数设置成功！")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
)
print("\n开始训练...")
print(f"训练样本数: {len(train_dataset)}")
print(f"批次大小: {training_args.per_device_train_batch_size}")
print(f"梯度累积: {training_args.gradient_accumulation_steps}")
print(f"有效批次大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

try:
    trainer.train()
    print("\n✓ 训练完成！")
except Exception as e:
    print(f"\n训练出错: {e}")
    print("尝试使用更简单的配置...")
    simple_args = TrainingArguments(
        output_dir='./results/stage2_supervised_simple',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=3e-6,
        logging_steps=10,
        save_steps=500,
    )
    
    simple_trainer = Trainer(
        model=model,
        args=simple_args,
        train_dataset=train_dataset,
    )
    
    simple_trainer.train()
model.save_pretrained('./models/supervised_finetuned')
tokenizer.save_pretrained('./models/supervised_finetuned')
print("\n✓ 模型保存到 ./models/supervised_finetuned")

def quick_test():
    """快速测试模型"""
    test_text = "A 54-year-old male patient presents with atypical angina. Resting blood pressure is 140 mmHg, cholesterol is 289 mg/dL."
    
    inputs = tokenizer(test_text, return_tensors='pt', max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prob = torch.softmax(outputs.logits, dim=-1)
        pred = outputs.logits.argmax(-1).item()
    
    print("\n" + "="*50)
    print("快速测试：")
    print(f"输入: {test_text}")
    print(f"预测: {'有心脏病' if pred == 1 else '无心脏病'}")
    print(f"置信度: {prob[0][pred].item():.4f}")
    print("="*50) 
quick_test()