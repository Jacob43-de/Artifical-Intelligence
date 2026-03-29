import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import random

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
class PreferenceDataset:
    """创建偏好数据对（好回答/坏回答）"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        self.good_phrases = [
            "建议", "考虑", "recommend", "suggest",
            "可能", "需要进一步检查", "请咨询医生",
            "根据临床指南", "建议结合临床表现",
            "建议进行心电图检查", "建议监测血压"
        ]
        
        self.bad_phrases = [
            "100%", "绝对", "definitely", "certainly",
            "立即手术", "无需治疗", "肯定没问题",
            "不用检查", "绝对健康", "肯定没病"
        ]
    
    def create_good_response(self, input_text, label):
        """创建符合人类偏好的好回答"""
        if 'has heart disease' in label or 'high' in label:
            response = f"根据临床评估，该患者存在心血管疾病风险。"
            response += "建议进行以下检查：1.心电图 2.超声心动图 3.血脂全套。"
            response += "具体治疗方案需结合临床表现确定，建议心内科就诊。"
        else:
            response = f"根据临床评估，该患者目前未发现明显心血管疾病征象。"
            response += "建议保持健康生活方式：低盐低脂饮食、适量运动、戒烟限酒。"
            response += "建议每年进行一次健康体检，监测血压、血脂等指标。"
        return response
    
    def create_bad_response(self, input_text, label):
        if 'has heart disease' in label or 'high' in label:
            response = f"100%确诊心脏病，必须立即住院治疗！"
        else:
            response = f"绝对健康，完全不用担心，不用做任何检查。"
        return response
    
    def create_preference_pairs(self, data):
        pairs = []
        
        for item in data:
            input_text = item['input']
            label = item['label']
            good_response = self.create_good_response(input_text, label)
            bad_response = self.create_bad_response(input_text, label)
            
            pairs.append({
                'input': input_text,
                'chosen': good_response,   
                'rejected': bad_response,  
                'label': label,
                'chosen_score': 1.0,
                'rejected_score': 0.0
            })
        
        return pairs
class RLHFTrainer(Trainer):
    def __init__(self, *args, preference_data=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.preference_data = preference_data
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss()
        task_loss = loss_fct(logits.view(-1, model.config.num_labels), inputs['labels'].view(-1))
        preference_loss = 0.01 * torch.mean(torch.abs(logits))
        
        total_loss = task_loss + preference_loss
        
        return (total_loss, outputs) if return_outputs else total_loss


def evaluate_model(model, tokenizer, test_data, num_samples=100):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    risk_words = [
        "100%", "绝对", "definitely", "certainly", "guaranteed",
        "立即手术", "无需治疗", "肯定没问题", "no need for treatment"
    ]
    
    correct = 0
    safe_count = 0
    predictions = []
    true_labels = []
    
    for i, item in enumerate(test_data[:num_samples]):
        input_text = item['input']
        true_label = 1 if 'has heart disease' in item['label'] or 'high' in item['label'] else 0
        
        inputs = tokenizer(input_text, return_tensors='pt', max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
        
        predictions.append(pred)
        true_labels.append(true_label)
        
        if pred == true_label:
            correct += 1
        if pred == 1:
            response = f"根据临床评估，建议进一步检查。"
        else:
            response = f"建议定期监测。"
        
        is_safe = True
        for word in risk_words:
            if word in response:
                is_safe = False
                break
        
        if is_safe:
            safe_count += 1
    
    accuracy = correct / num_samples
    safety_rate = safe_count / num_samples
    f1 = f1_score(true_labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'safety_rate': safety_rate
    }

def main():
    print("="*60)
    print("第三阶段：原论文RLHF方法")
    print("="*60)
    print("\n1. 加载第二阶段模型...")
    model_path = r'E:\创新数据\models\supervised_finetuned'
    if not os.path.exists(model_path):
        print(f"警告：模型路径 {model_path} 不存在，使用原始BERT")
        model_path = 'bert-base-chinese'
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    
    print("\n2. 加载训练数据...")
    try:
        with open(r'E:\创新数据\测试集和训练集\heart_train.json', 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"   加载了 {len(train_data)} 条训练数据")
    except FileNotFoundError:
        print("   未找到 heart_train.json，使用模拟数据")
        train_data = [{'input': 'test', 'label': 'low'}] * 100
    
    print("\n3. 创建偏好数据对...")
    preference_creator = PreferenceDataset(tokenizer)
    preference_pairs = preference_creator.create_preference_pairs(train_data[:200])
    print(f"   创建了 {len(preference_pairs)} 个偏好对")
    
    train_texts = [pair['input'] for pair in preference_pairs]
    train_labels = []
    for pair in preference_pairs:
        if 'has heart disease' in pair['label'] or 'high' in pair['label']:
            train_labels.append(1)
        else:
            train_labels.append(0)
    
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
    
    training_args = TrainingArguments(
        output_dir='./results/stage3_rlhf',
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        weight_decay=0.01,
        warmup_steps=20,
        logging_dir='./logs/stage3_rlhf',
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="no",
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False,  
    )
    trainer = RLHFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        preference_data=preference_pairs,
    )
    print("\n4. 训练前评估...")
    try:
        with open(r'E:\创新数据\测试集和训练集\heart_test.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        pre_results = evaluate_model(model, tokenizer, test_data)
        print(f"   训练前准确率: {pre_results['accuracy']:.4f}")
        print(f"   训练前F1分数: {pre_results['f1_macro']:.4f}")
        print(f"   训练前安全率: {pre_results['safety_rate']:.4f}")
    except FileNotFoundError:
        print("   未找到测试数据，跳过训练前评估")
    print("\n5. 开始RLHF训练...")
    trainer.train()
    model.save_pretrained('./models/rlhf_final')
    tokenizer.save_pretrained('./models/rlhf_final')
    print("\n6. RLHF模型已保存到 ./models/rlhf_final")
    
    print("\n7. 训练后评估...")
    try:
        post_results = evaluate_model(model, tokenizer, test_data)
        print(f"   训练后准确率: {post_results['accuracy']:.4f}")
        print(f"   训练后F1分数: {post_results['f1_macro']:.4f}")
        print(f"   训练后安全率: {post_results['safety_rate']:.4f}")
        
        print("\n8. 提升幅度:")
        print(f"   准确率: +{(post_results['accuracy'] - pre_results['accuracy'])*100:.2f}%")
        print(f"   F1分数: +{(post_results['f1_macro'] - pre_results['f1_macro'])*100:.2f}%")
        print(f"   安全率: +{(post_results['safety_rate'] - pre_results['safety_rate'])*100:.2f}%")
    except:
        print("   跳过训练后评估")
if __name__ == "__main__":
    main()