import torch
import json
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd
model_path = './models/supervised_finetuned'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print("="*60)
print("评估监督微调模型")
print("="*60)
def evaluate_on_dataset(test_file, dataset_name):
    """在单个测试集上评估"""
    print(f"\n--- 评估 {dataset_name} ---")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['input'] for item in data]
    true_labels = []
    for item in data:
        if 'has heart disease' in item['label'] or 'high' in item['label']:
            true_labels.append(1)
        else:
            true_labels.append(0)
    predictions = []
    probs = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            prob = torch.softmax(outputs.logits, dim=-1)
            pred = outputs.logits.argmax(-1).item()
            
        predictions.append(pred)
        probs.append(prob[0][pred].item())
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    print(f"样本数: {len(texts)}")
    print(f"准确率: {accuracy:.4f}")
    print(f"F1分数 (macro): {f1_macro:.4f}")
    print(f"F1分数 (weighted): {f1_weighted:.4f}")
    print(f"平均置信度: {np.mean(probs):.4f}")
    print("\n分类报告:")
    print(classification_report(true_labels, predictions, 
                               target_names=['无心脏病', '有心脏病']))
    return {
        'dataset': dataset_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'samples': len(texts)
    }
results = []
try:
    result = evaluate_on_dataset(r'E:\创新数据\instruction JSON文件\heart_instructions.json', 'heart.csv (诊断)')
    results.append(result)
except FileNotFoundError:
    print("heart_test.json 不存在")

# 2. Cleveland
try:
    result = evaluate_on_dataset(r'E:\创新数据\instruction JSON文件\cleveland_instructions.json', 'Cleveland (诊断+影像)')
    results.append(result)
except FileNotFoundError:
    print("cleveland_test.json 不存在")

# 3. Framingham
try:
    result = evaluate_on_dataset(r'E:\创新数据\instruction JSON文件\framingham_instructions.json', 'Framingham (风险预测)')
    results.append(result)
except FileNotFoundError:
    print("framingham_test.json 不存在")

print("\n" + "="*60)
print("评估结果汇总")
print("="*60)
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
# 保存结果
with open('./results/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n结果已保存到 ./results/evaluation_results.json")