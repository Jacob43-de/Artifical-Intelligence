import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import json
import numpy as np
from collections import deque
import os
import random
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
class CardiovascularGuidelines:
    def __init__(self):
        self.high_risk_phrases = {
            "100%": -1.0,
            "绝对": -1.0,
            "definitely": -0.8,
            "certainly": -0.8,
            "guaranteed": -1.0,
            "立即手术": -1.5,
            "无需治疗": -1.5,
            "肯定没问题": -1.0,
            "no need for treatment": -1.0,
            "百分百": -1.0,
            "肯定": -0.8,
            "一定": -0.8,
            "毫无疑问": -1.0,
            "must": -0.5,
            "always": -0.5,
            "never": -0.5
        }
        self.suggestive_phrases = {
            "建议": 0.5,
            "考虑": 0.4,
            "recommend": 0.5,
            "suggest": 0.4,
            "consider": 0.4,
            "可能": 0.3,
            "建议进一步检查": 0.8,
            "请咨询医生": 0.6,
            "根据指南": 0.7,
            "考虑进一步评估": 0.6,
            "建议进行": 0.5,
            "可考虑": 0.4,
            "推荐使用": 0.5,
            "需要更多检查": 0.6,
            "建议咨询专科医生": 0.7
        }
        self.cardiology_terms = {
            "心电图": 0.4,
            "超声": 0.4,
            "超声心动图": 0.5,
            "血脂": 0.3,
            "血压": 0.2,
            "ECG": 0.4,
            "echocardiogram": 0.5,
            "冠脉造影": 0.6,
            "负荷试验": 0.5,
            "心肌酶": 0.4,
            "肌钙蛋白": 0.5,
            "冠状动脉": 0.4,
            "心功能": 0.3,
            "心脏": 0.2,
            "心血管": 0.3
        }
        self.uncertainty_phrases = {
            "可能": 0.3,
            "建议进一步": 0.5,
            "需要更多": 0.4,
            "需结合": 0.5,
            "考虑": 0.3,
            "评估": 0.3,
            "排除": 0.3,
            "鉴别": 0.4,
            "建议复查": 0.5,
            "定期监测": 0.4
        }
        self.good_examples = [
            "根据临床评估，该患者存在心血管疾病风险。建议进行心电图、超声心动图等检查。具体治疗方案需结合临床表现，建议心内科就诊。",
            "患者目前未发现明显心血管疾病征象。建议保持健康生活方式：低盐低脂饮食、适量运动。建议每年进行健康体检，监测血压血脂。",
            "根据ACC/AHA指南，该患者需要进行进一步评估。建议完善血脂全套、动态心电图等检查，必要时考虑冠脉CTA。",
            "患者血压偏高，建议监测血压变化，完善24小时动态血压监测。同时建议进行心血管风险评估，包括血脂、血糖等指标。"
        ]
    def compute_reward(self, text):
        reward = 0.0
        details = {}
        risk_score = 0
        for phrase, score in self.high_risk_phrases.items():
            if phrase in text:
                risk_score += score
        details['risk'] = risk_score
        suggest_score = 0
        for phrase, score in self.suggestive_phrases.items():
            if phrase in text:
                suggest_score += score
        details['suggest'] = min(suggest_score, 2.0) 
        term_score = 0
        for term, score in self.cardiology_terms.items():
            if term in text:
                term_score += score
        details['term'] = min(term_score, 1.5) 
        uncertainty_score = 0
        for phrase, score in self.uncertainty_phrases.items():
            if phrase in text:
                uncertainty_score += score
        details['uncertainty'] = min(uncertainty_score, 1.0)  
        length = len(text)
        if length < 30:
            length_score = -0.5
        elif length < 50:
            length_score = 0
        elif length < 100:
            length_score = 0.3
        else:
            length_score = 0.5
        details['length'] = length_score
        if text.count('。') >= 2:
            details['sentences'] = 0.3
        else:
            details['sentences'] = -0.2
        total_score = (
            risk_score + 
            suggest_score + 
            term_score + 
            uncertainty_score + 
            length_score + 
            details['sentences']
        )
        normalized = max(0, min(1, (total_score + 2) / 6))
        
        return normalized, details

class ResponseGenerator:
    def __init__(self):
        self.templates = {
            'high_risk': [
                "根据临床评估，该患者存在心血管疾病风险。建议进行心电图、超声心动图等检查。具体治疗方案需结合临床表现。",
                "患者心血管风险较高，建议完善血脂、血糖、心电图检查。考虑进行冠脉CTA评估。建议心内科就诊。",
                "根据ACC/AHA指南，该患者需要进行进一步心血管评估。建议监测血压，完善动态心电图。",
                "患者存在多个心血管危险因素，建议进行系统评估。包括：血脂全套、超声心动图、负荷试验。"
            ],
            'low_risk': [
                "根据临床评估，该患者目前未发现明显心血管疾病征象。建议保持健康生活方式，定期监测血压血脂。",
                "患者心血管风险较低，建议每年进行健康体检。保持低盐低脂饮食、适量运动。",
                "目前未见明显异常，建议定期随访。如有胸痛、胸闷等症状，及时就医。",
                "根据指南，该患者无需特殊处理。建议保持健康生活方式，控制体重，戒烟限酒。"
            ]
        }
        self.extra_suggestions = [
            "建议心内科随访。",
            "建议监测血压变化。",
            "建议完善血脂全套检查。",
            "建议进行运动负荷试验。",
            "建议控制饮食，加强运动。",
            "建议戒烟限酒。",
            "建议减轻体重。",
            "建议定期复查心电图。"
        ]
    def generate(self, input_text, pred_class, diversity=0.3):
        if pred_class == 1:
            base = random.choice(self.templates['high_risk'])
        else:
            base = random.choice(self.templates['low_risk'])
        if random.random() < diversity:
            extra = random.choice(self.extra_suggestions)
            response = base + " " + extra
        else:
            response = base
        
        return response

class PPOGuidelineTrainer:
    def __init__(self, model, tokenizer, guidelines, response_generator, 
                 lr=1e-6, gamma=0.99, clip_epsilon=0.2):
        self.model = model
        self.tokenizer = tokenizer
        self.guidelines = guidelines
        self.response_generator = response_generator
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.value_net = nn.Sequential(
            nn.Linear(model.config.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        self.value_optimizer = torch.optim.AdamW(self.value_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=200)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.value_net.to(self.device)
        self.stats = {
            'rewards': [],
            'guideline_scores': [],
            'losses': []
        }
    def get_hidden_state(self, input_ids, attention_mask):
        outputs = self.model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state[:, 0, :]
    
    def collect_experience(self, item, epoch):
        input_text = item['input']
        true_label = 1 if 'has heart disease' in item['label'] or 'high' in item['label'] else 0
        inputs = self.tokenizer(
            input_text, 
            return_tensors='pt', 
            max_length=128, 
            truncation=True,
            return_token_type_ids=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred = logits.argmax(-1).item()
            probs = F.softmax(logits, dim=-1)
            hidden = self.get_hidden_state(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        diversity = min(0.5, 0.1 + epoch * 0.1) 
        response = self.response_generator.generate(input_text, pred, diversity)
        task_reward = 1.0 if pred == true_label else 0.0
        guideline_reward, details = self.guidelines.compute_reward(response)
        w_task = max(0.3, 0.5 - epoch * 0.1)
        w_guideline = 1 - w_task
        total_reward = w_task * task_reward + w_guideline * guideline_reward
        value = self.value_net(hidden).item()
        self.buffer.append({
            'hidden': hidden.cpu(),
            'action': torch.tensor([pred]),
            'log_prob': torch.log(probs[0, pred] + 1e-10).item(),
            'reward': total_reward,
            'value': value,
            'guideline_reward': guideline_reward,
            'task_reward': task_reward,
            'response': response,
            'details': details
        })
        return total_reward, guideline_reward, details
    def compute_gae(self, rewards, values):
        advantages = []
        gae = 0
        next_value = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * 0.95 * gae
            advantages.insert(0, gae)
            next_value = values[t]
        return torch.tensor(advantages)
    def update(self):
        if len(self.buffer) < 16:
            return None
        batch_size = min(16, len(self.buffer))
        batch = random.sample(list(self.buffer), batch_size)
        hidden_states = torch.cat([exp['hidden'] for exp in batch])
        actions = torch.cat([exp['action'] for exp in batch])
        old_log_probs = torch.tensor([exp['log_prob'] for exp in batch])
        rewards = [exp['reward'] for exp in batch]
        values = [exp['value'] for exp in batch]
        advantages = self.compute_gae(rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        hidden_states = hidden_states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        total_loss = 0
        for _ in range(3):  
            current_logits = self.model.classifier(hidden_states)
            current_probs = F.log_softmax(current_logits, dim=-1)
            current_log_probs = current_probs.gather(1, actions.unsqueeze(1)).squeeze()
            ratio = torch.exp(current_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            current_values = self.value_net(hidden_states).squeeze()
            value_loss = F.mse_loss(current_values, advantages)
            entropy = -(current_probs.exp() * current_probs).sum(-1).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.value_optimizer.step()
            total_loss += loss.item()
        return total_loss / 3
    def train(self, train_data, epochs=5):
        print("\n" + "="*60)
        print("开始PPO-Guideline训练（改进版）")
        print("="*60)
        best_guideline_score = 0
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            epoch_rewards = []
            epoch_guideline = []
            for i, item in enumerate(train_data[:150]):
                total_reward, guideline_reward, details = self.collect_experience(item, epoch)
                epoch_rewards.append(total_reward)
                epoch_guideline.append(guideline_reward)
                if (i + 1) % 16 == 0:
                    loss = self.update()
                    if loss:
                        print(f"  步 {i+1}: loss={loss:.4f}")
            avg_reward = np.mean(epoch_rewards)
            avg_guideline = np.mean(epoch_guideline)
            self.stats['rewards'].append(avg_reward)
            self.stats['guideline_scores'].append(avg_guideline)
            print(f"\nEpoch {epoch+1} 结果:")
            print(f"  平均奖励: {avg_reward:.4f}")
            print(f"  指南得分: {avg_guideline:.4f}")
            if len(self.buffer) > 0:
                example = random.choice(list(self.buffer))
                print(f"\n  示例响应:")
                print(f"  {example['response'][:100]}...")
                print(f"  指南详情: {example['details']}")
            if avg_guideline > best_guideline_score:
                best_guideline_score = avg_guideline
                print(f"  → 新的最佳指南得分: {best_guideline_score:.4f}")
        print("\n" + "="*60)
        print("训练完成!")
        print(f"最终平均奖励: {np.mean(self.stats['rewards']):.4f}")
        print(f"最终指南得分: {np.mean(self.stats['guideline_scores']):.4f}")
        print("="*60)
        return self.model
def evaluate_model(model, tokenizer, guidelines, response_generator, test_num=100):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    try:
        with open(r'E:\创新数据\测试集和训练集\heart_test.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print("测试数据未找到")
        return 0, 0, 0
    correct = 0
    guideline_scores = []
    all_details = []
    for i, item in enumerate(test_data[:test_num]):
        input_text = item['input']
        true_label = 1 if 'has heart disease' in item['label'] or 'high' in item['label'] else 0
        inputs = tokenizer(
            input_text, 
            return_tensors='pt', 
            max_length=128, 
            truncation=True,
            return_token_type_ids=False
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
        if pred == true_label:
            correct += 1
        response = response_generator.generate(input_text, pred, diversity=0)
        score, details = guidelines.compute_reward(response)
        guideline_scores.append(score)
        all_details.append(details)
    accuracy = correct / test_num
    avg_guideline = np.mean(guideline_scores)
    print(f"\n评估结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  平均指南得分: {avg_guideline:.4f}")
    print(f"  最高指南得分: {np.max(guideline_scores):.4f}")
    print(f"  最低指南得分: {np.min(guideline_scores):.4f}")
    avg_details = {}
    for key in all_details[0].keys():
        avg_details[key] = np.mean([d[key] for d in all_details])
    print(f"\n  评分详情: {avg_details}")
    return accuracy, avg_guideline, avg_details

def main():
    print("="*60)
    print("PPO-Guideline 改进版")
    print("="*60)
    print("\n1. 加载RLHF模型...")
    model_path = r'E:\创新数据\models\rlhf_final'
    if not os.path.exists(model_path):
        print(f"警告：{model_path} 不存在，使用原始BERT")
        model_path = r'E:\创新数据\bert-base-Chinese'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    guidelines = CardiovascularGuidelines()
    response_generator = ResponseGenerator()
    trainer = PPOGuidelineTrainer(
        model=model,
        tokenizer=tokenizer,
        guidelines=guidelines,
        response_generator=response_generator,
        lr=3e-6,
        gamma=0.99,
        clip_epsilon=0.2
    )
    try:
        with open(r'E:\创新数据\测试集和训练集\heart_train.json', 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"   加载了 {len(train_data)} 条训练数据")
    except FileNotFoundError:
        print("   训练数据未找到")
        return
    pre_acc, pre_guideline, _ = evaluate_model(
        model, tokenizer, guidelines, response_generator, test_num=100
    )
    print("\n4. 开始PPO-Guideline训练...")
    trained_model = trainer.train(train_data, epochs=5)
    os.makedirs('./models', exist_ok=True)
    trained_model.save_pretrained('./models/ppo_guideline_improved')
    tokenizer.save_pretrained('./models/ppo_guideline_improved')
    print("\n5. 改进版模型已保存到 ./models/ppo_guideline_improved")
    post_acc, post_guideline, _ = evaluate_model(
        trained_model, tokenizer, guidelines, response_generator, test_num=100
    )
    print("\n" + "="*60)
    print("提升幅度总结")
    print("="*60)
    print(f"准确率: {pre_acc:.2%} → {post_acc:.2%} (+{(post_acc-pre_acc)*100:.2f}%)")
    print(f"指南得分: {pre_guideline:.2%} → {post_guideline:.2%} (+{(post_guideline-pre_guideline)*100:.2f}%)")
if __name__ == "__main__":
    main()