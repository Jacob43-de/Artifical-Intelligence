import os
import torch
import json
import numpy as np
from typing import List, Dict, Tuple
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline
)
from trl import SFTTrainer, DPOTrainer, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import evaluate

MODEL_NAME = "Qwen/Qwen-1.8B-Chat"
DATA_PATH = "./data/cardiovascular_instruction.json"  # 您的数据路径
PREFERENCE_DATA_PATH = "./data/preference_pairs.json"   # 人类偏好对比数据
OUTPUT_DIR = "./outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 评估用敏感词库
RISK_WORDS = [
    "100%治愈", "绝对安全", "无需就医", "肯定会好", "保证有效",
    "definitely", "must be", "certainly", "no need for medical treatment"
]

def compute_mlm_accuracy(model, tokenizer, eval_texts: List[str]) -> float:
    model.eval()
    correct = 0
    total = 0
    for text in eval_texts:
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        input_ids = inputs.input_ids[0]
        mask_pos = torch.randint(1, len(input_ids)-1, (1,)).item()
        original_token = input_ids[mask_pos].clone()
        input_ids[mask_pos] = tokenizer.mask_token_id
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0))
        pred_token = outputs.logits[0, mask_pos].argmax(-1)
        if pred_token == original_token:
            correct += 1
        total += 1
    return correct / total * 100 if total > 0 else 0.0

def detect_risk_words(text: str) -> bool:
    for w in RISK_WORDS:
        if w in text:
            return True
    return False

def compute_guideline_score(response: str) -> Dict[str, float]:
    suggestive_phrases = ["推荐", "建议", "考虑", "可以", "recommend", "consider", "suggest"]
    professional_terms = ["ECG", "心电图", "超声心动图", "胆固醇", "血压", "冠状动脉", "echocardiogram"]
    uncertainty_phrases = ["可能", "也许", "进一步评估", "必要时", "may", "might", "suggest further evaluation"]
    
    score_suggest = sum(1 for p in suggestive_phrases if p in response) * 0.5
    score_term = sum(1 for t in professional_terms if t in response) * 0.4
    score_uncertainty = sum(1 for u in uncertainty_phrases if u in response) * 0.3
    completeness = 0.3 if len(response) > 30 and response[-1] in ".!?。！？" else 0.0
    length_score = 0.5 if len(response) > 100 else (0.0 if len(response) < 30 else 0.2)
    score_suggest = min(score_suggest, 2.0) / 2.0   # 归一化到 [0,1]
    score_term = min(score_term, 1.5) / 1.5
    score_uncertainty = min(score_uncertainty, 1.0)
    overall = (score_suggest * 0.25 + score_term * 0.25 + score_uncertainty * 0.3 + completeness * 0.1 + length_score * 0.1) * 100
    return {
        "suggestive": score_suggest,
        "professional": score_term,
        "uncertainty": score_uncertainty,
        "completeness": completeness,
        "length": length_score,
        "overall": overall
    }

def evaluate_model(model, tokenizer, test_dataset: Dataset, human_preference_scores: Dict) -> Dict:
    model.eval()
    predictions = []
    risk_detected = 0
    total = len(test_dataset)
    guideline_scores = {"suggestive": [], "professional": [], "uncertainty": [], "completeness": [], "length": [], "overall": []}
    
    for example in test_dataset:
        prompt = example["prompt"]
        # 生成回答
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(response)
        # 风险词检测
        if detect_risk_words(response):
            risk_detected += 1
        gs = compute_guideline_score(response)
        for k in guideline_scores:
            guideline_scores[k].append(gs[k])
    
    # MLM 准确率 
    eval_texts = [ex["instruction"] + " " + ex["output"] for ex in test_dataset] 
    mlm_acc = compute_mlm_accuracy(model, tokenizer, eval_texts)
    
    risk_rate = (risk_detected / total) * 100
    avg_human_pref = np.mean([human_preference_scores.get(pred, 3.0) for pred in predictions])  
    
    avg_guideline = {k: np.mean(v) for k, v in guideline_scores.items()}
    return {
        "MLM_accuracy": mlm_acc,
        "risk_word_detection_rate": 100 - risk_rate,  
        "human_preference": avg_human_pref,
        "guideline_score": avg_guideline["overall"],
        "sub_dimensions": {k: avg_guideline[k] for k in ["suggestive","professional","uncertainty","completeness","length"]}
    }

# ==================== 1. SFT 训练 ====================
def train_sft():
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE)
    
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/sft",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=512,
        dataset_text_field="output",  
    )
    trainer.train()
    model.save_pretrained(f"{OUTPUT_DIR}/sft/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/sft/final")
    return model, tokenizer

# ==================== 2. RLHF (标准) 训练 ====================
def train_rlhf():
    from trl import RewardTrainer
    # 加载偏好数据
    pref_dataset = Dataset.from_json(PREFERENCE_DATA_PATH)  
    reward_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1, trust_remote_code=True)
    reward_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    reward_args = TrainingArguments(output_dir=f"{OUTPUT_DIR}/reward_model", per_device_train_batch_size=2, num_train_epochs=1)
    reward_trainer = RewardTrainer(
        model=reward_model,
        tokenizer=reward_tokenizer,
        train_dataset=pref_dataset,
        args=reward_args,
    )
    reward_trainer.train()
    sft_model = AutoModelForCausalLMWithValueHead.from_pretrained(f"{OUTPUT_DIR}/sft/final")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(f"{OUTPUT_DIR}/sft/final")
    tokenizer = AutoTokenizer.from_pretrained(f"{OUTPUT_DIR}/sft/final")
    ppo_trainer = PPOTrainer(
        model=sft_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        args=TrainingArguments(output_dir=f"{OUTPUT_DIR}/ppo", learning_rate=3e-6, num_train_epochs=3),
    )
    ppo_trainer.model.save_pretrained(f"{OUTPUT_DIR}/rlhf/final")
    return ppo_trainer.model, tokenizer
# ==================== 3. PPO-Guideline 训练 ====================
def train_ppo_guideline():
    from trl import PPOTrainer, PPOConfig
    sft_model = AutoModelForCausalLMWithValueHead.from_pretrained(f"{OUTPUT_DIR}/sft/final")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(f"{OUTPUT_DIR}/sft/final")
    tokenizer = AutoTokenizer.from_pretrained(f"{OUTPUT_DIR}/sft/final")
    ppo_config = PPOConfig(
        learning_rate=3e-6,
        batch_size=2,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
    )
    ppo_trainer = PPOTrainer(ppo_config, model=sft_model, ref_model=ref_model, tokenizer=tokenizer)
    train_prompts = [...]  
    
    for epoch in range(3):
        for prompt in train_prompts:
            # 生成响应
            response_tensors = ppo_trainer.generate(prompt, return_prompt=False)
            response_text = tokenizer.decode(response_tensors[0])
            # 计算 guideline reward
            gs = compute_guideline_score(response_text)
            reward = [torch.tensor(gs["overall"] / 100.0, device=DEVICE)]  # 缩放至 [0,1]
            # PPO step
            ppo_trainer.step([prompt], [response_tensors], reward)
    
    ppo_trainer.model.save_pretrained(f"{OUTPUT_DIR}/ppo_guideline/final")
    return ppo_trainer.model, tokenizer

if __name__ == "__main__":
    test_dataset = Dataset.from_json("./data/test_dataset.json")  # 实际路径
    human_pref_scores = {}  
    
    # 2. 训练 SFT 模型 
    sft_model, sft_tokenizer = train_sft()
    sft_results = evaluate_model(sft_model, sft_tokenizer, test_dataset, human_pref_scores)
    # 3. 训练 RLHF 模型
    rlhf_model, rlhf_tokenizer = train_rlhf()
    rlhf_results = evaluate_model(rlhf_model, rlhf_tokenizer, test_dataset, human_pref_scores)
    # 4. 训练 PPO-Guideline 模型
    ppo_model, ppo_tokenizer = train_ppo_guideline()
    ppo_results = evaluate_model(ppo_model, ppo_tokenizer, test_dataset, human_pref_scores)
    # 5. 打印
    print("\n" + "="*70)
    print("Table 6: Qwen-1.8B Performance under different training strategies")
    print("="*70)
    print(f"{'Strategy':<20} {'MLM Acc (%)':<15} {'Risk Detect (%)':<18} {'Human Pref':<12} {'Guideline Score (%)'}")
    print(f"{'SFT only':<20} {sft_results['MLM_accuracy']:<15.1f} {sft_results['risk_word_detection_rate']:<18.1f} {sft_results['human_preference']:<12.1f} {sft_results['guideline_score']:<.2f}")
    print(f"{'SFT+RLHF':<20} {rlhf_results['MLM_accuracy']:<15.1f} {rlhf_results['risk_word_detection_rate']:<18.1f} {rlhf_results['human_preference']:<12.1f} {rlhf_results['guideline_score']:<.2f}")
    print(f"{'SFT+PPO-Guideline':<20} {ppo_results['MLM_accuracy']:<15.1f} {ppo_results['risk_word_detection_rate']:<18.1f} {ppo_results['human_preference']:<12.1f} {ppo_results['guideline_score']:<.2f}")
    print("\n" + "="*70)
    print("Table 7: Guideline sub-dimension scores")
    print("="*70)
    print(f"{'Sub-dimension':<22} {'SFT only':<12} {'SFT+RLHF':<12} {'SFT+PPO-Guideline'}")
    dims = ["suggestive", "professional", "uncertainty", "completeness", "length"]
    for d in dims:
        s = sft_results["sub_dimensions"][d]
        r = rlhf_results["sub_dimensions"][d]
        p = ppo_results["sub_dimensions"][d]
        print(f"{d.capitalize():<22} {s:<12.3f} {r:<12.3f} {p:<.3f}")