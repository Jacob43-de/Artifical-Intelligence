import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 固定随机种子
torch.manual_seed(42)

# 模型加载
model_path = "./qwen_sft_final"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, trust_remote_code=True).cuda()

ppo_config = PPOConfig(
    batch_size=4,
    learning_rate=1e-5,
    steps=5000,
    gradient_accumulation_steps=4,
    seed=42,
)

ppo_trainer = PPOTrainer(ppo_config, model, None, tokenizer)

def compute_guideline_reward(output_text):
    reward = 0.0
    if "risk" in output_text: reward += 0.2
    if "ECG" in output_text: reward += 0.1
    if "heart disease" in output_text: reward += 0.2
    if "guideline" in output_text: reward += 0.2
    if "uncertain" in output_text or "possible" in output_text: reward += 0.2
    return min(reward, 1.0)

# 训练循环
for step in range(1000):
    query = "Please perform cardiovascular disease risk assessment:"
    inputs = tokenizer(query, return_tensors="pt").input_ids.cuda()

    # 生成
    response = model.generate(inputs, max_new_tokens=128)
    response_text = tokenizer.decode(response[0], skip_add_special_tokens=True)

    # 计算奖励
    reward_val = compute_guideline_reward(response_text)
    rewards = torch.tensor([reward_val]).cuda()

    # PPO 更新
    ppo_trainer.step([inputs[0]], [response[0]], rewards)

# 保存最终模型
model.save_pretrained("./qwen_ppo_guideline_final")
tokenizer.save_pretrained("./qwen_ppo_guideline_final")