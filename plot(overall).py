import matplotlib.pyplot as plt
import numpy as np

dimensions = ['Suggestive\nlanguage', 'Professional\nterminology', 
              'Uncertainty\nexpression', 'Sentence\ncompleteness', 'Length score']
sft = [0.680, 0.730, 0.280, 0.240, 0.035]
rlhf = [0.785, 0.815, 0.380, 0.300, 0.045]
ppo = [0.801, 0.842, 0.415, 0.310, 0.060]

x = np.arange(len(dimensions))
width = 0.25
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x-width, sft, width, label='SFT only')
ax.bar(x, rlhf, width, label='SFT+RLHF')
ax.bar(x+width, ppo, width, label='SFT+PPO-Guideline')
ax.set_xticks(x)
ax.set_xticklabels(dimensions)
ax.set_ylabel('Score')
ax.legend()
plt.tight_layout()
plt.savefig('Figure4.png', dpi=300)