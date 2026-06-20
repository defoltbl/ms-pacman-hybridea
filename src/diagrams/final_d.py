import matplotlib.pyplot as plt
import numpy as np

# Data from your Table 4.1
labels = ['HYBRIDEA', 'DQN', 'Monte Carlo', 'A2C', 'PPO', 'BC']
mean_rewards = [1658.84, 1535.98, 1468.26, 1261.22, 1253.50, 1238.00]
max_rewards = [3890, 3720, 3140, 2190, 3700, 2690]

x = np.arange(len(labels))  # Label locations
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the bars
rects1 = ax.bar(x - width/2, mean_rewards, width, label='Mean Reward', color='#3498db', edgecolor='black', alpha=0.8)
rects2 = ax.bar(x + width/2, max_rewards, width, label='Max Reward', color='#e74c3c', edgecolor='black', alpha=0.8)

# Adding text, labels, and formatting
ax.set_ylabel('Score')
ax.set_title('Comparative Performance of Algorithms in Ms. Pac-Man')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Adding values on top of the bars (automatic labeling)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

# Saving the plot for the thesis/diploma
plt.savefig('performance_comparison.png', dpi=300)
plt.show()