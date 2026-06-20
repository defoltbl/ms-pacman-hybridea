import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ================================
# 1. Locate the file next to the script
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(BASE_DIR, "expert_dataset.pkl")

with open(FILE, "rb") as f:
    data = pickle.load(f)

print(f"Data loaded: {type(data)}, length: {len(data)}")

# ================================
# 2. Visualization Parameters
# ================================
num_elements = 6 

cols = 3 
rows = int(np.ceil(num_elements / cols))

plt.figure(figsize=(4*cols, 4*rows))

plt.suptitle(f"Visualization of {num_elements} states and actions (Action) from expert_dataset",
             fontsize=18, y=0.95) 

for idx, item in enumerate(data[:num_elements]):
    state = item['state'].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    action = item['action']
    mean_channels = state.mean(axis=(0, 1))

    ax = plt.subplot(rows, cols, idx + 1)
    ax.imshow(state)
    ax.axis('off')
    
    ax.set_title(
        f"Element: {idx}\n"
        f"Action: {action}\n"
        f"Channel average:\nR={mean_channels[0]:.1f}, G={mean_channels[1]:.1f}, B={mean_channels[2]:.1f}",
        fontsize=10
    )

plt.tight_layout(rect=[0, 0, 1, 0.93]) 
plt.show()