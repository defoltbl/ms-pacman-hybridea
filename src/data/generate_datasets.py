import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

np.random.seed(42) 

# Algorithm parameters with pros and cons
algorithms = {
    'DQN': {
        'max_reward': 8000, 'k': 0.005, 'midpoint': 500, 'noise_std': 500,
        'base_time': 12.0, 'min_time': 8.0,
        'pros': 'Sample-efficient, stable long-term',
        'cons': 'Slow convergence, sensitive to hyperparameters'
    },
    'MC': {
        'max_reward': 3000, 'k': 0.003, 'midpoint': 600, 'noise_std': 300,
        'base_time': 15.0, 'min_time': 10.0,
        'pros': 'Simple, unbiased estimates',
        'cons': 'High variance, slow learning'
    },
    'PPO': {
        'max_reward': 12000, 'k': 0.006, 'midpoint': 400, 'noise_std': 600,
        'base_time': 10.0, 'min_time': 5.0,
        'pros': 'Stable, good for Atari games',
        'cons': 'Computationally intensive, cold start'
    },
    'A2C': {
        'max_reward': 10000, 'k': 0.005, 'midpoint': 450, 'noise_std': 500,
        'base_time': 11.0, 'min_time': 6.0,
        'pros': 'Parallelizable, decent performance',
        'cons': 'Less stable, sensitive to learning rate'
    },
    'BC': {
        'max_reward': 7000, 'k': 0.001, 'midpoint': 800, 'noise_std': 200,
        'base_time': 10.0, 'min_time': 6.0, 'initial_factor': 0.8,
        'pros': 'Fast initial performance, mimics expert',
        'cons': 'No exploration, plateaus quickly'
    },
    'HYBRID_BC_PPO': {
        'max_reward': 15000, 'k': 0.007, 'midpoint': 350, 'noise_std': 700,
        'base_time': 8.0, 'min_time': 3.0, 'initial_factor': 0.7,
        'pros': 'Best performance, adaptive',
        'cons': 'Complex implementation'
    }
}

# Action distributions (from previous request, % of Stop actions for efficiency)
action_distributions = {
    'DQN': {'Stop': 5},
    'MC': {'Stop': 12},
    'PPO': {'Stop': 3},
    'A2C': {'Stop': 5},
    'BC': {'Stop': 5},
    'HYBRID_BC_PPO': {'Stop': 3}
}

# Generate synthetic data
episodes = np.arange(1, 1001)
datasets = {}

for algo, params in algorithms.items():
    # Rewards: Logistic growth
    max_reward = params['max_reward']
    k = params['k']
    midpoint = params['midpoint']
    noise_std = params['noise_std']
    
    if 'initial_factor' in params:  # BC and HYBRID_BC_PPO
        rewards = max_reward * params['initial_factor'] + (max_reward * (1 - params['initial_factor'])) / (1 + np.exp(-k * (episodes - midpoint)))
    else:
        rewards = max_reward / (1 + np.exp(-k * (episodes - midpoint)))
    
    rewards += np.random.normal(0, noise_std, size=episodes.shape)
    rewards = np.clip(rewards, 100, max_reward)
    
    # Pellets: Proportional to rewards (~10 points/pellet)
    pellets = np.clip(rewards / 10, 10, 480)
    pellets += np.random.normal(0, 20, size=episodes.shape)
    pellets = np.clip(pellets, 10, 480)
    
    # Time: Inversely proportional to rewards
    base_time = params['base_time']
    min_time = params['min_time']
    times = base_time / (1 + rewards / max_reward) + np.random.normal(0, 0.5, size=episodes.shape)
    times = np.clip(times, min_time, base_time)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Episode': episodes,
        'Reward': rewards,
        'Pellets': pellets,
        'Time': times
    })
    
    # Save to CSV
    os.makedirs('datasets', exist_ok=True)
    csv_path = f'datasets/{algo}_ms_pacman_1000_episodes.csv'
    df.to_csv(csv_path, index=False)
    datasets[algo] = df

# Compute strengths with baseline minimums
strengths = {'Reward': {}, 'Time Efficiency': {}, 'Action Efficiency': {}}
for algo in algorithms.keys():
    # Reward strength (baseline min: 0)
    strengths['Reward'][algo] = algorithms[algo]['max_reward']
    
    # Time efficiency (baseline min: 1/20 for 20s, a very high time)
    avg_time = datasets[algo]['Time'].mean()
    strengths['Time Efficiency'][algo] = 1 / avg_time
    
    # Action efficiency (baseline min: 1/100 for 100% Stop actions)
    stop_percentage = action_distributions[algo]['Stop']
    strengths['Action Efficiency'][algo] = 1 / (stop_percentage + 1)  # Avoid division by zero

# Normalize strengths to percentages with baseline minimums
baseline_mins = {
    'Reward': 0,               # Theoretical minimum: 0 rewards
    'Time Efficiency': 1/20,   # Theoretical minimum: 20s (inverse: 0.05)
    'Action Efficiency': 1/100 # Theoretical minimum: 100% Stop actions (inverse: 0.01)
}
for metric in strengths.keys():
    values = np.array(list(strengths[metric].values()))
    max_val = values.max()
    min_val = baseline_mins[metric]  # Use baseline minimum instead of actual minimum
    for algo in strengths[metric]:
        normalized = (strengths[metric][algo] - min_val) / (max_val - min_val) * 100
        strengths[metric][algo] = round(normalized, 1)

# Plotting: Individual plots for each algorithm
os.makedirs('plots/individual', exist_ok=True)
for algo, df in datasets.items():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    rewards_ma = df['Reward'].rolling(window=50).mean()
    times_ma = df['Time'].rolling(window=50).mean()
    
    ax1.plot(df['Episode'], times_ma, label=algo, color='blue')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Time per Episode (seconds)')
    ax1.set_title(f'{algo}: Time per Episode\nPros: {algorithms[algo]["pros"]}')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(df['Episode'], rewards_ma, label=algo, color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    strengths_text = (f"Strengths:\n- Reward: {strengths['Reward'][algo]}%\n"
                     f"- Time Efficiency: {strengths['Time Efficiency'][algo]}%\n"
                     f"- Action Efficiency: {strengths['Action Efficiency'][algo]}%")
    ax2.set_title(f'{algo}: Reward per Episode\nCons: {algorithms[algo]["cons"]}\n{strengths_text}')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/individual/{algo}_ms_pacman.png', bbox_inches='tight')
    plt.close()

# Plotting: Combined comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
for algo, df in datasets.items():
    rewards_ma = df['Reward'].rolling(window=50).mean()
    times_ma = df['Time'].rolling(window=50).mean()
    
    ax1.plot(df['Episode'], times_ma, label=algo)
    ax2.plot(df['Episode'], rewards_ma, label=algo)

ax1.set_xlabel('Episode')
ax1.set_ylabel('Time per Episode (seconds)')
ax1.set_title('Time per Episode Comparison Across Algorithms')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True)

ax2.set_xlabel('Episode')
ax2.set_ylabel('Average Reward')
ax2.set_title('Reward Comparison Across Algorithms')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True)

plt.tight_layout()
plt.savefig('plots/algorithm_comparison.png', bbox_inches='tight')
plt.close()

# Plotting: Histogram of Results with Strengths
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Data for histogram
algos = list(algorithms.keys())
final_rewards = [datasets[algo]['Reward'].iloc[-1] for algo in algos]  # Final episode reward
avg_times = [datasets[algo]['Time'].mean() for algo in algos]  # Average time

# Histogram for Rewards
x = np.arange(len(algos))
width = 0.35

ax1.bar(x, final_rewards, width, label='Final Reward', color='green')
ax1.set_xlabel('Algorithm')
ax1.set_ylabel('Final Reward')
ax1.set_title('Final Reward Comparison Across Algorithms')
ax1.set_xticks(x)
ax1.set_xticklabels(algos, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Histogram for Average Time
ax2.bar(x, avg_times, width, label='Average Time', color='blue')
ax2.set_xlabel('Algorithm')
ax2.set_ylabel('Average Time (seconds)')
ax2.set_title('Average Time Comparison Across Algorithms')
ax2.set_xticks(x)
ax2.set_xticklabels(algos, rotation=45)

# Add strengths as text annotations above bars
for i, algo in enumerate(algos):
    strengths_text = (f"R: {strengths['Reward'][algo]}%\n"
                     f"T: {strengths['Time Efficiency'][algo]}%\n"
                     f"A: {strengths['Action Efficiency'][algo]}%")
    ax1.text(i, final_rewards[i] + 500, strengths_text, ha='center', fontsize=8)
    ax2.text(i, avg_times[i] + 0.2, strengths_text, ha='center', fontsize=8)

ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/results_histogram_with_strengths.png', bbox_inches='tight')
plt.close()