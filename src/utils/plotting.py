import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_training_rewards(
    rewards,
    save_dir="plots",
    algorithm_name="dqn",
    run_id=None,
    rolling_window=10,
    show=False,
    level_completion_times=None
):
    """Plot and save the rewards curve across training episodes"""
    save_dir = Path(save_dir) / algorithm_name
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"reward_curve_{run_id}.png" if run_id is not None else "reward_curve.png"
    save_path = save_dir / filename

    plt.figure(figsize=(12, 6))
    
    # Create two subplots
    if level_completion_times is not None:
        plt.subplot(2, 1, 1)
    
    # Rewards plot
    plt.plot(rewards, label="Reward")
    if len(rewards) >= rolling_window:
        rolling = np.convolve(rewards, np.ones(rolling_window)/rolling_window, mode='valid')
        plt.plot(
            range(rolling_window-1, len(rewards)),
            rolling,
            label=f"Rolling Average ({rolling_window})",
            linestyle="--"
        )
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    
    # Level completion speed plot
    if level_completion_times is not None:
        plt.subplot(2, 1, 2)
        plt.plot(level_completion_times, label="Level Completion Time", color="tab:green")
        if len(level_completion_times) >= rolling_window:
            rolling_times = np.convolve(
                level_completion_times, 
                np.ones(rolling_window)/rolling_window, 
                mode='valid'
            )
            plt.plot(
                range(rolling_window-1, len(level_completion_times)),
                rolling_times,
                label=f"Rolling Average ({rolling_window})",
                linestyle="--",
                color="tab:orange"
            )
        plt.title("Level Completion Speed")
        plt.xlabel("Episode")
        plt.ylabel("Time (sec)")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)

    if show:
        plt.show()
    plt.close()