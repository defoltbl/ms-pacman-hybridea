import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import time
import inspect
import logging

# Path setup
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(ROOT_DIR))

from environment.pacman_env import PacmanEnv
from algorithms.dqn import DQNAgent
from algorithms.monte_carlo import MonteCarloAgent
from algorithms.ppo import PPOAgent
from algorithms.a2c import A2CAgent
from algorithms.bc import BCAgent
from algorithms.hybrid_bc_ppo import HybridBCPPOAgent
from utils.plotting import plot_training_rewards
from utils.system_monitor import SystemMonitor, plot_system_usage
from algorithms.heuristic_utils import get_pellet_heuristic, ghost_avoidance_heuristic, _find_pacman_position, _open_directions

AGENT_CLASSES = {
    "DQN": DQNAgent,
    "MC": MonteCarloAgent,
    "PPO": PPOAgent,
    "A2C": A2CAgent,
    "BC": BCAgent,
    "HYBRID_BC_PPO": HybridBCPPOAgent,
}

def choose_algorithm():
    """Interactive menu for selecting an algorithm."""
    print("\nAvailable algorithms:")
    options = list(AGENT_CLASSES.keys())
    for i, name in enumerate(options, start=1):
        print(f"{i}. {name}")
    
    while True:
        try:
            choice = input("\nEnter algorithm number (or name): ").strip()
            if choice.upper() in options:
                return choice.upper()
            
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

def choose_model_file(model_dir):
    if not os.path.exists(model_dir):
        return None
    files = sorted(Path(model_dir).rglob("*.pth"))
    if not files:
        return None
    
    print(f"\nFound existing models in {model_dir.name}:")
    print("0. Start from scratch (do not load)")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f.name}")
    
    try:
        choice = int(input("Select a model to load (or 0): "))
        if choice == 0: return None
        return str(files[choice-1])
    except (ValueError, IndexError):
        return None

def train(config, env, agent, device, run_id):
    active_algo = config['algorithm']['active']
    
    # Configure logs
    log_dir = Path("logs") / active_algo
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{run_id}.log"
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )

    logging.info(f"START: {active_algo.upper()} | Device: {device}")
    
    monitor = SystemMonitor()
    monitor.start()
    
    total_rewards = []
    level_times = []
    pellets_history = []
    losses = []
    
    log_interval = config["logging"].get("log_interval", 1)
    max_episodes = config["training"].get("episodes", 10)
    render_enabled = config["environment"].get("render_mode") == "human"

    # Set to True to print heuristic decisions every step for debugging
    DEBUG_HEURISTICS = True
    debug_step = 0

    try:
        pbar = tqdm(range(max_episodes), desc=f"Training {active_algo.upper()}")
        for episode in pbar:
            state = env.reset()
            done = False
            total_reward = 0
            episode_losses = []
            start_time = time.time()
            episode_pellets = 0  # track pellets from env info, not from reward

            # --- PHASE SWITCHING LOGIC FOR HYBRID ---
            # Use Behavioral Cloning (BC) for the first 50 episodes, then transition to pure PPO
            current_phase = 'bc' if (active_algo == 'hybrid_bc_ppo' and episode < 50) else 'ppo'
            
            if episode == 50 and active_algo == 'hybrid_bc_ppo':
                logging.info("\n" + "="*40)
                logging.info("PHASE 2: Disabling BC. Transitioning to pure PPO.")
                logging.info("="*40)

            while not done:
                if render_enabled:
                    env.render()
                
                # Collect extended info for the agent
                raw_screen = env.ale.getScreenRGB()
                ghost_pos = env.get_ghost_positions()
                pac_pos = _find_pacman_position(raw_screen)
                open_dirs = _open_directions(raw_screen, pac_pos) if pac_pos else []
                info = {
                    "ghost_positions": ghost_pos,
                    "blocked_actions": [],
                    "pellet_heuristic_action": get_pellet_heuristic(raw_screen, ghost_pos, pacman_pos=pac_pos, open_dirs=open_dirs),
                    "ghost_avoidance_action": ghost_avoidance_heuristic(raw_screen, ghost_pos, pacman_pos=pac_pos, open_dirs=open_dirs),
                    "unstuck_action": open_dirs[0] if len(open_dirs) == 1 else None,
                    "open_dirs": open_dirs,
                    "pac_pos": pac_pos,  # needed by hybrid BC phase distance check
                    "current_phase": current_phase,  # BC or PPO phase for hybrid
                }

                # Debug: print heuristic state every 30 steps
                if DEBUG_HEURISTICS and debug_step % 30 == 0:
                    pac = pac_pos
                    open_d = open_dirs
                    open_d = _open_directions(raw_screen, pac) if pac else []
                    ale_names = {0:'NOOP',1:'FIRE',2:'UP',3:'RIGHT',4:'LEFT',5:'DOWN'}
                    g_act = info['ghost_avoidance_action']
                    p_act = info['pellet_heuristic_action']
                    eps = f"{agent.epsilon:.3f}" if hasattr(agent, 'epsilon') else '-'
                    u_act = info.get('unstuck_action')
                    lives = getattr(env.ale, 'lives', lambda: '?')()
                    print(f"\n[DBG step={debug_step}] "
                          f"PAC={'YES '+str(pac) if pac else 'NOT FOUND'} | "
                          f"LIVES={lives} | "
                          f"GHOSTS={len(ghost_pos)} at {ghost_pos[:2]} | "
                          f"OPEN={[ale_names.get(a,a) for a in open_d]} | "
                          f"AVOID={ale_names.get(g_act, 'None')} | "
                          f"UNSTUCK={ale_names.get(u_act, 'None')} | "
                          f"PELLET={ale_names.get(p_act, 'None')} | "
                          f"EPS={eps}")
                debug_step += 1
                
                action = agent.act(state, info)
                next_state, reward, done, step_info = env.step(action)
                
                # Append environment frame updates to info dict
                if step_info:
                    info.update(step_info)
                
                # Cache transitions (Reward Shaping occurs within the agent)
                agent.cache(state, action, reward, next_state, done)
                
                # Train model considering active phase
                if active_algo == 'hybrid_bc_ppo':
                    loss = agent.learn(phase=current_phase)
                else:
                    loss = agent.learn()
                    
                if loss is not None and loss > 0:
                    episode_losses.append(loss)

                if step_info and not step_info.get('hit_wall', False) and reward > 0:
                    episode_pellets += 1

                state = next_state
                total_reward += reward

            duration = time.time() - start_time
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0

            # Decay DQN exploration rate once per episode (not per step)
            if hasattr(agent, 'decay_epsilon'):
                agent.decay_epsilon()
            
            total_rewards.append(total_reward)
            level_times.append(duration)
            pellets_history.append(episode_pellets)
            losses.append(avg_loss)

            if episode % log_interval == 0:
                mean_rew = np.mean(total_rewards[-log_interval:])
                loss_str = f"{avg_loss:8.4f}" if avg_loss > 0 else "N/A"
                phase_str = f" | Phase: {current_phase.upper()}" if active_algo == 'hybrid_bc_ppo' else ""
                
                log_msg = (f"Ep {episode:3d}{phase_str} | "
                           f"Reward: {int(total_reward):4d} (Avg: {mean_rew:6.1f}) | "
                           f"Pellets: {int(episode_pellets):3d} | "
                           f"Loss: {loss_str} | "
                           f"Time: {duration:5.1f}s")
                logging.info(log_msg)

            if device.type == 'mps':
                torch.mps.empty_cache()

    finally:
        monitor.stop()
        
    return total_rewards, level_times, pellets_history, monitor.get_data()

def evaluate(config, env, agent, num_episodes=5):
    """Pure evaluation mode running without model training."""
    print(f"\nLAUNCHING EVALUATION: {num_episodes} episodes")
    
    if hasattr(agent, 'policy_net'): agent.policy_net.eval()
    elif hasattr(agent, 'net'): agent.net.eval()

    eval_data = {
        'rewards': [],
        'steps': [],
        'times': []
    }

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        start_time = time.time()

        while not done:
            env.render()
            raw_screen = env.ale.getScreenRGB()
            ghost_pos = env.get_ghost_positions() if hasattr(env, 'get_ghost_positions') else []
            pac_pos_eval = _find_pacman_position(raw_screen)
            open_dirs_eval = _open_directions(raw_screen, pac_pos_eval) if pac_pos_eval else []
            info = {
                    "ghost_positions": ghost_pos,
                    "blocked_actions": [],
                    "pellet_heuristic_action": get_pellet_heuristic(raw_screen, ghost_pos, pacman_pos=pac_pos_eval, open_dirs=open_dirs_eval),
                    "ghost_avoidance_action": ghost_avoidance_heuristic(raw_screen, ghost_pos, pacman_pos=pac_pos_eval, open_dirs=open_dirs_eval),
                    "unstuck_action": open_dirs_eval[0] if len(open_dirs_eval) == 1 else None,
                    "open_dirs": open_dirs_eval,
                    "pac_pos": pac_pos_eval,
                    "current_phase": "ppo",
            }
            
            with torch.no_grad():
                action = agent.act(state, info)
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            time.sleep(0.01)

        duration = time.time() - start_time
        eval_data['rewards'].append(total_reward)
        eval_data['steps'].append(steps)
        eval_data['times'].append(duration)
        
        print(f"Test {episode + 1} | Reward: {int(total_reward):4d} | Steps: {steps:4d} | Time: {duration:4.1f}s")
    
    return eval_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pacman_env.yaml")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--algorithm", type=str, help="Name of the algorithm (DQN, PPO...)")
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        return

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Select active algorithm
    if args.algorithm:
        algorithm_name = args.algorithm.upper()
    else:
        algorithm_name = choose_algorithm()

    # Synchronize configuration file options
    active_algo_key = algorithm_name.lower()
    if 'algorithm' not in config:
        config['algorithm'] = {}
    config['algorithm']['active'] = active_algo_key

    run_id = time.strftime("%Y%m%d-%H%M%S")
    device = torch.device("mps" if config["training"].get("device") == "mps" and torch.backends.mps.is_available() else "cpu")
    
    # Initialize game environment
    env = PacmanEnv(
        render_mode=config["environment"].get("render_mode"),
        level=config["environment"].get("level", "easy"),
        frameskip=config["environment"].get("frameskip", 4),
        gray_scale=config["environment"].get("gray_scale", False),
        stack_frames=config["environment"].get("stack_frames", 1),
        resize_shape=tuple(config["environment"].get("resize_shape", [128, 128]))
    )

    # --- ADJUSTED INPUT CHANNEL CALCULATION ---
    stack_count = config["environment"].get("stack_frames", 1)
    is_gray = config["environment"].get("gray_scale", False)
    
    # 1 RGB frame = 3 channels. A stack of 4 RGB frames = 12 channels.
    total_channels = stack_count if is_gray else stack_count * 3
    state_dim = (total_channels, env.resize_shape[1], env.resize_shape[0])
    # --------------------------------------------

    save_dir = Path(f"models/{active_algo_key}")
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = choose_model_file(save_dir)

    # Gather required agent hyper-parameters
    algo_config = config['algorithms'].get(active_algo_key, {})
    all_params = {
        'state_dim': state_dim,
        'action_dim': len(env.action_space),
        'save_dir': str(save_dir),
        'device': device,
        'use_heuristics': config["training"].get("use_heuristics", True),
        **algo_config
    }

    # Dynamic initialization of the selected agent class
    agent_class = AGENT_CLASSES[algorithm_name]
    sig = inspect.signature(agent_class.__init__)
    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    init_params = all_params if has_kwargs else {k: v for k, v in all_params.items() if k in sig.parameters}

    print(f"Initializing {algorithm_name} (Input: {state_dim})...")
    agent = agent_class(**init_params)

    if model_path:
        agent.load(model_path)

    # Move underlying weights to accelerated hardware device (MPS/CPU)
    for attr in ['net', 'target_net', 'actor', 'critic', 'policy_net']:
        if hasattr(agent, attr):
            m = getattr(agent, attr)
            if isinstance(m, torch.nn.Module): m.to(device)

    if args.mode == "train":
        rewards, l_times, pellets, sys_data = train(config, env, agent, device, run_id)
        
        save_name = f"{algorithm_name}_{run_id}"
        agent.save(save_name)
        
        print("\nSaving graphical analysis...")
        plot_training_rewards(rewards, algorithm_name=algorithm_name, run_id=run_id, level_completion_times=l_times)
        plot_system_usage(sys_data, algorithm_name=algorithm_name, run_id=run_id)
        
        print(f"Done. Performance updates saved successfully.")

    elif args.mode == "eval":
        # STRICT VALIDATION: If evaluating performance, a trained model is mandatory
        if not model_path:
            print("\nERROR: Evaluation mode requires loading an optimized model path!")
            print("Please restart the process and pick a valid model reference ID (1, 2...).")
            env.close()
            return

        num_test_episodes = config["training"].get("eval_episodes", 5)
        eval_results = evaluate(config, env, agent, num_episodes=num_test_episodes)
        
        # --- STORAGE AND DATA VISUALIZATION ---
        # Generate an independent output metrics folder keyed by run_id
        eval_dir = Path(f"results/eval/{active_algo_key}/{run_id}")
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Text Summary File Logger
        with open(eval_dir / "report.txt", "w") as f:
            f.write(f"Algorithm: {algorithm_name}\n")
            f.write(f"Model used: {model_path}\n")
            f.write(f"Mean Reward: {np.mean(eval_results['rewards']):.2f}\n")
            f.write(f"Max Reward: {np.max(eval_results['rewards'])}\n")
            f.write(f"Rewards List: {eval_results['rewards']}\n")

        # 2. Performance Tracking Subplots (Matplotlib)
        plt.figure(figsize=(12, 5))
        
        # Score distribution across test episodes
        plt.subplot(1, 2, 1)
        plt.plot(eval_results['rewards'], marker='o', linestyle='-', color='royalblue')
        plt.axhline(y=np.mean(eval_results['rewards']), color='r', linestyle='--', label='Average')
        plt.title(f'Performance: {algorithm_name}')
        plt.xlabel('Test Episode')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Hit rate/Reward spectrum histogram
        plt.subplot(1, 2, 2)
        plt.hist(eval_results['rewards'], bins=8, color='mediumseagreen', edgecolor='black', alpha=0.7)
        plt.title('Reward Distribution')
        plt.xlabel('Score Range')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.savefig(eval_dir / "eval_metrics.png")
        plt.close()

        print(f"\nEvaluation metrics systematically written to: {eval_dir}")
    
    env.close()

if __name__ == "__main__":
    main()