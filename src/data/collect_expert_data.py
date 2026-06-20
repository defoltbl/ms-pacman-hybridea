import pickle
import numpy as np
import os
import random
from environment.pacman_env import PacmanEnv
from algorithms.heuristic_utils import ghost_avoidance_heuristic, get_pellet_heuristic, _find_pacman_position

def collect_expert_data(num_episodes=50, save_path="data/expert_dataset.pkl", epsilon=0.1):
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    env = PacmanEnv(render_mode=None, resize_shape=(128, 128))
    dataset = []

    print(f"Collecting CLEANED data... Episodes: {num_episodes}")
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        last_pos = None
        ep_steps = 0
        
        while not done:
            # Use raw ALE screen for position detection (not the stacked NN state)
            raw_screen = env.ale.getScreenRGB()
            current_pos = _find_pacman_position(raw_screen)
            
            # Action selection
            ghosts = env.get_ghost_positions() if hasattr(env, 'get_ghost_positions') else []
            
            if random.random() < epsilon:
                action = np.random.choice(env.action_space)  # Bug 3: use np.random for numpy array
            else:
                action = ghost_avoidance_heuristic(state, ghosts)
                if action is None:
                    # Bug 2: pass ghost positions, not pellet positions
                    action = get_pellet_heuristic(state, ghosts)
            
            if action is None:
                action = np.random.choice(env.action_space)

            next_state, _, done, _ = env.step(action)

            # Use raw ALE screen for next position detection
            next_raw = env.ale.getScreenRGB()
            next_pos = _find_pacman_position(next_raw)

            # Bug 4: filter only when both positions are known and pacman actually moved
            if last_pos is not None and next_pos is not None:
                dist = np.linalg.norm(np.array(next_pos) - np.array(last_pos))
                if dist > 0.8:
                    dataset.append({'state': state.copy(), 'action': action})
                    ep_steps += 1
            
            state = next_state
            # Bug 4: always update last_pos, using next_pos if available, else keep old
            if next_pos is not None:
                last_pos = next_pos
            
        print(f"Episode {ep+1:2d} | Recorded clean steps: {ep_steps}")

    random.shuffle(dataset)
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset ready! Total frames: {len(dataset)}")

if __name__ == "__main__":
    collect_expert_data()