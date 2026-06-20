import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import pickle
import random
from .heuristic_utils import _find_pacman_position

class PPOPolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PPOPolicyNetwork, self).__init__()
        # input_shape[0] теперь будет равен 12 (для RGB стека)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.conv_out_size = self._get_conv_out(input_shape)
        
        self.actor_fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        self.critic_fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            return self.conv(dummy).view(1, -1).size(1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # If last dim is smaller than spatial dims, it's HWC — permute to CHW
        if x.shape[-1] < x.shape[1] and x.shape[-1] < x.shape[2]:
            x = x.permute(0, 3, 1, 2)

        x = x.float() / 255.0
        conv_out = self.conv(x).view(x.size(0), -1)
        logits = self.actor_fc(conv_out)
        logits = torch.clamp(logits, -30, 30)
        value = self.critic_fc(conv_out)
        return logits, value

# ==============================
# PPO Agent с фиксацией действий
# ==============================
class PPOAgent:
    def __init__(self, **kwargs):
        # Параметры из YAML
        self.state_dim = kwargs.get('state_dim')
        self.action_dim = kwargs.get('action_dim')
        self.device = kwargs.get('device', 'cpu')
        self.save_dir = kwargs.get('save_dir', 'models/ppo')
        
        self.gamma = kwargs.get('gamma', 0.99)
        self.lam = kwargs.get('lam', 0.95)
        self.clip_eps = kwargs.get('clip_eps', 0.2)
        self.ppo_epochs = kwargs.get('ppo_epochs', 4)
        self.batch_size = kwargs.get('batch_size', 64)
        lr = kwargs.get('learning_rate', 2.5e-4)
        
        self.policy_net = PPOPolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.trajectories = []
        
        # ===== STATE FOR ACTION LOCKING =====
        self.prev_pos = None
        self.last_action = 0
        self.initialized = False

    def cache(self, state, action, reward, next_state, done):
        """Store transition. log_prob and value come from the last act() call."""
        state_t = torch.as_tensor(state, dtype=torch.float32)
        reward_clipped = float(np.clip(reward, -1.0, 1.0))
        pending = getattr(self, '_pending', {'log_prob': 0.0, 'value': 0.0})
        self.trajectories.append({
            'state': state_t,
            'action': action,
            'reward': reward_clipped,
            'value': pending['value'],
            'log_prob': pending['log_prob'],
            'done': done
        })

    def act(self, state, info=None) -> int:
        # Priority 1: heuristic overrides from main loop
        if info:
            g_act = info.get('ghost_avoidance_action')
            if g_act is not None:
                self.last_action = g_act
                self._pending = {'log_prob': np.log(0.99), 'value': 0.0}
                return g_act
            u_act = info.get('unstuck_action')
            if u_act is not None:
                self.last_action = u_act
                self._pending = {'log_prob': np.log(0.99), 'value': 0.0}
                return u_act

        open_dirs = info.get('open_dirs', []) if info else []
        self.policy_net.eval()
        with torch.no_grad():
            st_t = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            logits, value = self.policy_net(st_t)
            probs = F.softmax(logits, dim=-1)

            # Mask NOOP(0) and FIRE(1)
            probs[0, 0] = 0.0
            if probs.shape[1] > 1:
                probs[0, 1] = 0.0

            # Mask closed directions
            if open_dirs:
                closed = [a for a in range(probs.shape[1]) if a not in open_dirs]
                for a in closed:
                    probs[0, a] = 0.0

            total = probs.sum()
            if total < 1e-8:
                action = int(np.random.choice(open_dirs)) if open_dirs else random.randint(2, 5)
                log_prob = np.log(1.0 / max(len(open_dirs), 1))
            else:
                probs = probs / total
                dist = Categorical(probs)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()

        self.last_action = action
        self._pending = {'log_prob': log_prob, 'value': value.item()}
        return action

    def learn(self) -> float:
        # PPO is on-policy: only train at episode end with full trajectory
        if not self.trajectories or not self.trajectories[-1]['done']:
            return 0.0
        if len(self.trajectories) < self.batch_size:
            self.trajectories = []
            return 0.0
        
        # Подготовка данных
        states = torch.stack([t['state'] for t in self.trajectories]).to(self.device)
        actions = torch.tensor([t['action'] for t in self.trajectories], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor([t['log_prob'] for t in self.trajectories], dtype=torch.float32).to(self.device)
        rewards = [t['reward'] for t in self.trajectories]
        values = [t['value'] for t in self.trajectories]
        dones = [t['done'] for t in self.trajectories]

        # Расчет Returns и Advantages (GAE)
        returns = []
        gae = 0
        next_value = 0 
        
        for i in reversed(range(len(self.trajectories))):
            # TD Error
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i])
            next_value = values[i]

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - torch.tensor(values, dtype=torch.float32).to(self.device)
        # Нормализация преимуществ
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        self.policy_net.train()
        
        # Эпохи обучения PPO
        for _ in range(self.ppo_epochs):
            # Перемешивание индексов для батчей
            indices = np.random.permutation(len(self.trajectories))
            for start in range(0, len(self.trajectories), self.batch_size):
                idx = indices[start : start + self.batch_size]
                
                b_states = states[idx]
                b_actions = actions[idx]
                b_old_log_probs = old_log_probs[idx]
                b_advantages = advantages[idx]
                b_returns = returns[idx]

                logits, curr_values = self.policy_net(b_states)
                curr_probs = F.softmax(logits, dim=-1)
                curr_log_probs = torch.log(curr_probs.gather(1, b_actions.unsqueeze(1)).squeeze() + 1e-10)
                
                # PPO Clip Objective
                ratio = torch.exp(curr_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(curr_values.squeeze(), b_returns)
                
                # Энтропия для поощрения исследования
                entropy = Categorical(curr_probs).entropy().mean()
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                # Клиппинг градиентов для защиты от взрыва
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self.trajectories = []
        return total_loss / self.ppo_epochs

    def save(self, name):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"{name}.pth")
        torch.save(self.policy_net.state_dict(), path)
        print(f"💾 Модель PPO сохранена: {path}")

    def load(self, path):
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            print(f"📥 Модель PPO загружена: {path}")