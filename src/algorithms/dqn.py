import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

from .heuristic_utils import get_pellet_heuristic

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features, self.out_features, self.std_init = in_features, out_features, std_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, 
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        return F.linear(x, self.weight_mu, self.bias_mu)

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, noisy=False):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        with torch.no_grad():
            self.conv_out_size = self.conv(torch.zeros(1, *input_shape)).view(1, -1).size(1)
        
        if noisy:
            self.fc = nn.Sequential(
                NoisyLinear(self.conv_out_size, 512), nn.ReLU(), 
                NoisyLinear(512, n_actions)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.conv_out_size, 512), nn.ReLU(), 
                nn.Linear(512, n_actions)
            )

    def forward(self, x):
        # 1. Add batch dimension if a single frame is passed
        if x.dim() == 3: 
            x = x.unsqueeze(0)
            
        # If the last channel is smaller than the first two (Height/Width), we assume it's in HWC format and permute to CHW
        if x.shape[-1] < x.shape[1] and x.shape[-1] < x.shape[2]:
            x = x.permute(0, 3, 1, 2)

        # 3. Normalization (if data is still in 0-255 range)
        if x.max() > 1.0:
            x = x.float() / 255.0
            
        return self.fc(self.conv(x).reshape(x.size(0), -1))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity, self.alpha, self.buffer, self.priorities, self.pos = capacity, alpha, [], [], 0

    def add(self, transition, error):
        # MEMORY OPTIMIZATION: Save images as uint8 (0-255)
        s, a, r, ns, d = transition
        s_uint = s.astype(np.uint8) if isinstance(s, np.ndarray) else s
        ns_uint = ns.astype(np.uint8) if isinstance(ns, np.ndarray) else ns
        
        p = (abs(error) + 1e-6) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append((s_uint, a, r, ns_uint, d))
            self.priorities.append(p)
        else:
            self.buffer[self.pos] = (s_uint, a, r, ns_uint, d)
            self.priorities[self.pos] = p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        pri_arr = np.array(self.priorities, dtype=np.float64)
        probs = pri_arr / pri_arr.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        return [self.buffer[i] for i in indices], indices, weights / weights.max()

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = (abs(err) + 1e-6) ** self.alpha

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, **kwargs):
        self.state_dim = kwargs.get('state_dim')
        self.action_dim = kwargs.get('action_dim')
        self.device = kwargs.get('device')
        self.save_dir = kwargs.get('save_dir')

        if self.state_dim is None:
            raise ValueError("❌ state_dim was not passed to DQNAgent!")

        self.gamma = kwargs.get('gamma', 0.99)
        self.batch_size = kwargs.get('batch_size', 32)
        self.epsilon = kwargs.get('exploration_initial_eps', 1.0)
        self.epsilon_min = kwargs.get('exploration_final_eps', 0.05)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)  # per episode, not per step
        self.learn_every = kwargs.get('train_freq', 4)
        self.sync_every = kwargs.get('target_update_interval', 1000)
        self.use_heuristics = kwargs.get('use_heuristics', True)
        self.noisy = kwargs.get('noisy', False)
        
        self.net = DQN(self.state_dim, self.action_dim, self.noisy).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim, self.noisy).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=kwargs.get('learning_rate', 0.0001))
        self.memory = PrioritizedReplayBuffer(kwargs.get('buffer_size', 100000))
        self.frame_idx = 0

    def act(self, state, info=None):
        if self.use_heuristics and info:
            # Priority 1: flee ghost if close
            g_act = info.get('ghost_avoidance_action')
            if g_act is not None:
                return g_act

            # Priority 2: unstuck — if only 1 direction open, always take it
            # prevents oscillation in dead-end corridors regardless of epsilon
            u_act = info.get('unstuck_action')
            if u_act is not None:
                return u_act

            # Priority 3: pellet heuristic (fades with epsilon as network learns)
            pellet_prob = min(0.8, self.epsilon)
            p_act = info.get('pellet_heuristic_action')
            if p_act is not None and np.random.rand() < pellet_prob:
                return p_act

        # Random exploration — only from open directions, with momentum to avoid bouncing
        open_dirs = info.get('open_dirs', []) if info else []
        if not self.noisy and np.random.rand() < self.epsilon:
            if open_dirs:
                # 60% chance to continue last direction if still open (momentum)
                if hasattr(self, 'last_action') and self.last_action in open_dirs and np.random.rand() < 0.6:
                    return self.last_action
                action = int(np.random.choice(open_dirs))
                self.last_action = action
                return action
            return np.random.randint(self.action_dim)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if state_t.max() > 1.0: state_t /= 255.0
        with torch.no_grad():
            q_vals = self.net(state_t)[0]
            # Mask closed directions so network never picks a walled action
            if open_dirs:
                mask = torch.full(q_vals.shape, -1e9, device=q_vals.device)
                for a in open_dirs:
                    if a < len(mask):
                        mask[a] = 0.0
                q_vals = q_vals + mask
            return int(q_vals.argmax().item())

    def cache(self, state, action, reward, next_state, done):
        error = max(self.memory.priorities) if len(self.memory) > 0 else 1.0
        self.memory.add((state, action, reward, next_state, done), error)

    def learn(self):
        self.frame_idx += 1
        if len(self.memory) < self.batch_size or self.frame_idx % self.learn_every != 0:
            return None
        
        if self.frame_idx % self.sync_every == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        samples, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        # Convert from uint8 back to float32 for training
        states = torch.FloatTensor(np.array(states)).to(self.device) / 255.0
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device) / 255.0
        actions = torch.LongTensor(list(actions)).to(self.device)
        rewards = torch.FloatTensor(list(rewards)).clamp(-1.0, 1.0).to(self.device)
        dones = torch.BoolTensor(list(dones)).to(self.device)
        weights = torch.FloatTensor(list(weights)).to(self.device)

        curr_q = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q

        loss = (F.smooth_l1_loss(curr_q, target_q, reduction='none') * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_errors = (curr_q - target_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, np.abs(td_errors))
        return loss.item()

    def decay_epsilon(self):
        """Call once per episode end to decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, name):
        """Optimized save: weights only, without the replay memory buffer."""
        path = os.path.join(self.save_dir, f"{name}.pth")
        save_dict = {
            'net': self.net.state_dict(),
            'eps': self.epsilon,
            'frame_idx': self.frame_idx
        }
        # Using a higher level of compression in torch.save
        torch.save(save_dict, path, _use_new_zipfile_serialization=True)
        return 1

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.target_net.load_state_dict(self.net.state_dict())
        self.epsilon = checkpoint.get('eps', self.epsilon)
        self.frame_idx = checkpoint.get('frame_idx', 0)