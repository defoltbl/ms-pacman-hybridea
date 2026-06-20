import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from .heuristic_utils import get_pellet_heuristic

class A2CPolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(A2CPolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.conv_out_size = self._get_conv_out(input_shape)
        
        # Actor head
        self.actor_fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        # Critic head
        self.critic_fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
            return o.view(1, -1).size(1)

    def forward(self, x):
        # 1. Add batch dimension if a single frame is passed
        if x.dim() == 3: 
            x = x.unsqueeze(0)
            
        # 2. Smart check for axis ordering (CHW vs HWC)
        # If the last dimension is smaller than height and width, it indicates channels (HWC)
        if x.shape[-1] < x.shape[1] and x.shape[-1] < x.shape[2]:
            x = x.permute(0, 3, 1, 2)
            
        # 3. Cast to float and normalize
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0
            
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.actor_fc(conv_out), self.critic_fc(conv_out)

class A2CAgent:
    def __init__(self, **kwargs):
        self.state_dim = kwargs.get('state_dim')
        self.action_dim = kwargs.get('action_dim')
        self.device = kwargs.get('device')
        self.save_dir = kwargs.get('save_dir')
        
        if self.state_dim is None:
            raise ValueError("state_dim was not passed to A2CAgent!")

        self.gamma = kwargs.get('gamma', 0.99)
        self.use_heuristics = kwargs.get('use_heuristics', True)
        self.value_loss_coef = kwargs.get('value_loss_coef', 0.5)
        self.entropy_coef = kwargs.get('entropy_coef', 0.1)
        
        self.policy_net = A2CPolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=kwargs.get('learning_rate', 2.5e-4))
        
        # Current episode buffer
        self.episode_buffer = []

    def act(self, state, info=None):
        if self.use_heuristics and info:
            # Use pre-computed heuristics from main loop (correct raw screen coords)
            g_act = info.get('ghost_avoidance_action')
            if g_act is not None:
                return g_act
            u_act = info.get('unstuck_action')
            if u_act is not None:
                return u_act
            p_act = info.get('pellet_heuristic_action')
            if p_act is not None and np.random.rand() < 0.4:
                return p_act

        open_dirs = info.get('open_dirs', []) if info else []
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            logits, _ = self.policy_net(state_t)
            if open_dirs:
                mask = torch.full(logits.shape, -1e9, device=logits.device)
                for a in open_dirs:
                    if a < mask.shape[1]:
                        mask[0, a] = 0.0
                logits = logits + mask
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            return dist.sample().item()

    def cache(self, state, action, reward, next_state, done):
        # Save as uint8 to save RAM
        shaped_reward = reward if reward != 0 else -0.01
        self.episode_buffer.append({
            'state': state.astype(np.uint8),
            'action': action,
            'reward': shaped_reward,
            'done': done
        })

    def learn(self):
        if not self.episode_buffer or not self.episode_buffer[-1]['done']:
            return None
        
        # Data preparation
        states = torch.FloatTensor(np.array([t['state'] for t in self.episode_buffer])).to(self.device) / 255.0
        actions = torch.LongTensor([t['action'] for t in self.episode_buffer]).to(self.device)
        rewards = [t['reward'] for t in self.episode_buffer]
        dones = [t['done'] for t in self.episode_buffer]

        # Calculate Returns (G_t)
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d: R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)

        # Forward pass
        logits, values = self.policy_net(states)
        values = values.squeeze()
        
        # Calculate Advantages
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate Log Probs and Entropy
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Losses
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()
        
        self.episode_buffer = [] # Clear buffer
        return loss.item()

    def save(self, name):
        """Compact saving of weights only."""
        path = os.path.join(self.save_dir, f"{name}.pth")
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path, _use_new_zipfile_serialization=True)
        return 1

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])