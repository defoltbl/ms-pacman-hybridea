import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.distributions import Categorical
import numpy as np
from algorithms.heuristic_utils import ghost_avoidance_heuristic, get_pellet_heuristic


class MCPolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(MCPolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_out(self, shape):
        with torch.no_grad():
            return self.conv(torch.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.max() > 1.0:
            x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return torch.softmax(self.fc(x), dim=-1)


class MonteCarloAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, learning_rate=0.0001, batch_size=64,
                 save_dir='models/mc', device=None, use_heuristics=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.use_heuristics = use_heuristics
        os.makedirs(save_dir, exist_ok=True)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network
        self.policy_net = MCPolicyNetwork(state_dim, action_dim).to(self.device)
        # Aliases to ensure main.py works without modifications
        self.net = self.policy_net
        self.target_net = self.net

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

        # Episode buffer
        self.episode = []
        self._done = False
        self.memory = [] 

    def act(self, state, info=None):
        # Heuristics identical to DQN
        if self.use_heuristics and info is not None:
            g_act = info.get("ghost_avoidance_action")
            if g_act is not None:
                return g_act

        st = torch.FloatTensor(state / 255.0)\
                   .unsqueeze(0).to(self.device)
        probs = self.policy_net(st)
        dist = Categorical(probs)
        return dist.sample().item()

    def remember(self, state, action, reward):
        self.episode.append((state, action, reward))

    def cache(self, state, action, reward, next_state, done):
        # API mirroring DQNAgent
        self.remember(state, action, reward)
        self._done = done

    def learn(self):
        # Return loss only when the episode has ended
        if not self._done:
            return None

        # Compute G returns for each time step
        G = 0
        returns = []
        for _, _, r in reversed(self.episode):
            G = r + self.gamma * G
            returns.insert(0, G)

        # Normalization
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if returns.std() > 1e-5:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Collect states and actions
        states = torch.FloatTensor(
            np.array([s/255.0 for s,_,_ in self.episode])
        ).to(self.device)
        actions = torch.LongTensor([a for _,a,_ in self.episode]).to(self.device)

        # Policy gradient loss
        probs = self.policy_net(states)
        dist = Categorical(probs)
        logp = dist.log_prob(actions)
        loss = -(logp * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset variables before the next episode
        self.memory.append(self.episode.copy())
        self.episode = []
        self._done = False

        return loss.item()

    def save(self, algorithm_name='mc'):
        base = algorithm_name.replace('.pth','')
        dir_ = os.path.join(self.save_dir, base)
        os.makedirs(dir_, exist_ok=True)

        idx = len([f for f in os.listdir(dir_) if f.endswith('.pth')]) + 1
        model_path = os.path.join(dir_, f"{base}_{idx}.pth")
        torch.save({
            'policy_net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, model_path)

        mem_path = model_path.replace('.pth','_memory.pkl')
        with open(mem_path, 'wb') as f:
            pickle.dump(self.memory, f)

        print(f"Model saved: {model_path}")
        print(f"Memory saved: {mem_path}")
        return idx

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt['policy_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.net.eval()

        mem_path = path.replace('.pth','_memory.pkl')
        if os.path.exists(mem_path):
            with open(mem_path,'rb') as f:
                self.memory = pickle.load(f)
            print(f"Memory loaded: {mem_path}")
        print(f"Model loaded: {path}")