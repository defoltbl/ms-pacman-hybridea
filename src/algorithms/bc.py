import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import random

from .heuristic_utils import _find_pacman_position


# ==============================
# Policy Network (BC)
# ==============================

class BCPolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.input_channels = input_shape[0] 

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        if x.shape[1] not in (1, 3, 4, 12) and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)

        if x.shape[1] == 3 and self.input_channels == 12:
            x = x.repeat(1, 4, 1, 1)

        if x.shape[2:] != (128, 128):
            x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)

        x = x.float() / 255.0 if x.max() > 1.0 else x.float()
        return self.fc(self.conv(x).view(x.size(0), -1))


# ==============================
# BC Agent with Action Locking
# ==============================

class BCAgent:
    """
    Atari Ms. Pac-Man compatible BC agent.
    CRITICAL: direction is changed ONLY after movement.
    """

    def __init__(self, **kwargs):
        self.state_dim = kwargs["state_dim"]
        self.action_dim = kwargs["action_dim"]
        self.device = kwargs["device"]
        self.save_dir = kwargs["save_dir"]

        self.policy_net = BCPolicyNetwork(
            self.state_dim, self.action_dim
        ).to(self.device)

        self.batch_size = kwargs.get('batch_size', 128)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=kwargs.get('learning_rate', 1e-3)
        )

        self.expert_dataset = []
        if kwargs.get("expert_dataset_path"):
            self.load_expert_dataset(kwargs["expert_dataset_path"])

        # ===== ACTION LOCKING STATE =====
        self.prev_pos = None
        self.last_action = 0
        self.initialized = False

    def cache(self, *args):
        pass

    # ==============================
    # Action selection
    # ==============================

    def act(self, state, info=None):
        # Use blocked_actions from info for wall awareness; position detection
        self.policy_net.eval()
        with torch.no_grad():
            st = torch.from_numpy(state).to(self.device)
            logits = self.policy_net(st)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # sort actions by probability, excluding the "no-op" action (0)
        ranked_actions = [a for a in np.argsort(probs)[::-1] if a != 0]

        # mask wall-blocked actions if provided
        if info and 'blocked_actions' in info:
            ranked_actions = [a for a in ranked_actions if a not in info['blocked_actions']]
        if not ranked_actions:
            ranked_actions = [a for a in range(1, len(probs)) if a not in (info or {}).get('blocked_actions', [])]
        if not ranked_actions:
            ranked_actions = [random.choice([1, 2, 3, 4])]

        if not self.initialized:
            self.last_action = ranked_actions[0]
            self.initialized = True
            return self.last_action

        self.last_action = ranked_actions[0]
        return self.last_action

    # ==============================
    # BC Learning
    # ==============================

    def learn(self):
        if not self.expert_dataset:
            return 0.0

        self.policy_net.train()

        batch = random.sample(
            self.expert_dataset,
            min(len(self.expert_dataset), self.batch_size)
        )

        states = torch.stack([
            torch.as_tensor(s["state"], dtype=torch.float32)
            for s in batch
        ]).to(self.device)

        actions = torch.as_tensor(
            [s["action"] for s in batch],
            dtype=torch.long,
            device=self.device
        )

        logits = self.policy_net(states)
        loss = F.cross_entropy(
            logits, actions, label_smoothing=0.1
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ==============================
    # Save / Load
    # ==============================

    def save(self, name):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(
            self.policy_net.state_dict(),
            os.path.join(self.save_dir, f"{name}.pth")
        )

    def load(self, path):
        if os.path.exists(path):
            self.policy_net.load_state_dict(
                torch.load(path, map_location=self.device)
            )

    def load_expert_dataset(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.expert_dataset = pickle.load(f)