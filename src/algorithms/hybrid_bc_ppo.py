import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import random
from torch.distributions import Categorical

# ==============================
# 1. NETWORK
# ==============================
class HybridPolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.input_channels = input_shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out_size = self.conv(dummy).view(1, -1).size(1)

        self.actor_fc  = nn.Sequential(nn.Linear(out_size, 512), nn.ReLU(), nn.Linear(512, n_actions))
        self.critic_fc = nn.Sequential(nn.Linear(out_size, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[2:] != (128, 128):
            x = F.interpolate(x.float(), size=(128, 128), mode='bilinear', align_corners=False)
        if x.shape[1] == 3 and self.input_channels == 12:
            x = x.repeat(1, 4, 1, 1)
        x = x.float() / 255.0 if (x.dtype == torch.uint8 or x.max() > 1.0) else x.float()
        f = self.conv(x).view(x.size(0), -1)
        return self.actor_fc(f), self.critic_fc(f)


# ==============================
# 2. AGENT
# ==============================
class HybridBCPPOAgent:
    def __init__(self, **kwargs):
        self.state_dim   = kwargs["state_dim"]
        self.action_dim  = kwargs["action_dim"]
        self.device      = kwargs["device"]
        self.save_dir    = kwargs.get("save_dir", "models/hybrid_bc_ppo")
        self.use_heuristics = kwargs.get("use_heuristics", True)

        # Training hyper-params — all read from config via kwargs
        self.ppo_batch_size = kwargs.get('batch_size', 64)
        self.ppo_lr         = kwargs.get('ppo_learning_rate', 2e-5)
        self.bc_lr          = kwargs.get('bc_learning_rate', 1e-4)
        self.clip_eps       = kwargs.get('clip_eps', 0.2)
        self.entropy_coef   = kwargs.get('entropy_coef', 0.02)
        self.lam            = kwargs.get('lam', 0.95)
        self.ppo_epochs     = kwargs.get('ppo_epochs', 4)
        self.gamma          = 0.99

        self.policy_net    = HybridPolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.ppo_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.ppo_lr)
        self.bc_optimizer  = optim.Adam(self.policy_net.parameters(), lr=self.bc_lr)

        self.trajectories   = []
        self.expert_dataset = []
        self.last_action    = 2  # default: UP
        self._pending       = {'log_prob': 0.0, 'value': 0.0}

        if kwargs.get("expert_dataset_path"):
            self.load_expert_dataset(kwargs["expert_dataset_path"])
        os.makedirs(self.save_dir, exist_ok=True)

    # ==============================
    # ACTION SELECTION
    # ==============================
    def act(self, state, info=None):
        """
        BC phase: heuristic only for emergencies (ghost < 25px), else BC policy decides.
        PPO phase: full heuristic priority chain + PPO policy.
        """
        open_dirs = info.get('open_dirs', []) if info else []
        phase = info.get('current_phase', 'ppo') if info else 'ppo'

        # --- GHOST EMERGENCY (both phases) ---
        if self.use_heuristics and info:
            g_act = info.get('ghost_avoidance_action')
            if g_act is not None:
                # During BC phase only intervene for very close ghosts
                ghost_pos = info.get('ghost_positions', [])
                pac_pos   = info.get('pac_pos')
                if phase == 'bc' and pac_pos and ghost_pos:
                    from .heuristic_utils import _distance
                    min_dist = min(_distance(pac_pos, g) for g in ghost_pos)
                    if min_dist > 25:
                        g_act = None  # BC phase: let policy decide if ghost not imminent
                if g_act is not None:
                    self.last_action = int(g_act)
                    self._pending = {'log_prob': np.log(0.99), 'value': 0.0}
                    return self.last_action

        # --- UNSTUCK (both phases) ---
        if info:
            u_act = info.get('unstuck_action')
            if u_act is not None:
                self.last_action = int(u_act)
                self._pending = {'log_prob': np.log(0.99), 'value': 0.0}
                return self.last_action

        # --- PELLET HINT (PPO phase only, 30% rate) ---
        if phase == 'ppo' and self.use_heuristics and info:
            p_act = info.get('pellet_heuristic_action')
            if p_act is not None and np.random.rand() < 0.30:
                self.last_action = int(p_act)
                self._pending = {'log_prob': np.log(0.99), 'value': 0.0}
                return self.last_action

        # --- NEURAL NETWORK ---
        self.policy_net.eval()
        with torch.no_grad():
            st_t = torch.as_tensor(state, dtype=torch.uint8).to(self.device)
            logits, value = self.policy_net(st_t)

            # Mask closed directions + NOOP + FIRE
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            mask = np.zeros(self.action_dim)
            if open_dirs:
                for a in open_dirs:
                    if a < self.action_dim:
                        mask[a] = 1.0
            else:
                mask[:] = 1.0
            mask[0] = 0.0  # never NOOP
            if self.action_dim > 1:
                mask[1] = 0.0  # never FIRE

            probs = probs * mask
            total = probs.sum()
            if total < 1e-6:
                valid = open_dirs if open_dirs else [2, 3, 4, 5]
                action = int(random.choice(valid))
                log_prob = np.log(1.0 / max(len(valid), 1))
            else:
                probs /= total
                action = int(np.random.choice(self.action_dim, p=probs))
                log_prob = float(np.log(probs[action] + 1e-10))

        self.last_action = action
        self._pending = {'log_prob': log_prob, 'value': float(value.item())}
        return action

    # ==============================
    # EXPERIENCE CACHE
    # ==============================
    def cache(self, state, action, reward, next_state, done):
        # Reward shaping for PPO trajectories
        total_reward = float(np.clip(reward, -1.0, 1.0))
        if reward > 0:
            total_reward *= 1.5   # bonus for eating pellets/ghosts
        total_reward -= 0.005    # small time penalty (encourages efficiency)

        st_tensor = torch.as_tensor(state, dtype=torch.uint8)
        self.trajectories.append({
            'state':    st_tensor,
            'action':   int(action),
            'reward':   total_reward,
            'done':     bool(done),
            'value':    self._pending['value'],
            'log_prob': self._pending['log_prob'],
        })

    # ==============================
    # PPO LEARNING
    # ==============================
    def learn_ppo(self):
        # On-policy: only train when episode is complete
        if not self.trajectories or not self.trajectories[-1]['done']:
            return 0.0
        if len(self.trajectories) < self.ppo_batch_size:
            self.trajectories = []
            return 0.0

        self.policy_net.train()

        rewards = [t['reward'] for t in self.trajectories]
        values  = [t['value']  for t in self.trajectories]
        dones   = [t['done']   for t in self.trajectories]

        # GAE
        advantages, gae, next_v = [], 0, 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_v * (1 - dones[i]) - values[i]
            gae   = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_v = values[i]

        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = adv_t + torch.tensor(values, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        indices = np.arange(len(self.trajectories))
        losses  = []

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.ppo_batch_size):
                idx = indices[start:start + self.ppo_batch_size]

                b_states  = torch.stack([self.trajectories[i]['state']    for i in idx]).to(self.device)
                b_actions = torch.tensor([self.trajectories[i]['action']  for i in idx], device=self.device)
                b_old_lp  = torch.tensor([self.trajectories[i]['log_prob'] for i in idx], device=self.device)

                l, v = self.policy_net(b_states)
                dist = Categorical(F.softmax(l, dim=-1))
                new_lp = dist.log_prob(b_actions)

                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[idx]

                p_loss  = -torch.min(surr1, surr2).mean()
                v_loss  = F.mse_loss(v.squeeze(), ret_t[idx])
                entropy = dist.entropy().mean()

                loss = p_loss + 0.5 * v_loss - self.entropy_coef * entropy

                self.ppo_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.ppo_optimizer.step()
                losses.append(loss.item())

        self.trajectories = []
        return float(np.mean(losses)) if losses else 0.0

    # ==============================
    # BC LEARNING
    # ==============================
    def learn_bc(self):
        if not self.expert_dataset:
            return 0.0
        self.policy_net.train()

        batch = random.sample(self.expert_dataset, min(len(self.expert_dataset), self.ppo_batch_size))
        s = torch.stack([torch.as_tensor(x["state"], dtype=torch.uint8) for x in batch]).to(self.device)
        a = torch.tensor([int(x["action"]) for x in batch], dtype=torch.long, device=self.device)

        logits, _ = self.policy_net(s)
        # Label smoothing helps BC not overfit to expert noise
        loss = F.cross_entropy(logits, a, label_smoothing=0.05)

        self.bc_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.bc_optimizer.step()
        return loss.item()

    def learn(self, phase='ppo'):
        return self.learn_bc() if phase == 'bc' else self.learn_ppo()

    # ==============================
    # SAVE / LOAD
    # ==============================
    def save(self, name):
        path = os.path.join(self.save_dir, f"{name}.pth")
        torch.save({
            'policy_net':    self.policy_net.state_dict(),
            'ppo_optimizer': self.ppo_optimizer.state_dict(),
            'bc_optimizer':  self.bc_optimizer.state_dict(),
        }, path)
        print(f"Hybrid saved: {path}")

    def load(self, path):
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(ckpt['policy_net'])
            if 'ppo_optimizer' in ckpt:
                self.ppo_optimizer.load_state_dict(ckpt['ppo_optimizer'])
            if 'bc_optimizer' in ckpt:
                self.bc_optimizer.load_state_dict(ckpt['bc_optimizer'])
            print(f"Hybrid loaded: {path}")

    def load_expert_dataset(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.expert_dataset = pickle.load(f)
            print(f" The dataset is ready: {len(self.expert_dataset)} frames")