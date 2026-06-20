# HYBRIDEA — Hybrid RL/IL Game AI Agent

![Platform](https://img.shields.io/badge/platform-macOS%20%28Apple%20Silicon%29-000000?logo=apple&logoColor=white)
![Language](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Environment](https://img.shields.io/badge/env-Gymnasium%20%2F%20ALE-blueviolet)
![Acceleration](https://img.shields.io/badge/acceleration-Apple%20M4%20MPS-lightgrey)
![Conferences](https://img.shields.io/badge/presented%20at-2%20international%20conferences-success)

A research project exploring reinforcement learning, imitation learning, and a custom hybrid agent (**HYBRIDEA**) for learning level-completion strategies in **Ms. Pac-Man**, built as part of a master's qualification thesis on intelligent NPC behavior in video games.

---

## Overview

This project investigates how different learning paradigms — reinforcement learning (RL), imitation learning (IL), and a hybrid of the two — perform when training an agent to play Ms. Pac-Man. Six algorithms are implemented and benchmarked under identical conditions: **DQN**, **PPO**, **A2C**, **Monte Carlo**, **Behavioral Cloning (BC)**, and the project's main contribution, **HYBRIDEA** — a hybrid BC+PPO agent.

HYBRIDEA addresses a core limitation of pure RL (slow, unstable learning from a "cold start") by first imitating expert gameplay through Behavioral Cloning, then refining that policy through Proximal Policy Optimization once the agent has a solid baseline. The result is an agent that converges faster and ultimately outperforms every individual method it's built from.

## Results

All algorithms were trained for 500 episodes and evaluated under the same conditions. Across the board, HYBRIDEA achieved the best results:

| Algorithm | Mean Reward | Max Reward | Result |
|---|---|---|---|
| **HYBRIDEA (BC + PPO)** | **1658.84** | **3890** | Best overall |
| DQN | 1535.98 | 3720 | High |
| Monte Carlo | 1468.26 | 3140 | Stable, moderate |
| A2C | 1261.22 | 2190 | Moderate |
| PPO | 1253.50 | 3700 | High variance |
| BC | 1238.00 | 2690 | Baseline (expert imitation) |

HYBRIDEA's mean reward beat DQN (the strongest individual algorithm) by roughly 8%, and beat plain PPO by roughly 32%, while also reaching the highest peak score of any agent tested — evidence that the hybrid approach can exceed the quality of the expert demonstrations it was bootstrapped from.

## How HYBRIDEA Works

HYBRIDEA trains in two sequential phases on a shared Actor-Critic network:

1. **Behavioral Cloning phase (episodes 1–50)** — the agent learns from a dataset of 30,000+ expert game-state/action pairs using a cross-entropy loss, giving it safe baseline navigation behavior immediately, without the random, error-prone exploration typical of RL from scratch.
2. **PPO phase (episodes 51–300+)** — the agent continues training through direct interaction with the environment, using PPO's clipped objective and advantage estimation to improve on the expert's strategy rather than just imitate it.

The shared network (`HybridPolicyNetwork`) uses a Conv2D-based convolutional feature extractor feeding two heads: an **Actor** head (action logits) and a **Critic** head (state-value estimate). Game frames are normalized, resized to 128×128, and stacked into a 12-channel input (4 RGB frames) to give the model temporal/motion context. A lightweight heuristic layer (ghost-avoidance and stuck-state detection) helps stabilize behavior during both training and evaluation.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Deep learning | PyTorch |
| RL environment | Gymnasium / Arcade Learning Environment (ALE), included locally under `Arcade-Learning-Environment/` |
| Game | Ms. Pac-Man (Atari, `mspacman.bin` ROM) |
| Hardware acceleration | Apple M4 GPU via Metal Performance Shaders (MPS) — falls back cleanly to CPU |
| Data persistence | Pickle (expert dataset), `.pth` (trained model checkpoints) |

## Project Structure

```
.
├── Arcade-Learning-Environment/   # ALE emulator backend (external dependency)
├── configs/
│   └── pacman_env.yaml            # reward logic, maze/map config, game speed
├── data/
│   ├── expert_dataset.pkl         # serialized expert state-action pairs
│   └── preview_pkl.py             # utility to inspect/preview the dataset
├── datasets/                      # additional/raw dataset assets
├── demo_videos/                   # recorded gameplay demos
├── logs/                          # training run logs
├── models/                        # trained model checkpoints (.pth)
├── plots/                         # auto-generated loss/reward curves
├── results/
│   └── eval/                      # evaluation reports and performance metrics
├── src/
│   ├── algorithms/
│   │   ├── a2c.py                 # Advantage Actor-Critic
│   │   ├── bc.py                  # Behavioral Cloning
│   │   ├── dqn.py                 # Deep Q-Network (with experience replay)
│   │   ├── heuristic_utils.py     # wall-avoidance, ghost-avoidance, pathing heuristics
│   │   ├── hybrid_bc_ppo.py       # HYBRIDEA: BC pretraining → PPO fine-tuning
│   │   ├── monte_carlo.py         # Monte Carlo agent
│   │   └── ppo.py                 # Proximal Policy Optimization
│   ├── data/
│   │   ├── collect_expert_data.py # records expert gameplay trajectories for BC
│   │   └── generate_datasets.py   # builds training datasets
│   ├── diagrams/
│   │   ├── a2c_d.py               # architecture diagram generator — A2C
│   │   ├── bc_d.py                # architecture diagram generator — BC
│   │   ├── dqn_d.py               # architecture diagram generator — DQN
│   │   ├── monte_carlo_d.py       # architecture diagram generator — Monte Carlo
│   │   ├── ppo_d.py               # architecture diagram generator — PPO
│   │   ├── diagram.py             # shared diagram-rendering utilities
│   │   └── result/                # generated diagram output
│   ├── environment/
│   │   ├── roms/
│   │   │   └── mspacman.bin       # Atari ROM used by ALE
│   │   ├── level_config.py        # maze layout and key object coordinates
│   │   └── pacman_env.py          # Gymnasium-compliant env (reset(), step())
│   ├── utils/
│   │   ├── plotting.py            # shared plotting helpers for training/eval graphs
│   │   └── system_monitor.py      # tracks CPU / RAM / GPU load during training
│   ├── main.py                    # entry point — CLI flag selects algorithm and device
│   ├── make_demo_video.py         # renders a trained agent's playthrough to video
│   └── record_demo.py             # records a live/manual playthrough
├── temp/                          # scratch/working files
├── commands.txt                   # reference list of common run commands
└── requirements.txt                # Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.10+
- macOS with Apple Silicon recommended for MPS acceleration (CPU also supported)
- `pip`

### Installation

```bash
git clone https://github.com/AndreyMaksimenko/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Training an agent

```bash
python src/main.py --algo hybrid    # train HYBRIDEA (BC + PPO)
python src/main.py --algo ppo       # train PPO
python src/main.py --algo dqn       # train DQN
```

### Collecting expert data (for BC / HYBRIDEA)

```bash
python src/data/collect_expert_data.py
```

### Recording a demo

```bash
python src/record_demo.py          # record a manual/live playthrough
python src/make_demo_video.py      # render a trained agent's playthrough to video
```

## Evaluation Metrics

Agents are compared using:
- **Mean Reward** — average score across evaluation episodes (overall effectiveness)
- **Max Reward** — best single-episode score (peak performance ceiling)
- **Reward Distribution** — histogram of episode outcomes (consistency of the learned strategy)
- **Step Consistency** — variation in episode length (resistance to early failure)
- **Score Variance** — spread of results around the mean (predictability vs. risk)

## Experimental Setup

All training and evaluation was run on a Mac mini with an Apple M4 chip (10-core CPU, integrated GPU, 16 GB unified memory), using PyTorch's MPS backend to accelerate convolution and tensor operations. Each model took roughly 2–3 hours to train for 500 episodes.

## Conference Presentations

Results from this work were presented at two international academic conferences:
- Odesa, Ukraine — 2025
- Sofia, Bulgaria — 2026

## Thesis Context

This repository implements the practical component of a master's qualification thesis: *"Development and Research of a Methodology for Creating Intelligent Agents for Forming Strategies for Passing Levels in Video Games"* — Odesа Polytechnic National University, specialty 122 Computer Science.

## Developed by

**Maksymenko Andrii**
