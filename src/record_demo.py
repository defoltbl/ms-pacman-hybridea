"""
Video demonstration recorder for Ms. Pac-Man AI.

Records clean gameplay using a trained model and produces
a polished MP4 video with on-screen stats overlay.

Usage:
    python record_demo.py --algorithm DQN --model models/dqn/DQN_best.pth
    python record_demo.py --algorithm HYBRID_BC_PPO --model models/hybrid_bc_ppo/best.pth
    python record_demo.py --algorithm DQN --no-model   # run with heuristics only (no trained model needed)
"""

import argparse
import sys
import os
import time
import inspect
import yaml
import numpy as np
import cv2
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(ROOT_DIR))

from environment.pacman_env import PacmanEnv
from algorithms.heuristic_utils import (
    _find_pacman_position, _detect_ghosts,
    ghost_avoidance_heuristic, get_pellet_heuristic,
    _open_directions, ALE_UP, ALE_DOWN, ALE_LEFT, ALE_RIGHT,
)

# ─────────────────────────────────────────────
# Visual config
# ─────────────────────────────────────────────
SCALE       = 4           # 160×210 → 640×840
PANEL_W     = 300         # right-side stats panel width
FPS         = 30
FONT        = cv2.FONT_HERSHEY_SIMPLEX
CLR_WHITE   = (255, 255, 255)
CLR_YELLOW  = (0, 220, 255)
CLR_GREEN   = (80, 220, 80)
CLR_RED     = (80, 80, 220)
CLR_CYAN    = (220, 220, 80)
CLR_DARK    = (20, 20, 30)
CLR_PANEL   = (15, 15, 25)

ALE_NAMES = {2: 'UP', 3: 'RIGHT', 4: 'LEFT', 5: 'DOWN', 0: 'NOOP', 1: 'FIRE'}


# ─────────────────────────────────────────────
# Draw helpers
# ─────────────────────────────────────────────
def _bar(img, x, y, w, h, pct, color, bg=(50, 50, 60)):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    filled = max(1, int(w * min(pct, 1.0)))
    cv2.rectangle(img, (x, y), (x + filled, y + h), color, -1)

def _text(img, txt, x, y, scale=0.45, color=CLR_WHITE, thick=1):
    cv2.putText(img, txt, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

def _draw_panel(panel, stats: dict):
    """Draw the right-side statistics panel."""
    panel[:] = CLR_PANEL

    # Title
    cv2.rectangle(panel, (0, 0), (PANEL_W, 50), (30, 30, 50), -1)
    _text(panel, "Ms. Pac-Man AI", 10, 22, 0.6, CLR_YELLOW, 2)
    _text(panel, f"Algorithm: {stats['algo']}", 10, 44, 0.42, CLR_WHITE)

    y = 65
    # Lives
    _text(panel, "LIVES", 10, y); y += 18
    for i in range(3):
        col = CLR_GREEN if i < stats['lives'] else (60, 60, 70)
        cv2.circle(panel, (18 + i * 28, y), 10, col, -1)
    y += 28

    # Score
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (50, 50, 70), 1); y += 12
    _text(panel, f"SCORE", 10, y, 0.42, CLR_CYAN)
    _text(panel, f"{stats['score']:,}", 10, y + 18, 0.65, CLR_YELLOW, 2)
    y += 42

    # Pellets
    _text(panel, f"PELLETS  {stats['pellets']}", 10, y, 0.42, CLR_WHITE); y += 18
    _bar(panel, 10, y, PANEL_W - 20, 8, stats['pellets'] / max(stats['total_pellets'], 1),
         CLR_GREEN); y += 20

    # Episode / Step
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (50, 50, 70), 1); y += 12
    _text(panel, f"Episode:  {stats['episode']}", 10, y, 0.42, CLR_WHITE); y += 18
    _text(panel, f"Step:     {stats['step']}", 10, y, 0.42, CLR_WHITE); y += 18
    _text(panel, f"FPS:      {stats['fps']:.0f}", 10, y, 0.42, CLR_WHITE); y += 24

    # Ghosts detected
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (50, 50, 70), 1); y += 12
    _text(panel, f"GHOSTS VISIBLE: {stats['n_ghosts']}", 10, y, 0.42, CLR_RED); y += 22
    for gp in stats.get('ghost_pos', [])[:4]:
        _text(panel, f"  ({gp[0]:3d}, {gp[1]:3d})", 10, y, 0.38, (160, 100, 100)); y += 16
    y = max(y, 380)

    # Current action
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (50, 50, 70), 1); y += 12
    _text(panel, "LAST ACTION", 10, y, 0.42, CLR_CYAN); y += 20
    act_name = ALE_NAMES.get(stats['action'], str(stats['action']))
    act_col  = CLR_GREEN if stats['action_src'] == 'heuristic' else CLR_WHITE
    _text(panel, f"  {act_name}", 10, y, 0.65, act_col, 2); y += 28
    src_col = CLR_RED if stats['action_src'] == 'heuristic' else (120, 120, 180)
    _text(panel, f"  src: {stats['action_src']}", 10, y, 0.38, src_col); y += 22

    # Open directions
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (50, 50, 70), 1); y += 12
    _text(panel, "OPEN DIRS", 10, y, 0.42, CLR_CYAN); y += 20
    dir_str = "  " + " ".join(ALE_NAMES.get(a, str(a)) for a in stats.get('open_dirs', []))
    _text(panel, dir_str if dir_str.strip() else "  none", 10, y, 0.42, CLR_GREEN); y += 24

    # Algorithm description box
    cv2.line(panel, (8, y), (PANEL_W - 8, y), (50, 50, 70), 1); y += 12
    desc_lines = _wrap(stats.get('algo_desc', ''), 36)
    for line in desc_lines[:4]:
        _text(panel, line, 10, y, 0.36, (180, 180, 200)); y += 15

    # Bottom: phase indicator for hybrid
    if stats.get('phase'):
        bottom = panel.shape[0] - 35
        phase_col = CLR_GREEN if stats['phase'] == 'BC' else CLR_CYAN
        cv2.rectangle(panel, (0, bottom - 5), (PANEL_W, panel.shape[0]), (30, 30, 50), -1)
        _text(panel, f"Phase: {stats['phase']}", 10, bottom + 15, 0.5, phase_col, 2)


def _wrap(text, width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur += (" " if cur else "") + w
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines


def _overlay_debug(game_img, raw_screen, pac_pos, ghost_pos, action, open_dirs):
    """Draw detection circles and arrows directly on the game frame."""
    # Pac-Man green circle
    if pac_pos:
        sx, sy = pac_pos[0] * SCALE, pac_pos[1] * SCALE
        cv2.circle(game_img, (sx, sy), 10, (0, 255, 0), 2)

    # Ghost red circles
    for gp in ghost_pos:
        gx, gy = gp[0] * SCALE, gp[1] * SCALE
        cv2.circle(game_img, (gx, gy), 10, (0, 60, 255), 2)

    # Action arrow from Pac-Man
    if pac_pos and action in (ALE_UP, ALE_DOWN, ALE_LEFT, ALE_RIGHT):
        arrow = {ALE_UP:(0,-25), ALE_DOWN:(0,25), ALE_LEFT:(-25,0), ALE_RIGHT:(25,0)}
        dx, dy = arrow[action]
        sx, sy = pac_pos[0] * SCALE, pac_pos[1] * SCALE
        cv2.arrowedLine(game_img, (sx, sy), (sx+dx, sy+dy), (0, 255, 255), 2, tipLength=0.4)


# ─────────────────────────────────────────────
# Agent loader
# ─────────────────────────────────────────────
ALGO_DESCRIPTIONS = {
    'DQN':          'Deep Q-Network with Prioritized Experience Replay. Learns Q-values via Bellman updates.',
    'PPO':          'Proximal Policy Optimization. On-policy actor-critic with clipped surrogate objective.',
    'A2C':          'Advantage Actor-Critic. Synchronous on-policy with value baseline.',
    'MC':           'Monte Carlo Policy Gradient. Full-episode returns with REINFORCE.',
    'BC':           'Behavioral Cloning. Supervised imitation of expert demonstrations.',
    'HYBRID_BC_PPO':'Hybrid BC+PPO. Starts with expert imitation (BC), transitions to self-improvement (PPO).',
}

def load_agent(algorithm, model_path, config, env):
    from algorithms.dqn          import DQNAgent
    from algorithms.ppo          import PPOAgent
    from algorithms.a2c          import A2CAgent
    from algorithms.monte_carlo  import MonteCarloAgent
    from algorithms.bc           import BCAgent
    from algorithms.hybrid_bc_ppo import HybridBCPPOAgent

    CLASSES = {
        'DQN': DQNAgent, 'PPO': PPOAgent, 'A2C': A2CAgent,
        'MC': MonteCarloAgent, 'BC': BCAgent, 'HYBRID_BC_PPO': HybridBCPPOAgent,
    }

    key  = algorithm.lower()
    cfg  = config['algorithms'].get(key, {})
    stack = config['environment'].get('stack_frames', 4)
    gray  = config['environment'].get('gray_scale', False)
    ch    = stack if gray else stack * 3
    sz    = tuple(config['environment'].get('resize_shape', [128, 128]))
    state_dim = (ch, sz[1], sz[0])
    device = torch.device('cpu')

    params = {
        'state_dim': state_dim, 'action_dim': len(env.action_space),
        'save_dir': f'models/{key}', 'device': device,
        'use_heuristics': True, **cfg
    }

    cls = CLASSES[algorithm]
    sig = inspect.signature(cls.__init__)
    has_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    init_p = params if has_kw else {k: v for k, v in params.items() if k in sig.parameters}
    agent = cls(**init_p)

    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        print(f"Model loaded: {model_path}")
    else:
        print("No model loaded — running heuristic-only policy")

    # Set eval mode
    for attr in ['net', 'policy_net', 'actor']:
        m = getattr(agent, attr, None)
        if isinstance(m, torch.nn.Module):
            m.eval()

    return agent


# ─────────────────────────────────────────────
# Main recording loop
# ─────────────────────────────────────────────
def record(args):
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    out_dir = Path('demo_videos')
    out_dir.mkdir(exist_ok=True)
    ts       = time.strftime('%Y%m%d-%H%M%S')
    out_path = str(out_dir / f'{args.algorithm}_{ts}.mp4')

    env = PacmanEnv(
        render_mode=None,
        level=config['environment'].get('level', 'easy'),
        frameskip=config['environment'].get('frameskip', 4),
        gray_scale=config['environment'].get('gray_scale', False),
        stack_frames=config['environment'].get('stack_frames', 4),
        resize_shape=tuple(config['environment'].get('resize_shape', [128, 128])),
    )

    agent = None if args.no_model else load_agent(
        args.algorithm, args.model, config, env
    )

    # Video writer
    game_h = 210 * SCALE  # 840
    game_w = 160 * SCALE  # 640
    total_w = game_w + PANEL_W
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    writer  = cv2.VideoWriter(out_path, fourcc, FPS, (total_w, game_h))
    print(f"Recording → {out_path}  ({total_w}×{game_h} @ {FPS}fps)")

    algo_desc = ALGO_DESCRIPTIONS.get(args.algorithm, '')
    total_score = 0
    best_ep_score = 0

    for episode in range(args.episodes):
        state     = env.reset()
        done      = False
        ep_score  = 0
        ep_step   = 0
        ep_pellets= 0
        last_act  = ALE_RIGHT
        act_src   = 'network'
        frame_times = []
        phase = 'BC' if (args.algorithm == 'HYBRID_BC_PPO' and episode < 50) else 'PPO'

        while not done:
            t0 = time.time()

            raw_screen  = env.ale.getScreenRGB()
            pac_pos     = _find_pacman_position(raw_screen)
            ghost_pos   = env.get_ghost_positions()
            open_dirs   = _open_directions(raw_screen, pac_pos) if pac_pos else []

            info = {
                'ghost_positions':      ghost_pos,
                'ghost_avoidance_action': ghost_avoidance_heuristic(
                    raw_screen, ghost_pos, pacman_pos=pac_pos, open_dirs=open_dirs),
                'pellet_heuristic_action': get_pellet_heuristic(
                    raw_screen, ghost_pos, pacman_pos=pac_pos, open_dirs=open_dirs),
                'unstuck_action':   open_dirs[0] if len(open_dirs) == 1 else None,
                'open_dirs':        open_dirs,
                'pac_pos':          pac_pos,
                'current_phase':    phase.lower(),
                'blocked_actions':  [],
            }

            # Determine action
            g_act = info['ghost_avoidance_action']
            u_act = info['unstuck_action']

            if g_act is not None:
                action   = g_act
                act_src  = 'heuristic'
            elif u_act is not None:
                action   = u_act
                act_src  = 'unstuck'
            elif agent is not None:
                action   = agent.act(state, info)
                act_src  = 'network'
            else:
                # Pure heuristic demo
                p_act = info['pellet_heuristic_action']
                action = p_act if p_act else (last_act if last_act in open_dirs else
                         (open_dirs[0] if open_dirs else ALE_RIGHT))
                act_src = 'heuristic'

            last_act = action

            state, reward, done, step_info = env.step(action)
            ep_score   += reward
            ep_step    += 1
            if reward > 0:
                ep_pellets += 1

            # ── Build frame ──────────────────────────────────────
            game_bgr  = cv2.cvtColor(raw_screen, cv2.COLOR_RGB2BGR)
            game_big  = cv2.resize(game_bgr, (game_w, game_h), interpolation=cv2.INTER_NEAREST)

            # Debug overlay (can be toggled off with --no-overlay)
            if not args.no_overlay:
                _overlay_debug(game_big, raw_screen, pac_pos, ghost_pos, action, open_dirs)

            # Stats panel
            panel = np.zeros((game_h, PANEL_W, 3), dtype=np.uint8)
            fps_val = 1.0 / (time.time() - t0 + 1e-9)
            _draw_panel(panel, {
                'algo':         args.algorithm,
                'algo_desc':    algo_desc,
                'lives':        env.ale.lives(),
                'score':        int(ep_score),
                'pellets':      ep_pellets,
                'total_pellets':250,
                'episode':      episode + 1,
                'step':         ep_step,
                'fps':          fps_val,
                'n_ghosts':     len(ghost_pos),
                'ghost_pos':    ghost_pos,
                'action':       action,
                'action_src':   act_src,
                'open_dirs':    open_dirs,
                'phase':        phase if args.algorithm == 'HYBRID_BC_PPO' else None,
            })

            frame = np.hstack([game_big, panel])
            writer.write(frame)
            frame_times.append(time.time() - t0)

        total_score += ep_score
        best_ep_score = max(best_ep_score, ep_score)
        avg_fps = 1.0 / (np.mean(frame_times) + 1e-9)
        print(f"Ep {episode+1:3d} | Score: {int(ep_score):5d} | "
              f"Steps: {ep_step:4d} | Avg FPS: {avg_fps:.0f}")

    writer.release()
    env.close()

    print(f"\nVideo saved:  {out_path}")
    print(f"   Episodes:     {args.episodes}")
    print(f"   Best score:   {int(best_ep_score)}")
    print(f"   Total score:  {int(total_score)}")
    print()
    print("To convert to H.264 for presentations (requires ffmpeg):")
    print(f"  ffmpeg -i {out_path} -vcodec libx264 -crf 20 demo_final.mp4")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Record Ms. Pac-Man AI demo video')
    parser.add_argument('--algorithm', type=str, default='DQN',
                        choices=['DQN','PPO','A2C','MC','BC','HYBRID_BC_PPO'])
    parser.add_argument('--model',      type=str, default=None,
                        help='Path to .pth model file')
    parser.add_argument('--no-model',   action='store_true',
                        help='Run heuristic-only (no trained model needed)')
    parser.add_argument('--episodes',   type=int, default=3,
                        help='Number of episodes to record')
    parser.add_argument('--config',     type=str, default='configs/pacman_env.yaml')
    parser.add_argument('--no-overlay', action='store_true',
                        help='Disable debug circles/arrows overlay')
    args = parser.parse_args()

    record(args)