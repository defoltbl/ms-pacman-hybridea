"""
Presentation video maker for Ms. Pac-Man AI.
Records all algorithms and stitches into one comparison video.

Usage:
    python make_demo_video.py                          # record all algorithms
    python make_demo_video.py --algorithm DQN          # record one algorithm
    python make_demo_video.py --algorithm HYBRID_BC_PPO --model models/hybrid_bc_ppo/best.pth
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
    _find_pacman_position,
    ghost_avoidance_heuristic,
    get_pellet_heuristic,
    _open_directions,
    ALE_UP, ALE_DOWN, ALE_LEFT, ALE_RIGHT,
)

# ── Video constants ────────────────────────────────────────────────────────────
GAME_SCALE  = 4          # 160×210 → 640×840
PANEL_W     = 280
FPS_OUT     = 30
FONT        = cv2.FONT_HERSHEY_SIMPLEX

C_WHITE  = (255, 255, 255)
C_YELLOW = (30, 215, 255)
C_GREEN  = (60, 210, 60)
C_RED    = (60, 60, 220)
C_CYAN   = (210, 210, 40)
C_PANEL  = (16, 16, 28)
C_CARD   = (28, 28, 44)
C_LINE   = (55, 55, 75)
C_ORANGE = (40, 150, 255)

ALE_NAMES = {2:'UP', 3:'RIGHT', 4:'LEFT', 5:'DOWN', 0:'NOOP', 1:'FIRE'}

ALGO_META = {
    'DQN': {
        'color': (60, 180, 255),
        'short': 'DQN',
        'desc': ['Deep Q-Network', 'Prioritized Experience', 'Replay Buffer'],
    },
    'PPO': {
        'color': (60, 220, 120),
        'short': 'PPO',
        'desc': ['Proximal Policy', 'Optimization', 'On-policy Actor-Critic'],
    },
    'A2C': {
        'color': (200, 160, 60),
        'short': 'A2C',
        'desc': ['Advantage', 'Actor-Critic', 'Synchronous'],
    },
    'MC': {
        'color': (180, 60, 200),
        'short': 'MC',
        'desc': ['Monte Carlo', 'Policy Gradient', 'Full-episode Returns'],
    },
    'BC': {
        'color': (60, 200, 200),
        'short': 'BC',
        'desc': ['Behavioral Cloning', 'Imitation Learning', 'Expert Demonstrations'],
    },
    'HYBRID_BC_PPO': {
        'color': (30, 215, 255),
        'short': 'HYBRID',
        'desc': ['BC → PPO Hybrid', 'Expert Init + Self-', 'Improvement via PPO'],
    },
}


# ── Drawing helpers ────────────────────────────────────────────────────────────
def txt(img, text, x, y, scale=0.45, color=C_WHITE, thick=1):
    cv2.putText(img, str(text), (x, y), FONT, scale, color, thick, cv2.LINE_AA)

def bar(img, x, y, w, h, pct, color, bg=(45, 45, 60)):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    fill = max(1, int(w * min(max(pct, 0), 1)))
    cv2.rectangle(img, (x, y), (x+fill, y+h), color, -1)

def hline(img, y, x0=8, color=C_LINE):
    cv2.line(img, (x0, y), (PANEL_W-x0, y), color, 1)

def _draw_panel(panel, s):
    """Draw full stats panel. s = stats dict."""
    panel[:] = C_PANEL
    algo_color = ALGO_META.get(s['algo'], {}).get('color', C_WHITE)

    # ── Header ──
    cv2.rectangle(panel, (0, 0), (PANEL_W, 56), C_CARD, -1)
    cv2.rectangle(panel, (0, 0), (4, 56), algo_color, -1)
    txt(panel, 'Ms. Pac-Man  AI', 12, 22, 0.58, C_YELLOW, 2)
    short = ALGO_META.get(s['algo'], {}).get('short', s['algo'])
    txt(panel, short, 12, 44, 0.48, algo_color, 1)

    y = 64

    # ── Lives ──
    txt(panel, 'LIVES', 10, y, 0.38, C_CYAN)
    for i in range(3):
        c = C_GREEN if i < s['lives'] else (55, 55, 70)
        cx = 70 + i * 26
        cv2.circle(panel, (cx, y-5), 9, c, -1)
    y += 22

    # ── Score ──
    hline(panel, y); y += 14
    txt(panel, 'SCORE', 10, y, 0.38, C_CYAN)
    txt(panel, f"{int(s['score']):,}", 10, y+18, 0.70, C_YELLOW, 2)
    y += 42

    # ── Best score ──
    txt(panel, f"Best: {int(s['best_score']):,}", 10, y, 0.38, (160,160,200)); y += 20

    # ── Pellets ──
    hline(panel, y); y += 14
    txt(panel, f"PELLETS  {s['pellets']}", 10, y, 0.38, C_WHITE); y += 14
    bar(panel, 10, y, PANEL_W-20, 7, s['pellets']/max(s['total_pellets'],1), C_GREEN); y += 18

    # ── Episode / Step ──
    hline(panel, y); y += 14
    txt(panel, f"Episode  {s['episode']}", 10, y, 0.38, C_WHITE); y += 17
    txt(panel, f"Step     {s['step']}", 10, y, 0.38, C_WHITE); y += 17
    txt(panel, f"FPS      {s['fps']:.0f}", 10, y, 0.38, C_WHITE); y += 22

    # ── Ghost status ──
    hline(panel, y); y += 14
    gc = C_RED if s['n_ghosts'] > 0 else (80,80,80)
    txt(panel, f"GHOSTS: {s['n_ghosts']}", 10, y, 0.42, gc, 1); y += 22

    # Ghost danger indicator
    if s.get('ghost_danger'):
        cv2.rectangle(panel, (10, y), (PANEL_W-10, y+18), (50,20,20), -1)
        txt(panel, '  ⚠ GHOST NEARBY', 12, y+13, 0.38, C_RED); y += 24
    else:
        y += 4

    # ── Action ──
    hline(panel, y); y += 14
    txt(panel, 'ACTION', 10, y, 0.38, C_CYAN); y += 20
    act_name = ALE_NAMES.get(s['action'], '?')
    act_col  = C_RED if s['act_src'] == 'heuristic' else \
               C_ORANGE if s['act_src'] == 'unstuck' else C_GREEN
    txt(panel, f"  {act_name}", 14, y, 0.65, act_col, 2); y += 26
    txt(panel, f"  src: {s['act_src']}", 14, y, 0.36, act_col); y += 18

    # ── Open directions ──
    hline(panel, y); y += 12
    open_str = ' '.join(ALE_NAMES.get(a,'?') for a in s.get('open_dirs',[]))
    txt(panel, f"Open: {open_str or 'none'}", 10, y, 0.38, C_GREEN); y += 20

    # ── Algorithm description ──
    desc_y = panel.shape[0] - 100
    hline(panel, desc_y); desc_y += 12
    txt(panel, 'ABOUT', 10, desc_y, 0.38, C_CYAN); desc_y += 16
    for line in ALGO_META.get(s['algo'], {}).get('desc', []):
        txt(panel, line, 10, desc_y, 0.38, (170,170,200)); desc_y += 15

    # ── Phase badge (Hybrid only) ──
    if s.get('phase'):
        bh = panel.shape[0]
        cv2.rectangle(panel, (0, bh-28), (PANEL_W, bh), C_CARD, -1)
        pc = C_ORANGE if s['phase']=='BC' else C_CYAN
        txt(panel, f"Phase: {s['phase']}", 10, bh-10, 0.48, pc, 2)


def _overlay_game(frame, pac_pos, ghost_pos, action, open_dirs, show_overlay):
    if not show_overlay:
        return
    scale = GAME_SCALE
    if pac_pos:
        px, py = pac_pos[0]*scale, pac_pos[1]*scale
        cv2.circle(frame, (px, py), 12, (0,255,0), 2)
        if action in (ALE_UP, ALE_DOWN, ALE_LEFT, ALE_RIGHT):
            dmap = {ALE_UP:(0,-28), ALE_DOWN:(0,28), ALE_LEFT:(-28,0), ALE_RIGHT:(28,0)}
            dx, dy = dmap[action]
            cv2.arrowedLine(frame,(px,py),(px+dx,py+dy),(0,255,255),2,tipLength=0.4)
    for gp in ghost_pos:
        cv2.circle(frame, (gp[0]*scale, gp[1]*scale), 12, (0,60,255), 2)


# ── Intro / outro frames ───────────────────────────────────────────────────────
def _make_title_frames(algo, n_frames=60):
    """60 frames (~2 sec) title card."""
    W = 160*GAME_SCALE + PANEL_W
    H = 210*GAME_SCALE
    frames = []
    color = ALGO_META.get(algo, {}).get('color', C_WHITE)
    for i in range(n_frames):
        alpha = min(1.0, i / 20)
        f = np.zeros((H, W, 3), dtype=np.uint8)
        f[:] = (int(8*alpha), int(8*alpha), int(16*alpha))
        cv2.rectangle(f, (0, H//2-80), (W, H//2+80), C_CARD, -1)
        cv2.rectangle(f, (0, H//2-80), (6, H//2+80), color, -1)
        scale_t = 0.8 + 0.3*(1-alpha)
        title = ALGO_META.get(algo,{}).get('short', algo)
        tsz   = cv2.getTextSize(title, FONT, 1.4, 3)[0]
        tx    = (W - tsz[0])//2
        cv2.putText(f, title, (tx, H//2-10), FONT, 1.4, color, 3, cv2.LINE_AA)
        desc_lines = ALGO_META.get(algo,{}).get('desc',[])
        for di, dl in enumerate(desc_lines):
            dsz = cv2.getTextSize(dl, FONT, 0.6, 1)[0]
            dx  = (W - dsz[0])//2
            cv2.putText(f, dl, (dx, H//2+30+di*26), FONT, 0.6, C_WHITE, 1, cv2.LINE_AA)
        frames.append(f)
    return frames


def _make_score_frames(algo, score, best, n_frames=45):
    """45 frames score summary card."""
    W = 160*GAME_SCALE + PANEL_W
    H = 210*GAME_SCALE
    color = ALGO_META.get(algo, {}).get('color', C_WHITE)
    f = np.zeros((H, W, 3), dtype=np.uint8)
    f[:] = C_PANEL
    cv2.rectangle(f, (W//2-200, H//2-100), (W//2+200, H//2+100), C_CARD, -1)
    cv2.rectangle(f, (W//2-200, H//2-100), (W//2-194, H//2+100), color, -1)
    cv2.putText(f, 'EPISODE COMPLETE', (W//2-160, H//2-60),
                FONT, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(f, f'Score: {int(score):,}', (W//2-150, H//2),
                FONT, 1.0, C_YELLOW, 2, cv2.LINE_AA)
    cv2.putText(f, f'Best:  {int(best):,}', (W//2-150, H//2+45),
                FONT, 0.7, C_WHITE, 1, cv2.LINE_AA)
    return [f] * n_frames


# ── Agent loader ───────────────────────────────────────────────────────────────
def load_agent(algorithm, model_path, config, env):
    from algorithms.dqn           import DQNAgent
    from algorithms.ppo           import PPOAgent
    from algorithms.a2c           import A2CAgent
    from algorithms.monte_carlo   import MonteCarloAgent
    from algorithms.bc            import BCAgent
    from algorithms.hybrid_bc_ppo import HybridBCPPOAgent
    CLASSES = {'DQN':DQNAgent,'PPO':PPOAgent,'A2C':A2CAgent,
               'MC':MonteCarloAgent,'BC':BCAgent,'HYBRID_BC_PPO':HybridBCPPOAgent}
    key   = algorithm.lower()
    cfg   = config['algorithms'].get(key, {})
    stack = config['environment'].get('stack_frames', 4)
    gray  = config['environment'].get('gray_scale', False)
    ch    = stack if gray else stack*3
    sz    = tuple(config['environment'].get('resize_shape', [128,128]))
    state_dim = (ch, sz[1], sz[0])
    device    = torch.device('cpu')
    params = {'state_dim':state_dim,'action_dim':len(env.action_space),
              'save_dir':f'models/{key}','device':device,
              'use_heuristics':True, **cfg}
    cls  = CLASSES[algorithm]
    sig  = inspect.signature(cls.__init__)
    haskw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    init_p = params if haskw else {k:v for k,v in params.items() if k in sig.parameters}
    agent  = cls(**init_p)
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        print(f"  ✅ Model loaded: {model_path}")
    else:
        print(f"  ⚠️  No model — heuristic only")
    for attr in ['net','policy_net','actor']:
        m = getattr(agent, attr, None)
        if isinstance(m, torch.nn.Module): m.eval()
    return agent


# ── Record one algorithm ───────────────────────────────────────────────────────
def record_algo(algo, model_path, config, episodes, show_overlay, writer):
    print(f"\n{'─'*50}")
    print(f"  Recording: {algo}  ({episodes} episodes)")

    env = PacmanEnv(
        render_mode=None,
        level=config['environment'].get('level','easy'),
        frameskip=config['environment'].get('frameskip',4),
        gray_scale=config['environment'].get('gray_scale',False),
        stack_frames=config['environment'].get('stack_frames',4),
        resize_shape=tuple(config['environment'].get('resize_shape',[128,128])),
    )
    agent = load_agent(algo, model_path, config, env)

    # Title card
    for f in _make_title_frames(algo):
        writer.write(f)

    best_score = 0
    GH = 210*GAME_SCALE
    GW = 160*GAME_SCALE

    for ep in range(episodes):
        state     = env.reset()
        done      = False
        ep_score  = 0
        ep_step   = 0
        ep_pellets= 0
        last_act  = ALE_RIGHT
        phase = 'BC' if (algo == 'HYBRID_BC_PPO' and ep < 50) else 'PPO'
        frame_times = []

        while not done:
            t0 = time.time()

            raw   = env.ale.getScreenRGB()
            pac   = _find_pacman_position(raw)
            ghosts= env.get_ghost_positions()
            odirs = _open_directions(raw, pac) if pac else []

            g_act = ghost_avoidance_heuristic(raw, ghosts, pacman_pos=pac, open_dirs=odirs)
            p_act = get_pellet_heuristic(raw, ghosts, pacman_pos=pac, open_dirs=odirs)
            u_act = odirs[0] if len(odirs)==1 else None

            info = {
                'ghost_positions': ghosts,
                'ghost_avoidance_action': g_act,
                'pellet_heuristic_action': p_act,
                'unstuck_action': u_act,
                'open_dirs': odirs,
                'pac_pos': pac,
                'current_phase': phase.lower(),
                'blocked_actions': [],
            }

            if g_act is not None:
                action, act_src = g_act, 'heuristic'
            elif u_act is not None:
                action, act_src = u_act, 'unstuck'
            else:
                action  = agent.act(state, info)
                act_src = 'network'

            last_act = action
            state, reward, done, _ = env.step(action)
            ep_score   += reward
            ep_step    += 1
            if reward > 0: ep_pellets += 1

            # ── Build frame ──
            bgr  = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            game = cv2.resize(bgr, (GW, GH), interpolation=cv2.INTER_NEAREST)
            _overlay_game(game, pac, ghosts, action, odirs, show_overlay)

            panel = np.zeros((GH, PANEL_W, 3), np.uint8)
            fps_v = 1.0/(time.time()-t0+1e-9)
            _draw_panel(panel, {
                'algo': algo, 'lives': env.ale.lives(),
                'score': ep_score, 'best_score': best_score,
                'pellets': ep_pellets, 'total_pellets': 250,
                'episode': ep+1, 'step': ep_step, 'fps': fps_v,
                'n_ghosts': len(ghosts), 'ghost_pos': ghosts,
                'ghost_danger': g_act is not None,
                'action': action, 'act_src': act_src,
                'open_dirs': odirs,
                'phase': phase if algo=='HYBRID_BC_PPO' else None,
            })

            frame = np.hstack([game, panel])
            writer.write(frame)
            frame_times.append(time.time()-t0)

        best_score = max(best_score, ep_score)
        avg_fps = 1.0/(np.mean(frame_times)+1e-9)
        print(f"    Ep {ep+1} | Score {int(ep_score):5,} | "
              f"Steps {ep_step:4d} | FPS {avg_fps:.0f}")

        # Score card between episodes
        for f in _make_score_frames(algo, ep_score, best_score):
            writer.write(f)

    env.close()
    return best_score


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default=None,
                        help='Single algorithm to record (default: all)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to .pth model file')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Episodes per algorithm')
    parser.add_argument('--config', type=str, default='configs/pacman_env.yaml')
    parser.add_argument('--no-overlay', action='store_true',
                        help='Disable debug overlay circles/arrows')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (default: demo_videos/demo_<ts>.mp4)')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    algorithms = (
        [args.algorithm]
        if args.algorithm
        else ['DQN','PPO','A2C','MC','BC','HYBRID_BC_PPO']
    )

    out_dir = Path('demo_videos')
    out_dir.mkdir(exist_ok=True)
    ts       = time.strftime('%Y%m%d-%H%M%S')
    out_path = args.output or str(out_dir / f'demo_{ts}.mp4')

    W = 160*GAME_SCALE + PANEL_W
    H = 210*GAME_SCALE
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, FPS_OUT, (W, H))
    print(f"📹  Output → {out_path}  ({W}×{H} @ {FPS_OUT}fps)")

    model_map = {}
    if args.model and args.algorithm:
        model_map[args.algorithm] = args.model
    else:
        # Auto-detect latest model for each algorithm
        for algo in algorithms:
            key = algo.lower()
            mdir = Path(f'models/{key}')
            if mdir.exists():
                pths = sorted(mdir.rglob('*.pth'))
                if pths:
                    model_map[algo] = str(pths[-1])

    results = {}
    for algo in algorithms:
        model_path = model_map.get(algo)
        best = record_algo(algo, model_path, config,
                           args.episodes, not args.no_overlay, writer)
        results[algo] = best

    writer.release()

    print(f"\n{'═'*50}")
    print(f"Video saved: {out_path}")
    print(f"\n   Algorithm results (best episode score):")
    for algo, score in sorted(results.items(), key=lambda x: -x[1]):
        bar_w = int(score / max(results.values()) * 30) if results.values() else 0
        print(f"   {algo:18s} {'█'*bar_w} {int(score):,}")
    print()
    print("   To convert to H.264 (smaller file, better compatibility):")
    print(f"   ffmpeg -i {out_path} -vcodec libx264 -crf 18 demo_final.mp4")


if __name__ == '__main__':
    main()