from typing import Dict, Any, Set, Tuple
import numpy as np

# ==========================================
# Level difficulty configuration
# ==========================================
LEVELS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "ghost_speed": 0.5,          # maps to repeat_action_probability=0.0 (crisp control)
        "pellet_reward": 10,         # bonus reward on top of ALE's native 10
        "power_pellet_duration": 30, # used to give bonus reward when ghost eaten
        "lives": 3,                  # starting lives (written to ALE RAM on reset)
        "death_penalty": 50.0,       # reward penalty per life lost
    },
    "medium": {
        "ghost_speed": 0.75,
        "pellet_reward": 15,
        "power_pellet_duration": 20,
        "lives": 2,
        "death_penalty": 75.0,
    },
    "hard": {
        "ghost_speed": 1.0,          # maps to repeat_action_probability=0.25 (sticky actions)
        "pellet_reward": 20,
        "power_pellet_duration": 15,
        "lives": 1,
        "death_penalty": 100.0,
    }
}

# ==========================================
# MsPacman Maze 1 layout
# '#' = wall, '.' = corridor/pellet, ' ' = outside playfield
# This is used to build a reliable corridor map for heuristics,
# replacing unreliable pixel-color corridor detection.
#
# Grid: 21 cols x 13 rows
# Maps to raw ALE screen: x=0-159, y=16-187 (160x172 playfield)
# Cell size: ~7.6px wide x ~13.2px tall
# ==========================================
MAZE_LAYOUT = [
    "#####################",
    "#........#..........#",
    "#.##.###.#.###.##..#",
    "#..................#",
    "#.##.#.###.#.##..#.#",
    "#....#...#.#....#..#",
    "#####.###.##########",
    "#....#...#.#........",
    "#####.###.##########",
    "#...........#.......#",
    "#.##.###.##.#.......#",
    "#.....#.....#.......#",
    "#####################",
]

# Playfield bounds on raw 210x160 ALE screen
_PLAY_Y_START = 16
_PLAY_Y_END   = 188
_PLAY_X_START = 0
_PLAY_X_END   = 160

def build_corridor_map() -> Set[Tuple[int, int]]:
    """
    Build a set of (x, y) pixel positions that are open corridors,
    derived from MAZE_LAYOUT. Much more reliable than pixel color sampling.

    Returns a set of raw screen pixel coords that are walkable.
    Each maze cell maps to a cluster of pixels — we store the center.
    """
    rows = len(MAZE_LAYOUT)
    cols = max(len(r) for r in MAZE_LAYOUT)

    play_h = _PLAY_Y_END - _PLAY_Y_START  # 172
    play_w = _PLAY_X_END - _PLAY_X_START  # 160

    cell_h = play_h / rows   # ~13.2
    cell_w = play_w / cols   # ~7.6

    open_pixels: Set[Tuple[int, int]] = set()
    for row_i, row in enumerate(MAZE_LAYOUT):
        for col_i, ch in enumerate(row):
            if ch != '#':
                # Store center pixel of this cell
                px = int(_PLAY_X_START + col_i * cell_w + cell_w / 2)
                py = int(_PLAY_Y_START + row_i * cell_h + cell_h / 2)
                # Also mark a small radius around center as open
                for dy in range(-4, 5):
                    for dx in range(-3, 4):
                        nx = px + dx
                        ny = py + dy
                        if _PLAY_X_START <= nx < _PLAY_X_END and _PLAY_Y_START <= ny < _PLAY_Y_END:
                            open_pixels.add((nx, ny))
    return open_pixels


def get_repeat_action_probability(level: str) -> float:
    """
    Map ghost_speed to ALE repeat_action_probability.
    Higher value = stickier actions = harder to control = effectively harder.
    Easy: 0.0 (perfect control), Hard: 0.25 (standard Atari sticky actions)
    """
    speed = LEVELS.get(level, LEVELS["easy"])["ghost_speed"]
    # Linear interpolation: 0.5 -> 0.0, 1.0 -> 0.25
    return round((speed - 0.5) * 0.5, 3)


# Pre-build the corridor map once at import time (fast, ~1ms)
CORRIDOR_MAP: Set[Tuple[int, int]] = build_corridor_map()