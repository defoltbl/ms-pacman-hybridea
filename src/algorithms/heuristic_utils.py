import cv2
import numpy as np

# ============================================================
# MsPacman ALE action space (getLegalActionSet()):
#   0=NOOP, 1=FIRE(unused), 2=UP, 3=RIGHT, 4=LEFT, 5=DOWN
#
# Heuristic functions MUST return ALE action indices directly.
# Previous bug: returned 1=UP,2=RIGHT,3=LEFT,4=DOWN which mapped to
# FIRE,UP,RIGHT,LEFT — completely wrong directions.
# ============================================================

# ALE action constants for MsPacman
ALE_NOOP  = 0
ALE_UP    = 2
ALE_RIGHT = 3
ALE_LEFT  = 4
ALE_DOWN  = 5

def _preprocess_state(state):
    """Convert CHW stacked NN state to HWC RGB."""
    if state.ndim == 3 and state.shape[0] > 3:
        if state.shape[0] == 12:
            state = state[-3:, :, :]
        elif state.shape[0] == 4:
            return cv2.cvtColor(state[-1], cv2.COLOR_GRAY2RGB)
    if state.ndim == 3:
        if state.shape[0] == 3:
            return np.transpose(state, (1, 2, 0))
        elif state.shape[0] == 1:
            return cv2.cvtColor(state[0], cv2.COLOR_GRAY2RGB)
        elif state.shape[-1] == 3:
            return state
    if state.ndim == 2:
        return cv2.cvtColor(state, cv2.COLOR_GRAY2RGB)
    raise ValueError(f"Unsupported state shape: {state.shape}")

def _distance(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

def maze_distance_heuristic(a, b, maze_layout=None):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def _solidity(cnt):
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    return cv2.contourArea(cnt) / hull_area if hull_area > 0 else 0

def _is_corridor_map(x: int, y: int) -> bool:
    """
    Check if (x, y) is an open corridor using the precomputed CORRIDOR_MAP
    from level_config.py. Far more reliable than pixel color sampling.
    """
    try:
        from environment.level_config import CORRIDOR_MAP
        return (int(x), int(y)) in CORRIDOR_MAP
    except ImportError:
        try:
            from level_config import CORRIDOR_MAP
            return (int(x), int(y)) in CORRIDOR_MAP
        except ImportError:
            # Fallback to pixel sampling if import fails
            return False

def _is_corridor(screen_rgb, x, y):
    """Pixel-color corridor check — fallback only."""
    h_img, w_img = screen_rgb.shape[:2]
    x = int(np.clip(x, 0, w_img - 1))
    y = int(np.clip(y, 0, h_img - 1))
    px = screen_rgb[y, x].astype(np.float32)
    px_hsv = cv2.cvtColor(np.uint8([[px]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = int(px_hsv[0]), int(px_hsv[1]), int(px_hsv[2])
    return 108 <= h <= 130 and s > 100 and v > 60

def _open_directions(screen_rgb, pacman_pos, step=12):
    """
    Return list of ALE actions whose direction is an open corridor.
    Uses CORRIDOR_MAP from level_config for reliability.
    Falls back to pixel sampling if map unavailable.
    """
    if pacman_pos is None:
        return []
    x, y = pacman_pos
    checks = [
        (ALE_UP,    x,        y - step),
        (ALE_DOWN,  x,        y + step),
        (ALE_LEFT,  x - step, y       ),
        (ALE_RIGHT, x + step, y       ),
    ]
    # Try CORRIDOR_MAP first (reliable), then pixel fallback
    result = [action for action, tx, ty in checks if _is_corridor_map(tx, ty)]
    if not result:
        # Fallback: pixel-based check
        result = [action for action, tx, ty in checks if _is_corridor(screen_rgb, tx, ty)]
    return result

def _find_pacman_position(screen):
    """
    Find Pac-Man on a RAW ALE screen (H=210, W=160, HWC).
    Excludes top 16px (score) and bottom 22px (lives).
    Returns (x, y) in raw screen coords or None.
    """
    try:
        if screen.ndim == 3 and screen.shape[0] in (3, 12, 4):
            rgb = _preprocess_state(screen)
        else:
            rgb = screen

        h, w = rgb.shape[:2]
        y_min, y_max = 16, h - 22
        play = rgb[y_min:y_max, :]

        hsv = cv2.cvtColor(play.astype(np.uint8), cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([15, 75, 140]), np.array([30, 220, 255]))

        scale = (h * w) / (210 * 160)
        min_area = 15 * scale
        max_area = 90 * scale

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if min_area < a < max_area and a > best_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    best = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]) + y_min)
                    best_area = a
        return best
    except Exception:
        return None

def _detect_pellets(screen_rgb):
    return _detect_pellets_raw(screen_rgb)

def _detect_pellets_raw(screen_rgb):
    """
    Detect food pellets on raw ALE screen (160x210).
    Pellets are SINGLE PIXELS on the raw screen — cv2.findContours cannot detect them.
    Using np.where() instead. Color: H=130-175, S=50-120, V=130-215 (pink/mauve).
    """
    try:
        h = screen_rgb.shape[0]
        play = screen_rgb[16:h-22, :]
        hsv = cv2.cvtColor(play, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([130, 50, 130]), np.array([175, 120, 215]))
        ys, xs = np.where(mask > 0)
        # Cluster nearby pixels to avoid duplicates (pellets can be 1-2px wide)
        pellets = []
        used = set()
        for i in range(len(xs)):
            if i in used:
                continue
            x, y = int(xs[i]), int(ys[i]) + 16
            # Merge pixels within 2px
            cluster = [(j, xs[j], ys[j]) for j in range(len(xs))
                       if j not in used and abs(int(xs[j])-int(xs[i])) <= 2 
                       and abs(int(ys[j])-int(ys[i])) <= 2]
            for j, _, _ in cluster:
                used.add(j)
            cx = int(np.mean([xs[j] for j, _, _ in cluster]))
            cy = int(np.mean([ys[j] for j, _, _ in cluster])) + 16
            pellets.append((cx, cy))
        return pellets
    except Exception:
        return []

def _detect_ghosts(screen_rgb):
    """
    Detect ghosts on RAW ALE screen. Uses solidity>0.65 to separate
    ghost bodies from wall fragments (walls are same red hue as Blinky).
    """
    try:
        h, w = screen_rgb.shape[:2]
        play = screen_rgb[16:h-22, :]
        hsv = cv2.cvtColor(play, cv2.COLOR_RGB2HSV)
        scale = (h * w) / (210 * 160)
        min_a, max_a = 8 * scale, 150 * scale

        blinky = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,   140, 150]), np.array([8,   255, 255])),
            cv2.inRange(hsv, np.array([172, 140, 150]), np.array([180, 255, 255]))
        )
        inky  = cv2.inRange(hsv, np.array([72,  90, 130]), np.array([108, 170, 255]))
        pinky = cv2.inRange(hsv, np.array([138, 115, 145]), np.array([172, 175, 255]))
        clyde = cv2.inRange(hsv, np.array([10,  110, 145]), np.array([18,  195, 255]))

        ghosts = []
        for mask in [blinky, inky, pinky, clyde]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                a = cv2.contourArea(cnt)
                if min_a < a < max_a and _solidity(cnt) > 0.65:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        ghosts.append((int(M["m10"]/M["m00"]),
                                       int(M["m01"]/M["m00"]) + 16))
        return ghosts
    except Exception:
        return []

def _count_open_ahead(raw_screen, x, y, action, lookahead=24):
    """
    Count how many open positions exist along `action` direction,
    looking `lookahead` pixels ahead in steps of 8.
    More open steps = less likely to be a dead-end.
    """
    step_map = {ALE_UP: (0,-8), ALE_DOWN: (0,8), ALE_LEFT: (-8,0), ALE_RIGHT: (8,0)}
    dx, dy = step_map.get(action, (0,0))
    count = 0
    cx, cy = x, y
    for _ in range(lookahead // 8):
        cx += dx
        cy += dy
        if _is_corridor_map(cx, cy):
            count += 1
        else:
            break
    return count

def ghost_avoidance_heuristic(raw_screen, ghost_positions, safe_distance=55,
                               pacman_pos=None, open_dirs=None):
    """
    Flee from nearest active ghost when within safe_distance.
    - safe_distance increased to 55px for earlier reaction
    - overlapping_pac threshold reduced to 2px (was 6px, was filtering real threats)
    - Fallback prefers direction with most open corridor ahead (avoids dead-ends)
    """
    if pacman_pos is None:
        pacman_pos = _find_pacman_position(raw_screen)
    if pacman_pos is None or not ghost_positions:
        return None

    if open_dirs is None:
        open_dirs = _open_directions(raw_screen, pacman_pos)

    def in_spawn_house(g):
        return 60 <= g[0] <= 100 and 85 <= g[1] <= 115

    def overlapping_pac(g):
        # Only exclude exact detection noise (< 2px), not real threats
        return _distance(g, pacman_pos) < 2

    active_ghosts = [g for g in ghost_positions
                     if not in_spawn_house(g) and not overlapping_pac(g)]
    if not active_ghosts:
        return None

    closest = min(active_ghosts, key=lambda g: _distance(g, pacman_pos))
    if _distance(closest, pacman_pos) > safe_distance:
        return None

    dx = pacman_pos[0] - closest[0]
    dy = pacman_pos[1] - closest[1]

    if not open_dirs:
        return None

    # Preferred escape direction (directly away from ghost)
    if abs(dx) >= abs(dy):
        preferred = ALE_RIGHT if dx > 0 else ALE_LEFT
        bad = ALE_LEFT if dx > 0 else ALE_RIGHT
    else:
        preferred = ALE_DOWN if dy > 0 else ALE_UP
        bad = ALE_UP if dy > 0 else ALE_DOWN

    if preferred in open_dirs:
        return preferred

    # Smart fallback: among non-bad open directions, pick the one with most
    # open corridor ahead (avoids sending Pac-Man into dead-ends)
    candidates = [a for a in open_dirs if a != bad]
    if not candidates:
        candidates = open_dirs

    px, py = pacman_pos
    best = max(candidates,
               key=lambda a: _count_open_ahead(raw_screen, px, py, a))
    return best

def get_pellet_heuristic(raw_screen, ghost_positions, danger_distance=40,
                         pacman_pos=None, open_dirs=None):
    """
    Move toward nearest safe pellet using pre-computed pac_pos and open_dirs.
    """
    if pacman_pos is None:
        pacman_pos = _find_pacman_position(raw_screen)
    if pacman_pos is None:
        return None

    if open_dirs is None:
        open_dirs = _open_directions(raw_screen, pacman_pos)

    pellets = _detect_pellets_raw(raw_screen)
    if not pellets:
        return None

    active_ghosts = [g for g in ghost_positions
                     if not (60 <= g[0] <= 100 and 85 <= g[1] <= 115)]
    safe = [p for p in pellets
            if all(_distance(p, g) >= danger_distance for g in active_ghosts)]
    if not safe:
        return None

    closest = min(safe, key=lambda p: _distance(p, pacman_pos))
    dx = closest[0] - pacman_pos[0]
    dy = closest[1] - pacman_pos[1]

    # open_dirs already computed (passed in or computed above)
    if not open_dirs:
        return None

    # Preferred direction toward pellet
    if abs(dx) >= abs(dy):
        preferred = ALE_RIGHT if dx > 0 else ALE_LEFT
    else:
        preferred = ALE_UP if dy < 0 else ALE_DOWN

    if preferred in open_dirs:
        return preferred
    # Pick open direction that gets us closer
    return min(open_dirs, key=lambda a: {
        ALE_UP:    (pacman_pos[0] - closest[0])**2 + (pacman_pos[1]-1 - closest[1])**2,
        ALE_DOWN:  (pacman_pos[0] - closest[0])**2 + (pacman_pos[1]+1 - closest[1])**2,
        ALE_LEFT:  (pacman_pos[0]-1 - closest[0])**2 + (pacman_pos[1] - closest[1])**2,
        ALE_RIGHT: (pacman_pos[0]+1 - closest[0])**2 + (pacman_pos[1] - closest[1])**2,
    }[a])