import cv2
import numpy as np
from ale_py import ALEInterface
import os
from typing import Optional, Tuple, Dict, Any
from algorithms.heuristic_utils import _detect_pellets, _detect_ghosts, _find_pacman_position

class PacmanEnv:
    def __init__(
        self,
        render_mode: Optional[str] = None,
        level: str = "easy",
        frameskip: int = 1,
        gray_scale: bool = False,
        stack_frames: int = 4, # Optimal configuration for PPO
        resize_shape: Tuple[int, int] = (128, 128)
    ):
        self.ale = ALEInterface()
        self.ale.setInt("frame_skip", frameskip)

        rom_path = os.path.join(os.path.dirname(__file__), "roms", "mspacman.bin")
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM file not found: {rom_path}")
        self.ale.loadROM(rom_path)

        self.render_mode = render_mode
        self.level_config = self._load_level_config(level)
        self.gray_scale = gray_scale
        self.stack_frames = stack_frames
        self.resize_shape = resize_shape

        self.action_space = self.ale.getLegalActionSet()
        
        h, w = self.resize_shape
        channels = 1 if self.gray_scale else 3
        self.frame_buffer = np.zeros((stack_frames * channels, h, w), dtype=np.uint8)

    def _load_level_config(self, level: str) -> Dict[str, Any]:
        from .level_config import LEVELS, get_repeat_action_probability
        cfg = LEVELS.get(level, LEVELS["easy"])
        # Apply ghost_speed via sticky actions (repeat_action_probability)
        prob = get_repeat_action_probability(level)
        self.ale.setFloat("repeat_action_probability", prob)
        return cfg

    def reset(self) -> np.ndarray:
        self.ale.reset_game()
        self.prev_pellet_count = None  # clear stale count from previous episode

        # Apply starting lives from level config via ALE RAM write
        # MsPacman RAM address 0x77 = lives remaining
        starting_lives = self.level_config.get("lives", 3)
        try:
            ram = self.ale.getRAM()
            ram[0x77] = starting_lives
        except Exception:
            pass  # Graceful fallback if RAM write not supported

        self._prev_lives = self.ale.lives()  # track lives for death penalty
        obs = self._get_obs()
        
        channels = 1 if self.gray_scale else 3
        for i in range(self.stack_frames):
            self.frame_buffer[i*channels:(i+1)*channels] = obs
                
        return self._get_stacked_obs()

    def check_collision(self, action: int) -> bool:
        """Check whether the next cell in the given direction is a wall."""
        screen = self.ale.getScreenRGB()
        pos = _find_pacman_position(screen)
        if pos is None:
            return False

        # MsPacman playfield starts at approximately y=16 (above is score/UI).
        # Ignore anything in the top 16 rows to avoid false positives from digits/text.
        if pos[1] < 16:
            return False

        move = {2: (0, -8), 3: (8, 0), 4: (-8, 0), 5: (0, 8)}.get(action, (0, 0))
        h, w = screen.shape[:2]

        # Sample 3 pixels in the direction to reduce single-pixel noise
        hits = 0
        for step in [1.0, 1.5, 2.0]:
            tx = int(np.clip(pos[0] + move[0] * step, 0, w - 1))
            ty = int(np.clip(pos[1] + move[1] * step, 0, h - 1))
            pixel = screen[ty, tx]
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])

            is_black   = r < 20 and g < 20 and b < 20          # background
            is_yellow  = r > 150 and g > 150 and b < 80        # pellet / Pac-Man
            is_white   = r > 200 and g > 200 and b > 200       # score digits
            is_dark_bg = r + g + b < 40                         # very dark background

            is_wall = not is_black and not is_yellow and not is_white and not is_dark_bg
            if is_wall:
                hits += 1

        # Require at least 2 of 3 sample points to agree it's a wall
        return hits >= 2

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        reward = self.ale.act(action)
        done = self.ale.game_over()
        obs = self._get_obs()

        # Large penalty for losing a life — teaches ghost avoidance via reward signal
        current_lives = self.ale.lives()
        if not hasattr(self, '_prev_lives'):
            self._prev_lives = current_lives
        if current_lives < self._prev_lives:
            reward -= self.level_config.get("death_penalty", 50.0)
        self._prev_lives = current_lives

        # Reward for collecting a pellet
        if self._is_pellet_collected():
            reward += self.level_config["pellet_reward"]

        # Update frame stack
        channels = 1 if self.gray_scale else 3
        self.frame_buffer = np.roll(self.frame_buffer, shift=-channels, axis=0)
        self.frame_buffer[-channels:] = obs

        info = {
            "hit_wall": False,
            "lives": current_lives,
            "ghost_positions": self.get_ghost_positions(),
            "pellet_positions": self.get_pellet_positions()
        }

        return self._get_stacked_obs(), reward, done, info

    def _get_stacked_obs(self) -> np.ndarray:
        return self.frame_buffer.copy()

    def _get_obs(self) -> np.ndarray:
        rgb = self.ale.getScreenRGB()
        resized = cv2.resize(rgb, (self.resize_shape[1], self.resize_shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        if self.gray_scale:
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            return gray[np.newaxis, :, :]  # (1, H, W)
        
        return np.transpose(resized, (2, 0, 1))

    def render(self) -> None:
        if self.render_mode == "human":
            raw = self.ale.getScreenRGB()
            # Scale up for visibility: 160x210 -> 480x630
            scale = 3
            vis = cv2.resize(raw, (raw.shape[1] * scale, raw.shape[0] * scale),
                             interpolation=cv2.INTER_NEAREST)
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

            # --- DEBUG OVERLAY ---
            from algorithms.heuristic_utils import (
                _find_pacman_position, _detect_ghosts, ghost_avoidance_heuristic
            )
            pac = _find_pacman_position(raw)
            ghosts = _detect_ghosts(raw)

            # Draw Pac-Man position (green circle)
            if pac is not None:
                cv2.circle(vis, (pac[0] * scale, pac[1] * scale), 8, (0, 255, 0), 2)
                cv2.putText(vis, "PAC", (pac[0] * scale + 6, pac[1] * scale),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Draw ghost positions (red circles)
            for i, g in enumerate(ghosts):
                cv2.circle(vis, (g[0] * scale, g[1] * scale), 8, (0, 0, 255), 2)
                cv2.putText(vis, f"G{i}", (g[0] * scale + 6, g[1] * scale),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Draw avoidance arrow if heuristic fires
            if pac is not None:
                avoid = ghost_avoidance_heuristic(raw, ghosts)
                arrow_map = {2: (0, -20), 3: (20, 0), 4: (-20, 0), 5: (0, 20)}
                if avoid in arrow_map:
                    dx, dy = arrow_map[avoid]
                    px, py = pac[0] * scale, pac[1] * scale
                    cv2.arrowedLine(vis, (px, py), (px + dx, py + dy),
                                    (0, 255, 255), 2, tipLength=0.4)

            # Status text
            status = f"Ghosts found: {len(ghosts)}  PAC: {'YES' if pac else 'NO'}"
            cv2.putText(vis, status, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)
            # -------------------

            cv2.imshow("Ms. Pac-Man AI [DEBUG]", vis)
            cv2.waitKey(1)

    def get_pellet_positions(self):
        return _detect_pellets(self.ale.getScreenRGB())
    
    def get_ghost_positions(self):
        return _detect_ghosts(self.ale.getScreenRGB())

    def _is_pellet_collected(self) -> bool:
        screen = self.ale.getScreenRGB()
        hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        pellets = [cnt for cnt in contours if 2 < cv2.contourArea(cnt) < 30]

        if not hasattr(self, 'prev_pellet_count') or self.prev_pellet_count is None:
            self.prev_pellet_count = len(pellets)
            return False

        collected = len(pellets) < self.prev_pellet_count
        self.prev_pellet_count = len(pellets)
        return collected

    def close(self) -> None:
        if self.render_mode == "human":
            cv2.destroyAllWindows()
        self.ale = None