"""Deterministic Geometry Dash style simulator for offline training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import Env, spaces

from ..utils.seeding import set_seed
from .level_io import Obstacle, level_length, load_level, next_obstacle_features


@dataclass(frozen=True)
class SimulatorConfig:
    """Configuration values for :class:`GDSimEnv`."""

    dt: float = 1.0 / 240.0
    gravity: float = -55.0
    speed: float = 7.5
    jump_velocity: float = 20.0
    hold_velocity: float = 22.0
    max_steps: int = 2400


class GDSimEnv(Env[np.ndarray, int]):
    """Deterministic Geometry Dash inspired Gymnasium environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        level_path: str,
        config: SimulatorConfig | None = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config or SimulatorConfig()
        self.obstacles: list[Obstacle] = load_level(level_path)
        self._level_length = max(level_length(self.obstacles), 1.0)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng()
        self._last_progress = 0.0
        self._step_count = 0
        self._x = 0.0
        self._y = 0.0
        self._y_vel = 0.0
        self._on_ground = True
        if seed is not None:
            self.reset(seed=seed)
        else:
            self.reset()

    def seed(self, seed: Optional[int] = None) -> None:  # type: ignore[override]
        """Compat helper for classic Gym API."""

        self.reset(seed=seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            set_seed(seed)
            self._rng = np.random.default_rng(seed)
        self._x = 0.0
        self._y = 0.0
        self._y_vel = 0.0
        self._on_ground = True
        self._last_progress = 0.0
        self._step_count = 0
        obs = self._get_observation()
        info = {"progress": 0.0}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self._step_count += 1
        cfg = self.config

        if self._on_ground:
            if action == 1:
                self._y_vel = cfg.jump_velocity
                self._on_ground = False
            elif action == 2:
                self._y_vel = cfg.hold_velocity
                self._on_ground = False

        # Integrate velocity and position.
        self._y_vel += cfg.gravity * cfg.dt
        self._y += self._y_vel * cfg.dt
        if self._y <= 0.0:
            self._y = 0.0
            self._y_vel = 0.0
            self._on_ground = True

        self._x += cfg.speed * cfg.dt
        progress = min(self._x / self._level_length, 1.0)

        terminated = False
        truncated = False
        reward = progress - self._last_progress

        # Collision detection.
        for obstacle in self.obstacles:
            if obstacle.x_start <= self._x <= obstacle.x_end:
                if self._y <= obstacle.y_top:
                    terminated = True
                    reward -= 1.0
                    break

        # Check goal condition.
        if not terminated and self._x >= self._level_length:
            terminated = True
            reward += 100.0
            progress = 1.0

        if self._step_count >= cfg.max_steps and not terminated:
            truncated = True

        self._last_progress = progress
        obs = self._get_observation()
        info = {"progress": progress}

        return obs, float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        dist_next, height_next = next_obstacle_features(self._x, self.obstacles)
        obs = np.array(
            [
                self._x,
                self._y,
                self._y_vel,
                1.0 if self._on_ground else 0.0,
                dist_next,
                height_next,
                self.config.speed,
            ],
            dtype=np.float32,
        )
        return obs

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def y_velocity(self) -> float:
        return self._y_vel

    @property
    def on_ground(self) -> bool:
        return self._on_ground

    @property
    def level_length(self) -> float:
        return self._level_length
