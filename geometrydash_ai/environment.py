"""Simplified Geometry Dash-like environment for reinforcement learning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Transformation:
    """Represents a transformation (form) the player can take."""

    name: str
    gravity: float
    jump_strength: float
    terminal_velocity: float


@dataclass
class Obstacle:
    """Represents a simple obstacle within the level."""

    position: float
    height: float
    width: float

    def collides(self, player_pos: float, player_height: float) -> bool:
        within_horizontal = self.position <= player_pos <= self.position + self.width
        if not within_horizontal:
            return False
        return player_height <= self.height


@dataclass
class Level:
    """Data structure describing a level layout."""

    name: str
    length: float
    obstacles: List[Obstacle]
    checkpoints: Optional[List[float]] = None


@dataclass
class LevelProfile:
    """Stores persistent information about a level."""

    name: str
    best_distance: float = 0.0
    completed_runs: int = 0
    total_reward: float = 0.0
    episodes: int = 0

    def update(self, distance: float, reward: float, completed: bool) -> None:
        self.episodes += 1
        self.total_reward += reward
        if completed:
            self.completed_runs += 1
            self.best_distance = self.length
        else:
            self.best_distance = max(self.best_distance, distance)

    @property
    def length(self) -> float:
        """Proxy property to keep compatibility with the level length."""
        return getattr(self, "_level_length", self.best_distance)

    def set_level_length(self, length: float) -> None:
        self._level_length = length


class GeometryDashEnv:
    """A simplified continuous 1D Geometry Dash environment."""

    ACTIONS: Dict[int, str] = {0: "stay", 1: "jump", 2: "dash"}

    def __init__(self, level: Level, transformations: List[Transformation]):
        self.level = level
        self.transformations = transformations
        self.current_transformation = transformations[0]
        self.player_position = 0.0
        self.player_height = 0.0
        self.vertical_velocity = 0.0
        self.time_step = 0.05
        self.speed = 5.0
        self.gravity_multiplier = 1.0
        self.max_time = level.length / self.speed * 1.5
        self.elapsed_time = 0.0

    def reset(self, transformation_index: int = 0) -> np.ndarray:
        self.current_transformation = self.transformations[transformation_index]
        self.player_position = 0.0
        self.player_height = 0.0
        self.vertical_velocity = 0.0
        self.elapsed_time = 0.0
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        next_obstacles = sorted(
            [ob for ob in self.level.obstacles if ob.position >= self.player_position],
            key=lambda ob: ob.position,
        )[:3]
        obs = [
            self.player_position / self.level.length,
            self.player_height,
            self.vertical_velocity,
            self.transformations.index(self.current_transformation) / max(1, len(self.transformations) - 1),
        ]
        for obstacle in next_obstacles:
            obs.extend([
                (obstacle.position - self.player_position) / self.level.length,
                obstacle.height,
                obstacle.width,
            ])
        while len(obs) < 4 + 3 * 3:
            obs.append(0.0)
        return np.array(obs, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        self.elapsed_time += self.time_step
        reward = 0.0

        if action == 1:  # jump
            self.vertical_velocity = self.current_transformation.jump_strength
        elif action == 2:  # dash, increase forward speed temporarily
            reward -= 0.01  # discourage spam
            self.player_position += self.speed * self.time_step * 0.5

        self.vertical_velocity -= self.current_transformation.gravity * self.gravity_multiplier * self.time_step
        self.vertical_velocity = max(-self.current_transformation.terminal_velocity, self.vertical_velocity)
        self.player_height = max(0.0, self.player_height + self.vertical_velocity * self.time_step)

        self.player_position += self.speed * self.time_step
        distance_reward = self.speed * self.time_step / self.level.length
        reward += distance_reward

        for obstacle in self.level.obstacles:
            if obstacle.collides(self.player_position, self.player_height):
                reward -= 1.0
                obs = self._get_observation()
                return obs, reward, True, {"event": "collision", "position": self.player_position}

        completed = self.player_position >= self.level.length
        if completed:
            reward += 1.0

        timed_out = self.elapsed_time >= self.max_time
        done = completed or timed_out
        obs = self._get_observation()
        info = {"event": "completed" if completed else "timeout" if timed_out else "running"}
        return obs, reward, done, info

    def sample_level_state(self) -> Dict[str, float]:
        """Returns a snapshot of the current state for visualization."""
        return {
            "player_position": self.player_position,
            "player_height": self.player_height,
            "level_length": self.level.length,
            "obstacles": [(ob.position, ob.height, ob.width) for ob in self.level.obstacles],
            "transformation": self.current_transformation.name,
        }


DEFAULT_TRANSFORMATIONS = [
    Transformation(name="cube", gravity=9.8, jump_strength=8.5, terminal_velocity=15.0),
    Transformation(name="ship", gravity=6.0, jump_strength=10.0, terminal_velocity=12.0),
]


def demo_level(name: str = "training_ground") -> Level:
    """Creates a deterministic demo level for experimentation."""
    obstacles = [
        Obstacle(position=5.0, height=0.5, width=0.5),
        Obstacle(position=10.0, height=0.6, width=0.5),
        Obstacle(position=12.0, height=0.4, width=0.5),
        Obstacle(position=15.0, height=0.7, width=0.5),
        Obstacle(position=18.0, height=0.5, width=0.5),
        Obstacle(position=20.0, height=0.3, width=0.5),
    ]
    return Level(name=name, length=25.0, obstacles=obstacles)
