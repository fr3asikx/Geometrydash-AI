"""Level loading utilities for the deterministic Geometry Dash simulator."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    """Simple axis-aligned obstacle used by the simulator."""

    type: str
    x_start: float
    x_end: float
    y_top: float

    def contains_horizontal(self, x: float) -> bool:
        """Return ``True`` if ``x`` overlaps with the obstacle horizontally."""

        return self.x_start <= x <= self.x_end


def load_level(path: str | Path) -> List[Obstacle]:
    """Load a CSV level description into a list of :class:`Obstacle` objects."""

    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        obstacles: List[Obstacle] = []
        for row in reader:
            obstacle_type = row["type"].strip().lower()
            if obstacle_type not in {"block", "spike"}:
                raise ValueError(f"Unsupported obstacle type: {obstacle_type}")
            x_start = float(row["x_start"])
            x_end = float(row["x_end"])
            y_top = float(row["y_top"])
            if x_end < x_start:
                raise ValueError("x_end must be >= x_start")
            obstacles.append(
                Obstacle(
                    type=obstacle_type,
                    x_start=x_start,
                    x_end=x_end,
                    y_top=y_top,
                )
            )
    obstacles.sort(key=lambda obs: obs.x_start)
    return obstacles


def level_length(obstacles: Sequence[Obstacle]) -> float:
    """Return the overall length of the level as the maximum ``x_end`` value."""

    if not obstacles:
        return 0.0
    return max(obstacle.x_end for obstacle in obstacles)


def _obstacles_ahead(x: float, obstacles: Sequence[Obstacle]) -> Iterable[Obstacle]:
    for obstacle in obstacles:
        if obstacle.x_end >= x:
            yield obstacle


def next_obstacle_features(x: float, obstacles: Sequence[Obstacle]) -> tuple[float, float]:
    """Return the distance and height of the next obstacle relative to ``x``."""

    for obstacle in _obstacles_ahead(x, obstacles):
        if obstacle.x_end >= x:
            distance = max(obstacle.x_start - x, 0.0)
            height = obstacle.y_top
            return distance, height
    total_length = level_length(obstacles)
    remaining = max(total_length - x, 0.0)
    return remaining, 0.0


def uniform_progress_schedule(
    total_time: float,
    steps: int,
    start: float,
    obstacles: Sequence[Obstacle],
) -> np.ndarray:
    """Utility to create evenly spaced progress samples for transfer playback."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    total_length = max(level_length(obstacles), start)
    xs = np.linspace(start, total_length, steps, dtype=np.float32)
    features = np.zeros((steps, 2), dtype=np.float32)
    for idx, value in enumerate(xs):
        dist, height = next_obstacle_features(float(value), obstacles)
        features[idx, 0] = dist
        features[idx, 1] = height
    return features
