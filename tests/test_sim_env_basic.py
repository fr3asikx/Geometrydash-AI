"""Basic smoke tests for the deterministic simulator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from geometrydash_ai.envs.sim_env import GDSimEnv
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency missing
    pytest.skip(f"gymnasium not available: {exc}", allow_module_level=True)


def test_sim_env_runs_without_exceptions() -> None:
    env = GDSimEnv("levels/level1.csv")
    obs, info = env.reset()
    assert obs.shape == (7,)
    assert 0.0 <= info["progress"] <= 1.0

    rng = np.random.default_rng(0)
    for _ in range(10):
        action = int(rng.integers(0, 3))
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (7,)
        assert isinstance(reward, float)
        assert 0.0 <= info["progress"] <= 1.0
        if terminated or truncated:
            env.reset()
