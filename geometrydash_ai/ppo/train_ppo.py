"""Train a PPO agent inside the deterministic simulator."""

from __future__ import annotations

import argparse
import os
from typing import Callable, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from ..envs import GDSimEnv
from ..envs.sim_env import SimulatorConfig
from ..utils.seeding import set_seed


class ProgressCallback(BaseCallback):
    """Report episodic returns and best progress during training."""

    def __init__(self) -> None:
        super().__init__()
        self.best_progress = 0.0

    def _on_rollout_end(self) -> None:  # type: ignore[override]
        if len(self.model.ep_info_buffer) > 0:
            returns = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            mean_return = float(np.mean(returns))
            print(f"[PPO] Mean episodic return: {mean_return:.2f}")
        print(f"[PPO] Best progress so far: {self.best_progress * 100:.1f}%")

    def _on_step(self) -> bool:  # type: ignore[override]
        infos = self.locals.get("infos", [])
        for info in infos:
            progress = info.get("progress")
            if progress is not None and progress > self.best_progress:
                self.best_progress = progress
        return True


def make_env(level_path: str, cfg: SimulatorConfig, seed: int, rank: int) -> Callable[[], GDSimEnv]:
    def _init() -> GDSimEnv:
        env_seed = seed + rank
        return GDSimEnv(level_path, config=cfg, seed=env_seed)

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent in Geometry Dash simulator")
    parser.add_argument("--level", type=str, default="levels/level1.csv", help="Path to CSV level")
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total training timesteps")
    parser.add_argument("--dt", type=float, default=0.0041667, help="Physics timestep")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of vectorized environments")
    parser.add_argument("--save", type=str, default="models/ppo_gd_level1.zip", help="Model save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tensorboard", type=str, default=None, help="TensorBoard log directory")
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    cfg = SimulatorConfig(dt=args.dt)
    env_fns: List[Callable[[], GDSimEnv]] = [
        make_env(args.level, cfg, args.seed, rank) for rank in range(args.n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        n_steps=2048,
        batch_size=2048,
        n_epochs=10,
        tensorboard_log=args.tensorboard,
        verbose=1,
        seed=args.seed,
    )

    callback = ProgressCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback)

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    model.save(args.save)
    vec_env.close()
    print(f"[PPO] Training finished. Model saved to {args.save}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
