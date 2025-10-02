"""Entry point for training the Geometry Dash AI."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geometrydash_ai.agent import AgentConfig, DQNAgent
from geometrydash_ai.environment import DEFAULT_TRANSFORMATIONS, GeometryDashEnv, demo_level
from geometrydash_ai.profile_manager import ProfileManager
from geometrydash_ai.trainer import Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Geometry Dash inspired RL agent")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--level", type=str, default="training_ground", help="Level name")
    parser.add_argument("--no-visualization", action="store_true", help="Disable visualization windows")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory for saved models")
    parser.add_argument("--profiles-dir", type=Path, default=Path("profiles"), help="Directory for level profiles")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    level = demo_level(name=args.level)
    env = GeometryDashEnv(level=level, transformations=DEFAULT_TRANSFORMATIONS)
    dummy_state = env.reset()
    agent = DQNAgent(state_dim=dummy_state.shape[0], action_dim=len(env.ACTIONS), config=AgentConfig(device=args.device))
    profile_manager = ProfileManager(base_dir=args.profiles_dir)
    trainer = Trainer(
        env=env,
        agent=agent,
        level=level,
        profile_manager=profile_manager,
        config=TrainingConfig(
            episodes=args.episodes,
            model_dir=args.model_dir,
            visualization=not args.no_visualization,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
