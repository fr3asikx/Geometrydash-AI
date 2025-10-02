"""Training loop orchestration for the Geometry Dash agent."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .agent import DQNAgent
from .environment import GeometryDashEnv, Level, LevelProfile
from .profile_manager import ProfileManager
from .visualization import TrainingVisualizer


@dataclass
class TrainingConfig:
    episodes: int = 500
    max_steps_per_episode: int = 1_000
    save_interval: int = 50
    model_dir: Path = Path("models")
    visualization: bool = True


class Trainer:
    """Coordinates the training of the DQN agent."""

    def __init__(
        self,
        env: GeometryDashEnv,
        agent: DQNAgent,
        level: Level,
        profile_manager: Optional[ProfileManager] = None,
        config: TrainingConfig | None = None,
    ):
        self.env = env
        self.agent = agent
        self.level = level
        self.config = config or TrainingConfig()
        self.profile_manager = profile_manager or ProfileManager()
        self.profile = self.profile_manager.load_or_create(level)
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = TrainingVisualizer(self.config.visualization)
        self.rewards: List[float] = []
        self.losses: List[float] = []
        self.completions: List[bool] = []

    def train(self) -> None:
        for episode in range(1, self.config.episodes + 1):
            state = self.env.reset()
            total_reward = 0.0
            completed = False

            for step in range(self.config.max_steps_per_episode):
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.store_transition(state, action, reward, next_state, done)
                loss = self.agent.update()

                state = next_state
                total_reward += reward
                if loss:
                    self.losses.append(loss)
                if done:
                    completed = info.get("event") == "completed"
                    break

            self.rewards.append(total_reward)
            self.completions.append(completed)
            self.profile.update(self.env.player_position, total_reward, completed)
            self.profile_manager.save(self.profile)

            if episode % self.config.save_interval == 0:
                self._save_agent(episode)

            if self.visualizer.enabled:
                level_state = self.env.sample_level_state()
                self.visualizer.update(level_state, self.rewards, self.losses)

            print(
                f"Episode {episode:04d} | reward={total_reward:.3f} "
                f"| completed={completed} | best_distance={self.profile.best_distance:.2f}"
            )

        self._save_agent(self.config.episodes)
        self.visualizer.close()

    def _save_agent(self, episode: int) -> None:
        model_path = self.config.model_dir / f"{self.level.name}_episode_{episode}.pt"
        self.agent.save(str(model_path))
