"""Real-time visualization utilities for training."""
from __future__ import annotations

import itertools
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


class TrainingVisualizer:
    """Maintains two matplotlib windows: state and training progress."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            self.state_fig = None
            self.progress_fig = None
            return

        plt.ion()
        self.state_fig, self.state_ax = plt.subplots(num="Environment State")
        self.progress_fig, self.progress_ax = plt.subplots(num="Training Progress")
        self.progress_ax.set_xlabel("Episode")
        self.progress_ax.set_ylabel("Reward")
        self.progress_ax.grid(True)
        self.loss_ax = self.progress_ax.twinx()
        self.loss_ax.set_ylabel("Loss", color="tab:orange")
        self.loss_ax.tick_params(axis="y", labelcolor="tab:orange")
        self._reward_line = None
        self._loss_line = None

    def update(self, level_state, rewards: List[float], losses: List[float]) -> None:
        if not self.enabled:
            return
        self._draw_state(level_state)
        self._draw_progress(rewards, losses)
        plt.pause(0.001)

    def _draw_state(self, level_state) -> None:
        self.state_ax.clear()
        self.state_ax.set_title(f"Transformation: {level_state['transformation']}")
        self.state_ax.set_xlim(0, level_state["level_length"])
        self.state_ax.set_ylim(-1, 2)
        # Draw ground
        self.state_ax.axhline(0, color="black")
        # Draw player
        self.state_ax.plot(
            level_state["player_position"],
            level_state["player_height"],
            marker="o",
            color="blue",
        )
        # Draw obstacles
        for position, height, width in level_state["obstacles"]:
            self.state_ax.add_patch(
                plt.Rectangle(
                    (position, 0),
                    width,
                    height,
                    color="red",
                    alpha=0.6,
                )
            )
        self.state_ax.set_xlabel("Position")
        self.state_ax.set_ylabel("Height")

    def _draw_progress(self, rewards: List[float], losses: List[float]) -> None:
        episodes = np.arange(1, len(rewards) + 1)
        if self._reward_line is None:
            (self._reward_line,) = self.progress_ax.plot(episodes, rewards, label="Reward", color="tab:blue")
        else:
            self._reward_line.set_data(episodes, rewards)
        self.progress_ax.relim()
        self.progress_ax.autoscale_view()

        loss_x = np.arange(1, len(losses) + 1)
        if self._loss_line is None:
            (self._loss_line,) = self.loss_ax.plot(loss_x, losses, label="Loss", color="tab:orange", alpha=0.7)
        else:
            self._loss_line.set_data(loss_x, losses)
        self.loss_ax.relim()
        self.loss_ax.autoscale_view()

        if not self.progress_ax.get_legend():
            self.progress_ax.legend(loc="upper left")

    def close(self) -> None:
        if not self.enabled:
            return
        plt.ioff()
        plt.close(self.state_fig)
        plt.close(self.progress_fig)
