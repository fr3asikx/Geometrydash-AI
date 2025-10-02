"""Deep Q-Network agent for the Geometry Dash environment."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Simple feed-forward network."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class AgentConfig:
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 500
    buffer_size: int = 50_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    target_update_interval: int = 200
    device: str = "cpu"


class DQNAgent:
    """Implements a DQN agent with experience replay."""

    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig | None = None):
        self.config = config or AgentConfig()
        self.device = torch.device(self.config.device)
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.config.buffer_size)
        self.steps_done = 0
        self.action_dim = action_dim

    def select_action(self, state: np.ndarray) -> int:
        epsilon = self._epsilon_by_frame(self.steps_done)
        self.steps_done += 1
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_values = self.policy_net(state_t)
        return int(action_values.argmax().item())

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def update(self) -> float:
        if len(self.memory) < self.config.batch_size:
            return 0.0
        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze()
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + (1 - dones_t) * self.config.gamma * max_next_q

        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _epsilon_by_frame(self, frame_idx: int) -> float:
        epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
            np.exp(-1.0 * frame_idx / self.config.epsilon_decay)
        return float(epsilon)
