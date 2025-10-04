"""Timing utilities for real-time interaction with Geometry Dash."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class FrameTimer:
    """Helper to maintain a fixed frame rate loop."""

    target_fps: float
    _last_time: float | None = None

    def sleep(self) -> None:
        """Sleep the remainder of the frame to match ``target_fps``."""

        if self.target_fps <= 0:
            return
        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
            return
        elapsed = now - self._last_time
        frame_time = 1.0 / self.target_fps
        delay = frame_time - elapsed
        if delay > 0:
            time.sleep(delay)
            self._last_time = time.perf_counter()
        else:
            self._last_time = now


@dataclass
class Debounce:
    """Ensure that events are emitted no more frequently than ``min_interval``."""

    min_interval: float
    _last_time: float | None = None

    def ready(self) -> bool:
        """Return ``True`` if the debounce interval has elapsed."""

        now = time.perf_counter()
        if self._last_time is None or (now - self._last_time) >= self.min_interval:
            self._last_time = now
            return True
        return False
