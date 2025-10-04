"""Interfaces for capturing Geometry Dash gameplay and controlling inputs.

This module glues the reinforcement learning components to the actual game by
providing utilities to grab frames from the screen, extract a coarse state
description, and trigger keyboard inputs.  The implementation aims to stay as
light-weight as possible while keeping the pieces replaceable so experiments
can start quickly and be refined later on.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency, only available on Windows
    import dxcam  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    dxcam = None  # type: ignore

try:
    from mss import mss
except Exception:  # pragma: no cover - optional dependency
    mss = None  # type: ignore

try:  # pragma: no cover - optional dependency, not available in CI
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency, Windows friendly
    import pydirectinput
except Exception:  # pragma: no cover - optional dependency
    pydirectinput = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import pyautogui
except Exception:  # pragma: no cover - optional dependency
    pyautogui = None  # type: ignore


@dataclass
class CaptureConfig:
    """Configuration for screen capture."""

    monitor: int = 0
    region: Optional[Tuple[int, int, int, int]] = None
    downscale: int = 2
    frame_sleep: float = 1 / 60
    process_name: Optional[str] = "GeometryDash.exe"


class ScreenCapture:
    """Abstraction over dxcam/mss based screen capture."""

    def __init__(self, config: CaptureConfig | None = None):
        config = config or CaptureConfig()
        if config.region is None and config.process_name:
            config = self._with_process_region(config)
        if dxcam is not None:
            self._capture = dxcam.create(output_color="BGR")
            self._grab: Callable[[], np.ndarray | None] = self._grab_dxcam
            self._capture.start(region=config.region)
        elif mss is not None:
            self._capture = mss()
            self._grab = self._grab_mss
        else:  # pragma: no cover - requires OS specific deps
            raise RuntimeError("Install `dxcam` (Windows) or `mss` for screen capture.")
        self.config = config

    def _with_process_region(self, config: CaptureConfig) -> CaptureConfig:
        try:
            region = locate_process_window(config.process_name)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to locate window for process '{config.process_name}'."
            ) from exc
        return CaptureConfig(
            monitor=config.monitor,
            region=region,
            downscale=config.downscale,
            frame_sleep=config.frame_sleep,
            process_name=config.process_name,
        )

    def _grab_dxcam(self) -> np.ndarray | None:  # pragma: no cover - dxcam only
        return self._capture.get_latest_frame()

    def _grab_mss(self) -> np.ndarray:
        assert mss is not None  # mypy
        region = self.config.region
        monitor = self._capture.monitors[self.config.monitor + 1] if region is None else {
            "left": region[0],
            "top": region[1],
            "width": region[2] - region[0],
            "height": region[3] - region[1],
        }
        frame = np.array(self._capture.grab(monitor))[:, :, :3]  # discard alpha
        return frame[:, :, ::-1]  # convert RGBA -> BGR

    def grab(self) -> np.ndarray:
        frame = self._grab()
        if frame is None:
            raise RuntimeError("Failed to capture a frame from the screen.")
        if self.config.downscale > 1:
            frame = cv2.resize(
                frame,
                (frame.shape[1] // self.config.downscale, frame.shape[0] // self.config.downscale),
                interpolation=cv2.INTER_AREA,
            )
        return frame


def locate_process_window(process_name: str) -> Tuple[int, int, int, int]:
    """Locate the bounding box of a visible window owned by ``process_name``.

    Parameters
    ----------
    process_name:
        Executable name to match (case insensitive). Must include the extension,
        e.g. ``GeometryDash.exe``.

    Returns
    -------
    Tuple[int, int, int, int]
        The (left, top, right, bottom) rectangle of the first matching window.

    Raises
    ------
    RuntimeError
        If the process window cannot be located or the functionality is not
        available on the current platform.
    """

    if not hasattr(ctypes, "windll"):
        raise RuntimeError("Process-bound capture is only supported on Windows.")

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    psapi = ctypes.windll.psapi  # type: ignore[attr-defined]

    EnumWindows = user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    GetWindowThreadProcessId = user32.GetWindowThreadProcessId
    IsWindowVisible = user32.IsWindowVisible
    GetWindowRect = user32.GetWindowRect
    OpenProcess = kernel32.OpenProcess
    CloseHandle = kernel32.CloseHandle
    GetModuleFileNameExW = psapi.GetModuleFileNameExW

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_VM_READ = 0x0010

    target = process_name.lower()
    result: Optional[Tuple[int, int, int, int]] = None

    rect = wintypes.RECT()
    buffer = ctypes.create_unicode_buffer(260)

    def callback(hwnd: int, _lparam: int) -> bool:
        nonlocal result
        if not IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process_handle = OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,
            False,
            pid.value,
        )
        if not process_handle:
            return True
        try:
            buffer_length = GetModuleFileNameExW(process_handle, None, buffer, len(buffer))
            if buffer_length == 0:
                return True
            exe_name = Path(buffer.value).name.lower()
            if exe_name != target:
                return True
            if GetWindowRect(hwnd, ctypes.byref(rect)) == 0:
                return True
            result = (rect.left, rect.top, rect.right, rect.bottom)
            return False
        finally:
            CloseHandle(process_handle)

    EnumWindows(EnumWindowsProc(callback), 0)
    if result is None:
        raise RuntimeError(f"Window for process '{process_name}' not found.")
    return result


@dataclass
class EstimatorConfig:
    """Configuration for the state estimator."""

    player_hsv_lower: Tuple[int, int, int] = (80, 70, 120)
    player_hsv_upper: Tuple[int, int, int] = (140, 255, 255)
    obstacle_threshold: int = 60
    min_component_area: int = 25
    max_obstacles: int = 5


@dataclass
class GameState:
    """Coarse description of the level state derived from pixels."""

    player_pos: Tuple[float, float]
    player_velocity: Tuple[float, float]
    obstacles: Tuple[Tuple[float, float], ...]
    timestamp: float


class StateEstimator:
    """Extracts a coarse state representation from raw frames."""

    def __init__(self, config: EstimatorConfig | None = None):
        if cv2 is None:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "OpenCV (cv2) is required for StateEstimator. Install 'opencv-python'."
            )
        self.config = config or EstimatorConfig()
        self._last_player_center: Optional[Tuple[float, float]] = None
        self._last_time: Optional[float] = None

    def reset(self) -> None:
        self._last_player_center = None
        self._last_time = None

    def estimate(self, frame: np.ndarray) -> GameState:
        now = time.time()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.config.player_hsv_lower, self.config.player_hsv_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        player_center = (0.0, 0.0)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest)
            if moments["m00"] > 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                player_center = (cx / frame.shape[1], cy / frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.config.obstacle_threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstacle_centers: list[Tuple[float, float]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.min_component_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            obstacle_centers.append(((x + w / 2) / frame.shape[1], (y + h / 2) / frame.shape[0]))
        obstacle_centers.sort(key=lambda item: item[0])
        obstacle_centers = obstacle_centers[: self.config.max_obstacles]

        velocity = (0.0, 0.0)
        if self._last_player_center is not None and self._last_time:
            dt = max(now - self._last_time, 1e-3)
            velocity = (
                (player_center[0] - self._last_player_center[0]) / dt,
                (player_center[1] - self._last_player_center[1]) / dt,
            )

        self._last_player_center = player_center
        self._last_time = now

        return GameState(
            player_pos=player_center,
            player_velocity=velocity,
            obstacles=tuple(obstacle_centers),
            timestamp=now,
        )


class InputController:
    """Sends keyboard inputs to Geometry Dash."""

    PULSE_KEYS: Dict[str, int] = {"space": 0x39, "up": 0xC8, "mouse_left": 0x01}

    def __init__(self, pulse_duration: float = 0.08):
        self.pulse_duration = pulse_duration
        self._is_windows = self._check_windows()

    @staticmethod
    def _check_windows() -> bool:
        import platform

        return platform.system() == "Windows"

    def _send_windows_key(self, vk_code: int) -> None:  # pragma: no cover - Windows only
        extra = ctypes.c_ulong(0)

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_ulong),
                ("ii", KEYBDINPUT),
            ]

        SendInput = ctypes.windll.user32.SendInput
        key_down = INPUT(1, KEYBDINPUT(vk_code, 0, 0, 0, ctypes.pointer(extra)))
        key_up = INPUT(1, KEYBDINPUT(vk_code, 0, 2, 0, ctypes.pointer(extra)))
        SendInput(1, ctypes.byref(key_down), ctypes.sizeof(key_down))
        time.sleep(self.pulse_duration)
        SendInput(1, ctypes.byref(key_up), ctypes.sizeof(key_up))

    def _send_fallback(self, key: str) -> None:  # pragma: no cover - requires GUI
        if pydirectinput is not None:
            pydirectinput.press(key)
            return
        if pyautogui is not None:
            pyautogui.press(key)
            return
        raise RuntimeError(
            "Neither pydirectinput nor pyautogui is available for sending inputs."
        )

    def pulse(self, key: str = "space") -> None:
        if self._is_windows and key in self.PULSE_KEYS:
            self._send_windows_key(self.PULSE_KEYS[key])
        else:
            self._send_fallback(key)


class GeometryDashScreenInterface:
    """High level helper that couples capture, state estimation, and control."""

    def __init__(
        self,
        capture: ScreenCapture,
        estimator: StateEstimator,
        controller: InputController,
        reward_fn: Optional[Callable[[GameState, GameState], Tuple[float, bool, Dict[str, float]]]] = None,
    ):
        self.capture = capture
        self.estimator = estimator
        self.controller = controller
        self.reward_fn = reward_fn or self._default_reward
        self._last_state: Optional[GameState] = None

    def reset(self) -> Tuple[np.ndarray, GameState]:
        self.estimator.reset()
        frame = self.capture.grab()
        state = self.estimator.estimate(frame)
        self._last_state = state
        return self.to_observation(state), state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if action == 1:
            self.controller.pulse("space")
        elif action == 2:
            self.controller.pulse("mouse_left")

        time.sleep(self.capture.config.frame_sleep)
        frame = self.capture.grab()
        state = self.estimator.estimate(frame)
        last_state = self._last_state or state
        reward, done, info = self.reward_fn(last_state, state)
        self._last_state = state
        return self.to_observation(state), reward, done, info

    @property
    def last_state(self) -> Optional[GameState]:
        return self._last_state

    def to_observation(self, state: GameState) -> np.ndarray:
        obs = [
            state.player_pos[0],
            state.player_pos[1],
            state.player_velocity[0],
            state.player_velocity[1],
        ]
        for obstacle in state.obstacles:
            obs.extend([obstacle[0], obstacle[1]])
        while len(obs) < 4 + 2 * 5:
            obs.append(0.0)
        return np.array(obs, dtype=np.float32)

    @staticmethod
    def _default_reward(prev: GameState, curr: GameState) -> Tuple[float, bool, Dict[str, float]]:
        progressed = curr.player_pos[0] - prev.player_pos[0]
        fallen = curr.player_pos[1] > 0.9
        reward = progressed
        done = fallen
        info = {"event": "fallen" if fallen else "running"}
        return reward, done, info
