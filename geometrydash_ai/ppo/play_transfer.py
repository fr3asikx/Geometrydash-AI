"""Execute PPO policy actions inside the native Geometry Dash client."""

from __future__ import annotations

import argparse
from typing import Optional

from pynput import keyboard
from stable_baselines3 import PPO

from ..envs import GDSimEnv
from ..envs.sim_env import SimulatorConfig
from ..utils.timing import Debounce, FrameTimer
from .vision_obs import grab_obs

try:  # pragma: no cover - Windows only dependency
    import win32con
    import win32gui
except ImportError:  # pragma: no cover
    win32con = None  # type: ignore
    win32gui = None  # type: ignore


def focus_geometry_dash_window() -> bool:
    """Bring the Geometry Dash window to the foreground if running on Windows."""

    if win32gui is None or win32con is None:
        return False

    target_hwnd: Optional[int] = None

    def _callback(hwnd: int, _: int) -> bool:
        nonlocal target_hwnd
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if "geometry dash" in title.lower():
                target_hwnd = hwnd
                return False
        return True

    win32gui.EnumWindows(_callback, 0)
    if target_hwnd is None:
        return False
    try:
        win32gui.ShowWindow(target_hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(target_hwnd)
    except Exception:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer PPO policy into live Geometry Dash")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO model")
    parser.add_argument("--level", type=str, default="levels/level1.csv", help="Reference level file")
    parser.add_argument("--fps", type=int, default=240, help="Playback frame rate")
    parser.add_argument("--vision", action="store_true", help="Use vision based observations")
    parser.add_argument("--monitor-idx", type=int, default=1, help="Monitor index for screen capture")
    parser.add_argument("--dt", type=float, default=0.0041667, help="Simulator timestep for proxy state")
    return parser.parse_args()


def main_loop(args: argparse.Namespace) -> None:
    model = PPO.load(args.model)

    if args.vision:
        print("[Transfer] Running in vision observation mode.")
        obs = grab_obs(args.monitor_idx).reshape(-1).astype("float32")
    else:
        cfg = SimulatorConfig(dt=args.dt)
        env = GDSimEnv(args.level, config=cfg)
        obs, _ = env.reset()

    controller = keyboard.Controller()
    frame_timer = FrameTimer(target_fps=float(args.fps))
    hold_active = False
    debounce = Debounce(min_interval=0.05)

    focus_geometry_dash_window()

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)

            if action == 1:
                if debounce.ready():
                    controller.press(keyboard.Key.space)
                    controller.release(keyboard.Key.space)
            elif action == 2:
                if not hold_active:
                    controller.press(keyboard.Key.space)
                    hold_active = True
            else:
                if hold_active:
                    controller.release(keyboard.Key.space)
                    hold_active = False

            frame_timer.sleep()

            if args.vision:
                obs = grab_obs(args.monitor_idx).reshape(-1).astype("float32")
            else:
                next_obs, _, terminated, truncated, info = env.step(int(action))
                obs = next_obs
                if terminated or truncated:
                    print(
                        f"[Transfer] Proxy episode finished. Progress: {info['progress'] * 100:.1f}%"
                    )
                    obs, _ = env.reset()
                    if hold_active:
                        controller.release(keyboard.Key.space)
                        hold_active = False
    except KeyboardInterrupt:
        print("[Transfer] Interrupted by user, exiting.")
    finally:
        if hold_active:
            controller.release(keyboard.Key.space)


def main() -> None:
    args = parse_args()
    main_loop(args)


if __name__ == "__main__":
    main()
