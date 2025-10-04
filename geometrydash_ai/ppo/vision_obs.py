"""Vision based observation helpers for transfer playback."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
    import mss
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("vision_obs requires 'opencv-python' and 'mss' to be installed") from exc


def grab_obs(monitor_idx: int = 1, width: int = 128, height: int = 72) -> np.ndarray:
    """Capture a greyscale, normalized observation from the given monitor."""

    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_idx >= len(monitors):
            raise ValueError(f"monitor_idx {monitor_idx} is out of range for available monitors")
        monitor = monitors[monitor_idx]
        region = {
            "left": monitor["left"] + int(monitor["width"] * 0.2),
            "top": monitor["top"] + int(monitor["height"] * 0.2),
            "width": int(monitor["width"] * 0.6),
            "height": int(monitor["height"] * 0.6),
        }
        raw = np.array(sct.grab(region))
    # Convert BGRA to grayscale.
    gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized[..., None]


def read_progress_percentage(image: np.ndarray) -> Optional[float]:
    """Placeholder for OCR-based extraction of death-screen percentage."""

    # Future improvement: run OCR on the image to obtain the completion percent.
    return None
