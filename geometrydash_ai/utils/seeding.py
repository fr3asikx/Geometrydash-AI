"""Seeding helpers for reproducible experiments."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for tests
    torch = None  # type: ignore


def set_seed(seed: Optional[int]) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` if available."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[union-attr]
            torch.cuda.manual_seed_all(seed)  # type: ignore[union-attr]
