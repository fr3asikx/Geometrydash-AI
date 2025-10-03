"""Profile management for Geometry Dash levels."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

from .environment import Level, LevelProfile


@dataclass
class ProfileConfig:
    base_dir: Path


class ProfileManager:
    """Handles persistent profile storage per level."""

    def __init__(self, base_dir: str | Path = "profiles"):
        self.config = ProfileConfig(base_dir=Path(base_dir))
        self.config.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, LevelProfile] = {}

    def load_or_create(self, level: Level) -> LevelProfile:
        profile_path = self._profile_path(level.name)
        if profile_path.exists():
            with profile_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            profile = LevelProfile(**data)
        else:
            profile = LevelProfile(name=level.name)
        profile.set_level_length(level.length)
        self.cache[level.name] = profile
        return profile

    def save(self, profile: LevelProfile) -> None:
        profile_path = self._profile_path(profile.name)
        with profile_path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(profile), fh, indent=2)

    def _profile_path(self, level_name: str) -> Path:
        safe_name = level_name.replace(" ", "_")
        return self.config.base_dir / f"{safe_name}.json"
