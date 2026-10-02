"""Filesystem locations and process-wide configuration."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import APP_NAME


def _default_data_dir() -> Path:
    if override := os.environ.get("VCTTS_DATA_DIR"):
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_NAME.lower()


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def db(self) -> Path:
        return self.root / "app.sqlite3"

    @property
    def voices(self) -> Path:
        return self.root / "voices"

    @property
    def uploads(self) -> Path:
        return self.root / "uploads"

    @property
    def recordings(self) -> Path:
        return self.root / "recordings"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> "Paths":
        for p in (self.root, self.voices, self.uploads, self.recordings, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    return Paths(_default_data_dir()).ensure()


def frontend_dist_dir() -> Path | None:
    """Location of the built React UI, if present."""
    candidates = []
    if override := os.environ.get("VCTTS_FRONTEND_DIST"):
        candidates.append(Path(override))
    here = Path(__file__).resolve()
    candidates.append(here.parent / "static")  # packaged builds
    candidates.append(here.parents[2] / "frontend" / "dist")  # repo checkout
    for c in candidates:
        if (c / "index.html").exists():
            return c
    return None
