"""Resolve checkout paths without depending on a particular server."""
from pathlib import Path
import os


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def lerobot_root() -> Path:
    override = os.environ.get("LEROBOT_REPO")
    return Path(override).expanduser().resolve() if override else repository_root() / "lerobot_2.1"
