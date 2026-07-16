from __future__ import annotations
import json
from pathlib import Path


def log_run(log_path: str | Path) -> None:
    """Read persisted JSON training log and render in Rerun."""
    raise NotImplementedError
