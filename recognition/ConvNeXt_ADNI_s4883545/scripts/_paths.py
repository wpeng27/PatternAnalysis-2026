"""Resolve core and helper source files for provenance and frozen snapshots."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def source_path(name):
    """Locate a source basename without depending on the working directory."""
    if Path(name).name != name:
        raise ValueError("Expected a source basename")
    for directory in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(name)
