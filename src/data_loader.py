"""
data_loader.py

Purpose:
    Centralize dataset path handling and basic file existence checks.

Why this exists:
    The full end-to-end pipeline is implemented in main.py (entry point).
    This module provides a clean place for future refactoring and makes
    the repository structure match the course requirements.

Usage:
    from src.data_loader import required_raw_files, check_raw_data_present
"""

from __future__ import annotations
from pathlib import Path

RAW_DIR = Path("data/raw")

def required_raw_files() -> list[str]:
    """
    Kaggle input files required by this project.
    """
    return [
        "MNCAATourneySeeds.csv",
        "MNCAATourneyCompactResults.csv",
        "MNCAATourneySlots.csv",
        "MRegularSeasonDetailedResults.csv",
        "MTeamConferences.csv",
        "MTeams.csv",
    ]

def check_raw_data_present(raw_dir: Path = RAW_DIR) -> tuple[bool, list[str]]:
    """
    Returns (ok, missing_files).
    Does not load any data: only checks whether files exist locally.
    """
    missing = [f for f in required_raw_files() if not (raw_dir / f).exists()]
    return (len(missing) == 0, missing)

def raw_path(filename: str, raw_dir: Path = RAW_DIR) -> Path:
    """
    Convenience helper to build a path inside data/raw.
    """
    return raw_dir / filename
