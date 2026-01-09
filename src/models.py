"""
models.py

Purpose:
    Hold model-related definitions (training config, model factory).

Why this exists:
    The model training currently happens in main.py. This module provides
    a clear place to move model code later without changing repository structure.
"""

from __future__ import annotations

def default_random_state() -> int:
    """
    Single source of truth for reproducibility.
    """
    return 42

def describe_model() -> str:
    """
    Human-readable description of the model used in the project.
    """
    return "RandomForestClassifier (scikit-learn) trained on matchup-difference features"
