"""
evaluation.py

Purpose:
    Document evaluation choices and provide helpers for reporting.

Why this exists:
    The evaluation metrics/plots are implemented in main.py. This module
    records the evaluation approach and provides small helpers for consistent
    naming of outputs.
"""

from __future__ import annotations
from pathlib import Path

RESULTS_DIR = Path("results")

def ensure_results_dir(results_dir: Path = RESULTS_DIR) -> Path:
    """
    Ensure results directory exists locally.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def expected_outputs() -> list[str]:
    """
    Typical outputs produced by running main.py (may vary by run).
    """
    return [
        "model_metrics_by_season.csv",
        "model_odds_all_seasons.csv",
        "model_vs_baseline_by_season.csv",
        "Model_accuracy_over_time.png",
        "Model_probability_quality_over_time.png",
        "correlation_heatmap.png",
    ]
