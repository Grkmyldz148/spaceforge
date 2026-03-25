"""Regression detection — alert when metrics worsen beyond threshold."""

from __future__ import annotations

from .tracker import HistoryTracker


def check_regression(name: str, new_results: dict,
                     threshold_pct: float = 5.0,
                     history_dir: str = "spaceforge_history",
                     verbose: bool = True) -> list[str]:
    """Check for regressions against historical best.

    Args:
        name: Space name to filter history
        new_results: New evaluation results
        threshold_pct: Percentage threshold for regression
        history_dir: History directory

    Returns:
        List of regressed metric names.
    """
    tracker = HistoryTracker(history_dir)
    regressions = tracker.check_regression(name, new_results, threshold_pct)

    if verbose:
        if regressions:
            print(f"\n  REGRESSION ALERT: {len(regressions)} metrics regressed > {threshold_pct}%")
            for m in regressions:
                print(f"    - {m}")
        else:
            print(f"  No regressions detected (threshold: {threshold_pct}%)")

    return regressions
