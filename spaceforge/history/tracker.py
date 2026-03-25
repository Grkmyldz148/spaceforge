"""History tracker — auto-saves every evaluation for regression detection."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch


HISTORY_DIR = "spaceforge_history"


class HistoryTracker:
    """Tracks all evaluations for regression detection and model comparison."""

    def __init__(self, history_dir: str = HISTORY_DIR):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, params: dict, results: dict,
             metadata: dict | None = None) -> str:
        """Save an evaluation to history.

        Returns:
            Path to saved history entry.
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        param_hash = self._hash_params(params)[:8]
        filename = f"{name}_{ts}_{param_hash}.json"
        path = self.history_dir / filename

        entry = {
            "name": name,
            "timestamp": ts,
            "param_hash": param_hash,
            "params": self._serialize(params),
            "scores": self._extract_scores(results),
            "metadata": metadata or {},
        }

        with open(path, "w") as f:
            json.dump(entry, f, indent=2)

        return str(path)

    def list_entries(self, name_filter: str | None = None) -> list[dict]:
        """List all history entries, optionally filtered by name."""
        entries = []
        for p in sorted(self.history_dir.glob("*.json")):
            try:
                with open(p) as f:
                    entry = json.load(f)
                if name_filter and name_filter not in entry.get("name", ""):
                    continue
                entry["_path"] = str(p)
                entries.append(entry)
            except (json.JSONDecodeError, KeyError):
                pass
        return entries

    def best(self, metric_name: str, name_filter: str | None = None) -> dict | None:
        """Find the best model for a given metric.

        Args:
            metric_name: Metric to optimize for.
            name_filter: Only consider entries matching this name.
        """
        entries = self.list_entries(name_filter=name_filter)
        if not entries:
            return None

        # Use ColorBench's metric direction if available
        lower_is_better = self._get_metric_direction(metric_name)

        best_entry = None
        best_score = None

        for entry in entries:
            score = entry.get("scores", {}).get(metric_name)
            if score is None:
                continue

            if best_score is None:
                best_score = score
                best_entry = entry
            elif lower_is_better and score < best_score:
                best_score = score
                best_entry = entry
            elif not lower_is_better and score > best_score:
                best_score = score
                best_entry = entry

        return best_entry

    def _get_metric_direction(self, metric_name: str) -> bool:
        """Get whether lower is better for a metric. Uses ColorBench if available."""
        try:
            from ..metrics.registry import _get_colorbench_modules
            _, _, _, _, comparison_mod, _ = _get_colorbench_modules()
            for mdef in comparison_mod.METRIC_DEFS:
                if mdef.name == metric_name:
                    return mdef.lower_is_better
        except ImportError:
            pass
        # Fallback heuristic
        return any(kw in metric_name.lower()
                   for kw in ["cv", "error", "drift", "violation", "mud",
                              "shift", "cliff", "banding"])

    def diff(self, entry_a: dict, entry_b: dict) -> dict:
        """Compare two history entries metric by metric."""
        scores_a = entry_a.get("scores", {})
        scores_b = entry_b.get("scores", {})

        all_metrics = set(scores_a.keys()) | set(scores_b.keys())
        deltas = {}

        for m in sorted(all_metrics):
            sa = scores_a.get(m)
            sb = scores_b.get(m)
            if sa is not None and sb is not None:
                deltas[m] = {
                    "a": sa,
                    "b": sb,
                    "delta": sb - sa,
                    "pct": ((sb - sa) / (abs(sa) + 1e-30)) * 100,
                }

        return deltas

    def check_regression(self, name: str, new_results: dict,
                         threshold_pct: float = 5.0) -> list[str]:
        """Check if new results regress from the best historical scores.

        Returns list of regressed metric names.
        """
        entries = self.list_entries(name_filter=name)
        if not entries:
            return []

        new_scores = self._extract_scores(new_results)
        regressions = []

        for metric_name, new_score in new_scores.items():
            best = self.best(metric_name, name_filter=name)
            if best is None:
                continue

            best_score = best.get("scores", {}).get(metric_name)
            if best_score is None:
                continue

            # Check for regression (> threshold_pct worse)
            if abs(best_score) < 1e-30:
                continue

            lower_is_better = self._get_metric_direction(metric_name)
            pct_change = ((new_score - best_score) / abs(best_score)) * 100

            if lower_is_better and pct_change > threshold_pct:
                regressions.append(metric_name)
            elif not lower_is_better and pct_change < -threshold_pct:
                regressions.append(metric_name)

        return regressions

    def _extract_scores(self, results: dict) -> dict:
        """Extract metric scores from full results dict."""
        try:
            from ..metrics.registry import _get_colorbench_modules
            _, _, _, _, comparison_mod, _ = _get_colorbench_modules()

            scores = {}
            for mdef in comparison_mod.METRIC_DEFS:
                score = comparison_mod._extract_score(
                    results, mdef.result_key, mdef.score_path)
                if score is not None:
                    scores[mdef.name] = score
            return scores
        except ImportError:
            return {}

    def _hash_params(self, params: dict) -> str:
        s = json.dumps(self._serialize(params), sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()

    def _serialize(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()
                    if not k.startswith("_")}
        if isinstance(obj, (list, tuple)):
            return [self._serialize(v) for v in obj]
        return obj
