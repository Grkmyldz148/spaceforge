"""Pareto frontier computation — trade-off analysis between metrics."""

from __future__ import annotations

import copy

import torch

from ..core.pipeline import Pipeline


def compute_pareto(pipeline: Pipeline, params: dict, name: str,
                   x_metric: str, y_metric: str,
                   sweep_param: str, n_samples: int = 50,
                   device: str | None = None,
                   verbose: bool = True) -> dict:
    """Compute trade-off frontier between two metrics by sweeping a parameter.

    Args:
        pipeline: Pipeline to analyze
        params: Base parameters
        name: Space name
        x_metric: Metric name for X axis
        y_metric: Metric name for Y axis
        sweep_param: Parameter to sweep (e.g. "M2[1,0]" or "L_corr[0]")
        n_samples: Number of samples along sweep
        device: torch device
        verbose: Print progress

    Returns:
        Dict with sweep points, Pareto frontier, and plot data.
    """
    from ..metrics.registry import evaluate, _get_colorbench_modules

    _, _, _, _, comparison_mod, _ = _get_colorbench_modules()

    # Parse sweep parameter
    param_path, param_idx = _parse_param_ref(sweep_param)

    # Get current value and range
    cp = params.get("_checkpoint", {})
    current_val = _get_param_value(cp, param_path, param_idx)

    # Sweep range: ±50% of current value (or ±0.5 if near zero)
    half_range = max(abs(current_val) * 0.5, 0.5)
    sweep_values = torch.linspace(
        current_val - half_range, current_val + half_range,
        n_samples, dtype=torch.float64,
    )

    points = []

    for i, val in enumerate(sweep_values):
        if verbose:
            print(f"  [{i + 1}/{n_samples}] {sweep_param}={float(val):.4f}")

        # Create perturbed params
        perturbed = copy.deepcopy(params)
        if "_checkpoint" in perturbed:
            _set_param_value(perturbed["_checkpoint"], param_path, param_idx, float(val))

        try:
            results = evaluate(pipeline, perturbed, name=f"{name}_{i}",
                               device=device, verbose=False)

            x_score = _find_metric_score(results, x_metric, comparison_mod)
            y_score = _find_metric_score(results, y_metric, comparison_mod)

            if x_score is not None and y_score is not None:
                points.append({
                    "param_value": float(val),
                    "x": x_score,
                    "y": y_score,
                })
        except Exception as e:
            if verbose:
                print(f"    Failed: {e}")

    # Find Pareto frontier (non-dominated points)
    frontier = _compute_frontier(points, x_metric, y_metric, comparison_mod)

    result = {
        "x_metric": x_metric,
        "y_metric": y_metric,
        "sweep_param": sweep_param,
        "points": points,
        "frontier": frontier,
        "n_samples": n_samples,
    }

    if verbose:
        print(f"\n  Pareto frontier: {len(frontier)} non-dominated points out of {len(points)}")
        for p in frontier:
            print(f"    {sweep_param}={p['param_value']:.4f}  "
                  f"{x_metric}={p['x']:.4f}  {y_metric}={p['y']:.4f}")

    return result


def _parse_param_ref(ref: str) -> tuple[str, int | tuple | None]:
    """Parse 'M2[1,0]' into ('M2', (1, 0)) or 'L_corr[0]' into ('L_corr', 0)."""
    if "[" not in ref:
        return ref, None

    name = ref[:ref.index("[")]
    idx_str = ref[ref.index("[") + 1:ref.index("]")]

    if "," in idx_str:
        parts = idx_str.split(",")
        return name, tuple(int(p.strip()) for p in parts)
    else:
        return name, int(idx_str)


def _get_param_value(cp: dict, path: str, idx) -> float:
    if path not in cp:
        return 0.0
    val = cp[path]
    if idx is None:
        if isinstance(val, (list, tuple)):
            raise ValueError(
                f"Parameter '{path}' is a matrix/vector. "
                f"Use '{path}[i,j]' or '{path}[i]' to specify an element."
            )
        return float(val)
    elif isinstance(idx, tuple):
        return float(val[idx[0]][idx[1]])
    else:
        return float(val[idx])


def _set_param_value(cp: dict, path: str, idx, value: float):
    if path not in cp:
        return
    if idx is None:
        cp[path] = value
    elif isinstance(idx, tuple):
        cp[path][idx[0]][idx[1]] = value
    else:
        cp[path][idx] = value


def _find_metric_score(results: dict, metric_name: str, comparison_mod) -> float | None:
    """Find a metric score by name from results."""
    for mdef in comparison_mod.METRIC_DEFS:
        if mdef.name == metric_name:
            return comparison_mod._extract_score(results, mdef.result_key, mdef.score_path)
    return None


def _compute_frontier(points: list[dict], x_metric: str, y_metric: str,
                      comparison_mod) -> list[dict]:
    """Find Pareto-optimal points (non-dominated)."""
    if not points:
        return []

    # Determine directions
    x_lower = True
    y_lower = True
    for mdef in comparison_mod.METRIC_DEFS:
        if mdef.name == x_metric:
            x_lower = mdef.lower_is_better
        if mdef.name == y_metric:
            y_lower = mdef.lower_is_better

    frontier = []
    for p in points:
        dominated = False
        for q in points:
            if p is q:
                continue
            # q dominates p if q is at least as good in both and strictly better in one
            x_better = (q["x"] <= p["x"]) if x_lower else (q["x"] >= p["x"])
            y_better = (q["y"] <= p["y"]) if y_lower else (q["y"] >= p["y"])
            x_strict = (q["x"] < p["x"]) if x_lower else (q["x"] > p["x"])
            y_strict = (q["y"] < p["y"]) if y_lower else (q["y"] > p["y"])

            if x_better and y_better and (x_strict or y_strict):
                dominated = True
                break

        if not dominated:
            frontier.append(p)

    # Sort by x value
    frontier.sort(key=lambda p: p["x"])
    return frontier
