"""Parameter sensitivity analysis — Jacobian of metrics w.r.t. parameters."""

from __future__ import annotations

import copy
import json
import time

import torch

from ..core.pipeline import Pipeline


def compute_sensitivity(pipeline: Pipeline, params: dict, name: str,
                        param_names: list[str] | None = None,
                        epsilon: float = 0.001,
                        device: str | None = None,
                        verbose: bool = True,
                        quick: bool = False) -> dict:
    """Compute parameter → metric Jacobian.

    For each free parameter, perturb by ±epsilon and measure all metrics.
    The Jacobian J[i,j] = d(metric_i) / d(param_j).

    Args:
        pipeline: Pipeline to analyze
        params: Current parameters
        name: Space name
        param_names: Which parameter groups to analyze (e.g. ["M2"])
        epsilon: Perturbation size
        device: torch device
        verbose: Print progress
        quick: Use fast subset of metrics instead of full 46

    Returns:
        Dict with Jacobian matrix, parameter names, metric names.
    """
    from ..metrics.registry import evaluate, _get_colorbench_modules

    _, _, _, _, comparison_mod, _ = _get_colorbench_modules()

    # Get baseline metrics
    if verbose:
        print(f"  Computing baseline metrics...")
    baseline_results = evaluate(pipeline, params, name=name,
                                device=device, verbose=False)

    # Extract all metric scores from baseline
    metric_defs = comparison_mod.METRIC_DEFS
    baseline_scores = {}
    for mdef in metric_defs:
        score = comparison_mod._extract_score(baseline_results, mdef.result_key, mdef.score_path)
        if score is not None:
            baseline_scores[mdef.name] = score

    metric_names = list(baseline_scores.keys())

    # Find perturbable parameters
    cp = params.get("_checkpoint", {})
    free_params = _find_free_params(cp, param_names)

    if verbose:
        print(f"  {len(free_params)} free parameters, {len(metric_names)} metrics")

    # Compute Jacobian
    jacobian = {}
    param_labels = []

    for param_path, idx, current_val in free_params:
        label = f"{param_path}[{idx}]" if idx is not None else param_path
        param_labels.append(label)

        if verbose:
            print(f"  Perturbing {label} ({current_val:.6f} ± {epsilon})...")

        # Positive perturbation
        params_plus = _perturb_param(params, param_path, idx, epsilon)
        results_plus = evaluate(pipeline, params_plus, name=f"{name}+",
                                device=device, verbose=False)

        # Negative perturbation
        params_minus = _perturb_param(params, param_path, idx, -epsilon)
        results_minus = evaluate(pipeline, params_minus, name=f"{name}-",
                                 device=device, verbose=False)

        # Central difference
        sensitivities = {}
        for mdef in metric_defs:
            s_plus = comparison_mod._extract_score(results_plus, mdef.result_key, mdef.score_path)
            s_minus = comparison_mod._extract_score(results_minus, mdef.result_key, mdef.score_path)
            if s_plus is not None and s_minus is not None:
                deriv = (s_plus - s_minus) / (2 * epsilon)
                sensitivities[mdef.name] = deriv

        jacobian[label] = sensitivities

    # Build matrix form
    J_matrix = []
    for m_name in metric_names:
        row = []
        for p_label in param_labels:
            row.append(jacobian.get(p_label, {}).get(m_name, 0.0))
        J_matrix.append(row)

    result = {
        "metric_names": metric_names,
        "param_labels": param_labels,
        "jacobian": J_matrix,
        "baseline_scores": baseline_scores,
        "epsilon": epsilon,
    }

    if verbose:
        _print_sensitivity(result)

    return result


def _find_free_params(checkpoint: dict, param_names: list[str] | None = None) -> list:
    """Find all perturbable scalar parameters in checkpoint.

    Returns list of (param_path, index, current_value) tuples.
    """
    params_list = []

    for key, value in checkpoint.items():
        # Filter by param_names if specified
        if param_names and key not in param_names:
            continue

        if isinstance(value, list):
            if not value:
                continue  # Skip empty lists
            if isinstance(value[0], list):
                # Matrix (list of lists)
                for i, row in enumerate(value):
                    for j, val in enumerate(row):
                        try:
                            params_list.append((key, (i, j), float(val)))
                        except (TypeError, ValueError):
                            continue
            else:
                # Vector
                for i, val in enumerate(value):
                    try:
                        params_list.append((key, i, float(val)))
                    except (TypeError, ValueError):
                        continue
        elif isinstance(value, (int, float)):
            params_list.append((key, None, float(value)))

    return params_list


def _perturb_param(params: dict, param_path: str, idx, epsilon: float) -> dict:
    """Create a copy of params with one parameter perturbed."""
    new_params = copy.deepcopy(params)
    if "_checkpoint" not in new_params:
        return new_params
    cp = new_params["_checkpoint"]

    if param_path in cp:
        value = cp[param_path]
        if idx is None:
            cp[param_path] = value + epsilon
        elif isinstance(idx, tuple):
            i, j = idx
            cp[param_path][i][j] = value[i][j] + epsilon
        else:
            cp[param_path][idx] = value[idx] + epsilon

    return new_params


def _print_sensitivity(result: dict):
    """Print sensitivity summary."""
    metric_names = result["metric_names"]
    param_labels = result["param_labels"]
    J = result["jacobian"]

    print(f"\n  Sensitivity Analysis ({len(param_labels)} params × {len(metric_names)} metrics)")
    print(f"  {'─' * 70}")

    # Find most sensitive parameters for each metric
    for i, m_name in enumerate(metric_names):
        row = J[i]
        if not any(abs(v) > 1e-10 for v in row):
            continue

        # Sort by absolute sensitivity
        sensitivities = [(param_labels[j], row[j]) for j in range(len(row))]
        sensitivities.sort(key=lambda x: abs(x[1]), reverse=True)

        top = sensitivities[:3]
        top_str = ", ".join(f"{p}={v:+.4f}" for p, v in top if abs(v) > 1e-10)
        if top_str:
            print(f"  {m_name:35s} ← {top_str}")
