"""Feasibility analysis — can metric targets be achieved simultaneously?"""

from __future__ import annotations

import copy

import torch

from ..core.pipeline import Pipeline


def check_feasibility(pipeline: Pipeline, params: dict, name: str,
                      targets: dict, n_samples: int = 1000,
                      device: str | None = None,
                      verbose: bool = True) -> dict:
    """Check if metric targets are simultaneously achievable.

    Uses Latin hypercube sampling in parameter space to explore feasibility.

    Args:
        pipeline: Pipeline to analyze
        params: Base parameters
        name: Space name
        targets: Dict of {metric_name: {"min": val} | {"max": val}}
        n_samples: Number of Latin hypercube samples
        device: torch device
        verbose: Print progress

    Returns:
        Dict with feasibility status, closest feasible point, relaxation suggestions.
    """
    from ..metrics.registry import evaluate, _get_colorbench_modules

    _, _, _, _, comparison_mod, _ = _get_colorbench_modules()

    # Get free parameters
    cp = params.get("_checkpoint", {})
    from .sensitivity import _find_free_params
    free_params = _find_free_params(cp)

    if not free_params:
        return {"error": "No free parameters found"}

    if verbose:
        print(f"  Feasibility check: {len(targets)} targets, "
              f"{len(free_params)} free params, {n_samples} samples")

    # Latin hypercube sampling
    samples = _latin_hypercube(len(free_params), n_samples)

    # Scale samples to parameter ranges (±30% of current values)
    feasible_points = []
    closest_point = None
    min_violation = float("inf")

    for i in range(n_samples):
        if verbose and (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{n_samples}] {len(feasible_points)} feasible found")

        # Create perturbed params
        perturbed = copy.deepcopy(params)
        for j, (param_path, idx, current_val) in enumerate(free_params):
            range_size = max(abs(current_val) * 0.3, 0.1)
            new_val = current_val + (samples[i, j] - 0.5) * 2 * range_size
            _set_param_in_checkpoint(perturbed, param_path, idx, new_val)

        try:
            results = evaluate(pipeline, perturbed, name=f"sample_{i}",
                               device=device, verbose=False)

            # Check targets
            violation = _compute_violation(results, targets, comparison_mod)

            if violation == 0:
                feasible_points.append({
                    "sample_idx": i,
                    "violation": 0.0,
                })
            elif violation < min_violation:
                min_violation = violation
                closest_point = {
                    "sample_idx": i,
                    "violation": violation,
                }
        except Exception:
            pass

    # Compute relaxation suggestions
    relaxation = {}
    if not feasible_points:
        for metric_name, target in targets.items():
            relaxed = target.copy()
            if "min" in target:
                v = target["min"]
                relaxed["min"] = v - abs(v) * 0.1  # Relax downward by 10%
            if "max" in target:
                v = target["max"]
                relaxed["max"] = v + abs(v) * 0.1  # Relax upward by 10%
            relaxation[metric_name] = relaxed

    result = {
        "feasible": len(feasible_points) > 0,
        "n_feasible": len(feasible_points),
        "n_samples": n_samples,
        "targets": targets,
        "closest_infeasible": closest_point,
        "relaxation_suggestion": relaxation if not feasible_points else None,
    }

    if verbose:
        if feasible_points:
            print(f"\n  FEASIBLE: {len(feasible_points)}/{n_samples} samples satisfy all targets")
        else:
            print(f"\n  INFEASIBLE: No samples satisfy all targets "
                  f"(closest violation: {min_violation:.4f})")
            if relaxation:
                print(f"  Suggested relaxation:")
                for m, r in relaxation.items():
                    print(f"    {m}: {targets[m]} → {r}")

    return result


def _latin_hypercube(n_dims: int, n_samples: int) -> torch.Tensor:
    """Generate Latin hypercube samples in [0, 1]^n_dims."""
    samples = torch.zeros(n_samples, n_dims, dtype=torch.float64)
    for d in range(n_dims):
        perm = torch.randperm(n_samples)
        samples[:, d] = (perm.float() + torch.rand(n_samples)) / n_samples
    return samples


def _compute_violation(results: dict, targets: dict, comparison_mod) -> float:
    """Compute total constraint violation (0 = all satisfied)."""
    total = 0.0
    for metric_name, target in targets.items():
        score = None
        for mdef in comparison_mod.METRIC_DEFS:
            if mdef.name == metric_name:
                score = comparison_mod._extract_score(results, mdef.result_key, mdef.score_path)
                break

        if score is None:
            total += 1.0  # Penalty for missing metric
            continue

        if "min" in target and score < target["min"]:
            total += (target["min"] - score) / (abs(target["min"]) + 1e-30)
        if "max" in target and score > target["max"]:
            total += (score - target["max"]) / (abs(target["max"]) + 1e-30)

    return total


def _set_param_in_checkpoint(params: dict, path: str, idx, value: float):
    if "_checkpoint" not in params:
        return
    cp = params["_checkpoint"]
    if path not in cp:
        return
    if idx is None:
        cp[path] = value
    elif isinstance(idx, tuple):
        i, j = idx
        cp[path][i][j] = value
    else:
        cp[path][idx] = value
