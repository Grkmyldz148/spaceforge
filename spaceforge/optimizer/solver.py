"""Constraint-first optimization solver.

Unlike traditional "minimize loss" optimizers, this solver takes constraints
(hard structural) and targets (soft metric goals) as primary inputs.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch

from ..core.pipeline import Pipeline
from ..core.constraints import check_achromatic, check_white_L, check_monotonic_L


def solve(pipeline: Pipeline, params: dict, name: str,
          targets: dict, free_params: list[str] | None = None,
          fixed_params: list[str] | None = None,
          method: str = "cma_es",
          device: str | None = None,
          generations: int = 300,
          population: int = 128,
          sigma: float = 0.03,
          checkpoint_dir: str = "spaceforge_history",
          verbose: bool = True,
          **kwargs) -> dict:
    """Constraint-first optimization.

    Args:
        pipeline: Pipeline to optimize
        params: Initial parameters
        name: Space name
        targets: {metric_name: {"min": val} | {"max": val}}
        free_params: Parameter groups to optimize (None = all)
        fixed_params: Parameter groups to keep fixed
        method: "cma_es" | "adam"
        device: torch device
        generations: Max generations
        population: Population size (CMA-ES)
        sigma: Initial step size
        checkpoint_dir: Directory for saving progress
        verbose: Print progress

    Returns:
        Dict with optimized params, history, and final scores.
    """
    if method == "cma_es":
        return _solve_cma(pipeline, params, name, targets,
                          free_params, fixed_params,
                          device, generations, population, sigma,
                          checkpoint_dir, verbose)
    elif method == "adam":
        return _solve_adam(pipeline, params, name, targets,
                          free_params, fixed_params,
                          device, generations, checkpoint_dir, verbose)
    else:
        raise ValueError(f"Unknown method: {method}")


def _solve_cma(pipeline, params, name, targets, free_params, fixed_params,
               device, generations, population, sigma, checkpoint_dir, verbose):
    """CMA-ES optimization with constraint handling."""
    import cma

    from ..analysis.sensitivity import _find_free_params

    cp = params.get("_checkpoint", {})

    # Determine which params to optimize
    param_list = _find_free_params(cp, free_params)
    if fixed_params:
        param_list = [(p, i, v) for p, i, v in param_list
                      if p not in fixed_params]

    if not param_list:
        raise ValueError("No free parameters to optimize")

    n_params = len(param_list)
    x0 = [v for _, _, v in param_list]

    if verbose:
        print(f"  CMA-ES: {n_params} params, pop={population}, gen={generations}")
        print(f"  Targets: {targets}")

    # Setup CMA-ES
    opts = cma.CMAOptions()
    opts["maxiter"] = generations
    opts["popsize"] = population
    opts["tolfun"] = 1e-12
    opts["tolx"] = 1e-12
    opts["verbose"] = -1

    # Determine which metric functions to call based on targets
    from ..metrics.registry import _get_colorbench_modules, PipelineAdapter
    gpu_metrics, gpu_metrics_adv, gpu_metrics_perc, pairs_mod, comparison_mod, _ = _get_colorbench_modules()

    # Map target metric names → which result_keys we need
    needed_result_keys = set()
    target_metric_defs = []
    for metric_name in targets:
        for mdef in comparison_mod.METRIC_DEFS:
            if mdef.name == metric_name:
                needed_result_keys.add(mdef.result_key)
                target_metric_defs.append(mdef)
                break

    # Pre-generate pairs once (not per-candidate)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    pairs_xyz = pair_labels = None
    if "gradients" in needed_result_keys:
        pairs_xyz, pair_labels = pairs_mod.generate_all_pairs(dev)

    # Map result_keys to measurement functions
    METRIC_FUNCS = {
        "roundtrip": lambda a, d: gpu_metrics.measure_roundtrip(a, d),
        "achromatic": lambda a, d: gpu_metrics.measure_achromatic(a, d),
        "gradients": lambda a, d: gpu_metrics.measure_gradients(a, pairs_xyz, pair_labels, d),
        "gamut": lambda a, d: gpu_metrics.measure_gamut(a, d),
        "gamut_mapping": lambda a, d: gpu_metrics.measure_gamut_mapping(a, d),
        "hue": lambda a, d: gpu_metrics.measure_hue(a, d),
        "specials": lambda a, d: gpu_metrics.measure_special_gradients(a, d),
        "stability": lambda a, d: gpu_metrics.measure_stability(a, d),
        "cvd": lambda a, d: gpu_metrics_adv.measure_cvd(a, d),
        "animation": lambda a, d: gpu_metrics_adv.measure_animation(a, d),
        "jacobian": lambda a, d: gpu_metrics_adv.measure_jacobian(a, d),
        "double_rt": lambda a, d: gpu_metrics_adv.measure_double_roundtrip(a, d),
        "quantization": lambda a, d: gpu_metrics_adv.measure_quantization_symmetry(a, d),
        "channel_mono": lambda a, d: gpu_metrics_adv.measure_channel_monotonicity(a, d),
        "banding": lambda a, d: gpu_metrics_adv.measure_perceptual_banding(a, d),
        "hue_leaf": lambda a, d: gpu_metrics_adv.measure_hue_leaf(a, d),
        "3color": lambda a, d: gpu_metrics_adv.measure_3color_gradients(a, d),
        "cross_gamut": lambda a, d: gpu_metrics_adv.measure_cross_gamut_consistency(a, d),
        "munsell_value": lambda a, d: gpu_metrics_perc.measure_munsell_value(a, d),
        "munsell_hue": lambda a, d: gpu_metrics_perc.measure_munsell_hue(a, d),
        "macadam_isotropy": lambda a, d: gpu_metrics_perc.measure_macadam_isotropy(a, d),
        "hue_agreement": lambda a, d: gpu_metrics_perc.measure_hue_agreement(a, d),
        "palette_uniformity": lambda a, d: gpu_metrics_perc.measure_palette_uniformity(a, d),
        "tint_shade_hue": lambda a, d: gpu_metrics_perc.measure_tint_shade_hue(a, d),
        "dataviz_distinguish": lambda a, d: gpu_metrics_perc.measure_dataviz_distinguishability(a, d),
        "multistop_gradient": lambda a, d: gpu_metrics_perc.measure_multistop_gradient(a, d),
        "wcag_midpoint": lambda a, d: gpu_metrics_perc.measure_wcag_midpoint_contrast(a, d),
        "harmony_accuracy": lambda a, d: gpu_metrics_perc.measure_harmony_accuracy(a, d),
        "photo_gamut_map": lambda a, d: gpu_metrics_perc.measure_photo_gamut_map(a, d),
        "eased_animation": lambda a, d: gpu_metrics_perc.measure_eased_animation(a, d),
        "shade_hue_consistency": lambda a, d: gpu_metrics_perc.measure_shade_hue_consistency(a, d),
        "chroma_preservation": lambda a, d: gpu_metrics_perc.measure_chroma_preservation(a, d),
    }

    if verbose:
        n_full = len(METRIC_FUNCS)
        n_needed = len(needed_result_keys)
        print(f"  Quick-eval mode: running {n_needed}/{n_full} metric groups "
              f"(targets: {list(targets.keys())})")

    history = []
    best_loss = float("inf")
    best_x = None

    def objective(x):
        nonlocal best_loss, best_x

        # Build perturbed params
        trial_params = copy.deepcopy(params)
        for j, (path, idx, _) in enumerate(param_list):
            _set_param(trial_params, path, idx, x[j])

        # Check structural constraints (hard)
        try:
            ach = check_achromatic(pipeline, trial_params, device)
            wl = check_white_L(pipeline, trial_params, device)
        except Exception:
            return 999.0

        if not ach["pass"] or not wl["pass"]:
            return 999.0

        # Evaluate ONLY the metrics needed for targets (not all 46)
        try:
            adapter = PipelineAdapter(pipeline, trial_params, "trial")
            results = {}
            for rkey in needed_result_keys:
                if rkey in METRIC_FUNCS:
                    results[rkey] = METRIC_FUNCS[rkey](adapter, dev)
        except Exception:
            return 999.0

        # Compute loss from targets
        loss = 0.0
        for mdef in target_metric_defs:
            target = targets[mdef.name]
            score = comparison_mod._extract_score(results, mdef.result_key, mdef.score_path)

            if score is None:
                loss += 10.0
                continue

            if "min" in target and score < target["min"]:
                loss += ((target["min"] - score) / (abs(target["min"]) + 1e-30)) ** 2
            if "max" in target and score > target["max"]:
                loss += ((score - target["max"]) / (abs(target["max"]) + 1e-30)) ** 2

        # Achromatic penalty (soft)
        loss += 100.0 * ach["max_ab"]

        if loss < best_loss:
            best_loss = loss
            best_x = list(x)

        return loss

    es = cma.CMAEvolutionStrategy(x0, sigma, opts)

    t0 = time.time()
    gen = 0
    while not es.stop():
        X = es.ask()
        losses = [objective(x) for x in X]
        es.tell(X, losses)
        gen += 1

        if verbose and gen % 10 == 0:
            print(f"  gen {gen}: best={min(losses):.6f}, mean={sum(losses)/len(losses):.4f}")

        history.append({
            "gen": gen,
            "best": float(min(losses)),
            "mean": float(sum(losses) / len(losses)),
        })

    elapsed = time.time() - t0

    # Build optimized params
    if best_x is not None:
        opt_params = copy.deepcopy(params)
        for j, (path, idx, _) in enumerate(param_list):
            _set_param(opt_params, path, idx, best_x[j])
    else:
        opt_params = params

    # Save checkpoint
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = Path(checkpoint_dir) / f"{name}_{ts}.json"

    def _to_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, dict):
            return {k: _to_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_json(v) for v in obj]
        return obj

    with open(ckpt_path, "w") as f:
        json.dump({k: _to_json(v) for k, v in opt_params.get("_checkpoint", {}).items()},
                  f, indent=2)

    if verbose:
        print(f"\n  Optimization complete: {elapsed:.1f}s, {gen} generations")
        print(f"  Best loss: {best_loss:.6f}")
        print(f"  Checkpoint: {ckpt_path}")

    return {
        "optimized_params": opt_params,
        "best_loss": best_loss,
        "history": history,
        "generations": gen,
        "elapsed": elapsed,
        "checkpoint": str(ckpt_path),
    }


def _solve_adam(pipeline, params, name, targets, free_params, fixed_params,
                device, generations, checkpoint_dir, verbose):
    """Adam-based optimization for differentiable pipelines."""
    # Placeholder for future implementation
    raise NotImplementedError(
        "Adam optimization requires differentiable pipeline blocks. "
        "Use method='cma_es' for derivative-free optimization."
    )


def _set_param(params: dict, path: str, idx, value: float):
    if "_checkpoint" not in params:
        return
    cp = params["_checkpoint"]
    if path not in cp:
        return
    if idx is None:
        cp[path] = value
    elif isinstance(idx, tuple):
        cp[path][idx[0]][idx[1]] = value
    else:
        cp[path][idx] = value
