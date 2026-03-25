"""Constraint-first optimization solver.

Two modes:
1. Target mode: satisfy min/max constraints on specific metrics
2. Wins mode: maximize head-to-head wins against a reference space (e.g. OKLab)
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..core.pipeline import Pipeline
from ..core.constraints import check_achromatic, check_white_L


def solve(pipeline: Pipeline, params: dict, name: str,
          targets: dict | None = None,
          maximize_wins_vs: str | None = None,
          free_params: list[str] | None = None,
          fixed_params: list[str] | None = None,
          method: str = "cma_es",
          device: str | None = None,
          generations: int = 300,
          population: int = 64,
          sigma: float = 0.03,
          checkpoint_dir: str = "spaceforge_history",
          verbose: bool = True,
          **kwargs) -> dict:
    """Constraint-first optimization.

    Args:
        pipeline: Pipeline to optimize
        params: Initial parameters
        name: Space name
        targets: {metric_name: {"min": val} | {"max": val}} — constraint mode
        maximize_wins_vs: Reference space name (e.g. "oklab") — wins mode
        free_params: Parameter groups to optimize (None = all)
        fixed_params: Parameter groups to keep fixed
        method: "cma_es"
        device: torch device
        generations: Max generations
        population: Population size
        sigma: Initial step size
        checkpoint_dir: Directory for saving progress
        verbose: Print progress

    Returns:
        Dict with optimized params, history, and final scores.
    """
    if method != "cma_es":
        raise ValueError(f"Only 'cma_es' is supported, got '{method}'")

    if maximize_wins_vs:
        return _solve_maximize_wins(
            pipeline, params, name, maximize_wins_vs,
            free_params, fixed_params,
            device, generations, population, sigma,
            checkpoint_dir, verbose,
        )
    elif targets:
        return _solve_targets(
            pipeline, params, name, targets,
            free_params, fixed_params,
            device, generations, population, sigma,
            checkpoint_dir, verbose,
        )
    else:
        raise ValueError("Provide either 'targets' or 'maximize_wins_vs'")


# ── Shared helpers ───────────────────────────────────────────────

def _setup_cma(params, free_params, fixed_params, generations, population, sigma, verbose):
    """Setup CMA-ES: extract free params, create optimizer."""
    import cma
    from ..analysis.sensitivity import _find_free_params

    cp = params.get("_checkpoint", {})
    param_list = _find_free_params(cp, free_params)
    if fixed_params:
        param_list = [(p, i, v) for p, i, v in param_list
                      if p not in fixed_params]

    if not param_list:
        raise ValueError("No free parameters to optimize")

    x0 = [v for _, _, v in param_list]

    if verbose:
        print(f"  CMA-ES: {len(param_list)} params, pop={population}, gen={generations}, sigma={sigma}")

    opts = cma.CMAOptions()
    opts["maxiter"] = generations
    opts["popsize"] = population
    opts["tolfun"] = 1e-12
    opts["tolx"] = 1e-12
    opts["verbose"] = -1

    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    return es, param_list


def _build_trial(pipeline, params, param_list, x, device):
    """Build trial params from CMA-ES candidate vector."""
    trial_params = copy.deepcopy(params)
    for j, (path, idx, _) in enumerate(param_list):
        val = float(x[j])
        _set_param(trial_params, path, idx, val)
    return trial_params


def _check_structural(pipeline, trial_params, device):
    """Check hard structural constraints. Returns (pass, ach_error) or (False, None)."""
    try:
        ach = check_achromatic(pipeline, trial_params, device)
        wl = check_white_L(pipeline, trial_params, device)
    except Exception:
        return False, None

    if not ach["pass"] or not wl["pass"]:
        return False, None

    return True, ach["max_ab"]


def _get_metric_funcs(needed_keys, gpu_metrics, gpu_metrics_adv, gpu_metrics_perc,
                      pairs_xyz, pair_labels):
    """Build dict of result_key → measurement function for needed keys only."""
    ALL_FUNCS = {
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
    return {k: v for k, v in ALL_FUNCS.items() if k in needed_keys}


def _save_checkpoint(params, checkpoint_dir, name, extra_meta=None):
    """Save optimized checkpoint to JSON."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = Path(checkpoint_dir) / f"{name}_{ts}.json"

    def _to_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _to_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_json(v) for v in obj]
        return obj

    data = _to_json(params.get("_checkpoint", {}))
    if extra_meta:
        data.update(_to_json(extra_meta))

    with open(ckpt_path, "w") as f:
        json.dump(data, f, indent=2)

    return str(ckpt_path)


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


# ── Target-based solver ──────────────────────────────────────────

def _solve_targets(pipeline, params, name, targets, free_params, fixed_params,
                   device, generations, population, sigma, checkpoint_dir, verbose):
    """Minimize constraint violations on specific metric targets."""
    from ..metrics.registry import _get_colorbench_modules, PipelineAdapter

    es, param_list = _setup_cma(params, free_params, fixed_params,
                                generations, population, sigma, verbose)

    gpu_metrics, gpu_adv, gpu_perc, pairs_mod, comparison_mod, _ = _get_colorbench_modules()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Find needed metric groups
    needed_keys = set()
    target_defs = []
    for metric_name in targets:
        for mdef in comparison_mod.METRIC_DEFS:
            if mdef.name == metric_name:
                needed_keys.add(mdef.result_key)
                target_defs.append(mdef)
                break

    pairs_xyz = pair_labels = None
    if "gradients" in needed_keys:
        pairs_xyz, pair_labels = pairs_mod.generate_all_pairs(dev)

    metric_funcs = _get_metric_funcs(needed_keys, gpu_metrics, gpu_adv, gpu_perc,
                                     pairs_xyz, pair_labels)

    if verbose:
        print(f"  Quick-eval: {len(needed_keys)}/{len(comparison_mod.METRIC_DEFS)} metric groups")
        print(f"  Targets: {targets}")

    best_loss = float("inf")
    best_x = None
    history = []

    def objective(x):
        nonlocal best_loss, best_x

        trial = _build_trial(pipeline, params, param_list, x, device)
        ok, ach_err = _check_structural(pipeline, trial, device)
        if not ok:
            return 999.0

        try:
            adapter = PipelineAdapter(pipeline, trial, "trial")
            results = {}
            for rkey, func in metric_funcs.items():
                results[rkey] = func(adapter, dev)
        except Exception:
            return 999.0

        loss = 0.0
        for mdef in target_defs:
            target = targets[mdef.name]
            score = comparison_mod._extract_score(results, mdef.result_key, mdef.score_path)
            if score is None:
                loss += 10.0
                continue
            if "min" in target and score < target["min"]:
                loss += ((target["min"] - score) / (abs(target["min"]) + 1e-30)) ** 2
            if "max" in target and score > target["max"]:
                loss += ((score - target["max"]) / (abs(target["max"]) + 1e-30)) ** 2

        loss += 100.0 * (ach_err or 0)

        if loss < best_loss:
            best_loss = loss
            best_x = list(x)

        return loss

    t0 = time.time()
    gen = 0
    while not es.stop():
        X = es.ask()
        losses = [objective(x) for x in X]
        es.tell(X, losses)
        gen += 1
        if verbose and gen % 10 == 0:
            print(f"  gen {gen}: best={min(losses):.4f}")
        history.append({"gen": gen, "best": float(min(losses))})

    elapsed = time.time() - t0

    # Save
    if best_x is not None:
        opt_params = _build_trial(pipeline, params, param_list, best_x, device)
        ckpt_path = _save_checkpoint(opt_params, checkpoint_dir, name)
    else:
        opt_params = params
        ckpt_path = None

    if verbose:
        print(f"\n  Done: {elapsed:.0f}s, {gen} gen, best_loss={best_loss:.4f}")
        if ckpt_path:
            print(f"  Checkpoint: {ckpt_path}")

    return {
        "optimized_params": opt_params,
        "best_loss": best_loss,
        "history": history,
        "generations": gen,
        "elapsed": elapsed,
        "checkpoint": ckpt_path,
    }


# ── Maximize-wins solver ────────────────────────────────────────

def _solve_maximize_wins(pipeline, params, name, ref_name,
                         free_params, fixed_params,
                         device, generations, population, sigma,
                         checkpoint_dir, verbose):
    """Maximize head-to-head wins against a reference space."""
    from ..metrics.registry import _get_colorbench_modules, PipelineAdapter

    es, param_list = _setup_cma(params, free_params, fixed_params,
                                generations, population, sigma, verbose)

    gpu_metrics, gpu_adv, gpu_perc, pairs_mod, comparison_mod, spaces_mod = _get_colorbench_modules()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Build reference space
    ref_lower = ref_name.lower()
    if ref_lower == "oklab":
        ref_space = spaces_mod.OKLab(dev)
    elif ref_lower in ("cielab", "cie_lab"):
        ref_space = spaces_mod.CIELab(dev)
    else:
        raise ValueError(f"Unknown reference: {ref_name}")

    # Pre-generate pairs
    if verbose:
        print(f"  Generating gradient pairs...")
    pairs_xyz, pair_labels = pairs_mod.generate_all_pairs(dev)

    # All metric groups (we need all to count wins properly)
    all_keys = set()
    for mdef in comparison_mod.METRIC_DEFS:
        all_keys.add(mdef.result_key)

    metric_funcs = _get_metric_funcs(all_keys, gpu_metrics, gpu_adv, gpu_perc,
                                     pairs_xyz, pair_labels)

    # Precompute reference scores (ONCE)
    if verbose:
        print(f"  Precomputing {ref_space.name} baseline...")
    ref_results = {}
    for rkey, func in metric_funcs.items():
        ref_results[rkey] = func(ref_space, dev)

    ref_scores = {}
    for mdef in comparison_mod.METRIC_DEFS:
        score = comparison_mod._extract_score(ref_results, mdef.result_key, mdef.score_path)
        if score is not None:
            ref_scores[mdef.name] = (score, mdef.lower_is_better)

    if verbose:
        print(f"  {ref_space.name}: {len(ref_scores)} metric scores cached")
        print(f"  Optimizing to maximize wins...\n")

    best_wins = 0
    best_win_loss = (0, 999)  # (wins, losses) — maximize wins, minimize losses
    best_x = None
    history = []

    def objective(x):
        nonlocal best_wins, best_win_loss, best_x

        trial = _build_trial(pipeline, params, param_list, x, device)
        ok, ach_err = _check_structural(pipeline, trial, device)
        if not ok:
            return 999.0

        try:
            adapter = PipelineAdapter(pipeline, trial, "trial")
            results = {}
            for rkey, func in metric_funcs.items():
                results[rkey] = func(adapter, dev)
        except Exception:
            return 999.0

        # Count wins vs reference
        wins = 0
        losses = 0
        margin_penalty = 0.0  # How badly we lose on lost metrics

        for mdef in comparison_mod.METRIC_DEFS:
            if mdef.name not in ref_scores:
                continue

            ref_val, lower = ref_scores[mdef.name]
            my_val = comparison_mod._extract_score(results, mdef.result_key, mdef.score_path)
            if my_val is None:
                losses += 1
                continue

            # 1% tie tolerance
            if ref_val != 0:
                rel = abs(my_val - ref_val) / (abs(ref_val) + 1e-30)
                if rel <= 0.01:
                    continue  # tie

            if lower:
                if my_val < ref_val:
                    wins += 1
                else:
                    losses += 1
                    margin_penalty += (my_val - ref_val) / (abs(ref_val) + 1e-30)
            else:
                if my_val > ref_val:
                    wins += 1
                else:
                    losses += 1
                    margin_penalty += (ref_val - my_val) / (abs(ref_val) + 1e-30)

        # Loss function: prioritize wins, then reduce losses, then reduce margin
        loss = -wins * 3.0 + losses * 2.0 + margin_penalty * 0.1 + 50.0 * (ach_err or 0)

        if wins > best_wins or (wins == best_wins and losses < best_win_loss[1]):
            best_wins = wins
            best_win_loss = (wins, losses)
            best_x = [float(v) for v in x]

            if verbose:
                ties = len(ref_scores) - wins - losses
                print(f"  NEW BEST: {wins}-{losses} ({ties} tie) [loss={loss:.2f}]")

            # Auto-save best checkpoint
            trial_save = _build_trial(pipeline, params, param_list, best_x, device)
            _save_checkpoint(trial_save, checkpoint_dir, f"{name}_{wins}w",
                             {"_wins_vs": ref_name, "_wins": wins, "_losses": losses})

        return loss

    t0 = time.time()
    gen = 0
    while not es.stop():
        X = es.ask()
        losses = [objective(x) for x in X]
        es.tell(X, losses)
        gen += 1

        if verbose and gen % 10 == 0:
            print(f"  gen {gen}: best_wins={best_wins}, best_loss={min(losses):.2f}")

        history.append({
            "gen": gen,
            "best_loss": float(min(losses)),
            "best_wins": best_wins,
        })

    elapsed = time.time() - t0

    # Final result
    if best_x is not None:
        opt_params = _build_trial(pipeline, params, param_list, best_x, device)
        ckpt_path = _save_checkpoint(opt_params, checkpoint_dir, f"{name}_final",
                                     {"_wins_vs": ref_name,
                                      "_wins": best_win_loss[0],
                                      "_losses": best_win_loss[1]})
    else:
        opt_params = params
        ckpt_path = None

    if verbose:
        w, l = best_win_loss
        ties = len(ref_scores) - w - l
        print(f"\n  Done: {elapsed:.0f}s, {gen} gen")
        print(f"  Best: {w}-{l} ({ties} tie) vs {ref_name}")
        if ckpt_path:
            print(f"  Checkpoint: {ckpt_path}")

    return {
        "optimized_params": opt_params,
        "best_wins": best_win_loss[0],
        "best_losses": best_win_loss[1],
        "history": history,
        "generations": gen,
        "elapsed": elapsed,
        "checkpoint": ckpt_path,
    }
