"""Metric evaluation registry — delegates to ColorBench.

SpaceForge uses ColorBench as the single source of truth for all metrics.
This module wraps a SpaceForge Pipeline into a ColorBench-compatible ColorSpace
and runs the full 46-metric test suite.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch


def _ensure_colorbench_importable():
    """Add colorbench to sys.path if needed."""
    # Try direct import first
    try:
        from colorbench.core import spaces
        return
    except ImportError:
        pass

    # Try finding it relative to this file
    candidates = [
        Path(__file__).parent.parent.parent.parent / "colorbench",  # spaceforge/../colorbench
        Path.cwd() / "colorbench",
        Path.cwd().parent / "colorbench",
    ]
    for p in candidates:
        if (p / "core" / "spaces.py").exists():
            sys.path.insert(0, str(p.parent))
            return

    raise ImportError(
        "Cannot find colorbench. Expected at ../colorbench/ relative to spaceforge/ "
        "or in the current directory."
    )


class PipelineAdapter:
    """Adapts a SpaceForge Pipeline to the ColorBench ColorSpace protocol.

    ColorBench expects:
        .name: str
        .forward(xyz: Tensor) -> Tensor
        .inverse(lab: Tensor) -> Tensor
    """

    def __init__(self, pipeline, params: dict, name: str = "SpaceForge"):
        self.pipeline = pipeline
        self.params = params
        self.name = name

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.pipeline.forward(xyz, self.params)

    def inverse(self, lab: torch.Tensor) -> torch.Tensor:
        return self.pipeline.inverse(lab, self.params)


def _get_colorbench_modules():
    """Import and return all needed ColorBench modules."""
    _ensure_colorbench_importable()

    from colorbench.core import (
        gpu_metrics,
        gpu_metrics_advanced,
        gpu_metrics_perceptual,
        pairs,
        comparison,
        spaces,
    )
    return gpu_metrics, gpu_metrics_advanced, gpu_metrics_perceptual, pairs, comparison, spaces


def evaluate(pipeline, params: dict, name: str = "SpaceForge",
             device: str | None = None, cache_dir: str | None = None,
             verbose: bool = True) -> dict:
    """Run full 46-metric evaluation on a pipeline.

    Args:
        pipeline: SpaceForge Pipeline instance
        params: Parameter dict
        name: Space name for reporting
        device: torch device string (auto-detected if None)
        cache_dir: Directory to cache results (None = no cache)
        verbose: Print progress

    Returns:
        Full results dict compatible with ColorBench comparison.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Check cache
    cache_key = None
    if cache_dir:
        cache_key = _compute_cache_key(params, pipeline)
        cached = _load_cache(cache_dir, cache_key)
        if cached is not None:
            if verbose:
                print(f"  [{name}] Loaded from cache ({cache_key[:8]})")
            return cached

    gpu_metrics, gpu_metrics_adv, gpu_metrics_perc, pairs_mod, _, _ = _get_colorbench_modules()

    adapter = PipelineAdapter(pipeline, params, name)

    results = {}
    t0 = time.time()

    # Generate gradient pairs (deterministic)
    if verbose:
        print(f"  [{name}] Generating pairs...")
    pairs_xyz, pair_labels = pairs_mod.generate_all_pairs(dev)

    # Core metrics (8)
    if verbose:
        print(f"  [{name}] Core metrics...")
    results["roundtrip"] = gpu_metrics.measure_roundtrip(adapter, dev)
    results["achromatic"] = gpu_metrics.measure_achromatic(adapter, dev)
    results["gradients"] = gpu_metrics.measure_gradients(adapter, pairs_xyz, pair_labels, dev)
    results["gamut"] = gpu_metrics.measure_gamut(adapter, dev)
    results["gamut_mapping"] = gpu_metrics.measure_gamut_mapping(adapter, dev)
    results["hue"] = gpu_metrics.measure_hue(adapter, dev)
    results["specials"] = gpu_metrics.measure_special_gradients(adapter, dev)
    results["stability"] = gpu_metrics.measure_stability(adapter, dev)

    # Advanced metrics (12)
    if verbose:
        print(f"  [{name}] Advanced metrics...")
    results["cvd"] = gpu_metrics_adv.measure_cvd(adapter, dev)
    results["animation"] = gpu_metrics_adv.measure_animation(adapter, dev)
    results["jacobian"] = gpu_metrics_adv.measure_jacobian(adapter, dev)
    results["double_rt"] = gpu_metrics_adv.measure_double_roundtrip(adapter, dev)
    results["quantization"] = gpu_metrics_adv.measure_quantization_symmetry(adapter, dev)
    results["channel_mono"] = gpu_metrics_adv.measure_channel_monotonicity(adapter, dev)
    results["banding"] = gpu_metrics_adv.measure_perceptual_banding(adapter, dev)
    results["hue_leaf"] = gpu_metrics_adv.measure_hue_leaf(adapter, dev)
    results["3color"] = gpu_metrics_adv.measure_3color_gradients(adapter, dev)
    results["cross_gamut"] = gpu_metrics_adv.measure_cross_gamut_consistency(adapter, dev)

    # Perceptual metrics (12)
    if verbose:
        print(f"  [{name}] Perceptual metrics...")
    results["munsell_value"] = gpu_metrics_perc.measure_munsell_value(adapter, dev)
    results["munsell_hue"] = gpu_metrics_perc.measure_munsell_hue(adapter, dev)
    results["macadam_isotropy"] = gpu_metrics_perc.measure_macadam_isotropy(adapter, dev)
    results["hue_agreement"] = gpu_metrics_perc.measure_hue_agreement(adapter, dev)
    results["palette_uniformity"] = gpu_metrics_perc.measure_palette_uniformity(adapter, dev)
    results["tint_shade_hue"] = gpu_metrics_perc.measure_tint_shade_hue(adapter, dev)
    results["dataviz_distinguish"] = gpu_metrics_perc.measure_dataviz_distinguishability(adapter, dev)
    results["multistop_gradient"] = gpu_metrics_perc.measure_multistop_gradient(adapter, dev)
    results["wcag_midpoint"] = gpu_metrics_perc.measure_wcag_midpoint_contrast(adapter, dev)
    results["harmony_accuracy"] = gpu_metrics_perc.measure_harmony_accuracy(adapter, dev)
    results["photo_gamut_map"] = gpu_metrics_perc.measure_photo_gamut_map(adapter, dev)
    results["eased_animation"] = gpu_metrics_perc.measure_eased_animation(adapter, dev)
    results["shade_hue_consistency"] = gpu_metrics_perc.measure_shade_hue_consistency(adapter, dev)
    results["chroma_preservation"] = gpu_metrics_perc.measure_chroma_preservation(adapter, dev)

    elapsed = time.time() - t0
    if verbose:
        print(f"  [{name}] Done in {elapsed:.1f}s")

    # Save cache
    if cache_dir and cache_key:
        _save_cache(cache_dir, cache_key, results)

    return results


def evaluate_and_compare(spaces: dict[str, tuple], device: str | None = None,
                         verbose: bool = True) -> "Comparison":
    """Evaluate multiple pipelines and compare.

    Args:
        spaces: {name: (pipeline, params)} dict
        device: torch device
        verbose: Print progress

    Returns:
        ColorBench Comparison object.
    """
    _, _, _, _, comparison_mod, _ = _get_colorbench_modules()

    results_by_space = {}
    for name, (pipeline, params) in spaces.items():
        results_by_space[name] = evaluate(pipeline, params, name=name,
                                          device=device, verbose=verbose)

    return comparison_mod.compare_spaces(results_by_space)


def evaluate_vs_references(pipeline, params: dict, name: str = "SpaceForge",
                           refs: list[str] | None = None,
                           device: str | None = None,
                           verbose: bool = True) -> "Comparison":
    """Evaluate pipeline against reference spaces (OKLab, CIELab, etc.).

    Args:
        pipeline: SpaceForge Pipeline
        params: Parameters
        name: Name for this pipeline
        refs: List of reference names (default: ["oklab", "cielab"])
        device: torch device

    Returns:
        Comparison object.
    """
    if refs is None:
        refs = ["oklab", "cielab"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    _, _, _, _, comparison_mod, spaces_mod = _get_colorbench_modules()

    results_by_space = {}

    # Build reference spaces
    for ref in refs:
        ref_lower = ref.lower()
        if ref_lower == "oklab":
            ref_space = spaces_mod.OKLab(dev)
        elif ref_lower in ("cielab", "cie_lab", "cie-lab"):
            ref_space = spaces_mod.CIELab(dev)
        else:
            raise ValueError(f"Unknown reference: {ref}")

        results_by_space[ref_space.name] = evaluate(
            _ref_pipeline_wrapper(ref_space), {},
            name=ref_space.name, device=device, verbose=verbose,
        )

    # Evaluate target
    results_by_space[name] = evaluate(pipeline, params, name=name,
                                      device=device, verbose=verbose)

    return comparison_mod.compare_spaces(results_by_space)


class _RefSpaceWrapper:
    """Wraps a ColorBench space as a dummy pipeline for evaluate()."""

    def __init__(self, space):
        self._space = space

    def forward(self, xyz, params):
        return self._space.forward(xyz)

    def inverse(self, lab, params):
        return self._space.inverse(lab)


def _ref_pipeline_wrapper(space):
    return _RefSpaceWrapper(space)


# --- Caching ---

def _compute_cache_key(params: dict, pipeline=None) -> str:
    """Compute hash of params + pipeline identity for caching."""
    def _serialize(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, (list, tuple)):
            return [_serialize(v) for v in obj]
        return obj

    parts = [json.dumps(_serialize(params), sort_keys=True)]

    # Include pipeline structure in cache key so different architectures
    # with identical params don't collide
    if pipeline is not None:
        parts.append(repr(pipeline))

    s = "|".join(parts)
    return hashlib.sha256(s.encode()).hexdigest()


def _load_cache(cache_dir: str, key: str) -> dict | None:
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(cache_dir: str, key: str, results: dict):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / f"{key}.json"

    def _to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, (float, int, str, bool, type(None))):
            return obj
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        return str(obj)

    with open(path, "w") as f:
        json.dump(_to_serializable(results), f)
