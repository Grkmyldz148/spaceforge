"""Root cause analysis — trace metric problems to pipeline stages."""

from __future__ import annotations

import math

import torch

from ..core.pipeline import Pipeline
from ..core.inverse import _srgb_to_xyz


def analyze_root_cause(pipeline: Pipeline, params: dict, name: str,
                       metric_name: str, device: str | None = None,
                       verbose: bool = True) -> dict:
    """Decompose a metric problem across pipeline stages.

    For gradient-related metrics: traces hue drift / CV through each stage.
    For gamut metrics: traces cusp geometry through each stage.
    For hue metrics: traces hue angle changes through each stage.

    Args:
        pipeline: Pipeline to analyze
        params: Parameters
        name: Space name
        metric_name: Name of the metric to decompose
        device: torch device
        verbose: Print progress

    Returns:
        Dict with per-stage analysis.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if "hue" in metric_name.lower():
        return _analyze_hue_drift(pipeline, params, dev, verbose)
    elif "gradient" in metric_name.lower() or "cv" in metric_name.lower():
        return _analyze_gradient_cv(pipeline, params, dev, verbose)
    elif "cusp" in metric_name.lower() or "gamut" in metric_name.lower():
        return _analyze_gamut(pipeline, params, dev, verbose)
    elif "achromatic" in metric_name.lower() or "gray" in metric_name.lower():
        return _analyze_achromatic(pipeline, params, dev, verbose)
    else:
        return _analyze_generic(pipeline, params, dev, metric_name, verbose)


def _analyze_hue_drift(pipeline: Pipeline, params: dict,
                       device: torch.device, verbose: bool) -> dict:
    """Trace hue angle changes through pipeline for key colors."""
    # Test colors: 12 hues at max saturation
    import colorsys
    colors = []
    hue_labels = []
    for h_deg in range(0, 360, 30):
        r, g, b = colorsys.hls_to_rgb(h_deg / 360, 0.5, 1.0)
        colors.append([r, g, b])
        hue_labels.append(f"{h_deg}°")

    rgb = torch.tensor(colors, dtype=torch.float64, device=device)
    xyz = _srgb_to_xyz(rgb.cpu()).to(device)

    intermediates = pipeline.forward_intermediates(xyz, params)

    results = {"stages": [], "colors": hue_labels}

    for stage_name, values in intermediates:
        # Compute hue angle from a,b channels (if Lab-like)
        if values.shape[1] >= 3:
            a = values[:, 1] if values.shape[1] > 1 else torch.zeros(len(values), device=device)
            b = values[:, 2] if values.shape[1] > 2 else torch.zeros(len(values), device=device)
            hue_deg = torch.atan2(b, a) * 180 / math.pi
            chroma = torch.sqrt(a * a + b * b)
            L = values[:, 0]

            stage_data = {
                "name": stage_name,
                "hue_deg": hue_deg.cpu().tolist(),
                "chroma": chroma.cpu().tolist(),
                "L": L.cpu().tolist(),
            }
            results["stages"].append(stage_data)

    if verbose:
        _print_hue_trace(results)

    return results


def _analyze_gradient_cv(pipeline: Pipeline, params: dict,
                         device: torch.device, verbose: bool) -> dict:
    """Trace gradient uniformity through pipeline stages."""
    # Blue → White gradient (commonly problematic)
    N = 25
    t = torch.linspace(0, 1, N, dtype=torch.float64, device=device)
    blue = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64, device=device)
    white = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64, device=device)
    rgb = blue * (1 - t.unsqueeze(1)) + white * t.unsqueeze(1)
    xyz = _srgb_to_xyz(rgb.cpu()).to(device)

    intermediates = pipeline.forward_intermediates(xyz, params)

    results = {"gradient": "Blue→White", "stages": []}

    for stage_name, values in intermediates:
        diffs = values[1:] - values[:-1]
        step_sizes = torch.sqrt((diffs ** 2).sum(dim=1))
        mean_step = float(step_sizes.mean())
        std_step = float(step_sizes.std())
        cv = std_step / (mean_step + 1e-30)

        results["stages"].append({
            "name": stage_name,
            "cv": cv,
            "mean_step": mean_step,
            "std_step": std_step,
            "min_step": float(step_sizes.min()),
            "max_step": float(step_sizes.max()),
        })

    if verbose:
        print(f"\n  Gradient CV decomposition ({results['gradient']}):")
        print(f"  {'Stage':25s} {'CV':>8s} {'Mean':>10s} {'Std':>10s}")
        for s in results["stages"]:
            print(f"  {s['name']:25s} {s['cv']:>8.4f} {s['mean_step']:>10.4f} {s['std_step']:>10.4f}")

    return results


def _analyze_gamut(pipeline: Pipeline, params: dict,
                   device: torch.device, verbose: bool) -> dict:
    """Trace gamut boundary through pipeline stages."""
    # Sample gamut boundary at 36 hues
    results = {"stages": []}

    hues = torch.linspace(0, 360, 37, dtype=torch.float64, device=device)[:-1]
    import colorsys
    colors = []
    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(float(h) / 360, 1.0, 1.0)
        colors.append([r, g, b])

    rgb = torch.tensor(colors, dtype=torch.float64, device=device)
    xyz = _srgb_to_xyz(rgb.cpu()).to(device)

    intermediates = pipeline.forward_intermediates(xyz, params)

    for stage_name, values in intermediates:
        if values.shape[1] >= 3:
            a = values[:, 1]
            b_ch = values[:, 2]
            chroma = torch.sqrt(a * a + b_ch * b_ch)
            results["stages"].append({
                "name": stage_name,
                "max_chroma": float(chroma.max()),
                "min_chroma": float(chroma.min()),
                "mean_chroma": float(chroma.mean()),
            })

    if verbose:
        print(f"\n  Gamut boundary through stages:")
        for s in results["stages"]:
            print(f"  {s['name']:25s} chroma: {s['min_chroma']:.4f} – {s['max_chroma']:.4f} (mean {s['mean_chroma']:.4f})")

    return results


def _analyze_achromatic(pipeline: Pipeline, params: dict,
                        device: torch.device, verbose: bool) -> dict:
    """Trace achromatic axis through pipeline stages."""
    D65 = torch.tensor([[0.95047, 1.00000, 1.08883]], dtype=torch.float64, device=device)
    Y = torch.linspace(0.01, 1.0, 100, dtype=torch.float64, device=device)
    xyz = D65 * Y.unsqueeze(1)

    intermediates = pipeline.forward_intermediates(xyz, params)

    results = {"stages": []}
    for stage_name, values in intermediates:
        if values.shape[1] >= 3:
            a = values[:, 1]
            b_ch = values[:, 2]
            max_ab = float(torch.max(torch.abs(a).max(), torch.abs(b_ch).max()))
            results["stages"].append({
                "name": stage_name,
                "max_a": float(torch.abs(a).max()),
                "max_b": float(torch.abs(b_ch).max()),
                "max_ab": max_ab,
            })

    if verbose:
        print(f"\n  Achromatic error through stages:")
        for s in results["stages"]:
            print(f"  {s['name']:25s} max|a|={s['max_a']:.2e}  max|b|={s['max_b']:.2e}")

    return results


def _analyze_generic(pipeline: Pipeline, params: dict,
                     device: torch.device, metric_name: str,
                     verbose: bool) -> dict:
    """Generic stage-by-stage analysis."""
    # Use D65 white and a few test colors
    test_colors = torch.tensor([
        [0.95047, 1.0, 1.08883],  # D65 white
        [0.4124, 0.2127, 0.0193],  # sRGB red
        [0.3576, 0.7152, 0.1192],  # sRGB green
        [0.1804, 0.0722, 0.9503],  # sRGB blue
    ], dtype=torch.float64, device=device)

    intermediates = pipeline.forward_intermediates(test_colors, params)

    results = {"metric": metric_name, "stages": []}
    for stage_name, values in intermediates:
        results["stages"].append({
            "name": stage_name,
            "values": values.cpu().tolist(),
        })

    if verbose:
        print(f"\n  Stage values for {metric_name}:")
        labels = ["White", "Red", "Green", "Blue"]
        for s in results["stages"]:
            print(f"  {s['name']:25s}")
            for i, label in enumerate(labels):
                vals = s["values"][i]
                print(f"    {label:8s}: [{vals[0]:+.6f}, {vals[1]:+.6f}, {vals[2]:+.6f}]")

    return results


def _print_hue_trace(results: dict):
    """Print hue angle trace through stages."""
    print(f"\n  Hue angle through pipeline stages:")
    labels = results["colors"]

    header = f"  {'Color':>8s}"
    for s in results["stages"]:
        header += f" {s['name'][:12]:>12s}"
    print(header)

    for i, label in enumerate(labels):
        line = f"  {label:>8s}"
        for s in results["stages"]:
            h = s["hue_deg"][i]
            line += f" {h:>12.1f}"
        print(line)
