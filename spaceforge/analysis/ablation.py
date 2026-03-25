"""Ablation study — measure impact of removing pipeline blocks."""

from __future__ import annotations

from ..core.pipeline import Pipeline


def run_ablation(pipeline: Pipeline, params: dict, name: str,
                 block_name: str, device: str | None = None,
                 verbose: bool = True) -> dict:
    """Evaluate pipeline with and without a specific block.

    Args:
        pipeline: Full pipeline
        params: Parameters
        name: Space name
        block_name: Name of block to remove
        device: torch device
        verbose: Print progress

    Returns:
        Dict with baseline scores, ablated scores, and deltas.
    """
    from ..metrics.registry import evaluate, _get_colorbench_modules

    _, _, _, _, comparison_mod, _ = _get_colorbench_modules()

    # Baseline evaluation
    if verbose:
        print(f"  Evaluating baseline ({name})...")
    baseline_results = evaluate(pipeline, params, name=name,
                                device=device, verbose=False)

    # Ablated pipeline
    ablated = pipeline.remove_block(block_name)
    if len(ablated.blocks) == len(pipeline.blocks):
        raise ValueError(f"Block '{block_name}' not found in pipeline. "
                         f"Available: {pipeline.block_names()}")

    if verbose:
        print(f"  Evaluating without '{block_name}'...")
    ablated_results = evaluate(ablated, params, name=f"{name}-{block_name}",
                               device=device, verbose=False)

    # Compute deltas
    metric_defs = comparison_mod.METRIC_DEFS
    deltas = {}

    for mdef in metric_defs:
        score_base = comparison_mod._extract_score(baseline_results, mdef.result_key, mdef.score_path)
        score_abl = comparison_mod._extract_score(ablated_results, mdef.result_key, mdef.score_path)

        if score_base is not None and score_abl is not None:
            delta = score_abl - score_base
            # Determine if the change is good or bad
            if mdef.lower_is_better:
                improvement = delta < 0
            else:
                improvement = delta > 0

            deltas[mdef.name] = {
                "baseline": score_base,
                "ablated": score_abl,
                "delta": delta,
                "pct_change": (delta / (abs(score_base) + 1e-30)) * 100,
                "improved": improvement,
                "unit": mdef.unit,
            }

    result = {
        "block_removed": block_name,
        "pipeline_before": str(pipeline),
        "pipeline_after": str(ablated),
        "deltas": deltas,
    }

    if verbose:
        _print_ablation(result)

    return result


def _print_ablation(result: dict):
    """Print ablation summary."""
    block = result["block_removed"]
    deltas = result["deltas"]

    print(f"\n  Ablation: Removing '{block}'")
    print(f"  {result['pipeline_before']}")
    print(f"  → {result['pipeline_after']}")
    print(f"  {'─' * 70}")

    # Sort by impact
    items = sorted(deltas.items(), key=lambda x: abs(x[1]["pct_change"]), reverse=True)

    improved = sum(1 for _, d in items if d["improved"])
    worsened = sum(1 for _, d in items if not d["improved"] and abs(d["delta"]) > 1e-10)
    unchanged = len(items) - improved - worsened

    print(f"  Impact: {improved} improved, {worsened} worsened, {unchanged} unchanged")
    print()

    for name, d in items[:20]:  # Top 20 by impact
        if abs(d["pct_change"]) < 0.01:
            continue
        arrow = "+" if d["improved"] else "−"
        print(f"  {arrow} {name:35s} {d['baseline']:>10.4f} → {d['ablated']:>10.4f} "
              f"({d['pct_change']:>+7.1f}%)")
