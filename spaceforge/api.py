"""One-liner API for SpaceForge.

Usage:
    from spaceforge import forge

    # Evaluate a checkpoint
    results = forge("checkpoints/v7b_nodelta.json")

    # Compare against OKLab
    comp = forge("checkpoints/v7b_nodelta.json", vs=["oklab"])

    # Evaluate a YAML preset
    results = forge("presets/helmgen_v7b.yaml")

    # Compare two checkpoints
    comp = forge("a.json", "b.json")

    # Visual report
    forge("checkpoints/v7b_nodelta.json", visual=True)
"""

from __future__ import annotations

import os
from pathlib import Path


def forge(*paths: str,
          vs: list[str] | None = None,
          device: str | None = None,
          visual: bool = False,
          visual_output: str = "visual_report.html",
          verbose: bool = True):
    """Evaluate one or more color spaces with zero boilerplate.

    Args:
        *paths: One or more paths to YAML presets or JSON checkpoints.
        vs: Reference spaces to compare against (e.g. ["oklab", "cielab"]).
        device: torch device ("cuda" / "cpu" / None for auto).
        visual: If True, generate visual HTML report.
        visual_output: Output path for visual report.
        verbose: Print progress.

    Returns:
        - Single path, no vs: raw results dict
        - Single path + vs: Comparison object
        - Multiple paths: Comparison object
        - visual=True: path to HTML report

    Examples:
        >>> from spaceforge import forge
        >>> r = forge("checkpoints/v7b_nodelta.json")
        >>> comp = forge("checkpoints/v7b_nodelta.json", vs=["oklab"])
        >>> forge("my_model.json", visual=True)
    """
    if not paths:
        raise ValueError("Provide at least one path (YAML preset or JSON checkpoint)")

    from .engine import SpaceForge

    engines = []
    for p in paths:
        engines.append(_load_any(p, device))

    if visual:
        from .report.visualize import generate_visual_report

        sf = engines[0]
        ref_pipeline = ref_params = None
        ref_name = "OKLab"

        if vs:
            from .cli import _build_reference_sf
            ref_sf = _build_reference_sf(vs[0], device=device)
            ref_pipeline = ref_sf.pipeline
            ref_params = ref_sf.params
            ref_name = ref_sf.name

        out = generate_visual_report(
            sf.pipeline, sf.params, sf.name,
            output_path=visual_output,
            reference_pipeline=ref_pipeline,
            reference_params=ref_params,
            reference_name=ref_name,
        )
        if verbose:
            print(f"Visual report saved to {out}")
        return out

    if len(engines) == 1 and not vs:
        # Single eval
        return engines[0].eval(verbose=verbose)

    if len(engines) == 1 and vs:
        # Single vs references
        return engines[0].eval(vs=vs, verbose=verbose)

    # Multiple engines → compare all
    from .metrics.registry import evaluate_and_compare
    spaces = {sf.name: (sf.pipeline, sf.params) for sf in engines}

    if vs:
        from .cli import _build_reference_sf
        for ref in vs:
            ref_sf = _build_reference_sf(ref, device=device)
            spaces[ref_sf.name] = (ref_sf.pipeline, ref_sf.params)

    comp = evaluate_and_compare(spaces, device=device, verbose=verbose)

    if verbose:
        from colorbench.core.comparison import print_summary
        print_summary(comp)

    return comp


def _load_any(path: str, device: str | None = None):
    """Load a SpaceForge engine from either a YAML preset or JSON checkpoint."""
    from .engine import SpaceForge
    from .core.pipeline import Pipeline
    from .core.blocks import MatrixBlock, CbrtTransfer, LCorrectionBlock

    p = Path(path)

    if p.suffix in (".yaml", ".yml"):
        return SpaceForge(str(p), device=device)

    elif p.suffix == ".json":
        # Auto-detect pipeline from JSON checkpoint keys
        import json
        with open(p) as f:
            cp = json.load(f)

        blocks = [MatrixBlock(name="M1"), CbrtTransfer(), MatrixBlock(name="M2")]

        if "L_corr" in cp or "c1" in cp:
            blocks.append(LCorrectionBlock(degree=3))

        if "n" in cp and "sigma" in cp:
            # Naka-Rushton checkpoint — rebuild with NR pipeline
            from .core.blocks import NakaRushtonTransfer, ChromaEnrichmentBlock
            blocks = [MatrixBlock(name="M1"), NakaRushtonTransfer(), MatrixBlock(name="M2")]
            if "cp" in cp or "k" in cp:
                blocks.append(ChromaEnrichmentBlock())
            if "c1" in cp:
                blocks.append(LCorrectionBlock(degree=3))

        pipeline = Pipeline(blocks)
        params = {"_checkpoint": cp}
        name = p.stem

        return SpaceForge(pipeline=pipeline, params=params, name=name, device=device)

    else:
        raise ValueError(f"Unknown file type: {p.suffix}. Use .yaml or .json")
