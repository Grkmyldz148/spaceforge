"""SpaceForge CLI — command-line interface for color space development."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.group()
@click.version_option(version="0.1.0")
def main():
    """SpaceForge — Color Space Development Engine."""
    pass


@main.command()
@click.argument("yaml_path")
@click.option("--vs", multiple=True, help="Reference spaces to compare against (e.g. oklab, cielab)")
@click.option("--device", default=None, help="Device (cuda/cpu)")
@click.option("--cache", default=None, help="Cache directory")
@click.option("--json-out", default=None, help="Save raw results to JSON")
def eval(yaml_path, vs, device, cache, json_out):
    """Full 46-metric evaluation."""
    from .engine import SpaceForge

    sf = SpaceForge(yaml_path, device=device)
    click.echo(f"Evaluating: {sf.name} ({sf.pipeline})")

    if vs:
        comp = sf.eval(vs=list(vs), cache_dir=cache)
        if json_out:
            # Save comparison as JSON
            comp_dict = {
                "space_names": comp.space_names,
                "solo_wins": comp.solo_wins,
                "shared_wins": comp.shared_wins,
                "head_to_head": {f"{k[0]}_vs_{k[1]}": v
                                 for k, v in comp.head_to_head.items()},
            }
            _save_json(comp_dict, json_out)
            click.echo(f"Comparison saved to {json_out}")
    else:
        results = sf.eval(cache_dir=cache)
        if json_out:
            _save_json(results, json_out)
            click.echo(f"Results saved to {json_out}")


@main.command()
@click.argument("yaml_path")
@click.option("--n", default=1000, help="Number of random samples")
def check(yaml_path, n):
    """Run structural constraint checks."""
    from .engine import SpaceForge

    sf = SpaceForge(yaml_path)
    sf.check()
    sf.roundtrip(n_samples=n)


@main.command()
@click.argument("yaml_path")
@click.option("--params", multiple=True, help="Parameter groups to analyze (e.g. M2)")
@click.option("--epsilon", default=0.001, type=float, help="Perturbation size")
def sensitivity(yaml_path, params, epsilon):
    """Parameter sensitivity analysis."""
    from .engine import SpaceForge

    sf = SpaceForge(yaml_path)
    sf.sensitivity(param_names=list(params) if params else None, epsilon=epsilon)


@main.command()
@click.argument("yaml_path")
@click.option("--remove", required=True, help="Block to remove")
def ablation(yaml_path, remove):
    """Ablation study — evaluate without a block."""
    from .engine import SpaceForge

    sf = SpaceForge(yaml_path)
    sf.ablation(remove=remove)


@main.command()
@click.argument("yaml_path")
@click.option("--x", "x_metric", required=True, help="X-axis metric")
@click.option("--y", "y_metric", required=True, help="Y-axis metric")
@click.option("--sweep", required=True, help="Parameter to sweep")
@click.option("--samples", default=50, type=int, help="Number of samples")
def pareto(yaml_path, x_metric, y_metric, sweep, samples):
    """Trade-off frontier between two metrics."""
    from .engine import SpaceForge

    sf = SpaceForge(yaml_path)
    sf.pareto(x_metric, y_metric, sweep, n_samples=samples)


@main.command()
@click.argument("yaml_path")
@click.option("--targets", required=True, help="Comma-separated targets (e.g. cusps>350,munsv<3)")
@click.option("--samples", default=1000, type=int)
def feasibility(yaml_path, targets, samples):
    """Check if metric targets are simultaneously achievable."""
    from .engine import SpaceForge

    targets_dict = _parse_targets(targets)
    sf = SpaceForge(yaml_path)
    sf.feasibility(targets_dict, n_samples=samples)


@main.command()
@click.argument("yaml_a")
@click.argument("yaml_b")
@click.option("--device", default=None)
def diff(yaml_a, yaml_b, device):
    """Compare two models metric-by-metric."""
    from .engine import SpaceForge

    sf_a = SpaceForge(yaml_a, device=device)
    sf_b = SpaceForge(yaml_b, device=device)
    sf_a.diff(sf_b)


@main.command()
@click.argument("yaml_path")
@click.option("--targets", required=True, help="YAML file with optimization targets")
@click.option("--method", default="cma_es", help="Optimization method")
@click.option("--free", multiple=True, help="Free parameter groups")
@click.option("--fixed", multiple=True, help="Fixed parameter groups")
@click.option("--generations", default=300, type=int)
@click.option("--population", default=128, type=int)
def solve(yaml_path, targets, method, free, fixed, generations, population):
    """Constraint-first optimization."""
    import yaml as yaml_mod
    from .engine import SpaceForge

    with open(targets) as f:
        targets_config = yaml_mod.safe_load(f)

    sf = SpaceForge(yaml_path)
    sf.solve(
        targets=targets_config.get("targets", {}),
        free_params=list(free) if free else None,
        fixed_params=list(fixed) if fixed else None,
        method=method,
        generations=generations,
        population=population,
    )


@main.command("rootcause")
@click.argument("yaml_path")
@click.option("--metric", required=True, help="Metric to decompose")
def rootcause(yaml_path, metric):
    """Root cause analysis for a specific metric."""
    from .engine import SpaceForge

    sf = SpaceForge(yaml_path)
    from .analysis.root_cause import analyze_root_cause
    analyze_root_cause(
        sf.pipeline, sf.params, sf.name,
        metric_name=metric, device=sf.device,
    )


@main.command()
@click.argument("yaml_path")
@click.option("--format", "fmt", default="json", help="Export format (json/helmlab/css)")
@click.option("--output", "-o", default=None, help="Output file path")
def export(yaml_path, fmt, output):
    """Export model in various formats."""
    from .engine import SpaceForge

    sf = SpaceForge(yaml_path)
    result = sf.export(format=fmt, output_path=output)
    if isinstance(result, str):
        click.echo(result)
    elif isinstance(result, dict):
        click.echo(json.dumps(result, indent=2))


@main.command()
@click.argument("yaml_path")
@click.option("--output", "-o", default="report.html", help="Output HTML file")
@click.option("--vs", multiple=True, help="Reference spaces")
def report(yaml_path, output, vs):
    """Generate HTML dashboard report."""
    from .engine import SpaceForge

    sf = SpaceForge(yaml_path)
    path = sf.report(output_path=output, vs=list(vs) if vs else None)
    click.echo(f"Report saved to {path}")


@main.command()
@click.argument("yaml_path")
@click.option("--output", "-o", default="visual_report.html", help="Output HTML file")
@click.option("--vs", default=None, help="Reference space for side-by-side (e.g. oklab)")
def visual(yaml_path, output, vs):
    """Generate visual report with gradient strips, gamut cusps, hue wheel."""
    from .engine import SpaceForge
    from .report.visualize import generate_visual_report

    sf = SpaceForge(yaml_path)

    ref_pipeline = ref_params = None
    ref_name = "OKLab"
    if vs:
        ref_sf = _build_reference_sf(vs)
        ref_pipeline = ref_sf.pipeline
        ref_params = ref_sf.params
        ref_name = ref_sf.name

    path = generate_visual_report(
        sf.pipeline, sf.params, sf.name,
        output_path=output,
        reference_pipeline=ref_pipeline,
        reference_params=ref_params,
        reference_name=ref_name,
    )
    click.echo(f"Visual report saved to {path}")


@main.command("batch")
@click.argument("yaml_paths", nargs=-1, required=True)
@click.option("--vs", multiple=True, help="Reference spaces")
@click.option("--device", default=None)
@click.option("--output", "-o", default=None, help="Save HTML comparison report")
def batch(yaml_paths, vs, device, output):
    """Evaluate and compare multiple spaces at once.

    Example: spaceforge batch presets/helmgen_v7b.yaml presets/helmgen_h_v2.yaml --vs oklab
    """
    from .engine import SpaceForge
    from .metrics.registry import evaluate, evaluate_and_compare, _get_colorbench_modules

    spaces = {}
    for yp in yaml_paths:
        sf = SpaceForge(yp, device=device)
        spaces[sf.name] = (sf.pipeline, sf.params)

    # Add reference spaces
    if vs:
        _, _, _, _, _, spaces_mod = _get_colorbench_modules()
        dev_str = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        dev = __import__("torch").device(dev_str)
        for ref in vs:
            ref_sf = _build_reference_sf(ref, device=dev_str)
            spaces[ref_sf.name] = (ref_sf.pipeline, ref_sf.params)

    comp = evaluate_and_compare(spaces, device=device)

    from colorbench.core.comparison import print_summary
    print_summary(comp)

    if output:
        from .report.html import generate_report
        generate_report(comp, "batch", output)
        click.echo(f"\nReport saved to {output}")


@main.command()
@click.argument("watch_dir")
@click.option("--vs", multiple=True, default=["oklab"], help="Reference spaces")
@click.option("--interval", default=10, type=int, help="Poll interval in seconds")
@click.option("--device", default=None)
def watch(watch_dir, vs, interval, device):
    """Watch a directory for new checkpoints and auto-evaluate.

    Example: spaceforge watch checkpoints/ --vs oklab --interval 5
    """
    import time as _time
    from pathlib import Path

    watch_path = Path(watch_dir)
    if not watch_path.exists():
        click.echo(f"Error: directory '{watch_dir}' does not exist")
        return

    seen = set(str(p) for p in watch_path.glob("*.json"))
    click.echo(f"Watching {watch_dir} for new .json files (interval: {interval}s)")
    click.echo(f"Already seen: {len(seen)} files")
    click.echo(f"References: {', '.join(vs)}")
    click.echo("Press Ctrl+C to stop.\n")

    try:
        while True:
            current = set(str(p) for p in watch_path.glob("*.json"))
            new_files = current - seen

            for fpath in sorted(new_files):
                click.echo(f"\n{'=' * 60}")
                click.echo(f"  New checkpoint: {fpath}")
                click.echo(f"{'=' * 60}")

                try:
                    _eval_checkpoint(fpath, list(vs), device)
                except Exception as e:
                    click.echo(f"  Error: {e}")

                seen.add(fpath)

            _time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nStopped.")


def _eval_checkpoint(json_path: str, refs: list[str], device: str | None):
    """Evaluate a raw JSON checkpoint against references."""
    from .engine import SpaceForge
    from .metrics.registry import evaluate_vs_references
    from .core.pipeline import Pipeline
    from .core.blocks import MatrixBlock, CbrtTransfer, LCorrectionBlock
    import json as _json

    with open(json_path) as f:
        cp = _json.load(f)

    # Auto-detect pipeline from checkpoint keys
    blocks = [MatrixBlock(name="M1"), CbrtTransfer(), MatrixBlock(name="M2")]
    if "L_corr" in cp or "c1" in cp:
        blocks.append(LCorrectionBlock(degree=3))

    pipeline = Pipeline(blocks)
    params = {"_checkpoint": cp}

    import os
    name = os.path.basename(json_path).replace(".json", "")
    comp = evaluate_vs_references(pipeline, params, name=name,
                                  refs=refs, device=device, verbose=True)

    from colorbench.core.comparison import print_summary
    print_summary(comp)


def _build_reference_sf(ref_name: str, device: str | None = None):
    """Build a SpaceForge instance for a reference space."""
    from .core.pipeline import Pipeline
    from .core.blocks import MatrixBlock, CbrtTransfer
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ref_lower = ref_name.lower()

    if ref_lower == "oklab":
        M1_srgb = torch.tensor([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ], dtype=torch.float64)
        M_SRGB = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=torch.float64)
        M1 = M1_srgb @ torch.linalg.inv(M_SRGB)
        M2 = torch.tensor([
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ], dtype=torch.float64)

        pipeline = Pipeline([
            MatrixBlock(name="M1"), CbrtTransfer(), MatrixBlock(name="M2"),
        ])
        params = {"M1": M1, "M2": M2}
        from .engine import SpaceForge
        return SpaceForge(pipeline=pipeline, params=params, name="OKLab", device=dev)
    else:
        raise ValueError(f"Unknown reference: {ref_name}. Use 'oklab'.")


# --- Helpers ---

def _parse_targets(s: str) -> dict:
    """Parse 'cusps>350,munsv<3,hue_agree<15' into dict."""
    targets = {}
    for part in s.split(","):
        part = part.strip()
        if ">" in part:
            name, val = part.split(">", 1)
            targets[name.strip()] = {"min": float(val)}
        elif "<" in part:
            name, val = part.split("<", 1)
            targets[name.strip()] = {"max": float(val)}
    return targets


def _save_json(data: dict, path: str):
    import torch

    def _convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(data), f, indent=2)


if __name__ == "__main__":
    main()
