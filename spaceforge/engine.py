"""SpaceForge engine — main entry point for pipeline evaluation and analysis."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .core.pipeline import Pipeline, load_pipeline, load_from_yaml
from .core.constraints import check_all as check_constraints
from .core.inverse import verify_roundtrip


class SpaceForge:
    """Main engine for color space development.

    Usage:
        sf = SpaceForge("preset.yaml")
        sf.eval()
        sf.eval(vs=["oklab", "cielab"])
        sf.sensitivity(params=["M2"])
        sf.ablation(remove="l_correction")
    """

    def __init__(self, yaml_path: str | None = None,
                 pipeline: Pipeline | None = None,
                 params: dict | None = None,
                 name: str | None = None,
                 device: str | None = None):
        """Initialize from YAML file or direct pipeline/params."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {}

        if yaml_path:
            self.pipeline, self.params, self.config = load_from_yaml(yaml_path)
            self.name = self.config.get("name", Path(yaml_path).stem)
            self.yaml_path = yaml_path
        elif pipeline and params is not None:
            self.pipeline = pipeline
            self.params = params
            self.name = name or "custom"
            self.yaml_path = None
        else:
            raise ValueError("Provide either yaml_path or (pipeline, params)")

    def eval(self, vs: list[str] | None = None, verbose: bool = True,
             cache_dir: str | None = None) -> dict:
        """Run full 46-metric evaluation.

        Args:
            vs: Reference spaces to compare against (e.g. ["oklab", "cielab"])
            verbose: Print progress
            cache_dir: Cache directory for results

        Returns:
            If vs is None: raw results dict
            If vs is provided: Comparison object
        """
        from .metrics.registry import evaluate, evaluate_vs_references

        if vs:
            comp = evaluate_vs_references(
                self.pipeline, self.params, name=self.name,
                refs=vs, device=self.device, verbose=verbose,
            )
            if verbose:
                from colorbench.core.comparison import print_summary
                print_summary(comp)
            return comp
        else:
            results = evaluate(
                self.pipeline, self.params, name=self.name,
                device=self.device, cache_dir=cache_dir, verbose=verbose,
            )
            return results

    def check(self, verbose: bool = True) -> dict:
        """Run structural constraint checks."""
        results = check_constraints(self.pipeline, self.params, self.device)

        if verbose:
            print(f"\n  Structural Checks for '{self.name}':")
            for name, result in results.items():
                status = "PASS" if result.get("pass", False) else "FAIL"
                print(f"    {name:20s}: {status}")
                for k, v in result.items():
                    if k != "pass":
                        print(f"      {k}: {v}")

        return results

    def roundtrip(self, n_samples: int = 1000, verbose: bool = True) -> dict:
        """Quick round-trip verification."""
        result = verify_roundtrip(
            self.pipeline, self.params,
            n_samples=n_samples, device=self.device,
        )
        if verbose:
            status = "PASS" if result["pass"] else "FAIL"
            print(f"  Round-trip ({n_samples} samples): {status} (max_err={result['max_error']:.2e})")
        return result

    def intermediates(self, xyz: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Get intermediate values at each pipeline stage (for root cause)."""
        return self.pipeline.forward_intermediates(xyz, self.params)

    def sensitivity(self, param_names: list[str] | None = None,
                    epsilon: float = 0.001, verbose: bool = True) -> dict:
        """Parameter sensitivity analysis."""
        from .analysis.sensitivity import compute_sensitivity
        return compute_sensitivity(
            self.pipeline, self.params, self.name,
            param_names=param_names, epsilon=epsilon,
            device=self.device, verbose=verbose,
        )

    def ablation(self, remove: str, verbose: bool = True) -> dict:
        """Ablation study — evaluate pipeline without named block."""
        from .analysis.ablation import run_ablation
        return run_ablation(
            self.pipeline, self.params, self.name,
            block_name=remove, device=self.device, verbose=verbose,
        )

    def pareto(self, x_metric: str, y_metric: str,
               sweep_param: str, n_samples: int = 50,
               verbose: bool = True) -> dict:
        """Trade-off frontier between two metrics."""
        from .analysis.pareto import compute_pareto
        return compute_pareto(
            self.pipeline, self.params, self.name,
            x_metric=x_metric, y_metric=y_metric,
            sweep_param=sweep_param, n_samples=n_samples,
            device=self.device, verbose=verbose,
        )

    def feasibility(self, targets: dict, n_samples: int = 1000,
                    verbose: bool = True) -> dict:
        """Check feasibility of metric targets."""
        from .analysis.feasibility import check_feasibility
        return check_feasibility(
            self.pipeline, self.params, self.name,
            targets=targets, n_samples=n_samples,
            device=self.device, verbose=verbose,
        )

    def diff(self, other: "SpaceForge", verbose: bool = True) -> dict:
        """Compare this model against another."""
        from .metrics.registry import evaluate_and_compare
        comp = evaluate_and_compare(
            {self.name: (self.pipeline, self.params),
             other.name: (other.pipeline, other.params)},
            device=self.device, verbose=verbose,
        )
        if verbose:
            from colorbench.core.comparison import print_summary
            print_summary(comp)
        return comp

    def export(self, format: str = "json", output_path: str | None = None) -> str | dict:
        """Export model in various formats."""
        from .export.checkpoint import export_checkpoint
        from .export.helmlab import export_helmlab
        from .export.css import export_css, check_css_compatibility

        if format == "json":
            return export_checkpoint(self.params, output_path)
        elif format == "helmlab":
            return export_helmlab(self.params, output_path)
        elif format == "css":
            return export_css(self.pipeline, self.params, output_path)
        elif format == "css-check":
            return check_css_compatibility(self.pipeline)
        else:
            raise ValueError(f"Unknown format: {format}")

    def solve(self, targets: dict | None = None,
              maximize_wins_vs: str | None = None,
              free_params: list[str] | None = None,
              fixed_params: list[str] | None = None,
              method: str = "cma_es", **kwargs) -> dict:
        """Optimize the color space.

        Two modes:
            targets: satisfy min/max constraints on metrics
            maximize_wins_vs: maximize head-to-head wins vs reference
        """
        from .optimizer.solver import solve
        return solve(
            self.pipeline, self.params, self.name,
            targets=targets, maximize_wins_vs=maximize_wins_vs,
            free_params=free_params, fixed_params=fixed_params,
            method=method, device=self.device, **kwargs,
        )

    def report(self, output_path: str = "report.html",
               vs: list[str] | None = None) -> str:
        """Generate HTML report."""
        from .report.html import generate_report
        results = self.eval(vs=vs, verbose=False)
        return generate_report(results, self.name, output_path)

    def __repr__(self) -> str:
        return f"SpaceForge('{self.name}', {self.pipeline})"
