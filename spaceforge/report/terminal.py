"""Terminal output formatting for SpaceForge results."""

from __future__ import annotations


def print_eval_summary(results: dict, name: str):
    """Print concise terminal summary of evaluation results."""
    try:
        from ..metrics.registry import _get_colorbench_modules
        _, _, _, _, comparison_mod, _ = _get_colorbench_modules()
        metric_defs = comparison_mod.METRIC_DEFS
    except ImportError:
        print(f"  {name}: Results available (install colorbench for summary)")
        return

    print(f"\n  {'=' * 60}")
    print(f"  {name} — Evaluation Summary")
    print(f"  {'=' * 60}")

    current_cat = ""
    for mdef in metric_defs:
        score = comparison_mod._extract_score(results, mdef.result_key, mdef.score_path)
        if score is None:
            continue

        if mdef.category != current_cat:
            current_cat = mdef.category
            print(f"\n  {current_cat}:")

        fmt = mdef.format_str or ".4f"
        if fmt == "d":
            val_str = str(int(score))
        elif fmt == ".2e":
            val_str = f"{score:.2e}"
        else:
            val_str = f"{score:{fmt}}"

        unit = f" {mdef.unit}" if mdef.unit else ""
        print(f"    {mdef.name:35s} {val_str:>12s}{unit}")


def print_constraint_check(constraints: dict, name: str):
    """Print constraint check results."""
    print(f"\n  Constraints — {name}")
    print(f"  {'─' * 50}")

    all_pass = True
    for cname, result in constraints.items():
        passed = result.get("pass", False)
        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(f"  [{status}] {cname}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILED'}")
