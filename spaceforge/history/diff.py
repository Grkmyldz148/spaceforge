"""Model diff — metric-by-metric comparison between two models."""

from __future__ import annotations


def diff_models(results_a: dict, results_b: dict,
                name_a: str = "A", name_b: str = "B",
                verbose: bool = True) -> dict:
    """Compare two evaluation result dicts metric by metric.

    Returns dict with per-metric deltas and summary.
    """
    try:
        from ..metrics.registry import _get_colorbench_modules
        _, _, _, _, comparison_mod, _ = _get_colorbench_modules()
        metric_defs = comparison_mod.METRIC_DEFS
    except ImportError:
        return {"error": "colorbench not available"}

    deltas = {}
    a_wins = b_wins = ties = 0

    for mdef in metric_defs:
        score_a = comparison_mod._extract_score(results_a, mdef.result_key, mdef.score_path)
        score_b = comparison_mod._extract_score(results_b, mdef.result_key, mdef.score_path)

        if score_a is None or score_b is None:
            continue

        delta = score_b - score_a
        rel = abs(delta) / (abs(score_a) + 1e-30)

        if rel <= 0.01:
            winner = "tie"
            ties += 1
        elif mdef.lower_is_better:
            if score_a < score_b:
                winner = name_a
                a_wins += 1
            else:
                winner = name_b
                b_wins += 1
        else:
            if score_a > score_b:
                winner = name_a
                a_wins += 1
            else:
                winner = name_b
                b_wins += 1

        deltas[mdef.name] = {
            "a": score_a,
            "b": score_b,
            "delta": delta,
            "pct": rel * 100 * (1 if delta >= 0 else -1),
            "winner": winner,
            "category": mdef.category,
        }

    result = {
        "name_a": name_a,
        "name_b": name_b,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "deltas": deltas,
    }

    if verbose:
        print(f"\n  {name_a} vs {name_b}: {a_wins}-{b_wins} ({ties} ties)")
        print(f"  {'─' * 70}")

        # Sort by absolute change
        items = sorted(deltas.items(), key=lambda x: abs(x[1]["pct"]), reverse=True)

        for mname, d in items:
            if abs(d["pct"]) < 0.1:
                continue
            arrow = "<<" if d["winner"] == name_a else (">>" if d["winner"] == name_b else "==")
            print(f"  {mname:35s} {d['a']:>10.4f} {arrow} {d['b']:>10.4f} ({d['pct']:>+6.1f}%)")

    return result
