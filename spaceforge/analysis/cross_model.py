"""Cross-model analysis — compare N models and find best-in-class per metric."""

from __future__ import annotations


def cross_model_analysis(models: dict[str, tuple],
                         device: str | None = None,
                         verbose: bool = True) -> dict:
    """Compare N models and find best-in-class per metric.

    Args:
        models: {name: (pipeline, params)} dict
        device: torch device
        verbose: Print progress

    Returns:
        Dict with best-in-class table, overlap analysis, hybrid suggestions.
    """
    from ..metrics.registry import evaluate, _get_colorbench_modules

    _, _, _, _, comparison_mod, _ = _get_colorbench_modules()

    # Evaluate all models
    results_by_model = {}
    for name, (pipeline, params) in models.items():
        if verbose:
            print(f"  Evaluating {name}...")
        results_by_model[name] = evaluate(pipeline, params, name=name,
                                          device=device, verbose=False)

    # Best-in-class table
    metric_defs = comparison_mod.METRIC_DEFS
    best_in_class = {}
    model_wins = {name: 0 for name in models}

    for mdef in metric_defs:
        scores = {}
        for name, results in results_by_model.items():
            score = comparison_mod._extract_score(results, mdef.result_key, mdef.score_path)
            if score is not None:
                scores[name] = score

        if not scores:
            continue

        if mdef.lower_is_better:
            best_name = min(scores, key=scores.get)
        else:
            best_name = max(scores, key=scores.get)

        best_in_class[mdef.name] = {
            "winner": best_name,
            "scores": scores,
            "lower_is_better": mdef.lower_is_better,
        }
        model_wins[best_name] += 1

    # Overlap analysis: pairwise win counts
    overlap = {}
    for m1 in models:
        for m2 in models:
            if m1 >= m2:
                continue
            w1 = sum(1 for e in best_in_class.values() if e["winner"] == m1)
            w2 = sum(1 for e in best_in_class.values() if e["winner"] == m2)
            overlap[(m1, m2)] = {"m1_wins": w1, "m2_wins": w2}

    result = {
        "best_in_class": best_in_class,
        "model_wins": model_wins,
        "pairwise": {f"{k[0]}_vs_{k[1]}": v for k, v in overlap.items()},
    }

    if verbose:
        print(f"\n  Best-in-class across {len(models)} models:")
        print(f"  {'Model':20s} {'Wins':>6s}")
        for name, wins in sorted(model_wins.items(), key=lambda x: -x[1]):
            print(f"  {name:20s} {wins:>6d}")

        print(f"\n  Category breakdown:")
        categories = {}
        for mname, entry in best_in_class.items():
            cat = next((m.category for m in metric_defs if m.name == mname), "?")
            if cat not in categories:
                categories[cat] = {}
            w = entry["winner"]
            categories[cat][w] = categories[cat].get(w, 0) + 1

        for cat, winners in categories.items():
            parts = ", ".join(f"{n}={c}" for n, c in sorted(winners.items(), key=lambda x: -x[1]))
            print(f"    {cat:20s}: {parts}")

    return result
