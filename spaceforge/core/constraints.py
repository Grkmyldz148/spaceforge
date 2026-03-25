"""Structural constraints for pipeline optimization."""

import torch

from .pipeline import Pipeline


# D65 white point in XYZ
D65_XYZ = torch.tensor([[0.95047, 1.00000, 1.08883]], dtype=torch.float64)


def check_achromatic(pipeline: Pipeline, params: dict,
                     device: str | None = None, n_grays: int = 257) -> dict:
    """Check that achromatic colors map to a=b=0.

    Returns max |a|, max |b| across gray ramp.
    """
    dev = _dev(device)
    # Gray ramp: Y from 0 to 1
    Y = torch.linspace(0, 1, n_grays, dtype=torch.float64, device=dev)
    # D65-proportional XYZ
    xyz = D65_XYZ.to(dev) * Y.unsqueeze(1)

    lab = pipeline.forward(xyz, params)
    max_a = float(torch.abs(lab[:, 1]).max())
    max_b = float(torch.abs(lab[:, 2]).max())
    max_ab = max(max_a, max_b)

    return {
        "max_a": max_a,
        "max_b": max_b,
        "max_ab": max_ab,
        "structural": max_ab < 1e-10,
        "pass": max_ab < 0.01,
    }


def check_white_L(pipeline: Pipeline, params: dict,
                  device: str | None = None) -> dict:
    """Check that D65 white maps to L=1.0 (or L=100 depending on scale)."""
    dev = _dev(device)
    white = D65_XYZ.to(dev)
    lab = pipeline.forward(white, params)
    L_white = float(lab[0, 0])

    return {
        "L_white": L_white,
        "pass": 0.9 < L_white < 1.1 or 90 < L_white < 110,
    }


def check_monotonic_L(pipeline: Pipeline, params: dict,
                      device: str | None = None, n_steps: int = 1000) -> dict:
    """Check that L is monotonically increasing along the gray axis."""
    dev = _dev(device)
    Y = torch.linspace(0.001, 1.0, n_steps, dtype=torch.float64, device=dev)
    xyz = D65_XYZ.to(dev) * Y.unsqueeze(1)

    lab = pipeline.forward(xyz, params)
    L = lab[:, 0]

    diffs = L[1:] - L[:-1]
    violations = int((diffs <= 0).sum())
    min_diff = float(diffs.min())

    return {
        "monotonic": violations == 0,
        "violations": violations,
        "min_step": min_diff,
        "pass": violations == 0,
    }


def check_invertibility(pipeline: Pipeline, params: dict,
                        device: str | None = None, n_samples: int = 1000) -> dict:
    """Check pipeline invertibility via random round-trip."""
    from .inverse import verify_roundtrip
    return verify_roundtrip(pipeline, params, n_samples=n_samples, device=device)


def check_condition_numbers(params: dict) -> dict:
    """Check condition numbers of M1 and M2 matrices."""
    cp = params.get("_checkpoint", {})
    results = {}

    for name in ["M1", "M2"]:
        if name in cp:
            M = cp[name]
            if isinstance(M, list):
                M = torch.tensor(M, dtype=torch.float64)
            cond = float(torch.linalg.cond(M))
            results[name] = {
                "condition_number": cond,
                "pass": cond < 100,
            }

    return results


def check_all(pipeline: Pipeline, params: dict,
              device: str | None = None) -> dict:
    """Run all structural constraint checks."""
    return {
        "achromatic": check_achromatic(pipeline, params, device),
        "white_L": check_white_L(pipeline, params, device),
        "monotonic_L": check_monotonic_L(pipeline, params, device),
        "invertibility": check_invertibility(pipeline, params, device),
        "condition_numbers": check_condition_numbers(params),
    }


def _dev(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
