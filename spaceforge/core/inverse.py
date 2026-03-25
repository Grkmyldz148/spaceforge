"""Automatic inverse generation — Newton iteration, bisection, analytical."""

import torch

from .pipeline import Pipeline


def verify_roundtrip(pipeline: Pipeline, params: dict,
                     n_samples: int = 1000, device: str | None = None,
                     seed: int = 42) -> dict:
    """Verify pipeline round-trip accuracy on random XYZ colors.

    Returns dict with max_error, mean_error, and per-sample errors.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Random sRGB -> XYZ
    rgb = torch.rand(n_samples, 3, dtype=torch.float64, generator=gen)
    xyz = _srgb_to_xyz(rgb).to(dev)

    lab = pipeline.forward(xyz, params)
    xyz_rt = pipeline.inverse(lab, params)

    errors = torch.sqrt(((xyz - xyz_rt) ** 2).sum(dim=1))
    return {
        "max_error": float(errors.max()),
        "mean_error": float(errors.mean()),
        "median_error": float(errors.median()),
        "errors": errors.cpu(),
        "n_samples": n_samples,
        "pass": float(errors.max()) < 1e-10,
    }


def verify_roundtrip_full_srgb(pipeline: Pipeline, params: dict,
                                device: str | None = None) -> dict:
    """Full 16.7M sRGB round-trip verification."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Generate all 8-bit sRGB values in batches
    max_error = 0.0
    total = 0
    batch_size = 65536

    for r_start in range(0, 256, 16):
        r_end = min(r_start + 16, 256)
        coords = []
        for r in range(r_start, r_end):
            for g in range(256):
                for b in range(256):
                    coords.append([r / 255.0, g / 255.0, b / 255.0])

        rgb = torch.tensor(coords, dtype=torch.float64)
        xyz = _srgb_to_xyz(rgb).to(dev)

        for i in range(0, len(xyz), batch_size):
            batch = xyz[i:i + batch_size]
            lab = pipeline.forward(batch, params)
            xyz_rt = pipeline.inverse(lab, params)
            err = torch.sqrt(((batch - xyz_rt) ** 2).sum(dim=1)).max()
            max_error = max(max_error, float(err))
            total += len(batch)

    return {
        "max_error": max_error,
        "total_colors": total,
        "pass": max_error < 1e-10,
    }


def _srgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB [0,1] to XYZ D65."""
    # sRGB gamma decode
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )
    # sRGB to XYZ matrix (D65)
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=rgb.dtype, device=rgb.device)
    return linear @ M.T


def _xyz_to_srgb(xyz: torch.Tensor) -> torch.Tensor:
    """Convert XYZ D65 to sRGB [0,1]."""
    M_fwd = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=xyz.dtype, device=xyz.device)
    M_inv = torch.linalg.inv(M_fwd)
    linear = xyz @ M_inv.T
    # sRGB gamma encode
    srgb = torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * linear.clamp(min=0).pow(1.0 / 2.4) - 0.055,
    )
    return srgb
