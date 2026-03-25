"""Export to Helmlab gen_params.json format."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def export_helmlab(params: dict, output_path: str | None = None) -> dict:
    """Export parameters in Helmlab gen_params.json format.

    This format is used by:
    - src/helmlab/data/gen_params.json (Python)
    - packages/helmlab-js/src/data/gen-params.json (JS)
    """
    cp = params.get("_checkpoint", {})

    out = {}

    # Core matrices
    if "M1" in cp:
        out["M1"] = _to_list(cp["M1"])
    if "M2" in cp:
        out["M2"] = _to_list(cp["M2"])
    if "gamma" in cp:
        out["gamma"] = _to_list(cp["gamma"])

    # Precomputed inverses
    if "M1" in cp:
        M1 = torch.tensor(cp["M1"], dtype=torch.float64)
        out["M1_inv"] = torch.linalg.inv(M1).tolist()
    if "M2" in cp:
        M2 = torch.tensor(cp["M2"], dtype=torch.float64)
        out["M2_inv"] = torch.linalg.inv(M2).tolist()

    # Enrichment
    if "L_corr" in cp:
        out["L_corr"] = _to_list(cp["L_corr"])
    if "delta" in cp:
        out["delta"] = float(cp["delta"])
    if "hue_correction" in cp:
        out["hue_correction"] = _to_list(cp["hue_correction"])
    if "dark_L" in cp:
        out["dark_L"] = _to_list(cp["dark_L"])
    if "L_chroma" in cp:
        out["L_chroma"] = _to_list(cp["L_chroma"])
    if "chroma_power" in cp:
        out["chroma_power"] = float(cp["chroma_power"])

    # Naka-Rushton params
    for key in ["n", "sigma", "s_gain"]:
        if key in cp:
            out[key] = float(cp[key])

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)

    return out


def _to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, list):
        return [_to_list(v) if isinstance(v, torch.Tensor) else v for v in obj]
    return obj
