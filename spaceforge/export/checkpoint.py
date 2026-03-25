"""JSON checkpoint export — compatible with existing checkpoint format."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def export_checkpoint(params: dict, output_path: str | None = None) -> dict:
    """Export parameters as a JSON checkpoint.

    Compatible with the existing helmlab checkpoint format.
    """
    cp = params.get("_checkpoint", {})

    # Clean and serialize
    out = {}
    for key, value in cp.items():
        if key.startswith("_"):
            continue
        out[key] = _to_json(value)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)

    return out


def _to_json(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    return obj
