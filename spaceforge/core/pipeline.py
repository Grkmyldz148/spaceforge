"""Composable pipeline system for color space construction."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
import yaml


class PipelineBlock(ABC):
    """Abstract base for pipeline stages."""

    name: str
    n_params: int

    @abstractmethod
    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """Forward transform (N,3) -> (N,3)."""

    @abstractmethod
    def inverse(self, y: torch.Tensor, params: dict) -> torch.Tensor:
        """Inverse transform (N,3) -> (N,3)."""

    def jacobian(self, x: torch.Tensor, params: dict, eps: float = 1e-7) -> torch.Tensor:
        """Numerical Jacobian via finite differences. Returns (N,3,3)."""
        N = x.shape[0]
        J = torch.zeros(N, 3, 3, dtype=x.dtype, device=x.device)
        y0 = self.forward(x, params)
        for j in range(3):
            x_plus = x.clone()
            x_plus[:, j] += eps
            y_plus = self.forward(x_plus, params)
            J[:, :, j] = (y_plus - y0) / eps
        return J

    def param_count(self) -> int:
        return self.n_params


class Pipeline:
    """Composable color space pipeline built from blocks."""

    def __init__(self, blocks: list[PipelineBlock] | None = None):
        self.blocks: list[PipelineBlock] = blocks or []

    def add(self, block: PipelineBlock) -> "Pipeline":
        self.blocks.append(block)
        return self

    def forward(self, xyz: torch.Tensor, params: dict) -> torch.Tensor:
        """XYZ (N,3) -> Lab (N,3) through all blocks."""
        x = xyz
        for block in self.blocks:
            x = block.forward(x, params)
        return x

    def inverse(self, lab: torch.Tensor, params: dict) -> torch.Tensor:
        """Lab (N,3) -> XYZ (N,3) through blocks in reverse."""
        x = lab
        for block in reversed(self.blocks):
            x = block.inverse(x, params)
        return x

    def forward_intermediates(self, xyz: torch.Tensor, params: dict) -> list[tuple[str, torch.Tensor]]:
        """Forward with intermediate values for root cause analysis."""
        intermediates = [("input_XYZ", xyz.clone())]
        x = xyz
        for block in self.blocks:
            x = block.forward(x, params)
            intermediates.append((block.name, x.clone()))
        return intermediates

    def total_params(self) -> int:
        return sum(b.param_count() for b in self.blocks)

    def block_names(self) -> list[str]:
        return [b.name for b in self.blocks]

    def remove_block(self, name: str) -> "Pipeline":
        """Return new pipeline without the named block (for ablation)."""
        new_blocks = [b for b in self.blocks if b.name != name]
        return Pipeline(new_blocks)

    def __repr__(self) -> str:
        parts = " -> ".join(b.name for b in self.blocks)
        return f"Pipeline({parts}, {self.total_params()} params)"


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve device string to torch.device, auto-detecting CUDA."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- YAML Loading ---

def _build_block(spec: dict) -> PipelineBlock:
    """Build a PipelineBlock from a YAML spec dict."""
    from .blocks import (
        MatrixBlock, CbrtTransfer, PowerTransfer, NakaRushtonTransfer,
        LogTransfer, CrossTermBlock, LCorrectionBlock,
        ChromaEnrichmentBlock, HueRotationBlock, BlueFixBlock,
    )

    btype = spec["type"]

    if btype == "matrix":
        return MatrixBlock(
            name=spec.get("name", "matrix"),
            achromatic_constraint=spec.get("achromatic_constraint", False),
        )
    elif btype == "transfer":
        func = spec.get("function", "cbrt")
        if func == "cbrt":
            return CbrtTransfer(name=spec.get("name", "cbrt"))
        elif func == "power":
            return PowerTransfer(
                name=spec.get("name", "power"),
                per_channel=spec.get("per_channel", False),
            )
        elif func == "naka_rushton":
            return NakaRushtonTransfer(name=spec.get("name", "naka_rushton"))
        elif func == "log":
            return LogTransfer(name=spec.get("name", "log"))
        else:
            raise ValueError(f"Unknown transfer function: {func}")
    elif btype == "cross_term":
        return CrossTermBlock(
            name=spec.get("name", "cross_term"),
            formula=spec.get("formula", "lms[0] += d*(Z - k*Y)"),
        )
    elif btype == "l_correction":
        return LCorrectionBlock(
            name=spec.get("name", "l_correction"),
            degree=spec.get("degree", 3),
        )
    elif btype == "chroma_enrichment":
        return ChromaEnrichmentBlock(
            name=spec.get("name", "chroma_enrichment"),
            power=spec.get("power", True),
            l_dependent=spec.get("l_dependent", True),
        )
    elif btype == "hue_rotation":
        return HueRotationBlock(
            name=spec.get("name", "hue_rotation"),
            mode=spec.get("mode", "global"),
        )
    elif btype == "blue_fix":
        return BlueFixBlock(
            name=spec.get("name", "blue_fix"),
            da=spec.get("da", 0.025),
            db=spec.get("db", 0.030),
            h0=spec.get("h0", 250.0),
            sigma=spec.get("sigma", 35.0),
            c_gate=spec.get("c_gate", 0.12),
        )
    else:
        raise ValueError(f"Unknown block type: {btype}")


def _load_config(yaml_path: str) -> dict:
    """Load and return raw YAML config."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _build_pipeline_from_config(config: dict, yaml_dir: str) -> tuple[Pipeline, dict]:
    """Build pipeline and params from parsed config dict.

    Args:
        config: Parsed YAML config
        yaml_dir: Directory of the YAML file (for resolving relative paths)
    """
    import json
    import os

    # Handle builtin spaces
    if "builtin" in config:
        raise ValueError(
            f"'{config['builtin']}' is a built-in reference space, not a pipeline. "
            f"Use --vs {config['builtin']} to compare against it."
        )

    pipeline_specs = config.get("pipeline", [])
    if not isinstance(pipeline_specs, list):
        raise ValueError(f"'pipeline' must be a list of block specs, got {type(pipeline_specs).__name__}")

    blocks = []
    for i, spec in enumerate(pipeline_specs):
        if not isinstance(spec, dict):
            raise ValueError(f"Pipeline block {i} must be a dict, got {type(spec).__name__}")
        if "type" not in spec:
            raise ValueError(f"Pipeline block {i} missing required 'type' key")
        blocks.append(_build_block(spec))

    pipeline = Pipeline(blocks)

    # Extract params from YAML (flattened into a single dict)
    params = {}
    for spec in pipeline_specs:
        block_name = spec.get("name", spec["type"])
        block_params = spec.get("params", {})
        if block_params:
            params[block_name] = block_params

    # Top-level params (matrices, etc.)
    if "params" in config:
        params.update(config["params"])

    # Load from external JSON checkpoint if specified
    if "checkpoint" in config:
        ckpt_path = config["checkpoint"]
        # Resolve relative paths from the YAML file's directory
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(yaml_dir, ckpt_path)
        with open(ckpt_path) as f:
            checkpoint = json.load(f)
        params["_checkpoint"] = checkpoint

    return pipeline, params


def load_pipeline(yaml_path: str) -> tuple[Pipeline, dict]:
    """Load pipeline definition and parameters from YAML file.

    Returns:
        (pipeline, params) where params is a dict with all parameter values.
    """
    import os
    config = _load_config(yaml_path)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    return _build_pipeline_from_config(config, yaml_dir)


def load_from_yaml(yaml_path: str) -> tuple[Pipeline, dict, dict]:
    """Load pipeline, params, and full config from YAML.

    Returns:
        (pipeline, params, config)
    """
    import os
    config = _load_config(yaml_path)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    pipeline, params = _build_pipeline_from_config(config, yaml_dir)
    return pipeline, params, config
