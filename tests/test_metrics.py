"""Tests for metrics system — pipeline adapter and evaluation."""

import pytest
import torch

from spaceforge.core.pipeline import Pipeline
from spaceforge.core.blocks import MatrixBlock, CbrtTransfer
from spaceforge.metrics.registry import PipelineAdapter


class TestPipelineAdapter:
    @pytest.fixture
    def oklab_adapter(self):
        M1 = torch.tensor([
            [0.8189330101, 0.3618667424, -0.1288597137],
            [0.0329845436, 0.9293118715, 0.0361456387],
            [0.0482003018, 0.2643662691, 0.6338517070],
        ], dtype=torch.float64)
        M2 = torch.tensor([
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ], dtype=torch.float64)

        p = Pipeline([
            MatrixBlock(name="M1"),
            CbrtTransfer(),
            MatrixBlock(name="M2"),
        ])
        params = {"M1": M1, "M2": M2}
        return PipelineAdapter(p, params, name="TestOKLab")

    def test_adapter_has_required_attrs(self, oklab_adapter):
        assert hasattr(oklab_adapter, "name")
        assert hasattr(oklab_adapter, "forward")
        assert hasattr(oklab_adapter, "inverse")

    def test_adapter_forward(self, oklab_adapter):
        xyz = torch.tensor([[0.95047, 1.0, 1.08883]], dtype=torch.float64)
        lab = oklab_adapter.forward(xyz)
        assert lab.shape == (1, 3)
        # White should have L close to 1
        assert abs(float(lab[0, 0]) - 1.0) < 0.01

    def test_adapter_roundtrip(self, oklab_adapter):
        gen = torch.Generator().manual_seed(42)
        xyz = torch.rand(50, 3, dtype=torch.float64, generator=gen)
        lab = oklab_adapter.forward(xyz)
        xyz_rt = oklab_adapter.inverse(lab)
        assert torch.allclose(xyz, xyz_rt, atol=1e-13)
