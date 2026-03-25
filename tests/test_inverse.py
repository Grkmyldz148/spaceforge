"""Tests for inverse verification and round-trip accuracy."""

import pytest
import torch

from spaceforge.core.pipeline import Pipeline
from spaceforge.core.blocks import (
    MatrixBlock, CbrtTransfer, LCorrectionBlock,
    PowerTransfer, NakaRushtonTransfer,
)
from spaceforge.core.inverse import verify_roundtrip, _srgb_to_xyz, _xyz_to_srgb
from spaceforge.core.constraints import (
    check_achromatic, check_white_L, check_monotonic_L,
    check_condition_numbers,
)


class TestVerifyRoundtrip:
    def test_identity_pipeline(self):
        p = Pipeline()
        result = verify_roundtrip(p, {}, n_samples=100, device="cpu")
        assert result["pass"]
        assert result["max_error"] < 1e-15

    def test_oklab_roundtrip(self):
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

        result = verify_roundtrip(p, params, n_samples=500, device="cpu")
        assert result["pass"]
        assert result["max_error"] < 1e-13


class TestSRGBConversion:
    def test_white(self):
        white = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
        xyz = _srgb_to_xyz(white)
        # D65 white point ≈ [0.9505, 1.0000, 1.0890]
        assert abs(float(xyz[0, 1]) - 1.0) < 0.001

    def test_black(self):
        black = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        xyz = _srgb_to_xyz(black)
        assert torch.allclose(xyz, torch.zeros_like(xyz), atol=1e-15)

    def test_roundtrip(self):
        rgb = torch.rand(100, 3, dtype=torch.float64)
        xyz = _srgb_to_xyz(rgb)
        rgb_rt = _xyz_to_srgb(xyz)
        assert torch.allclose(rgb, rgb_rt, atol=1e-12)


class TestConstraints:
    @pytest.fixture
    def oklab_pipeline(self):
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
        return p, params

    def test_achromatic(self, oklab_pipeline):
        p, params = oklab_pipeline
        result = check_achromatic(p, params, device="cpu")
        assert result["pass"]

    def test_white_L(self, oklab_pipeline):
        p, params = oklab_pipeline
        result = check_white_L(p, params, device="cpu")
        assert result["pass"]

    def test_monotonic_L(self, oklab_pipeline):
        p, params = oklab_pipeline
        result = check_monotonic_L(p, params, device="cpu")
        assert result["pass"]

    def test_condition_numbers(self):
        M1 = [[0.8189, 0.3619, -0.1289],
               [0.0330, 0.9293, 0.0361],
               [0.0482, 0.2644, 0.6339]]
        result = check_condition_numbers({"_checkpoint": {"M1": M1}})
        assert result["M1"]["pass"]
        assert result["M1"]["condition_number"] < 100
