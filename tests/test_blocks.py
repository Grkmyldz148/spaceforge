"""Tests for individual pipeline blocks — edge cases and numerical stability."""

import pytest
import torch

from spaceforge.core.blocks import (
    MatrixBlock, CbrtTransfer, PowerTransfer, NakaRushtonTransfer,
    LogTransfer, CrossTermBlock, LCorrectionBlock,
    ChromaEnrichmentBlock, HueRotationBlock,
)


class TestEdgeCases:
    """Test blocks with extreme values."""

    def test_cbrt_near_zero(self):
        block = CbrtTransfer()
        x = torch.tensor([[1e-20, 0.0, -1e-20]], dtype=torch.float64)
        y = block.forward(x, {})
        x_rt = block.inverse(y, {})
        assert torch.allclose(x, x_rt, atol=1e-30)

    def test_cbrt_negative(self):
        block = CbrtTransfer()
        x = torch.tensor([[-0.5, -1.0, -0.001]], dtype=torch.float64)
        y = block.forward(x, {})
        assert (y < 0).all()
        x_rt = block.inverse(y, {})
        assert torch.allclose(x, x_rt, atol=1e-14)

    def test_matrix_singular_detection(self):
        block = MatrixBlock(name="M")
        M = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # singular
            [0.0, 0.0, 1.0],
        ], dtype=torch.float64)
        params = {"M": M}
        x = torch.rand(5, 3, dtype=torch.float64)
        with pytest.raises(Exception):
            block.inverse(block.forward(x, params), params)

    def test_power_per_channel_different_gammas(self):
        block = PowerTransfer(per_channel=True)
        params = {"power": {"gamma": [0.2, 0.5, 0.8]}}
        x = torch.rand(50, 3, dtype=torch.float64).clamp(min=0.01)
        y = block.forward(x, params)

        # Each channel should be scaled differently
        ratios = y / x
        assert not torch.allclose(ratios[:, 0], ratios[:, 1], atol=0.01)

        x_rt = block.inverse(y, params)
        assert torch.allclose(x, x_rt, atol=1e-12)

    def test_l_correction_pure_black(self):
        block = LCorrectionBlock(degree=3)
        params = {"l_correction": {"coeffs": [-0.1, 0.1, 0.3]}}
        x = torch.tensor([[0.0, 0.5, -0.3]], dtype=torch.float64)
        y = block.forward(x, params)
        # L=0 should stay 0 (since t = L*(1-L) = 0)
        assert abs(float(y[0, 0])) < 1e-15

    def test_l_correction_pure_white(self):
        block = LCorrectionBlock(degree=3)
        params = {"l_correction": {"coeffs": [-0.1, 0.1, 0.3]}}
        x = torch.tensor([[1.0, 0.5, -0.3]], dtype=torch.float64)
        y = block.forward(x, params)
        # L=1 should stay 1 (since t = 1*(1-1) = 0)
        assert abs(float(y[0, 0]) - 1.0) < 1e-15

    def test_nr_small_values(self):
        block = NakaRushtonTransfer()
        params = {"naka_rushton": {"n": 0.76, "sigma": 0.33, "s_gain": 0.71}}
        x = torch.tensor([[1e-10, 1e-5, 0.001]], dtype=torch.float64)
        y = block.forward(x, params)
        x_rt = block.inverse(y, params)
        assert torch.allclose(x, x_rt, rtol=1e-6)

    def test_hue_rotation_preserves_chroma(self):
        block = HueRotationBlock(mode="global")
        params = {"hue_rotation": {"theta": 60.0}}
        x = torch.rand(50, 3, dtype=torch.float64) * 2 - 1
        C_before = torch.sqrt(x[:, 1] ** 2 + x[:, 2] ** 2)
        y = block.forward(x, params)
        C_after = torch.sqrt(y[:, 1] ** 2 + y[:, 2] ** 2)
        assert torch.allclose(C_before, C_after, atol=1e-14)


class TestBlockParamCount:
    def test_param_counts(self):
        assert MatrixBlock().n_params == 9
        assert CbrtTransfer().n_params == 0
        assert PowerTransfer(per_channel=False).n_params == 1
        assert PowerTransfer(per_channel=True).n_params == 3
        assert NakaRushtonTransfer().n_params == 3
        assert LogTransfer().n_params == 1
        assert CrossTermBlock().n_params == 2
        assert LCorrectionBlock(degree=3).n_params == 3
        assert LCorrectionBlock(degree=5).n_params == 5
        assert HueRotationBlock(mode="global").n_params == 1
        assert HueRotationBlock(mode="hue_dependent").n_params == 4
