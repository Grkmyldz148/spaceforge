"""Tests for pipeline system and block composition."""

import pytest
import torch

from spaceforge.core.pipeline import Pipeline, PipelineBlock
from spaceforge.core.blocks import (
    MatrixBlock, CbrtTransfer, PowerTransfer, NakaRushtonTransfer,
    LogTransfer, CrossTermBlock, LCorrectionBlock,
    ChromaEnrichmentBlock, HueRotationBlock,
)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def random_xyz(device):
    """Random XYZ test data (100 samples)."""
    gen = torch.Generator().manual_seed(42)
    rgb = torch.rand(100, 3, dtype=torch.float64, generator=gen)
    # Simple sRGB->XYZ
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=torch.float64)
    return rgb @ M.T


class TestPipelineComposition:
    def test_empty_pipeline(self, random_xyz):
        p = Pipeline()
        result = p.forward(random_xyz, {})
        assert torch.allclose(result, random_xyz)

    def test_single_block(self, random_xyz):
        p = Pipeline([CbrtTransfer()])
        lab = p.forward(random_xyz, {})
        assert lab.shape == random_xyz.shape

    def test_roundtrip_cbrt(self, random_xyz):
        p = Pipeline([CbrtTransfer()])
        lab = p.forward(random_xyz, {})
        xyz_rt = p.inverse(lab, {})
        assert torch.allclose(random_xyz, xyz_rt, atol=1e-14)

    def test_pipeline_add(self):
        p = Pipeline()
        p.add(CbrtTransfer())
        p.add(MatrixBlock(name="M2"))
        assert len(p.blocks) == 2
        assert p.total_params() == 9  # cbrt=0 + matrix=9

    def test_remove_block(self):
        p = Pipeline([CbrtTransfer(), MatrixBlock(name="M2")])
        p2 = p.remove_block("M2")
        assert len(p2.blocks) == 1
        assert p2.blocks[0].name == "cbrt"

    def test_intermediates(self, random_xyz):
        p = Pipeline([CbrtTransfer(), MatrixBlock(name="M2")])
        M2 = torch.eye(3, dtype=torch.float64)
        params = {"M2": M2}
        intermediates = p.forward_intermediates(random_xyz, params)
        assert len(intermediates) == 3  # input + cbrt + M2
        assert intermediates[0][0] == "input_XYZ"
        assert intermediates[1][0] == "cbrt"
        assert intermediates[2][0] == "M2"


class TestMatrixBlock:
    def test_identity(self, random_xyz):
        block = MatrixBlock(name="M")
        M = torch.eye(3, dtype=torch.float64)
        params = {"M": M}
        result = block.forward(random_xyz, params)
        assert torch.allclose(result, random_xyz)

    def test_roundtrip(self, random_xyz):
        block = MatrixBlock(name="M")
        M = torch.tensor([
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ], dtype=torch.float64)
        params = {"M": M}

        y = block.forward(random_xyz, params)
        x_rt = block.inverse(y, params)
        assert torch.allclose(random_xyz, x_rt, atol=1e-13)

    def test_checkpoint_loading(self, random_xyz):
        block = MatrixBlock(name="M1")
        params = {
            "_checkpoint": {
                "M1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            }
        }
        result = block.forward(random_xyz, params)
        assert torch.allclose(result, random_xyz, atol=1e-14)


class TestTransferBlocks:
    def test_cbrt_roundtrip(self, random_xyz):
        block = CbrtTransfer()
        y = block.forward(random_xyz, {})
        x = block.inverse(y, {})
        assert torch.allclose(random_xyz, x, atol=1e-14)

    def test_power_roundtrip(self, random_xyz):
        block = PowerTransfer(per_channel=True)
        params = {"power": {"gamma": [0.3, 0.35, 0.33]}}
        y = block.forward(random_xyz, params)
        x = block.inverse(y, params)
        assert torch.allclose(random_xyz, x, atol=1e-13)

    def test_power_default_cbrt(self, random_xyz):
        block = PowerTransfer()
        y = block.forward(random_xyz, {})
        cbrt = CbrtTransfer()
        y2 = cbrt.forward(random_xyz, {})
        assert torch.allclose(y, y2, atol=1e-14)

    def test_naka_rushton_roundtrip(self):
        block = NakaRushtonTransfer()
        params = {"naka_rushton": {"n": 0.76, "sigma": 0.33, "s_gain": 0.71}}
        x = torch.rand(50, 3, dtype=torch.float64).clamp(min=0.001)
        y = block.forward(x, params)
        x_rt = block.inverse(y, params)
        assert torch.allclose(x, x_rt, atol=1e-10)

    def test_log_roundtrip(self, random_xyz):
        block = LogTransfer()
        params = {"log": {"k": 10.0}}
        x = random_xyz.clamp(min=0.001)
        y = block.forward(x, params)
        x_rt = block.inverse(y, params)
        assert torch.allclose(x, x_rt, atol=1e-13)


class TestCrossTermBlock:
    def test_roundtrip(self, random_xyz):
        block = CrossTermBlock()
        params = {"cross_term": {"d": -0.6, "k": 1.08883}}
        y = block.forward(random_xyz, params)
        x = block.inverse(y, params)
        assert torch.allclose(random_xyz, x, atol=1e-14)

    def test_identity_when_d_zero(self, random_xyz):
        block = CrossTermBlock()
        params = {"cross_term": {"d": 0.0, "k": 1.0}}
        y = block.forward(random_xyz, params)
        assert torch.allclose(y, random_xyz, atol=1e-15)


class TestLCorrectionBlock:
    def test_roundtrip_degree3(self):
        block = LCorrectionBlock(degree=3)
        params = {"l_correction": {"coeffs": [-0.098, 0.133, 0.304]}}
        x = torch.rand(100, 3, dtype=torch.float64)
        x[:, 0] = x[:, 0].clamp(0.01, 0.99)  # L in (0,1)
        y = block.forward(x, params)
        x_rt = block.inverse(y, params)
        assert torch.allclose(x, x_rt, atol=1e-12)

    def test_roundtrip_degree5(self):
        block = LCorrectionBlock(degree=5)
        params = {"l_correction": {"coeffs": [0.05, -0.02, 0.01, 0.005, -0.003]}}
        x = torch.rand(100, 3, dtype=torch.float64)
        x[:, 0] = x[:, 0].clamp(0.01, 0.99)
        y = block.forward(x, params)
        x_rt = block.inverse(y, params)
        assert torch.allclose(x, x_rt, atol=1e-11)

    def test_identity_when_zero_coeffs(self, random_xyz):
        block = LCorrectionBlock(degree=3)
        params = {"l_correction": {"coeffs": [0.0, 0.0, 0.0]}}
        y = block.forward(random_xyz, params)
        assert torch.allclose(y, random_xyz)

    def test_checkpoint_loading(self):
        block = LCorrectionBlock(degree=3)
        params = {"_checkpoint": {"L_corr": [-0.098, 0.133, 0.304]}}
        x = torch.rand(10, 3, dtype=torch.float64)
        x[:, 0] = x[:, 0].clamp(0.01, 0.99)
        y = block.forward(x, params)
        # L should be modified, a/b unchanged
        assert not torch.allclose(y[:, 0], x[:, 0])
        assert torch.allclose(y[:, 1:], x[:, 1:])


class TestChromaEnrichmentBlock:
    def test_roundtrip_power_only(self):
        block = ChromaEnrichmentBlock(power=True, l_dependent=False)
        params = {"chroma_enrichment": {"chroma_power": 0.75}}
        x = torch.rand(50, 3, dtype=torch.float64) * 2 - 1  # Lab-like
        x[:, 0] = x[:, 0].abs()  # L positive
        y = block.forward(x, params)
        x_rt = block.inverse(y, params)
        assert torch.allclose(x, x_rt, atol=1e-12)

    def test_identity_when_defaults(self, random_xyz):
        block = ChromaEnrichmentBlock()
        y = block.forward(random_xyz, {})
        assert torch.allclose(y, random_xyz)


class TestHueRotationBlock:
    def test_global_roundtrip(self):
        block = HueRotationBlock(mode="global")
        params = {"hue_rotation": {"theta": 45.0}}
        x = torch.rand(50, 3, dtype=torch.float64) * 2 - 1
        y = block.forward(x, params)
        x_rt = block.inverse(y, params)
        assert torch.allclose(x, x_rt, atol=1e-14)

    def test_zero_rotation(self, random_xyz):
        block = HueRotationBlock(mode="global")
        params = {"hue_rotation": {"theta": 0.0}}
        y = block.forward(random_xyz, params)
        assert torch.allclose(y, random_xyz)

    def test_L_preserved(self):
        block = HueRotationBlock(mode="global")
        params = {"hue_rotation": {"theta": 30.0}}
        x = torch.rand(50, 3, dtype=torch.float64)
        y = block.forward(x, params)
        assert torch.allclose(y[:, 0], x[:, 0])  # L unchanged


class TestFullPipeline:
    def test_oklab_pipeline_roundtrip(self, random_xyz):
        """Test OKLab-style pipeline: M1 → cbrt → M2."""
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

        lab = p.forward(random_xyz, params)
        xyz_rt = p.inverse(lab, params)
        assert torch.allclose(random_xyz, xyz_rt, atol=1e-13)

    def test_genspace_v7b_pipeline_roundtrip(self):
        """Test GenSpace v7b pipeline: M1 → cbrt → M2 → L_corr."""
        M1 = torch.tensor([
            [6.2137, -0.5042, -0.4042],
            [-1.1592, 4.3502, 0.5255],
            [0.0008, 0.7227, 2.2278],
        ], dtype=torch.float64)
        M2 = torch.tensor([
            [0.4675, 0.2092, -0.0849],
            [0.4844, -0.3666, -0.1727],
            [-0.0442, 0.3938, -0.3686],
        ], dtype=torch.float64)

        p = Pipeline([
            MatrixBlock(name="M1"),
            CbrtTransfer(),
            MatrixBlock(name="M2"),
            LCorrectionBlock(degree=3),
        ])
        params = {
            "M1": M1, "M2": M2,
            "l_correction": {"coeffs": [-0.098, 0.133, 0.304]},
        }

        # Use sRGB colors for test
        gen = torch.Generator().manual_seed(42)
        rgb = torch.rand(200, 3, dtype=torch.float64, generator=gen)
        M_srgb = torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=torch.float64)
        xyz = rgb @ M_srgb.T

        lab = p.forward(xyz, params)
        xyz_rt = p.inverse(lab, params)
        assert torch.allclose(xyz, xyz_rt, atol=1e-11)
