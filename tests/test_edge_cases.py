"""Edge case battery — automatic inverse bug detection."""

import pytest
import torch

from spaceforge.core.pipeline import Pipeline
from spaceforge.core.blocks import (
    MatrixBlock, CbrtTransfer, LCorrectionBlock,
    PowerTransfer, NakaRushtonTransfer, ChromaEnrichmentBlock,
    HueRotationBlock, CrossTermBlock,
)
from spaceforge.core.inverse import _srgb_to_xyz


def _make_oklab_pipeline():
    """Standard OKLab pipeline for testing."""
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


def _make_enriched_pipeline():
    """Pipeline with L correction for testing."""
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
    return p, params


class TestPurePrimaries:
    """Test pure RGB primaries and their XYZ equivalents."""

    @pytest.fixture(params=[
        ("black", [0.0, 0.0, 0.0]),
        ("white", [1.0, 1.0, 1.0]),
        ("red", [1.0, 0.0, 0.0]),
        ("green", [0.0, 1.0, 0.0]),
        ("blue", [0.0, 0.0, 1.0]),
        ("yellow", [1.0, 1.0, 0.0]),
        ("cyan", [0.0, 1.0, 1.0]),
        ("magenta", [1.0, 0.0, 1.0]),
    ])
    def primary(self, request):
        name, rgb = request.param
        rgb_t = torch.tensor([rgb], dtype=torch.float64)
        xyz = _srgb_to_xyz(rgb_t)
        return name, xyz

    def test_oklab_roundtrip(self, primary):
        name, xyz = primary
        p, params = _make_oklab_pipeline()
        lab = p.forward(xyz, params)
        xyz_rt = p.inverse(lab, params)
        err = float(torch.sqrt(((xyz - xyz_rt) ** 2).sum()))
        assert err < 1e-13, f"{name}: roundtrip error {err:.2e}"

    def test_enriched_roundtrip(self, primary):
        name, xyz = primary
        if name == "black":
            pytest.skip("Black requires special handling for L_corr at boundary")
        p, params = _make_enriched_pipeline()
        lab = p.forward(xyz, params)
        xyz_rt = p.inverse(lab, params)
        err = float(torch.sqrt(((xyz - xyz_rt) ** 2).sum()))
        assert err < 1e-10, f"{name}: roundtrip error {err:.2e}"


class TestNearBoundary:
    """Test near-black and near-white colors."""

    def test_near_black(self):
        p, params = _make_oklab_pipeline()
        rgb = torch.tensor([
            [0.001, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, 0.001],
            [0.001, 0.001, 0.001],
        ], dtype=torch.float64)
        xyz = _srgb_to_xyz(rgb)
        lab = p.forward(xyz, params)
        xyz_rt = p.inverse(lab, params)
        err = torch.sqrt(((xyz - xyz_rt) ** 2).sum(dim=1))
        assert err.max() < 1e-12

    def test_near_white(self):
        p, params = _make_oklab_pipeline()
        rgb = torch.tensor([
            [0.999, 0.999, 0.999],
            [1.0, 0.999, 0.999],
            [0.999, 1.0, 0.999],
            [0.999, 0.999, 1.0],
        ], dtype=torch.float64)
        xyz = _srgb_to_xyz(rgb)
        lab = p.forward(xyz, params)
        xyz_rt = p.inverse(lab, params)
        err = torch.sqrt(((xyz - xyz_rt) ** 2).sum(dim=1))
        assert err.max() < 1e-13


class TestGamutBoundary:
    """Test maximum chroma colors at gamut boundary."""

    def test_saturated_colors(self):
        p, params = _make_oklab_pipeline()
        # Maximum saturation sRGB colors
        import colorsys
        colors = []
        for h in range(0, 360, 15):
            r, g, b = colorsys.hsv_to_rgb(h / 360, 1.0, 1.0)
            colors.append([r, g, b])

        rgb = torch.tensor(colors, dtype=torch.float64)
        xyz = _srgb_to_xyz(rgb)
        lab = p.forward(xyz, params)
        xyz_rt = p.inverse(lab, params)
        err = torch.sqrt(((xyz - xyz_rt) ** 2).sum(dim=1))
        assert err.max() < 1e-13


class TestNegativeXYZ:
    """Test with negative XYZ values (wide gamut)."""

    def test_rec2020_primaries(self):
        p, params = _make_oklab_pipeline()
        # Rec.2020 primaries in XYZ
        xyz = torch.tensor([
            [0.6370, 0.2627, 0.0000],  # Red
            [0.1446, 0.6780, 0.0281],   # Green
            [0.1689, 0.0593, 1.0694],    # Blue
        ], dtype=torch.float64)
        lab = p.forward(xyz, params)
        xyz_rt = p.inverse(lab, params)
        err = torch.sqrt(((xyz - xyz_rt) ** 2).sum(dim=1))
        assert err.max() < 1e-13

    def test_wide_gamut_negative_xyz(self):
        """Some wide gamut colors produce negative XYZ."""
        p, params = _make_oklab_pipeline()
        xyz = torch.tensor([
            [-0.1, 0.5, 0.2],
            [0.3, -0.05, 0.8],
            [0.2, 0.3, -0.1],
        ], dtype=torch.float64)
        lab = p.forward(xyz, params)
        xyz_rt = p.inverse(lab, params)
        err = torch.sqrt(((xyz - xyz_rt) ** 2).sum(dim=1))
        assert err.max() < 1e-13


class TestBatchConsistency:
    """Test that batch processing gives same results as individual."""

    def test_batch_vs_single(self):
        p, params = _make_oklab_pipeline()
        gen = torch.Generator().manual_seed(42)
        xyz = torch.rand(50, 3, dtype=torch.float64, generator=gen)

        # Batch forward
        lab_batch = p.forward(xyz, params)

        # Single forward
        for i in range(50):
            lab_single = p.forward(xyz[i:i + 1], params)
            assert torch.allclose(lab_batch[i:i + 1], lab_single, atol=1e-15), \
                f"Batch/single mismatch at index {i}"
