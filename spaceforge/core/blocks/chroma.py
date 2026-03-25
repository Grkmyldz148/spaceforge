"""Chroma enrichment block — power + L-dependent scaling."""

import torch

from ..pipeline import PipelineBlock


class ChromaEnrichmentBlock(PipelineBlock):
    """Chroma enrichment in Lab polar coordinates.

    Components:
    1. Chroma power: C_out = C^p  (sublinear compression, p < 1)
    2. L-dependent scaling: C_out *= exp(k * (L - 0.5))

    Forward: polar decomposition, scale, recombine
    Inverse: analytical (divide by scale, raise to 1/p)
    """

    def __init__(self, name: str = "chroma_enrichment",
                 power: bool = True, l_dependent: bool = True):
        self.name = name
        self.has_power = power
        self.has_l_dep = l_dependent
        self.n_params = (1 if power else 0) + (1 if l_dependent else 0)

    def _get_params(self, params: dict) -> tuple[float, float]:
        bp = params.get(self.name, {})
        if isinstance(bp, dict):
            bp_power = bp.get("chroma_power")
            bp_k = bp.get("k")
        else:
            bp_power = bp_k = None

        cp = params.get("_checkpoint", {})

        # chroma_power: block params → cp namespaced → cp "cp" (H_v2) → default
        if bp_power is not None:
            chroma_power = float(bp_power)
        elif "chroma_power" in cp:
            chroma_power = float(cp["chroma_power"])
        elif "cp" in cp:
            chroma_power = float(cp["cp"])
        else:
            chroma_power = 1.0

        # l_chroma_k: block params → cp namespaced "chroma_k" → cp "l_chroma_k" → cp "k" (H_v2, only if no cross_term conflict)
        if bp_k is not None:
            l_chroma_k = float(bp_k)
        elif "chroma_k" in cp:
            l_chroma_k = float(cp["chroma_k"])
        elif "l_chroma_k" in cp:
            l_chroma_k = float(cp["l_chroma_k"])
        elif "k" in cp and "cross_d" not in cp:
            # Use bare "k" only if there's no cross_term in checkpoint
            # (cross_term uses "cross_k", so bare "k" is safe for chroma)
            l_chroma_k = float(cp["k"])
        elif "k" in cp:
            l_chroma_k = float(cp["k"])
        else:
            l_chroma_k = 0.0

        return chroma_power, l_chroma_k

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        chroma_power, l_chroma_k = self._get_params(params)

        if abs(chroma_power - 1.0) < 1e-12 and abs(l_chroma_k) < 1e-12:
            return x

        out = x.clone()
        L, a, b = x[:, 0], x[:, 1], x[:, 2]
        C = torch.sqrt(a * a + b * b).clamp(min=1e-30)

        scale = torch.ones_like(C)

        # Chroma power
        if self.has_power and abs(chroma_power - 1.0) > 1e-12:
            scale = scale * C.pow(chroma_power - 1.0)

        # L-dependent scaling
        if self.has_l_dep and abs(l_chroma_k) > 1e-12:
            scale = scale * torch.exp(l_chroma_k * (L - 0.5))

        out[:, 1] = a * scale
        out[:, 2] = b * scale
        return out

    def inverse(self, y: torch.Tensor, params: dict) -> torch.Tensor:
        chroma_power, l_chroma_k = self._get_params(params)

        if abs(chroma_power - 1.0) < 1e-12 and abs(l_chroma_k) < 1e-12:
            return y

        out = y.clone()
        L, a, b = y[:, 0], y[:, 1], y[:, 2]
        C_out = torch.sqrt(a * a + b * b).clamp(min=1e-30)

        # Inverse L-dependent scaling
        l_scale = torch.ones_like(C_out)
        if self.has_l_dep and abs(l_chroma_k) > 1e-12:
            l_scale = torch.exp(-l_chroma_k * (L - 0.5))

        # After removing L-dep: C_after_power = C_out * l_scale_inv
        # C_after_power = C_orig^chroma_power
        # C_orig = C_after_power^(1/chroma_power)

        if self.has_power and abs(chroma_power - 1.0) > 1e-12:
            C_mid = C_out * l_scale
            C_orig = C_mid.pow(1.0 / chroma_power)
            total_scale = C_orig / C_out.clamp(min=1e-30)
        else:
            total_scale = l_scale

        out[:, 1] = a * total_scale
        out[:, 2] = b * total_scale
        return out
