"""Cross-term nonlinear block for M1 augmentation."""

import torch

from ..pipeline import PipelineBlock


class CrossTermBlock(PipelineBlock):
    """Nonlinear cross-term in LMS domain.

    Default formula: lms[0] += d * (Z - k * Y)
    where Z = lms[2], Y = lms[1] in XYZ input.

    This adds a blue-selective correction to the L channel.
    Forward: analytical
    Inverse: Newton iteration with analytical Jacobian
    """

    def __init__(self, name: str = "cross_term",
                 formula: str = "lms[0] += d*(Z - k*Y)"):
        self.name = name
        self.formula = formula
        self.n_params = 2  # d, k

    def _get_params(self, params: dict) -> tuple[float, float]:
        bp = params.get(self.name, {})
        cp = params.get("_checkpoint", {})
        d = float(bp.get("d", cp.get("cross_d", -0.6)))
        k = float(bp.get("k", cp.get("cross_k", 1.08883)))
        return d, k

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        d, k = self._get_params(params)
        out = x.clone()
        # lms[0] += d * (lms[2] - k * lms[1])
        out[:, 0] = x[:, 0] + d * (x[:, 2] - k * x[:, 1])
        return out

    def inverse(self, y: torch.Tensor, params: dict) -> torch.Tensor:
        d, k = self._get_params(params)
        # Since lms[1] and lms[2] are unchanged:
        # y[0] = x[0] + d * (x[2] - k * x[1])
        # x[0] = y[0] - d * (y[2] - k * y[1])
        out = y.clone()
        out[:, 0] = y[:, 0] - d * (y[:, 2] - k * y[:, 1])
        return out
