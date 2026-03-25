"""3x3 matrix block."""

import torch

from ..pipeline import PipelineBlock


class MatrixBlock(PipelineBlock):
    """3x3 matrix multiplication block.

    Forward:  y = x @ M.T
    Inverse:  x = y @ M_inv.T
    """

    def __init__(self, name: str = "matrix", achromatic_constraint: bool = False):
        self.name = name
        self.n_params = 9
        self.achromatic_constraint = achromatic_constraint

    def _get_matrix(self, params: dict) -> torch.Tensor:
        """Extract matrix from params dict."""
        # Check for direct matrix under block name
        if self.name in params:
            M = params[self.name]
            if isinstance(M, torch.Tensor):
                return M
            if isinstance(M, list):
                return torch.tensor(M, dtype=torch.float64)

        # Check checkpoint
        cp = params.get("_checkpoint", {})
        if self.name in cp:
            M = cp[self.name]
            if isinstance(M, list):
                M = torch.tensor(M, dtype=torch.float64)
            return M

        raise KeyError(f"Matrix '{self.name}' not found in params")

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        M = self._get_matrix(params).to(x.device)
        return x @ M.T

    def inverse(self, y: torch.Tensor, params: dict) -> torch.Tensor:
        M = self._get_matrix(params).to(y.device)

        # Check for precomputed inverse
        inv_key = f"{self.name}_inv"
        cp = params.get("_checkpoint", {})
        if inv_key in cp:
            M_inv = cp[inv_key]
            if isinstance(M_inv, list):
                M_inv = torch.tensor(M_inv, dtype=torch.float64)
            M_inv = M_inv.to(y.device)
        elif inv_key in params and isinstance(params[inv_key], torch.Tensor):
            M_inv = params[inv_key].to(y.device)
        else:
            M_inv = torch.linalg.inv(M)

        return y @ M_inv.T
