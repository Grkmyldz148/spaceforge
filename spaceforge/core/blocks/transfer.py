"""Transfer function blocks: cbrt, power, naka_rushton, log."""

import torch

from ..pipeline import PipelineBlock


class CbrtTransfer(PipelineBlock):
    """Cube root transfer: sign(x) * |x|^(1/3).

    Zero parameters — structural choice.
    """

    def __init__(self, name: str = "cbrt"):
        self.name = name
        self.n_params = 0

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return torch.sign(x) * torch.abs(x).pow(1.0 / 3.0)

    def inverse(self, y: torch.Tensor, params: dict) -> torch.Tensor:
        return torch.sign(y) * torch.abs(y).pow(3.0)


class PowerTransfer(PipelineBlock):
    """Signed power transfer: sign(x) * |x|^gamma.

    Per-channel or shared gamma.
    """

    def __init__(self, name: str = "power", per_channel: bool = False):
        self.name = name
        self.per_channel = per_channel
        self.n_params = 3 if per_channel else 1

    def _get_gamma(self, params: dict) -> torch.Tensor:
        cp = params.get("_checkpoint", {})

        # Check for 'gamma' in checkpoint
        if "gamma" in cp:
            g = cp["gamma"]
            if isinstance(g, (list, tuple)):
                return torch.tensor(g, dtype=torch.float64)
            return torch.tensor([g, g, g], dtype=torch.float64)

        # Check block-level params
        bp = params.get(self.name, {})
        if isinstance(bp, dict) and "gamma" in bp:
            g = bp["gamma"]
            if isinstance(g, (list, tuple)):
                return torch.tensor(g, dtype=torch.float64)
            return torch.tensor([g, g, g], dtype=torch.float64)

        # Default: cube root
        return torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=torch.float64)

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        gamma = self._get_gamma(params).to(x.device)
        return torch.sign(x) * torch.abs(x).pow(gamma.unsqueeze(0))

    def inverse(self, y: torch.Tensor, params: dict) -> torch.Tensor:
        gamma = self._get_gamma(params).to(y.device)
        inv_gamma = 1.0 / gamma
        return torch.sign(y) * torch.abs(y).pow(inv_gamma.unsqueeze(0))


class NakaRushtonTransfer(PipelineBlock):
    """Naka-Rushton neurophysiological transfer.

    Forward: y = s * x^n / (x^n + sigma^n)
    Inverse: x = sigma * (y / (s - y))^(1/n)

    When clamp_input=True (default for physiological models), input is clamped
    to >= 0 since cone responses are non-negative. When False, signed transfer
    is used: sign(x) * NR(|x|).
    """

    def __init__(self, name: str = "naka_rushton", clamp_input: bool = True):
        self.name = name
        self.n_params = 3  # n, sigma, s_gain
        self.clamp_input = clamp_input

    def _get_params(self, params: dict) -> tuple[float, float, float]:
        bp = params.get(self.name, {})
        cp = params.get("_checkpoint", {})

        # Check block params first, then checkpoint (both "n" and "nr_n" formats)
        n = bp.get("n", cp.get("n", cp.get("nr_n", 0.76)))
        sigma = bp.get("sigma", cp.get("sigma", cp.get("nr_sigma", 0.33)))
        s_gain = bp.get("s_gain", cp.get("s_gain", cp.get("nr_s_gain", 0.71)))
        return float(n), float(sigma), float(s_gain)

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        n, sigma, s_gain = self._get_params(params)
        if self.clamp_input:
            x_pos = x.clamp(min=0)
            x_n = x_pos.pow(n)
        else:
            x_pos = torch.abs(x).clamp(min=1e-30)
            x_n = x_pos.pow(n)
        sigma_n = sigma ** n
        y = s_gain * x_n / (x_n + sigma_n)
        if not self.clamp_input:
            y = torch.sign(x) * y
        return y

    def inverse(self, y: torch.Tensor, params: dict) -> torch.Tensor:
        n, sigma, s_gain = self._get_params(params)
        if self.clamp_input:
            y_pos = y.clamp(min=0)
            upper = torch.tensor(s_gain - 1e-10, dtype=y.dtype, device=y.device)
            y_pos = torch.minimum(y_pos, upper)
            ratio = (y_pos / (s_gain - y_pos).clamp(min=1e-30)).clamp(min=0)
            return sigma * ratio.pow(1.0 / n)
        else:
            y_abs = torch.abs(y).clamp(min=1e-30)
            upper = torch.tensor(s_gain - 1e-10, dtype=y.dtype, device=y.device)
            y_abs = torch.minimum(y_abs, upper)
            ratio = y_abs / (s_gain - y_abs).clamp(min=1e-30)
            x = sigma * ratio.pow(1.0 / n)
            return torch.sign(y) * x


class LogTransfer(PipelineBlock):
    """Logarithmic transfer.

    Forward: y = sign(x) * log(1 + k*|x|) / log(1 + k)
    Inverse: x = sign(y) * (exp(|y| * log(1+k)) - 1) / k
    """

    def __init__(self, name: str = "log"):
        self.name = name
        self.n_params = 1  # k

    def _get_k(self, params: dict) -> float:
        bp = params.get(self.name, {})
        cp = params.get("_checkpoint", {})
        return float(bp.get("k", cp.get("log_k", 10.0)))

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        import math
        k = self._get_k(params)
        denom = math.log(1.0 + k)
        return torch.sign(x) * torch.log(1.0 + k * torch.abs(x)) / denom

    def inverse(self, y: torch.Tensor, params: dict) -> torch.Tensor:
        import math
        k = self._get_k(params)
        log1pk = math.log(1.0 + k)
        return torch.sign(y) * (torch.exp(torch.abs(y) * log1pk) - 1.0) / k
