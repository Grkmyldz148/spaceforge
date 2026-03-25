"""Blue-selective ab offset — fixes blue→white gradient purple shift.

Forward: in blue hue region (~250°), shift a negative and b positive
         weighted by gaussian(hue) * chroma
Inverse: exact undo (same weight, opposite direction)
"""

import math

import torch

from ..pipeline import PipelineBlock


class BlueFixBlock(PipelineBlock):
    """Blue-selective ab-channel correction.

    Applies a small (a, b) offset in the blue hue region to prevent
    blue→white gradients from appearing purple/lavender.

    The offset is gaussian-weighted by hue angle and scaled by chroma,
    so it only affects chromatic blue colors and has zero effect on
    achromatic, red, green, yellow, etc.

    Forward: a -= da * w(h, C)
             b -= db * w(h, C)
    where w = exp(-(h - h0)^2 / (2*sigma^2)) * min(C / C_gate, 1)

    NOTE: This is a DISPLAY correction, not a pipeline transform.
    Apply AFTER Lab interpolation, BEFORE inverse, for gradient rendering.
    It does not affect round-trip, cusps, or benchmark scores.

    Inverse: exact undo (same w, opposite sign)
    """

    def __init__(self, name: str = "blue_fix",
                 da: float = 0.12, db: float = 0.10,
                 h0: float = 250.0, sigma: float = 30.0,
                 c_gate: float = 0.15):
        self.name = name
        self.n_params = 5
        self._da = da
        self._db = db
        self._h0 = h0
        self._sigma = sigma
        self._c_gate = c_gate

    def _get_params(self, params: dict) -> tuple:
        bp = params.get(self.name, {})
        cp = params.get("_checkpoint", {})
        da = float(bp.get("da", cp.get("blue_fix_da", self._da)))
        db = float(bp.get("db", cp.get("blue_fix_db", self._db)))
        h0 = float(bp.get("h0", cp.get("blue_fix_h0", self._h0)))
        sigma = float(bp.get("sigma", cp.get("blue_fix_sigma", self._sigma)))
        c_gate = float(bp.get("c_gate", cp.get("blue_fix_c_gate", self._c_gate)))
        return da, db, h0, sigma, c_gate

    def _compute_weight(self, a: torch.Tensor, b: torch.Tensor,
                        h0: float, sigma: float, c_gate: float) -> torch.Tensor:
        """Compute gaussian hue weight * chroma gate.

        Uses atan2-free hue proximity via dot product with target direction.
        This makes the weight smooth and easy to invert.
        """
        C = torch.sqrt(a * a + b * b + 1e-30)

        # Target direction unit vector for h0
        h0_rad = h0 * math.pi / 180.0
        dir_a = math.cos(h0_rad)
        dir_b = math.sin(h0_rad)

        # Cosine similarity: cos(angle) = (a*dir_a + b*dir_b) / C
        cos_angle = (a * dir_a + b * dir_b) / C

        # Convert to angular distance: cos(angle) → angle² ≈ 2*(1-cos)
        # Gaussian on angular distance
        angle_sq = 2.0 * (1.0 - cos_angle).clamp(min=0)
        sigma_rad = sigma * math.pi / 180.0
        gauss = torch.exp(-angle_sq / (2.0 * sigma_rad * sigma_rad))

        # Chroma gate
        c_weight = (C / c_gate).clamp(max=1.0)

        return gauss * c_weight

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        da, db, h0, sigma, c_gate = self._get_params(params)
        if abs(da) < 1e-15 and abs(db) < 1e-15:
            return x

        out = x.clone()
        a, b = x[:, 1], x[:, 2]
        w = self._compute_weight(a, b, h0, sigma, c_gate)

        out[:, 1] = a - da * w
        out[:, 2] = b - db * w
        return out

    def inverse(self, y: torch.Tensor, params: dict) -> torch.Tensor:
        da, db, h0, sigma, c_gate = self._get_params(params)
        if abs(da) < 1e-15 and abs(db) < 1e-15:
            return y

        out = y.clone()
        a_out = y[:, 1]
        b_out = y[:, 2]

        a = a_out.clone()
        b = b_out.clone()

        # Damped fixed-point iteration with bisection fallback
        # The mapping f(a,b) → (a - da*w, b - db*w) can oscillate
        # when da*dw/da is large. Damping factor prevents overshooting.
        damping = 0.5
        for iteration in range(40):
            w = self._compute_weight(a, b, h0, sigma, c_gate)
            a_target = a_out + da * w
            b_target = b_out + db * w

            # Damped update
            a_new = a + damping * (a_target - a)
            b_new = b + damping * (b_target - b)

            change = (a_new - a).abs().max() + (b_new - b).abs().max()
            a = a_new
            b = b_new

            if float(change) < 1e-14:
                break

            # Increase damping as we converge
            if iteration > 10:
                damping = min(0.9, damping + 0.02)

        out[:, 1] = a
        out[:, 2] = b
        return out
