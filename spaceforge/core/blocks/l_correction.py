"""L (lightness) correction block — polynomial correction in Lab domain."""

import torch

from ..pipeline import PipelineBlock


class LCorrectionBlock(PipelineBlock):
    """Polynomial L correction in Lab space.

    Degree 3 (Bernstein-like basis):
        L' = L + p1*L*(1-L) + p2*L*(1-L)*(2L-1) + p3*L^2*(1-L)^2

    Degree 5 (extended):
        L' = L + p1*t + p2*t*(0.5-L) + p3*t^2 + p4*t^2*(0.5-L) + p5*t^3
        where t = L*(1-L)

    Forward: polynomial evaluation
    Inverse: Newton iteration (converges in 3-5 iterations for typical params)
    """

    def __init__(self, name: str = "l_correction", degree: int = 3):
        self.name = name
        self.degree = degree
        self.n_params = degree

    def _get_coeffs(self, params: dict) -> list[float]:
        bp = params.get(self.name, {})
        cp = params.get("_checkpoint", {})

        if "coeffs" in bp:
            return [float(c) for c in bp["coeffs"]]

        if "L_corr" in cp:
            return [float(c) for c in cp["L_corr"]]

        # H_v2 format: single "c1" coefficient → [c1, 0, 0]
        if "c1" in cp and self.degree == 3:
            return [float(cp["c1"]), 0.0, 0.0]

        return [0.0] * self.degree

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        coeffs = self._get_coeffs(params)
        if all(abs(c) < 1e-15 for c in coeffs):
            return x

        out = x.clone()
        L = x[:, 0]

        if self.degree == 3:
            p1, p2, p3 = coeffs[0], coeffs[1], coeffs[2]
            t = L * (1.0 - L)
            L_new = L + p1 * t + p2 * t * (2.0 * L - 1.0) + p3 * t * t
        elif self.degree == 5:
            p1, p2, p3, p4, p5 = coeffs
            t = L * (1.0 - L)
            half_minus_L = 0.5 - L
            L_new = (L + p1 * t + p2 * t * half_minus_L
                     + p3 * t * t + p4 * t * t * half_minus_L
                     + p5 * t * t * t)
        else:
            L_new = L

        out[:, 0] = L_new
        return out

    def inverse(self, y: torch.Tensor, params: dict) -> torch.Tensor:
        coeffs = self._get_coeffs(params)
        if all(abs(c) < 1e-15 for c in coeffs):
            return y

        out = y.clone()
        L_target = y[:, 0]

        # Newton iteration to invert L correction
        L = L_target.clone()
        for _ in range(20):
            if self.degree == 3:
                p1, p2, p3 = coeffs[0], coeffs[1], coeffs[2]
                t = L * (1.0 - L)
                f_L = L + p1 * t + p2 * t * (2.0 * L - 1.0) + p3 * t * t - L_target
                # Derivative
                dt = 1.0 - 2.0 * L
                df = (1.0 + p1 * dt
                      + p2 * (dt * (2.0 * L - 1.0) + t * 2.0)
                      + p3 * 2.0 * t * dt)
            elif self.degree == 5:
                p1, p2, p3, p4, p5 = coeffs
                t = L * (1.0 - L)
                dt = 1.0 - 2.0 * L
                half_minus_L = 0.5 - L
                f_L = (L + p1 * t + p2 * t * half_minus_L
                       + p3 * t * t + p4 * t * t * half_minus_L
                       + p5 * t * t * t - L_target)
                df = (1.0 + p1 * dt
                      + p2 * (dt * half_minus_L - t)
                      + p3 * 2.0 * t * dt
                      + p4 * (2.0 * t * dt * half_minus_L - t * t)
                      + p5 * 3.0 * t * t * dt)
            else:
                break

            # Sign-preserving clamp: avoid division by ~0 without flipping sign
            df_safe = torch.where(df.abs() < 1e-12, torch.ones_like(df), df)
            step = f_L / df_safe
            L = L - step

            # Early exit when converged
            if float(f_L.abs().max()) < 1e-15:
                break

        out[:, 0] = L
        return out
