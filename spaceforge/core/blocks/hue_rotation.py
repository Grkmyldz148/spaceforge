"""Hue rotation block — global or hue-dependent ab-plane rotation."""

import math

import torch

from ..pipeline import PipelineBlock


class HueRotationBlock(PipelineBlock):
    """Rotation in the ab-plane.

    Global mode: single angle theta (1 param)
    Hue-dependent mode: Fourier series rotation (N params)

    Forward: [a', b'] = R(theta) @ [a, b]
    Inverse: [a, b] = R(-theta) @ [a', b']
    """

    def __init__(self, name: str = "hue_rotation", mode: str = "global"):
        self.name = name
        self.mode = mode
        if mode == "global":
            self.n_params = 1
        else:
            self.n_params = 4  # c1, s1, c2, s2 Fourier coefficients

    def _get_theta(self, params: dict, h: torch.Tensor | None = None) -> torch.Tensor | float:
        bp = params.get(self.name, {})
        cp = params.get("_checkpoint", {})

        if self.mode == "global":
            theta_deg = float(bp.get("theta", cp.get("rotation_deg", 0.0)))
            return math.radians(theta_deg)
        else:
            # Hue-dependent: theta(h) = c1*cos(h) + s1*sin(h) + c2*cos(2h) + s2*sin(2h)
            c1 = float(bp.get("c1", cp.get("hue_rot_c1", 0.0)))
            s1 = float(bp.get("s1", cp.get("hue_rot_s1", 0.0)))
            c2 = float(bp.get("c2", cp.get("hue_rot_c2", 0.0)))
            s2 = float(bp.get("s2", cp.get("hue_rot_s2", 0.0)))
            if h is None:
                return 0.0
            return (c1 * torch.cos(h) + s1 * torch.sin(h)
                    + c2 * torch.cos(2 * h) + s2 * torch.sin(2 * h))

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        out = x.clone()
        a, b = x[:, 1], x[:, 2]

        if self.mode == "global":
            theta = self._get_theta(params)
            if abs(theta) < 1e-12:
                return x
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            out[:, 1] = a * cos_t - b * sin_t
            out[:, 2] = a * sin_t + b * cos_t
        else:
            h = torch.atan2(b, a)
            theta = self._get_theta(params, h)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            out[:, 1] = a * cos_t - b * sin_t
            out[:, 2] = a * sin_t + b * cos_t

        return out

    def inverse(self, y: torch.Tensor, params: dict) -> torch.Tensor:
        out = y.clone()
        a, b = y[:, 1], y[:, 2]

        if self.mode == "global":
            theta = self._get_theta(params)
            if abs(theta) < 1e-12:
                return y
            # R(-theta)
            cos_t = math.cos(-theta)
            sin_t = math.sin(-theta)
            out[:, 1] = a * cos_t - b * sin_t
            out[:, 2] = a * sin_t + b * cos_t
        else:
            # For hue-dependent, iterate: compute h from current a,b -> get theta -> unrotate
            for _ in range(10):
                h = torch.atan2(out[:, 2], out[:, 1])
                theta = self._get_theta(params, h)
                cos_t = torch.cos(-theta)
                sin_t = torch.sin(-theta)
                out[:, 1] = a * cos_t - b * sin_t
                out[:, 2] = a * sin_t + b * cos_t

        return out
