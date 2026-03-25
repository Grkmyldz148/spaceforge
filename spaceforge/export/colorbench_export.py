"""Export pipeline as a ColorBench-compatible space class."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


def export_colorbench_space(pipeline, params: dict, class_name: str = "ForgedSpace",
                            output_path: str | None = None) -> str:
    """Generate a Python file implementing a ColorBench-compatible space class.

    The generated class can be used directly with `colorbench run.py --custom`.
    """
    cp = params.get("_checkpoint", {})

    # Build imports and class
    lines = [
        '"""Auto-generated ColorBench space from SpaceForge."""',
        "",
        "import torch",
        "from colorbench.core.spaces import ColorSpace",
        "",
        "",
        f"class {class_name}(ColorSpace):",
        f'    name = "{class_name}"',
        "",
        "    def __init__(self, device: torch.device):",
    ]

    # Emit matrices
    if "M1" in cp:
        lines.append(f"        self.M1 = torch.tensor({json.dumps(cp['M1'])}, "
                     "device=device, dtype=torch.float64)")
        lines.append("        self.M1_inv = torch.linalg.inv(self.M1)")

    if "M2" in cp:
        lines.append(f"        self.M2 = torch.tensor({json.dumps(cp['M2'])}, "
                     "device=device, dtype=torch.float64)")
        lines.append("        self.M2_inv = torch.linalg.inv(self.M2)")

    if "gamma" in cp:
        g = cp["gamma"]
        if isinstance(g, (int, float)):
            g = [g, g, g]
        lines.append(f"        self.gamma = {json.dumps(g)}")

    if "L_corr" in cp:
        lines.append(f"        self.L_corr = {json.dumps(cp['L_corr'])}")

    # Forward method
    lines.extend([
        "",
        "    def forward(self, xyz):",
        "        lms = xyz @ self.M1.T",
    ])

    # Transfer function
    if "gamma" in cp:
        lines.append("        lms_c = torch.sign(lms) * torch.abs(lms).pow("
                     "torch.tensor(self.gamma, device=lms.device, dtype=torch.float64))")
    else:
        lines.append("        lms_c = torch.sign(lms) * torch.abs(lms).pow(1.0 / 3.0)")

    lines.append("        lab = lms_c @ self.M2.T")

    # L correction
    if "L_corr" in cp:
        lc = cp["L_corr"]
        if len(lc) >= 3:
            lines.extend([
                f"        p1, p2, p3 = {lc[0]}, {lc[1]}, {lc[2]}",
                "        L = lab[:, 0]",
                "        t = L * (1.0 - L)",
                "        lab = lab.clone()",
                "        lab[:, 0] = L + p1 * t + p2 * t * (2.0 * L - 1.0) + p3 * t * t",
            ])

    lines.append("        return lab")

    # Inverse method
    lines.extend([
        "",
        "    def inverse(self, lab):",
    ])

    if "L_corr" in cp and len(cp["L_corr"]) >= 3:
        lc = cp["L_corr"]
        lines.extend([
            "        lab = lab.clone()",
            f"        p1, p2, p3 = {lc[0]}, {lc[1]}, {lc[2]}",
            "        L_target = lab[:, 0]",
            "        L = L_target.clone()",
            "        for _ in range(20):",
            "            t = L * (1.0 - L)",
            "            dt = 1.0 - 2.0 * L",
            "            f_L = L + p1 * t + p2 * t * (2.0 * L - 1.0) + p3 * t * t - L_target",
            "            df = 1.0 + p1 * dt + p2 * (dt * (2.0 * L - 1.0) + t * 2.0) + p3 * 2.0 * t * dt",
            "            df_safe = torch.where(df.abs() < 1e-12, torch.ones_like(df), df)",
            "            L = L - f_L / df_safe",
            "        lab[:, 0] = L",
        ])

    lines.extend([
        "        lms_c = lab @ self.M2_inv.T",
    ])

    if "gamma" in cp:
        lines.append("        inv_gamma = [1.0 / g for g in self.gamma]")
        lines.append("        lms = torch.sign(lms_c) * torch.abs(lms_c).pow("
                     "torch.tensor(inv_gamma, device=lms_c.device, dtype=torch.float64))")
    else:
        lines.append("        lms = torch.sign(lms_c) * torch.abs(lms_c).pow(3.0)")

    lines.append("        return lms @ self.M1_inv.T")

    code = "\n".join(lines) + "\n"

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(code)

    return code
