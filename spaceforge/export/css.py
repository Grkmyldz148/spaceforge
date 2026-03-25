"""CSS compatibility check and export."""

from __future__ import annotations

from ..core.pipeline import Pipeline


def check_css_compatibility(pipeline: Pipeline) -> dict:
    """Check if a pipeline is CSS-compatible.

    CSS color spaces need:
    - All blocks have closed-form inverses (no Newton/bisection)
    - Reasonable computational complexity
    - Float32 precision safe
    """
    from ..core.blocks import (
        MatrixBlock, CbrtTransfer, PowerTransfer,
        HueRotationBlock, LCorrectionBlock,
        NakaRushtonTransfer, CrossTermBlock,
        ChromaEnrichmentBlock,
    )

    issues = []
    warnings = []
    total_ops = 0

    for block in pipeline.blocks:
        # Closed-form inverse check
        if isinstance(block, (MatrixBlock, CbrtTransfer, PowerTransfer, HueRotationBlock)):
            # These have exact analytical inverses
            pass
        elif isinstance(block, LCorrectionBlock):
            issues.append(f"'{block.name}': L correction requires Newton iteration for inverse")
        elif isinstance(block, NakaRushtonTransfer):
            warnings.append(f"'{block.name}': Naka-Rushton has analytical inverse but "
                            "numerical stability concerns near sigma boundary")
        elif isinstance(block, CrossTermBlock):
            # Cross term has analytical inverse for the standard formula
            pass
        elif isinstance(block, ChromaEnrichmentBlock):
            if block.has_power:
                warnings.append(f"'{block.name}': Chroma power requires "
                                "iterative inverse when combined with L-dependent scaling")
        else:
            issues.append(f"'{block.name}': Unknown block type, cannot verify inverse")

        # Op count
        if isinstance(block, MatrixBlock):
            total_ops += 9  # matrix multiply
        elif isinstance(block, CbrtTransfer):
            total_ops += 3  # 3 cube roots
        elif isinstance(block, PowerTransfer):
            total_ops += 3  # 3 pow()
        elif isinstance(block, NakaRushtonTransfer):
            total_ops += 12  # pow + div per channel
        else:
            total_ops += 6  # estimate

    # Float32 check
    float32_safe = total_ops < 50 and not any("Newton" in i for i in issues)

    result = {
        "compatible": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "total_ops": total_ops,
        "float32_safe": float32_safe,
        "blocks": len(pipeline.blocks),
        "summary": "CSS-compatible" if not issues else f"{len(issues)} issues found",
    }

    return result


def export_css(pipeline: Pipeline, params: dict,
               output_path: str | None = None) -> str:
    """Export pipeline as CSS custom properties (color-mix compatible).

    This generates a CSS @color-profile declaration with the matrices.
    """
    cp = params.get("_checkpoint", {})

    lines = ["/* SpaceForge — auto-generated CSS color space */"]
    lines.append("/* Note: Custom color spaces require CSS Color Level 5 support */")
    lines.append("")

    # Matrix custom properties
    if "M1" in cp:
        M1 = cp["M1"]
        for i, row in enumerate(M1):
            vals = ", ".join(f"{v:.10f}" for v in row)
            lines.append(f"--sf-m1-row{i}: {vals};")

    if "M2" in cp:
        M2 = cp["M2"]
        for i, row in enumerate(M2):
            vals = ", ".join(f"{v:.10f}" for v in row)
            lines.append(f"--sf-m2-row{i}: {vals};")

    css_text = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(css_text)

    return css_text
