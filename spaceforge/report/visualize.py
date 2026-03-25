"""Visual components — gradient strips, gamut cusps, hue wheel.

All output is self-contained HTML/inline SVG. No external dependencies.
"""

from __future__ import annotations

import colorsys
import math
from pathlib import Path

import torch


def _srgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    linear = torch.where(rgb <= 0.04045, rgb / 12.92,
                         ((rgb + 0.055) / 1.055).pow(2.4))
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=rgb.dtype, device=rgb.device)
    return linear @ M.T


def _xyz_to_srgb(xyz: torch.Tensor) -> torch.Tensor:
    M_fwd = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=xyz.dtype, device=xyz.device)
    M_inv = torch.linalg.inv(M_fwd)
    linear = xyz @ M_inv.T
    srgb = torch.where(linear <= 0.0031308, linear * 12.92,
                       1.055 * linear.clamp(min=0).pow(1.0 / 2.4) - 0.055)
    return srgb.clamp(0, 1)


def _to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(round(max(0, min(1, r)) * 255)),
        int(round(max(0, min(1, g)) * 255)),
        int(round(max(0, min(1, b)) * 255)),
    )


# ── Gradient Strips ──────────────────────────────────────────────

# Key test gradients: (name, start_rgb, end_rgb)
KEY_GRADIENTS = [
    ("Red → White", (1, 0, 0), (1, 1, 1)),
    ("Green → White", (0, 1, 0), (1, 1, 1)),
    ("Blue → White", (0, 0, 1), (1, 1, 1)),
    ("Yellow → White", (1, 1, 0), (1, 1, 1)),
    ("Cyan → White", (0, 1, 1), (1, 1, 1)),
    ("Magenta → White", (1, 0, 1), (1, 1, 1)),
    ("Red → Blue", (1, 0, 0), (0, 0, 1)),
    ("Red → Green", (1, 0, 0), (0, 1, 0)),
    ("Blue → Yellow", (0, 0, 1), (1, 1, 0)),
    ("Black → White", (0, 0, 0), (1, 1, 1)),
]


def render_gradient_strips(pipeline, params: dict,
                           n_steps: int = 64,
                           strip_height: int = 40,
                           reference_pipeline=None,
                           reference_params: dict | None = None,
                           reference_name: str = "Reference") -> str:
    """Render gradient strips as HTML.

    If reference_pipeline is provided, shows both side-by-side for comparison.
    """
    rows = []

    for gname, start_rgb, end_rgb in KEY_GRADIENTS:
        # Generate gradient colors
        colors = _compute_gradient(pipeline, params, start_rgb, end_rgb, n_steps)
        strip = _gradient_strip_html(colors, strip_height)

        if reference_pipeline is not None:
            ref_colors = _compute_gradient(reference_pipeline, reference_params or {},
                                           start_rgb, end_rgb, n_steps)
            ref_strip = _gradient_strip_html(ref_colors, strip_height)
            rows.append(
                f'<div class="grad-row">'
                f'<span class="grad-label">{gname}</span>'
                f'<div class="grad-pair">'
                f'<div class="grad-strip">{strip}<div class="grad-tag">SpaceForge</div></div>'
                f'<div class="grad-strip">{ref_strip}<div class="grad-tag">{reference_name}</div></div>'
                f'</div></div>'
            )
        else:
            rows.append(
                f'<div class="grad-row">'
                f'<span class="grad-label">{gname}</span>'
                f'{strip}'
                f'</div>'
            )

    return (
        '<div class="gradient-strips">'
        '<h2>Gradient Strips</h2>'
        + "\n".join(rows)
        + '</div>'
    )


def _compute_gradient(pipeline, params, start_rgb, end_rgb, n_steps):
    """Interpolate in Lab space, convert back to sRGB hex."""
    t = torch.linspace(0, 1, n_steps, dtype=torch.float64).unsqueeze(1)
    start = torch.tensor([start_rgb], dtype=torch.float64)
    end = torch.tensor([end_rgb], dtype=torch.float64)

    # Convert endpoints to XYZ
    xyz_start = _srgb_to_xyz(start)
    xyz_end = _srgb_to_xyz(end)

    # Forward to Lab
    lab_start = pipeline.forward(xyz_start, params)
    lab_end = pipeline.forward(xyz_end, params)

    # Interpolate in Lab
    lab_interp = lab_start * (1 - t) + lab_end * t

    # Inverse to XYZ, then to sRGB
    xyz_interp = pipeline.inverse(lab_interp, params)
    rgb_out = _xyz_to_srgb(xyz_interp)

    return [_to_hex(float(rgb_out[i, 0]), float(rgb_out[i, 1]), float(rgb_out[i, 2]))
            for i in range(n_steps)]


def _gradient_strip_html(colors: list[str], height: int) -> str:
    """Render a list of hex colors as an inline gradient bar."""
    n = len(colors)
    stops = ", ".join(f"{c} {i * 100 / (n - 1):.1f}%" for i, c in enumerate(colors))
    return (f'<div style="height:{height}px;width:100%;border-radius:4px;'
            f'background:linear-gradient(to right, {stops})"></div>')


# ── Gamut Cusp Curves ────────────────────────────────────────────

def render_gamut_cusps(pipeline, params: dict,
                       gamuts: list[str] | None = None,
                       size: int = 400) -> str:
    """Render gamut cusp curves as inline SVG.

    For each hue angle, finds the maximum chroma at the cusp L value.
    """
    if gamuts is None:
        gamuts = ["sRGB"]

    GAMUT_PRIMARIES = {
        "sRGB": torch.tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=torch.float64),
    }

    colors_map = {"sRGB": "#58a6ff", "P3": "#7ee787", "Rec2020": "#d2a8ff"}

    cx, cy = size // 2, size // 2
    r_max = size // 2 - 30

    paths = []

    for gamut_name in gamuts:
        color = colors_map.get(gamut_name, "#e6edf3")

        # Sample 360 hue angles, find max chroma
        cusps = _compute_cusp_curve(pipeline, params, gamut_name)

        if not cusps:
            continue

        # Normalize chroma to fit in SVG
        max_c = max(c for _, c in cusps) if cusps else 1.0
        if max_c < 1e-10:
            max_c = 1.0

        points = []
        for h_deg, chroma in cusps:
            r = (chroma / max_c) * r_max
            angle = math.radians(h_deg - 90)  # 0° at top
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            points.append(f"{px:.1f},{py:.1f}")

        if points:
            points.append(points[0])  # close the path
            path_d = "M " + " L ".join(points)
            paths.append(
                f'<path d="{path_d}" fill="none" stroke="{color}" '
                f'stroke-width="2" opacity="0.9"/>'
                f'<text x="{size - 10}" y="{len(paths) * 20 + 20}" '
                f'fill="{color}" font-size="12" text-anchor="end">{gamut_name}</text>'
            )

    # Background: hue angle labels
    labels = []
    for h in range(0, 360, 30):
        angle = math.radians(h - 90)
        lx = cx + (r_max + 15) * math.cos(angle)
        ly = cy + (r_max + 15) * math.sin(angle)
        labels.append(
            f'<text x="{lx:.0f}" y="{ly:.0f}" fill="#8b949e" font-size="10" '
            f'text-anchor="middle" dominant-baseline="central">{h}°</text>'
        )

    # Concentric circles
    circles = ""
    for frac in [0.25, 0.5, 0.75, 1.0]:
        cr = r_max * frac
        circles += (f'<circle cx="{cx}" cy="{cy}" r="{cr:.0f}" '
                    f'fill="none" stroke="#21262d" stroke-width="1"/>')

    return (
        f'<div class="gamut-cusps"><h2>Gamut Cusp Curves</h2>'
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'{circles}'
        + "\n".join(labels)
        + "\n".join(paths)
        + '</svg></div>'
    )


def _compute_cusp_curve(pipeline, params, gamut_name, n_hues=360):
    """Find max-chroma cusp at each hue angle."""
    cusps = []

    for h_deg in range(n_hues):
        # Generate saturated color at this hue
        r, g, b = colorsys.hsv_to_rgb(h_deg / 360, 1.0, 1.0)
        rgb = torch.tensor([[r, g, b]], dtype=torch.float64)
        xyz = _srgb_to_xyz(rgb)

        try:
            lab = pipeline.forward(xyz, params)
            a, b_ch = float(lab[0, 1]), float(lab[0, 2])
            chroma = math.sqrt(a * a + b_ch * b_ch)
            cusps.append((h_deg, chroma))
        except Exception:
            cusps.append((h_deg, 0.0))

    return cusps


# ── Hue Wheel ────────────────────────────────────────────────────

def render_hue_wheel(pipeline, params: dict, size: int = 400) -> str:
    """Render hue wheel showing primary/secondary positions in Lab space."""
    cx, cy = size // 2, size // 2
    r_wheel = size // 2 - 50

    # 6 primaries + 6 secondaries
    test_colors = [
        ("Red", (1, 0, 0)), ("Yellow", (1, 1, 0)), ("Green", (0, 1, 0)),
        ("Cyan", (0, 1, 1)), ("Blue", (0, 0, 1)), ("Magenta", (1, 0, 1)),
    ]

    # Background: continuous hue ring
    ring_segments = []
    for h in range(360):
        r, g, b = colorsys.hsv_to_rgb(h / 360, 0.8, 0.9)
        hex_c = _to_hex(r, g, b)
        a1 = math.radians(h - 90)
        a2 = math.radians(h + 1 - 90)
        x1 = cx + r_wheel * math.cos(a1)
        y1 = cy + r_wheel * math.sin(a1)
        x2 = cx + r_wheel * math.cos(a2)
        y2 = cy + r_wheel * math.sin(a2)
        ring_segments.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{hex_c}" stroke-width="20" stroke-linecap="round" opacity="0.3"/>'
        )

    # Compute Lab hue angles for test colors
    markers = []
    for name, (r, g, b) in test_colors:
        rgb = torch.tensor([[r, g, b]], dtype=torch.float64)
        xyz = _srgb_to_xyz(rgb)
        lab = pipeline.forward(xyz, params)
        a_val, b_val = float(lab[0, 1]), float(lab[0, 2])
        h_deg = math.degrees(math.atan2(b_val, a_val)) % 360
        chroma = math.sqrt(a_val ** 2 + b_val ** 2)

        # Draw marker
        angle = math.radians(h_deg - 90)
        mx = cx + r_wheel * math.cos(angle)
        my = cy + r_wheel * math.sin(angle)
        hex_c = _to_hex(r, g, b)

        # Label position (outside ring)
        lx = cx + (r_wheel + 30) * math.cos(angle)
        ly = cy + (r_wheel + 30) * math.sin(angle)

        markers.append(
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="8" fill="{hex_c}" '
            f'stroke="#e6edf3" stroke-width="2"/>'
            f'<text x="{lx:.0f}" y="{ly:.0f}" fill="#e6edf3" font-size="11" '
            f'text-anchor="middle" dominant-baseline="central">{name}<tspan '
            f'font-size="9" fill="#8b949e"> {h_deg:.0f}°</tspan></text>'
        )

    # Cross-hairs
    cross = (
        f'<line x1="{cx}" y1="{cy - r_wheel - 5}" x2="{cx}" y2="{cy + r_wheel + 5}" '
        f'stroke="#21262d" stroke-width="1"/>'
        f'<line x1="{cx - r_wheel - 5}" y1="{cy}" x2="{cx + r_wheel + 5}" y2="{cy}" '
        f'stroke="#21262d" stroke-width="1"/>'
    )

    return (
        f'<div class="hue-wheel"><h2>Hue Wheel</h2>'
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        + "\n".join(ring_segments)
        + cross
        + "\n".join(markers)
        + '</svg></div>'
    )


# ── Full Visual Report ───────────────────────────────────────────

def generate_visual_report(pipeline, params: dict, name: str,
                           output_path: str = "visual_report.html",
                           reference_pipeline=None,
                           reference_params: dict | None = None,
                           reference_name: str = "OKLab") -> str:
    """Generate full visual HTML report with all components."""
    gradients_html = render_gradient_strips(
        pipeline, params,
        reference_pipeline=reference_pipeline,
        reference_params=reference_params,
        reference_name=reference_name,
    )
    gamut_html = render_gamut_cusps(pipeline, params)
    hue_html = render_hue_wheel(pipeline, params)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SpaceForge Visual Report — {name}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1000px; margin: 40px auto; padding: 0 20px;
       background: #0d1117; color: #e6edf3; }}
h1 {{ color: #58a6ff; }}
h2 {{ color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
.gradient-strips {{ margin: 24px 0; }}
.grad-row {{ display: flex; align-items: center; gap: 12px; margin: 6px 0; }}
.grad-label {{ min-width: 130px; font-size: 13px; color: #8b949e; text-align: right; }}
.grad-row > div:not(.grad-pair) {{ flex: 1; }}
.grad-pair {{ flex: 1; display: flex; flex-direction: column; gap: 2px; }}
.grad-strip {{ position: relative; }}
.grad-tag {{ position: absolute; right: 4px; top: 2px; font-size: 9px; color: #0d1117;
             background: rgba(255,255,255,0.6); padding: 1px 4px; border-radius: 2px; }}
.gamut-cusps, .hue-wheel {{ margin: 32px 0; }}
svg {{ display: block; margin: 0 auto; }}
.meta {{ color: #8b949e; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>SpaceForge Visual Report</h1>
<p class="meta">Space: <strong>{name}</strong></p>

{gradients_html}
{gamut_html}
{hue_html}

<p class="meta" style="margin-top:40px">Generated by SpaceForge v0.1.0</p>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    return output_path
