# SpaceForge

**Color Space Development Engine** — end-to-end toolkit for designing, evaluating, optimizing, and comparing perceptual color spaces.

SpaceForge replaces the blind trial-and-error cycle of color space research (write optimizer script, run on GPU, wait, benchmark, repeat) with a systematic pipeline: define your space in YAML, evaluate against 46 metrics, analyze trade-offs, and optimize with constraints — all from one tool.

## Why SpaceForge?

Color space development involves a tight loop of matrix optimization, perceptual benchmarking, and architectural exploration. Without tooling, each iteration requires:

1. Writing a new Python class for the pipeline
2. Implementing forward and inverse transforms (and debugging inverse bugs)
3. Running a separate benchmark suite
4. Manually comparing results across experiments
5. Losing track of which checkpoint was best for which metric

SpaceForge collapses this into:

```bash
spaceforge eval my_space.yaml --vs oklab cielab
```

One command. 46 metrics. Head-to-head comparison. Visual report.

## Features

- **Composable Pipeline** — define color spaces as YAML block chains (matrix, cube root, power, Naka-Rushton, L correction, chroma enrichment, hue rotation)
- **Automatic Inverse** — every block has a verified inverse (analytical or Newton iteration), tested to machine precision (1e-15)
- **46-Metric Evaluation** — full [ColorBench](https://github.com/Grkmyldz148/colorbench) integration: gradient uniformity, gamut geometry, hue accuracy, perceptual uniformity, CVD accessibility, and more
- **Multi-Reference Comparison** — compare against OKLab, CIE Lab, or any custom space with head-to-head scoring
- **Sensitivity Analysis** — parameter-to-metric Jacobian: "changing M2[1,0] by 0.01 shifts Hue RMS by +2.3 degrees"
- **Ablation Study** — remove a block, measure the impact on all 46 metrics
- **Pareto Frontier** — sweep a parameter, find the trade-off curve between two metrics
- **Feasibility Analysis** — "can I get cusps > 350 AND hue RMS < 15 AND gradient CV < 36% simultaneously?"
- **Root Cause Analysis** — trace a metric problem through each pipeline stage
- **Constraint-First Optimizer** — CMA-ES with hard structural constraints (achromatic axis, white point) and soft metric targets
- **Visual Report** — gradient strips, gamut cusp curves, hue wheel — all as inline HTML/SVG
- **Watch Mode** — monitor a directory for new checkpoints, auto-evaluate, alert on regressions
- **History & Regression** — every evaluation is saved; detect when a change makes things worse
- **Export** — JSON checkpoint, Helmlab gen_params.json, CSS custom properties, ColorBench space class
- **One-Liner API** — `from spaceforge import forge; forge("checkpoint.json", vs=["oklab"])`

## Installation

```bash
git clone https://github.com/Grkmyldz148/spaceforge.git
cd spaceforge
pip install -e .
```

Requires Python 3.10+, PyTorch 2.0+. CUDA auto-detected for GPU acceleration.

## Quick Start

### Evaluate a color space

```bash
# Single evaluation (46 metrics)
spaceforge eval presets/oklab.yaml

# Compare against OKLab and CIE Lab
spaceforge eval presets/helmgen_v7b.yaml --vs oklab --vs cielab

# Batch: compare multiple spaces at once
spaceforge batch presets/helmgen_v7b.yaml presets/helmgen_h_v2.yaml --vs oklab
```

### Visual report

```bash
# Gradient strips + gamut cusps + hue wheel
spaceforge visual presets/helmgen_v7b.yaml -o report.html

# Side-by-side comparison with OKLab
spaceforge visual presets/helmgen_v7b.yaml --vs oklab -o comparison.html
```

### Python API

```python
from spaceforge import forge

# Evaluate a JSON checkpoint
results = forge("checkpoints/model.json")

# Compare against OKLab
comparison = forge("checkpoints/model.json", vs=["oklab"])

# Compare two models
comparison = forge("model_a.json", "model_b.json")

# Visual report
forge("checkpoints/model.json", visual=True)
```

### Structural checks

```bash
# Verify achromatic axis, white point, monotonicity, invertibility
spaceforge check presets/helmgen_v7b.yaml
```

### Analysis

```bash
# Which parameters affect which metrics?
spaceforge sensitivity presets/helmgen_v7b.yaml --params M2

# What happens without L correction?
spaceforge ablation presets/helmgen_v7b.yaml --remove l_correction

# Trade-off: gradient CV vs gamut cusps
spaceforge pareto presets/helmgen_v7b.yaml --x "Gradient CV (mean)" --y "sRGB valid cusps" --sweep "M2[1,0]"

# Can I satisfy all these targets?
spaceforge feasibility presets/helmgen_v7b.yaml --targets "sRGB valid cusps>350,Hue RMS<15"

# Why is Munsell Hue spacing bad? Trace through pipeline stages.
spaceforge rootcause presets/helmgen_v7b.yaml --metric munsell_hue
```

### Optimization

```bash
# Constraint-first: satisfy targets, don't just minimize loss
spaceforge solve presets/helmgen_v7b.yaml --targets targets.yaml --method cma_es --free M2 L_corr --fixed M1
```

### Watch mode

```bash
# Monitor for new checkpoints, auto-evaluate
spaceforge watch checkpoints/ --vs oklab --interval 5
```

## Defining a Color Space

Color spaces are defined as YAML pipelines — ordered chains of composable blocks:

```yaml
name: MyNewSpace

checkpoint: path/to/params.json   # relative to this YAML file

pipeline:
  - type: matrix
    name: M1                      # 3x3 XYZ → LMS
  - type: transfer
    function: cbrt                 # cube root compression
  - type: matrix
    name: M2                      # LMS' → Lab
  - type: l_correction
    degree: 3                      # polynomial L refinement
```

### Available blocks

| Block | Forward | Inverse | Params |
|-------|---------|---------|--------|
| `matrix` | `x @ M.T` | `x @ M_inv.T` | 9 |
| `transfer: cbrt` | `sign(x) * \|x\|^(1/3)` | `sign(x) * \|x\|^3` | 0 |
| `transfer: power` | `sign(x) * \|x\|^gamma` | exact | 1-3 |
| `transfer: naka_rushton` | `s * x^n / (x^n + sigma^n)` | exact | 3 |
| `transfer: log` | `sign(x) * log(1+k\|x\|) / log(1+k)` | exact | 1 |
| `cross_term` | `lms[0] += d*(Z - k*Y)` | analytical | 2 |
| `l_correction` | polynomial (degree 3 or 5) | Newton | 3-5 |
| `chroma_enrichment` | `C^p * exp(k*(L-0.5))` | analytical | 2 |
| `hue_rotation` | `R(theta) @ [a, b]` | `R(-theta)` | 1-4 |

Every block has a verified inverse. Round-trip error is at machine precision (< 1e-14 for all built-in pipelines).

### Loading from JSON checkpoints

SpaceForge auto-detects the pipeline architecture from checkpoint keys:

```python
from spaceforge import forge

# Auto-detects: M1 + cbrt + M2 + L_corr
forge("genspace_checkpoint.json")

# Auto-detects: M1 + Naka-Rushton + M2 + chroma + L_corr
forge("naka_rushton_checkpoint.json")
```

## Metrics (46 total)

SpaceForge uses [ColorBench](https://github.com/Grkmyldz148/colorbench) as its evaluation backend — the same 46 metrics, 3038 gradient pairs, 3 gamuts (sRGB, Display P3, Rec.2020), and winner logic.

| Category | Metrics |
|----------|---------|
| Numerical Stability | Round-trip sRGB/P3/Rec2020 (16.7M colors) |
| Achromatic | Gray ramp chroma (sRGB, pure D65) |
| Gradient Quality | CV mean/p95/max, hue drift, banding, 3-color CV |
| Hue | RMS, primary L range, leaf constancy, CIE Lab agreement |
| Gamut Geometry | Valid cusps, mono violations, cliff, volume, smoothness |
| Special Gradients | Yellow chroma, blue-white midpoint, red-white midpoint |
| Banding | Invisible steps, duplicate 8-bit steps |
| CVD Accessibility | Protan/deutan min step dE |
| Perceptual | Munsell Value/Hue, MacAdam isotropy, hue agreement |
| Application | Palette spacing, tint/shade hue, data viz dE, multi-stop CV, WCAG contrast, harmony accuracy, gamut mapping, animation CV, shade consistency, chroma preservation |
| Advanced | Jacobian condition, 1000-trip RT, quantization, channel monotonicity, cross-gamut amplification |

## Architecture

```
spaceforge/
├── core/
│   ├── pipeline.py          # Pipeline + Block ABC + YAML loader
│   ├── blocks/              # 7 composable block types
│   ├── inverse.py           # Round-trip verification
│   └── constraints.py       # Achromatic, white point, monotonicity
├── metrics/
│   └── registry.py          # ColorBench integration (46 metrics)
├── analysis/
│   ├── sensitivity.py       # Parameter → metric Jacobian
│   ├── ablation.py          # Block removal impact
│   ├── pareto.py            # Trade-off frontier
│   ├── feasibility.py       # Constraint satisfaction
│   ├── root_cause.py        # Per-stage metric decomposition
│   └── cross_model.py       # N-model best-in-class
├── optimizer/
│   └── solver.py            # CMA-ES constraint-first solver
├── report/
│   ├── html.py              # Scorecard + comparison dashboard
│   └── visualize.py         # Gradient strips, gamut cusps, hue wheel
├── history/
│   ├── tracker.py           # Auto-save, regression detection
│   └── diff.py              # Model-to-model comparison
├── export/                  # JSON, Helmlab, CSS, ColorBench class
├── api.py                   # forge() one-liner API
├── engine.py                # SpaceForge main class
├── cli.py                   # 14 CLI commands
└── presets/                 # OKLab, HelmGen-v7b, HelmGen-H-v2
```

## Related Projects

- [Helmlab](https://github.com/Grkmyldz148/helmlab) — the color space family that SpaceForge was built to develop
- [ColorBench](https://github.com/Grkmyldz148/colorbench) — 46-metric benchmark suite (used as SpaceForge's evaluation backend)

## License

MIT

## Author

[Gorkem Yildiz](https://gorkemyildiz.com)
