"""Architecture search — generate and evaluate 100 pipeline variants.

Systematically explores the design space of color space pipelines:
- Transfer functions: cbrt, power (per-channel), NR, log, sinh-like
- Enrichment: none, L_corr, chroma, hue_rotation, combinations
- Matrix count: 1x M2, 2x M2 (dual matrix)
- Cross-terms: none, blue-selective, full nonlinear

Each architecture gets random M1/M2 from known good seeds (OKLab, v7b, H_v2),
then evaluated on a fast subset of metrics vs OKLab.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import numpy as np

from .core.pipeline import Pipeline
from .core.blocks import (
    MatrixBlock, CbrtTransfer, PowerTransfer, NakaRushtonTransfer,
    LogTransfer, CrossTermBlock, LCorrectionBlock,
    ChromaEnrichmentBlock, HueRotationBlock,
)


# Known good M1 matrices (XYZ domain)
SEEDS = {
    "oklab": {
        "M1": [[0.8187985883032540, 0.3620277493294354, -0.1288275302451293],
               [0.0328604587632766, 0.9293629304937852, 0.0361893924939526],
               [0.0481337081895724, 0.2642425309122435, 0.6337149022717601]],
        "M2": [[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]],
    },
    "v7b": {
        "M1": [[6.213663274448127, -0.5041794153770129, -0.40416891025666857],
               [-1.1592256796157883, 4.350194381717271, 0.5254938968299478],
               [0.0008170122534259527, 0.7226718820884986, 2.227799849833172]],
        "M2": [[0.4675499211910323, 0.20915320090703618, -0.08488334505679182],
               [0.4843952725673558, -0.3665958307304812, -0.17266206907852755],
               [-0.04418360083197623, 0.39383739736845824, -0.36863136176600936]],
    },
    "h_v2": {
        "M1": [[0.6690879943531067, 0.34567929166443445, 0.01687374736438943],
               [-0.002956225032678894, 0.7378780762054156, 0.24331780627039545],
               [0.5498057592517742, -0.02761619550379546, 0.46383945657977976]],
        "M2": [[0.43705284002153927, 0.7814975215075335, 0.7806848276448284],
               [1.8275120492659194, -2.026071711749756, 0.19855966248383705],
               [0.4664431037955333, 0.7554432958065151, -1.221886399602048]],
    },
}


def _make_pipeline(transfer: str, enrichments: list[str],
                   cross_term: bool = False) -> Pipeline:
    """Build a pipeline from architecture spec."""
    blocks = []

    blocks.append(MatrixBlock(name="M1"))

    if cross_term:
        blocks.append(CrossTermBlock(name="cross_term"))

    if transfer == "cbrt":
        blocks.append(CbrtTransfer())
    elif transfer == "power":
        blocks.append(PowerTransfer(name="power", per_channel=True))
    elif transfer == "power_shared":
        blocks.append(PowerTransfer(name="power", per_channel=False))
    elif transfer == "nr":
        blocks.append(NakaRushtonTransfer(name="naka_rushton"))
    elif transfer == "log":
        blocks.append(LogTransfer(name="log"))
    else:
        raise ValueError(f"Unknown transfer: {transfer}")

    blocks.append(MatrixBlock(name="M2"))

    for e in enrichments:
        if e == "l_corr":
            blocks.append(LCorrectionBlock(name="l_correction", degree=3))
        elif e == "l_corr5":
            blocks.append(LCorrectionBlock(name="l_correction", degree=5))
        elif e == "chroma":
            blocks.append(ChromaEnrichmentBlock(name="chroma_enrichment"))
        elif e == "hue_rot":
            blocks.append(HueRotationBlock(name="hue_rotation", mode="global"))
        elif e == "hue_rot_dep":
            blocks.append(HueRotationBlock(name="hue_rotation", mode="hue_dependent"))
        elif e == "chroma_then_lcorr":
            blocks.append(ChromaEnrichmentBlock(name="chroma_enrichment"))
            blocks.append(LCorrectionBlock(name="l_correction", degree=3))

    return Pipeline(blocks)


def _make_params(seed_name: str, transfer: str, enrichments: list[str],
                 cross_term: bool = False, gamma: list | None = None,
                 perturb_m: bool = False, arch_id: int = 0) -> dict:
    """Build params dict from seed and architecture spec."""
    seed = SEEDS[seed_name]

    if perturb_m:
        # Perturb M1/M2 by small random amounts
        rng = np.random.RandomState(arch_id * 7 + 13)
        M1 = np.array(seed["M1"]) + rng.randn(3, 3) * 0.02
        M2 = np.array(seed["M2"]) + rng.randn(3, 3) * 0.02
        cp = {
            "M1": M1.tolist(),
            "M2": M2.tolist(),
        }
    else:
        cp = {
            "M1": [list(row) for row in seed["M1"]],
            "M2": [list(row) for row in seed["M2"]],
        }

    if transfer == "power" or transfer == "power_shared":
        cp["gamma"] = gamma or [1/3, 1/3, 1/3]
    elif transfer == "nr":
        cp["n"] = 0.76
        cp["sigma"] = 0.33
        cp["s_gain"] = 0.71
    elif transfer == "log":
        cp["log_k"] = 10.0

    if cross_term:
        cp["cross_d"] = -0.6
        cp["cross_k"] = 1.08883

    # Seed enrichment params with small random values (not zero!)
    # This way each architecture actually exercises its enrichment blocks
    rng = np.random.RandomState(hash(seed_name + transfer + str(enrichments)) % 2**31)

    for e in enrichments:
        if e in ("l_corr", "chroma_then_lcorr"):
            cp["L_corr"] = (rng.randn(3) * 0.05).tolist()
        elif e == "l_corr5":
            cp["L_corr"] = (rng.randn(5) * 0.03).tolist()
        if e in ("chroma", "chroma_then_lcorr"):
            cp["chroma_power"] = float(0.7 + rng.rand() * 0.3)  # 0.7-1.0
            cp["chroma_k"] = float(rng.randn() * 0.2)            # small L-dep
        if e == "hue_rot":
            cp["rotation_deg"] = float(rng.randn() * 15)         # ±15°
        if e == "hue_rot_dep":
            cp["hue_rot_c1"] = float(rng.randn() * 0.1)
            cp["hue_rot_s1"] = float(rng.randn() * 0.1)
            cp["hue_rot_c2"] = float(rng.randn() * 0.05)
            cp["hue_rot_s2"] = float(rng.randn() * 0.05)

    return {"_checkpoint": cp}


def _quick_score(pipeline: Pipeline, params: dict, device: torch.device) -> dict:
    """Fast evaluation — gradient CV, achromatic, cusps, hue RMS only."""
    from .metrics.registry import PipelineAdapter

    adapter = PipelineAdapter(pipeline, params, "test")

    results = {}

    # 1. Achromatic
    D65 = torch.tensor([[0.95047, 1.0, 1.08883]], dtype=torch.float64, device=device)
    Y = torch.linspace(0.01, 1.0, 100, dtype=torch.float64, device=device)
    gray_xyz = D65 * Y.unsqueeze(1)
    gray_lab = adapter.forward(gray_xyz)
    ach_err = float(torch.sqrt(gray_lab[:, 1]**2 + gray_lab[:, 2]**2).max())

    # 2. White point
    white_lab = adapter.forward(D65)
    white_L = float(white_lab[0, 0])

    # 3. Round-trip (100 random)
    gen = torch.Generator(device="cpu").manual_seed(42)
    rgb = torch.rand(100, 3, dtype=torch.float64, generator=gen)
    M_srgb = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=torch.float64)
    xyz_test = (rgb @ M_srgb.T).to(device)

    try:
        lab_test = adapter.forward(xyz_test)
        xyz_rt = adapter.inverse(lab_test)
        rt_err = float((xyz_test - xyz_rt).abs().max())
    except Exception:
        rt_err = 999.0

    # 4. Gradient CV (10 key gradients, 25 steps)
    import colorsys
    test_grads = [
        ((1, 0, 0), (1, 1, 1)),  # R→W
        ((0, 0, 1), (1, 1, 1)),  # B→W
        ((0, 1, 0), (1, 1, 1)),  # G→W
        ((1, 0, 0), (0, 0, 1)),  # R→B
        ((0, 0, 0), (1, 1, 1)),  # K→W
        ((1, 1, 0), (1, 1, 1)),  # Y→W
    ]
    cvs = []
    for start_rgb, end_rgb in test_grads:
        t = torch.linspace(0, 1, 25, dtype=torch.float64, device=device).unsqueeze(1)
        start = torch.tensor([start_rgb], dtype=torch.float64, device=device)
        end = torch.tensor([end_rgb], dtype=torch.float64, device=device)
        rgb_interp = start * (1 - t) + end * t
        xyz_interp = rgb_interp @ M_srgb.to(device).T

        lab_start = adapter.forward(xyz_interp[:1])
        lab_end = adapter.forward(xyz_interp[-1:])
        lab_lerp = lab_start * (1 - t) + lab_end * t

        xyz_back = adapter.inverse(lab_lerp)
        # Compute step sizes in XYZ
        diffs = xyz_back[1:] - xyz_back[:-1]
        step_sizes = torch.sqrt((diffs**2).sum(dim=1))
        mean_step = float(step_sizes.mean())
        if mean_step > 1e-10:
            cv = float(step_sizes.std()) / mean_step
            cvs.append(cv)

    grad_cv = np.mean(cvs) if cvs else 999.0

    # 5. Cusps (36 hues)
    valid_cusps = 0
    for h_deg in range(0, 360, 10):
        r, g, b = colorsys.hsv_to_rgb(h_deg / 360, 1.0, 1.0)
        rgb_t = torch.tensor([[r, g, b]], dtype=torch.float64, device=device)
        xyz_h = rgb_t @ M_srgb.to(device).T
        try:
            lab_h = adapter.forward(xyz_h)
            if not lab_h.isnan().any() and not lab_h.isinf().any():
                valid_cusps += 1
        except Exception:
            pass

    # 6. Hue RMS (6 primaries)
    primaries = [(1,0,0), (1,1,0), (0,1,0), (0,1,1), (0,0,1), (1,0,1)]
    hue_angles = []
    for r, g, b in primaries:
        rgb_t = torch.tensor([[r, g, b]], dtype=torch.float64, device=device)
        xyz_h = rgb_t @ M_srgb.to(device).T
        lab_h = adapter.forward(xyz_h)
        h = float(torch.atan2(lab_h[0, 2], lab_h[0, 1]))
        hue_angles.append(h)

    # Check hue ordering (R < Y < G < C < B < M in standard)
    hue_order_ok = all(hue_angles[i] < hue_angles[i+1] or
                       abs(hue_angles[i] - hue_angles[i+1]) > 3.0
                       for i in range(len(hue_angles)-1))

    # Monotonic L
    gray_L = gray_lab[:, 0]
    mono_violations = int((gray_L[1:] - gray_L[:-1] <= 0).sum())

    results = {
        "ach_err": ach_err,
        "white_L": white_L,
        "rt_err": rt_err,
        "grad_cv": grad_cv,
        "valid_cusps": valid_cusps,  # out of 36
        "hue_order_ok": hue_order_ok,
        "mono_violations": mono_violations,
        "viable": (ach_err < 0.1 and
                   0.8 < white_L < 1.2 and
                   rt_err < 1e-6 and
                   mono_violations == 0),
    }

    return results


# Architecture definitions — all combinations to try
TRANSFERS = ["cbrt", "power", "power_shared", "nr", "log"]
ENRICHMENTS = [
    [],
    ["l_corr"],
    ["chroma"],
    ["hue_rot"],
    ["l_corr", "chroma"],
    ["l_corr", "hue_rot"],
    ["chroma", "hue_rot"],
    ["l_corr", "chroma", "hue_rot"],
    ["chroma_then_lcorr"],
    ["hue_rot_dep"],
    ["l_corr", "hue_rot_dep"],
    ["l_corr5"],
    ["l_corr5", "chroma"],
    ["l_corr5", "hue_rot"],
    ["l_corr5", "chroma", "hue_rot"],
    ["l_corr5", "hue_rot_dep"],
    ["chroma", "hue_rot_dep"],
    ["l_corr", "chroma", "hue_rot_dep"],
]
CROSS_TERMS = [False, True]
GAMMA_VARIANTS = [
    [1/3, 1/3, 1/3],
    [0.30, 0.36, 0.33],   # M-cone boost
    [0.35, 0.35, 0.30],   # S-cone compress
    [0.40, 0.33, 0.28],   # aggressive L boost
    [0.28, 0.38, 0.34],   # strong M-cone
    [0.33, 0.33, 0.25],   # S-cone strong compress
    [0.25, 0.40, 0.35],   # extreme M-cone
    [0.36, 0.30, 0.34],   # L-cone boost
]


def generate_architectures() -> list[dict]:
    """Generate all architecture specs to test."""
    archs = []
    arch_id = 0

    for transfer in TRANSFERS:
        for enrichment in ENRICHMENTS:
            for cross_term in CROSS_TERMS:
                for seed_name in SEEDS:
                    # Skip invalid combos
                    if cross_term and transfer == "nr":
                        continue  # NR already handles nonlinearity

                    if transfer in ("power", "power_shared"):
                        for gamma in GAMMA_VARIANTS:
                            archs.append({
                                "id": arch_id,
                                "transfer": transfer,
                                "enrichments": enrichment,
                                "cross_term": cross_term,
                                "seed": seed_name,
                                "gamma": gamma,
                                "name": f"a{arch_id:03d}_{transfer}_{seed_name}_{'_'.join(enrichment) or 'bare'}_{'ct' if cross_term else 'no'}_{'-'.join(f'{g:.2f}' for g in gamma)}",
                            })
                            arch_id += 1
                    else:
                        archs.append({
                            "id": arch_id,
                            "transfer": transfer,
                            "enrichments": enrichment,
                            "cross_term": cross_term,
                            "seed": seed_name,
                            "gamma": None,
                            "name": f"a{arch_id:03d}_{transfer}_{seed_name}_{'_'.join(enrichment) or 'bare'}_{'ct' if cross_term else 'no'}",
                        })
                        arch_id += 1

    # Add random M1/M2 perturbations of best seeds
    rng = np.random.RandomState(2026)
    for base_seed in SEEDS:
        for _ in range(20):  # 20 random perturbations per seed
            perturbed_name = f"rnd_{base_seed}_{arch_id}"
            # Pick random transfer + enrichment combo from top performers
            transfer = rng.choice(["cbrt", "power", "nr", "log"])
            enrich_options = [
                ["l_corr", "chroma", "hue_rot"],
                ["l_corr", "hue_rot_dep"],
                ["l_corr", "chroma"],
                ["chroma_then_lcorr"],
                ["l_corr5", "chroma", "hue_rot"],
            ]
            enrichment = enrich_options[rng.randint(len(enrich_options))]
            gamma = None
            if transfer in ("power", "power_shared"):
                # Random gamma around 1/3
                gamma = [float(1/3 + rng.randn() * 0.04) for _ in range(3)]
                gamma = [max(0.2, min(0.5, g)) for g in gamma]

            archs.append({
                "id": arch_id,
                "transfer": transfer,
                "enrichments": list(enrichment),
                "cross_term": bool(rng.rand() < 0.3),
                "seed": base_seed,
                "gamma": gamma,
                "perturb_m": True,  # Flag to perturb matrices
                "name": f"a{arch_id:03d}_rnd_{base_seed}_{transfer}_{'_'.join(enrichment)}",
            })
            arch_id += 1

    return archs


def run_architecture_search(max_archs: int = 100,
                            device: str | None = None,
                            output_dir: str = "arch_search_results",
                            verbose: bool = True) -> list[dict]:
    """Run architecture search — evaluate many pipeline variants.

    Returns list of results sorted by viability and gradient CV.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    archs = generate_architectures()
    if len(archs) > max_archs:
        # Prioritize: diverse transfers, diverse seeds, diverse enrichments
        archs = archs[:max_archs]

    if verbose:
        print(f"Architecture search: {len(archs)} variants on {device}")

    results = []
    t0 = time.time()

    for i, arch in enumerate(archs):
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(archs) - i - 1) / rate
            print(f"  [{i+1}/{len(archs)}] {rate:.1f} arch/s, ETA {eta:.0f}s")

        try:
            pipeline = _make_pipeline(arch["transfer"], arch["enrichments"],
                                      arch["cross_term"])
            params = _make_params(arch["seed"], arch["transfer"],
                                  arch["enrichments"], arch["cross_term"],
                                  arch.get("gamma"),
                                  perturb_m=arch.get("perturb_m", False),
                                  arch_id=arch["id"])

            scores = _quick_score(pipeline, params, dev)
            scores["arch"] = arch
            scores["pipeline"] = str(pipeline)
            results.append(scores)

        except Exception as e:
            results.append({
                "arch": arch,
                "pipeline": "FAILED",
                "viable": False,
                "error": str(e),
            })

    # Sort: viable first, then by grad_cv
    results.sort(key=lambda r: (
        not r.get("viable", False),
        r.get("grad_cv", 999),
    ))

    elapsed = time.time() - t0

    if verbose:
        print(f"\n  Done: {len(results)} architectures in {elapsed:.1f}s")
        viable = [r for r in results if r.get("viable", False)]
        print(f"  Viable: {len(viable)}/{len(results)}")

        print(f"\n  Top 20 viable architectures:")
        print(f"  {'#':>3} {'Name':50s} {'CV':>8} {'Cusps':>6} {'Ach':>10} {'RT':>10}")
        for i, r in enumerate(viable[:20]):
            name = r["arch"]["name"][:50]
            print(f"  {i+1:>3} {name:50s} "
                  f"{r.get('grad_cv', 999):>8.4f} "
                  f"{r.get('valid_cusps', 0):>6} "
                  f"{r.get('ach_err', 999):>10.2e} "
                  f"{r.get('rt_err', 999):>10.2e}")

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "arch_search.json"

    def _ser(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump([{k: _ser(v) for k, v in r.items()} for r in results],
                  f, indent=2, default=str)

    if verbose:
        print(f"\n  Results saved: {out_path}")

    return results
