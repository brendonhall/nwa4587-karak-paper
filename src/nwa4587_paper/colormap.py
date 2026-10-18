"""The colormap round trip: scalar -> jet RGB -> nearest-palette scalar, and its error."""
from __future__ import annotations

import matplotlib
import numpy as np
from scipy.spatial import cKDTree

from . import constants as C
from . import numbers as N


def _palette(cmap_name: str, lut_n: int) -> tuple[np.ndarray, np.ndarray]:
    scalars = np.linspace(0, 1, lut_n, dtype=np.float32)
    cmap = matplotlib.colormaps[cmap_name]
    rgb = (np.array([cmap(s)[:3] for s in scalars]) * 255).astype(np.uint8)
    return rgb, scalars


def build_lut(cmap_name: str, lut_n: int) -> np.ndarray:
    """(256, 256, 256) float32 table: each RGB triplet to the scalar of its nearest palette entry."""
    rgb, scalars = _palette(cmap_name, lut_n)
    tree = cKDTree(rgb.astype(np.float64))
    r, g, b = np.mgrid[0:256, 0:256, 0:256]
    allc = np.column_stack([r.ravel(), g.ravel(), b.ravel()]).astype(np.float64)
    idx = np.empty(len(allc), dtype=np.intp)
    chunk = 1_000_000
    for s in range(0, len(allc), chunk):
        _, idx[s:s + chunk] = tree.query(allc[s:s + chunk])
    lut = scalars[idx].reshape(256, 256, 256)
    lut[0, 0, 0] = 0.0
    return lut


def forward_map(scalars: np.ndarray, cmap_name: str) -> np.ndarray:
    """Scalars in [0, 1] to uint8 (N, 3) RGB."""
    cmap = matplotlib.colormaps[cmap_name]
    return (cmap(np.asarray(scalars))[:, :3] * 255).astype(np.uint8)


def invert_via_lut(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[rgb[:, 0], rgb[:, 1], rgb[:, 2]]


def classify_jet_region(scalar: float) -> str:
    for upper, name in C.COLORMAP_REGION_BOUNDS:
        if scalar < upper:
            return name
    return C.COLORMAP_REGION_BOUNDS[-1][1]


def analyse(cmap_name: str, n_test: int, lut: np.ndarray) -> dict:
    scalars = np.linspace(0, 1, n_test)
    rgb = forward_map(scalars, cmap_name)
    ae = np.abs(invert_via_lut(rgb, lut).astype(np.float64) - scalars)
    out = {
        "mae": float(ae.mean()),
        "median_ae": float(np.median(ae)),
        "max_ae": float(ae.max()),
        "rmse": float(np.sqrt((ae ** 2).mean())),
        "p90": float(np.percentile(ae, 90)),
        "p99": float(np.percentile(ae, 99)),
        "n_exact": int((ae < 1e-6).sum()),
        "n_unique_rgb": int(len(np.unique(rgb, axis=0))),
    }
    if cmap_name == "jet":
        names = np.array([classify_jet_region(v) for v in scalars])
        out["_regions"] = {
            name: {"n": int((names == name).sum()),
                   "mae": float(ae[names == name].mean()) if (names == name).any() else float("nan"),
                   "max_ae": float(ae[names == name].max()) if (names == name).any() else float("nan")}
            for _, name in C.COLORMAP_REGION_BOUNDS
        }
    return out


def run(run, out, data, *, settings: C.ColormapSettings | None = None) -> dict:
    """`run`, `out` and `data` are unused; the CLI writes numbers/colormap.json."""
    settings = settings or C.COLORMAP
    nums: dict = {}
    for name in settings.cmaps:
        lut = build_lut(name, settings.lut_n)
        res = analyse(name, settings.n_test, lut)
        regions = res.pop("_regions", None)
        nums[name] = res
        if regions is not None:
            nums["jet_regions"] = regions
    nums["lut_step"] = 1 / (settings.lut_n - 1)
    return nums


N.register({
    "colormap.lut_step": N.Spec("float", tol=1e-9, fmt="{:.6f}"),
    "colormap.*.mae": N.Spec("float", tol=1e-5, fmt="{:.2e}"),
    "colormap.*.median_ae": N.Spec("float", tol=1e-5, fmt="{:.2e}"),
    "colormap.*.max_ae": N.Spec("float", tol=1e-4, fmt="{:.2e}"),
    "colormap.*.rmse": N.Spec("float", tol=1e-5, fmt="{:.2e}"),
    "colormap.*.p90": N.Spec("float", tol=1e-5, fmt="{:.2e}"),
    "colormap.*.p99": N.Spec("float", tol=1e-5, fmt="{:.2e}"),
    "colormap.*.n_exact": N.Spec("int"), "colormap.*.n_unique_rgb": N.Spec("int"),
    "colormap.jet_regions.*.n": N.Spec("int"),
    "colormap.jet_regions.*.mae": N.Spec("float", tol=1e-5, fmt="{:.2e}"),
    "colormap.jet_regions.*.max_ae": N.Spec("float", tol=1e-4, fmt="{:.2e}"),
})
