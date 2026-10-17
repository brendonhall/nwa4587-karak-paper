"""Fig 2: mean elemental intensity (with standard deviation) per phase."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from . import constants as C  # noqa: E402
from . import numbers as N  # noqa: E402
from .io import DataPaths, Run  # noqa: E402

N.register({
    "fingerprints.element_order": N.Spec("str"),
    "fingerprints.phases.*.n_pixels": N.Spec("int"),
    "fingerprints.phases.*.area_pct": N.Spec("float", tol=0.0005, fmt="{:.2f}"),
    "fingerprints.phases.*.mean.*": N.Spec("float", tol=0.002, fmt="{:.3f}"),
    "fingerprints.phases.*.std.*": N.Spec("float", tol=0.002, fmt="{:.3f}"),
})

N_COLS = 3
FIG_WIDTH_IN = 17 / 2.54
ROW_HEIGHT_IN = 1.30
MARGIN_BOTTOM_IN = 0.60
BAR_WIDTH = 0.70
Y_TICK_STEP = 0.25
Y_HEADROOM = 1.08
TITLE_PT = 7
TICK_PT = 6
XLABEL_ROTATION = 50


def compute(run: Run) -> dict[int, dict]:
    """Per label mean, std, n_pixels and area_pct (over all mineral pixels), excluded keys removed."""
    from karak.identification.fingerprint import compute_fingerprints

    cube = run.f["denoised/cube"][:]
    res = compute_fingerprints(cube, run.cleaned_labels_flat, run.mineral_indices,
                               run.element_order, accumulate="float32")
    drop = {run.phase(k) for k in C.FINGERPRINT_EXCLUDED_KEYS if run.has(k)}
    total = run.n_mineral_pixels
    out = {}
    for lab, fp in res["fingerprints"].items():
        if lab < 0 or lab in drop or lab not in run.keys or fp["n_pixels"] <= 0:
            continue
        out[int(lab)] = {"mean": np.asarray(fp["mean"], float), "std": np.asarray(fp["std"], float),
                         "n_pixels": int(fp["n_pixels"]), "area_pct": 100.0 * fp["n_pixels"] / total}
    return out


def element_order(fps: dict[int, dict], elements: list[str]) -> list[str]:
    """Elements by descending variance of the per-phase means; ties keep cube order."""
    means = np.array([fp["mean"] for fp in fps.values()])
    order = np.argsort(-np.var(means, axis=0), kind="stable")
    return [elements[i] for i in order]


def _pct_text(pct: float) -> str:
    if pct >= 1:
        return f"{pct:.1f}"
    if pct >= 0.1:
        return f"{pct:.2f}"
    if pct >= 0.01:
        return f"{pct:.3f}"
    return "<0.01"


def _draw(run: Run, fps: dict[int, dict], order: list[str], path: Path) -> None:
    elements = list(run.element_order)
    idx = [elements.index(e) for e in order]
    labs = sorted(fps, key=lambda lab: (-fps[lab]["n_pixels"], lab))
    n_rows = math.ceil(len(labs) / N_COLS)
    fig_h = n_rows * ROW_HEIGHT_IN + MARGIN_BOTTOM_IN
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(FIG_WIDTH_IN, fig_h), squeeze=False)
    x = np.arange(len(order))
    for i, ax in enumerate(axes.flat):
        if i >= len(labs):
            ax.set_visible(False)
            continue
        fp = fps[labs[i]]
        mean, std = fp["mean"][idx], fp["std"][idx]
        ax.bar(x, mean, width=BAR_WIDTH, color=C.FIG2_BAR_COLOR, edgecolor=C.FIG2_BAR_EDGE_COLOR,
               linewidth=0.4, yerr=std, error_kw={"ecolor": C.FIG2_ERROR_COLOR, "elinewidth": 0.45,
                                                  "capsize": 1.0})
        key = run.keys[labs[i]]
        ax.set_title(f"{C.DISPLAY_NAMES.get(key, key)}  ({_pct_text(fp['area_pct'])})",
                     fontsize=TITLE_PT, fontweight="bold", loc="left")
        ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))
        ax.set_ylim(0, min(1, Y_HEADROOM * float(np.max(mean + std))))
        ax.tick_params(axis="y", labelsize=TICK_PT)
        ax.set_xticks(x)
        if i + N_COLS >= len(labs):
            ax.set_xticklabels(order, rotation=XLABEL_ROTATION, ha="right", fontsize=TICK_PT)
        else:
            ax.set_xticklabels([])
        if i % N_COLS == 0:
            ax.set_ylabel("Mean intensity", fontsize=TICK_PT)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=MARGIN_BOTTOM_IN / fig_h, top=0.98,
                        hspace=0.55, wspace=0.30)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight",
                    facecolor="white", pad_inches=0.04)
    plt.close(fig)


def run(run: Run, out: Path, data: DataPaths) -> dict:
    d = Path(out) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    fps = compute(run)
    order = element_order(fps, list(run.element_order))
    _draw(run, fps, order, d / "fig2_fingerprints")
    return {
        "element_order": order,
        "phases": {run.keys[lab]: {
            "n_pixels": fp["n_pixels"], "area_pct": float(fp["area_pct"]),
            "mean": {e: float(v) for e, v in zip(run.element_order, fp["mean"])},
            "std": {e: float(v) for e, v in zip(run.element_order, fp["std"])},
        } for lab, fp in fps.items()},
    }
