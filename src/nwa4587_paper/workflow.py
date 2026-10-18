"""Fig 1: the five-step workflow from the jet-RGB export to the named phases."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402

from . import constants as C  # noqa: E402
from . import numbers as N  # noqa: E402
from . import render  # noqa: E402
from .io import DataPaths, Run, find_png, require  # noqa: E402

N.register({
    "workflow.n_clusters_before_pyx": N.Spec("int"),
    "workflow.n_phases_final": N.Spec("int"),
})

SPLIT_NODE = "pyx"
FIG_NAME = "fig1_workflow"
DENOISE_ELEMENT = C.FIG1_ELEMENT
SCALAR_PERCENTILE = 99.5

FIG_W = 17 / 2.54
MARGIN, GAP = 0.06, 0.20
N_PANELS = 5
PANEL_W = (FIG_W - 2 * MARGIN - (N_PANELS - 1) * GAP) / N_PANELS
TITLE_H, NOTE_H, LEGEND_H = 0.22, 0.16, 0.44
MASKED_GREY = 0.95
TITLE_PT, NOTE_PT, LEGEND_PT = 6.5, 5.5, 5.5
ARROW_STYLE, ARROW_LW = "->,head_width=2.5,head_length=4", 1.1
LEGEND_NCOL = 5
NOTE_TEXT = "colormap inversion: only for false-colour exports"
TITLES = ("(a) Jet RGB input", "(b) Inverted scalar", "(c) Mask + denoise")


def _jet_thumb(eds: Path, trim: int) -> np.ndarray:
    import imageio.v3 as iio

    rgb = iio.imread(find_png(eds, C.FIG1_ELEMENT))[trim:, :, :3]
    return rgb[::C.FIG1_PNG_STRIDE, ::C.FIG1_PNG_STRIDE]


def _panels(run: Run, eds: Path) -> tuple[list[np.ndarray], int]:
    s = C.FIG1_H5_STRIDE
    raw = run.raw_channel(C.FIG1_ELEMENT)[::s, ::s]
    pos = raw[raw > 0]
    vmax = float(np.percentile(pos, SCALAR_PERCENTILE)) if pos.size else 1.0
    scalar = np.repeat(np.clip(raw / vmax, 0, 1)[:, :, None], 3, axis=2)

    ca = run.channel(DENOISE_ELEMENT)[::s, ::s]
    vmax_dn = float(ca[run.mask[::s, ::s]].max()) if run.mask[::s, ::s].any() else 1.0
    gray = np.where(run.mask[::s, ::s], np.clip(ca / (vmax_dn or 1.0), 0, 1), MASKED_GREY)
    denoised = np.repeat(gray[:, :, None], 3, axis=2)

    pre_full = run.labels_before(SPLIT_NODE)
    n_clusters = int(np.unique(pre_full[pre_full >= 0]).size)    # counted before the thumbnail stride
    pre = pre_full[::s, ::s]
    final = run.labels[::s, ::s]
    imgs = [_jet_thumb(eds, run.geometry.header_trim_px), scalar, denoised,
            render.colorize(pre, run.keys_before(SPLIT_NODE)), render.colorize(final, run.keys)]
    return imgs, n_clusters


def _legend_entries(run: Run) -> list[tuple[str, str]]:
    s = C.FIG1_H5_STRIDE
    thumb = run.labels[::s, ::s]
    labs, n = np.unique(thumb[thumb >= 0], return_counts=True)
    order = [int(lab) for lab in labs[np.argsort(-n, kind="stable")]]
    keys = [run.keys[lab] for lab in order if lab in run.keys]
    return [(k, C.PALETTE[k]) for k in keys if k in C.PALETTE]


def _draw(run: Run, imgs: list[np.ndarray], n_clusters: int, path: Path) -> None:
    h, w = imgs[1].shape[:2]
    panel_h = PANEL_W * h / w
    fig_h = MARGIN + TITLE_H + panel_h + NOTE_H + LEGEND_H + MARGIN
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor="white")
    titles = [*TITLES, f"(d) {n_clusters} clusters", "(e) Named phases"]
    y0 = MARGIN + LEGEND_H + NOTE_H
    for i, (img, title) in enumerate(zip(imgs, titles)):
        x0 = MARGIN + i * (PANEL_W + GAP)
        ax = fig.add_axes([x0 / FIG_W, y0 / fig_h, PANEL_W / FIG_W, panel_h / fig_h])
        ax.imshow(img, interpolation="nearest", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=TITLE_PT, fontweight="bold")
        if i < N_PANELS - 1:
            mid = y0 + panel_h / 2
            xa, xb = x0 + PANEL_W, x0 + PANEL_W + GAP
            fig.add_artist(FancyArrowPatch((xa / FIG_W, mid / fig_h), (xb / FIG_W, mid / fig_h),
                                           transform=fig.transFigure, arrowstyle=ARROW_STYLE,
                                           color=C.FIG1_ARROW_COLOR, linewidth=ARROW_LW))
    fig.text((MARGIN + PANEL_W) / FIG_W, (MARGIN + LEGEND_H + NOTE_H / 2) / fig_h, NOTE_TEXT,
             fontsize=NOTE_PT, style="italic", color=C.FIG1_NOTE_COLOR, ha="left", va="center")
    handles = [Patch(color=c, label=C.DISPLAY_NAMES.get(k, k)) for k, c in _legend_entries(run)]
    fig.legend(handles=handles, loc="lower center", ncol=LEGEND_NCOL, fontsize=LEGEND_PT, frameon=False,
               bbox_to_anchor=(0.5, MARGIN / fig_h))
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, facecolor="white")
    plt.close(fig)


def run(run: Run, out: Path, data: DataPaths) -> dict:
    eds = require(data.eds_dir, "elemental PNG directory", "data --from PATH")
    imgs, n_clusters = _panels(run, eds)
    figs = Path(out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    _draw(run, imgs, n_clusters, figs / FIG_NAME)
    return {"n_clusters_before_pyx": n_clusters,
            "n_phases_final": sum(1 for lab in run.counts if lab >= 0)}
