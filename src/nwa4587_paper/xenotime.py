"""Fig 6: the xenotime grain on BSE, the phase map, and the Y and P intensities."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage  # noqa: E402

from . import constants as C  # noqa: E402
from . import numbers as N  # noqa: E402
from . import render  # noqa: E402
from .io import DataPaths, MissingInput, Run  # noqa: E402

N.register({
    "xenotime.n_pixels_2x": N.Spec("int"),
    "xenotime.n_components": N.Spec("int"),
    "xenotime.largest_component_px": N.Spec("int"),
    "xenotime.bbox_2x": N.Spec("int"),
    "xenotime.roi_2x": N.Spec("int"),
    "xenotime.centroid_2x": N.Spec("float", tol=0.5, fmt="{:.0f}"),
})

Y_ELEMENT, P_ELEMENT = "Y", "P"
FIG_NAME = "fig6_xenotime_callout"
FIG_SIZE_IN = (3.54, 4.0)
GRID_KW = dict(height_ratios=[1, 1, 0.03], hspace=0.22, wspace=0.08,
               left=0.02, right=0.98, top=0.93, bottom=0.18)
TITLE_PT, LEGEND_PT, CBAR_PT = 6, 4.5, 5
CONTOUR_WIDTH = 1.0
BSE_CONTOUR_COLOR = C.XENOTIME_CONTOUR_COLOR
PHASE_CONTOUR_COLOR, CHANNEL_CONTOUR_COLOR = "white", "cyan"
LABEL_FACE, LABEL_TEXT = "black", "white"
SCALEBAR_COLOR = "white"
CHANNEL_CMAP = "hot"
CBAR_LABEL = "Relative intensity"
PANEL_LABEL_XY = (0.04, 0.96)
LEGEND_MAX_COLS = 5


def region(labels: np.ndarray, label: int, pad: int) -> tuple[dict, slice, slice]:
    """Components of `label` (4-connectivity), the largest one's padded square window."""
    comp, n = ndimage.label(labels == label)
    sizes = np.bincount(comp.ravel())[1:]
    big = int(np.argmax(sizes)) + 1
    rows, cols = np.nonzero(comp == big)
    r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
    h, w = labels.shape
    a0, a1, b0, b1 = max(r0 - pad, 0), min(r1 + pad, h), max(c0 - pad, 0), min(c1 + pad, w)
    diff = (a1 - a0) - (b1 - b0)
    if diff > 0:
        b0, b1 = max(b0 - diff // 2, 0), min(b1 + diff - diff // 2, w)
    elif diff < 0:
        diff = -diff
        a0, a1 = max(a0 - diff // 2, 0), min(a1 + diff - diff // 2, h)
    info = {"n_components": int(n), "largest_component_px": int(sizes.max()),
            "bbox_2x": [r0, c0, r1, c1]}
    return info, slice(a0, a1), slice(b0, b1)


def _add_scalebar(
    ax: plt.Axes,
    pixel_size_um: float,
    bar_length_um: float | None = None,
    loc: str = "lower right",
) -> None:
    """Add a simple scalebar to an axes.

    Parameters
    ----------
    ax : plt.Axes
    pixel_size_um : float
        Size of one pixel in micrometres.
    bar_length_um : float or None
        Desired physical length.  If None, auto-select a round number.
    loc : str
        "lower right" or "lower left".
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    view_width_px = abs(xlim[1] - xlim[0])
    view_height_px = abs(ylim[0] - ylim[1])  # imshow has inverted y

    # Auto-select bar length: ~20% of view width, rounded to nice number
    if bar_length_um is None:
        target_um = view_width_px * pixel_size_um * 0.20
        nice = [10, 20, 25, 50, 100, 200, 250, 500, 1000]
        bar_length_um = min(nice, key=lambda x: abs(x - target_um))

    bar_length_px = bar_length_um / pixel_size_um
    margin_x = view_width_px * 0.05
    margin_y = view_height_px * 0.07

    if loc == "lower right":
        x_end = xlim[1] - margin_x
        x_start = x_end - bar_length_px
    else:
        x_start = xlim[0] + margin_x
        x_end = x_start + bar_length_px

    y_bar = ylim[0] - margin_y  # ylim[0] is bottom in imshow coords (large y)

    ax.plot([x_start, x_end], [y_bar, y_bar], color=SCALEBAR_COLOR, linewidth=2.0,
            solid_capstyle="butt")
    ax.text(
        (x_start + x_end) / 2, y_bar - view_height_px * 0.03,
        f"{bar_length_um:.0f} \u00b5m",
        color=SCALEBAR_COLOR, fontsize=7, ha="center", va="top",
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none",
                  boxstyle="round,pad=0.1"),
    )


def _need(run: Run, element: str) -> None:
    if element not in run.element_order:
        raise MissingInput(f"channel {element} not in denoised cube "
                           f"(elements: {', '.join(run.element_order)})")


def _draw(run: Run, mask: np.ndarray, rs: slice, cs: slice, path: Path) -> None:
    y, p = run.channel(Y_ELEMENT)[rs, cs], run.channel(P_ELEMENT)[rs, cs]
    vmax = float(max(y.max(), p.max()))
    m = ndimage.binary_dilation(mask)[rs, cs]
    lab = run.labels[rs, cs]
    fig = plt.figure(figsize=FIG_SIZE_IN)
    gs = GridSpec(3, 2, figure=fig, **GRID_KW)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax_a, ax_b, ax_c, ax_d = axes

    ax_a.imshow(render.bse_gray(run.bse[rs, cs]), interpolation="nearest")
    ax_a.contour(m, levels=[0.5], colors=[BSE_CONTOUR_COLOR], linewidths=CONTOUR_WIDTH)
    ax_a.set_title("BSE", fontsize=TITLE_PT)
    _add_scalebar(ax_a, run.geometry.pixel_size_um_2x)

    ax_b.imshow(render.colorize(lab, run.keys), interpolation="nearest")
    ax_b.contour(m, levels=[0.5], colors=[PHASE_CONTOUR_COLOR], linewidths=CONTOUR_WIDTH)
    ax_b.set_title("Phase map", fontsize=TITLE_PT)

    ims = []
    for ax, img, title in ((ax_c, y, "Y-L intensity"), (ax_d, p, "P-K intensity")):
        ims.append(ax.imshow(img, cmap=CHANNEL_CMAP, vmin=0, vmax=vmax, interpolation="nearest"))
        ax.contour(m, levels=[0.5], colors=[CHANNEL_CONTOUR_COLOR], linewidths=CONTOUR_WIDTH)
        ax.set_title(title, fontsize=TITLE_PT)
    for col, im in enumerate(ims):
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[2, col]), orientation="horizontal")
        cb.set_label(CBAR_LABEL, fontsize=CBAR_PT)
        cb.ax.tick_params(labelsize=CBAR_PT)

    for ax, tag in zip(axes, "abcd"):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(*PANEL_LABEL_XY, f"({tag})", transform=ax.transAxes, fontsize=TITLE_PT, fontweight="bold",
                color=LABEL_TEXT, va="top", ha="left", bbox=dict(facecolor=LABEL_FACE, edgecolor="none", pad=1.5))

    present = [run.keys[int(v)] for v in np.unique(lab) if int(v) in run.keys and run.keys[int(v)] in C.PALETTE]
    if present:
        fig.legend(handles=[Patch(color=C.PALETTE[k], label=C.DISPLAY_NAMES.get(k, k)) for k in present],
                   loc="lower center", ncol=min(len(present), LEGEND_MAX_COLS), fontsize=LEGEND_PT,
                   frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(run: Run, out: Path, data: DataPaths) -> dict:
    for element in (Y_ELEMENT, P_ELEMENT):
        _need(run, element)
    label = run.phase(C.XENOTIME_KEY)
    labels = run.labels
    mask = labels == label
    info, rs, cs = region(labels, label, C.XENOTIME_PAD_PX_2X)
    figs = Path(out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    _draw(run, mask, rs, cs, figs / FIG_NAME)
    rows, cols = np.nonzero(mask)
    return {"n_pixels_2x": int(mask.sum()), **info,
            "roi_2x": [rs.start, cs.start, rs.stop, cs.stop],
            "centroid_2x": [float(rows.mean()), float(cols.mean())]}
