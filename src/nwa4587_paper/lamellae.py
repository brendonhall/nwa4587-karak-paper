"""Fig 8 and the section 3.5 numbers: pyroxene exsolution lamellae."""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from . import constants as C  # noqa: E402
from . import lamellae_core  # noqa: E402
from . import numbers as N  # noqa: E402
from .io import DataPaths, Run  # noqa: E402

N.register({
    "lamellae.n_grains_total": N.Spec("int"), "lamellae.n_grains_analyzed": N.Spec("int"),
    "lamellae.pixel_size_um": N.Spec("float", tol=0.0, fmt="{:.1f}"),
    "lamellae.nyquist_limit_um": N.Spec("float", tol=0.0, fmt="{:.1f}"),
    "lamellae.orientation.circular_mean_deg": N.Spec("float", tol=0.5, fmt="{:.1f}"),
    "lamellae.orientation.rayleigh_R_bar": N.Spec("float", tol=0.005, fmt="{:.2f}"),
    "lamellae.orientation.rayleigh_p_value": N.Spec("float", tol=0.01, fmt="{:.2f}"),
    "lamellae.orientation.uniform": N.Spec("bool"),
    "lamellae.spacing.*": N.Spec("float", tol=0.5, fmt="{:.0f}"),
    "lamellae.spacing.n_near_nyquist": N.Spec("int"),
    "lamellae.grain_size_vs_spacing.spearman_rho": N.Spec("float", tol=0.01, fmt="{:.2f}"),
    "lamellae.grain_size_vs_spacing.spearman_p": N.Spec("float", tol=0.5, rel=True, fmt="{:.3f}"),
    "lamellae.coherence.*": N.Spec("float", tol=0.01, fmt="{:.2f}"),
})

AUGITE_KEY, PIGEONITE_KEY = C.FIG5_KARAK_HIGH_KEY, C.FIG5_KARAK_LOW_KEY
FIG_NAME = "fig8_lamellae"
TABLE_NAME = "lamellae_grain_table.csv"
SUMMARY_NAME = "lamellae_summary.json"
CSV_FIELDS = ("grain_id", "area_px", "area_um2", "equiv_diameter_um", "centroid_row", "centroid_col",
              "augite_fraction", "orientation_deg", "coherence", "spacing_px", "spacing_um",
              "spacing_iqr_um", "n_profiles", "near_nyquist")

FIG_WIDTH_IN, FIG_HEIGHT_RATIO = 6.693, 0.85
GRID_KW = dict(hspace=0.35, wspace=0.35)
PANEL_LABEL_XY, PANEL_LABEL_PT = (-0.06, 1.04), 10
TITLE_PT, TICK_PT, LABEL_PT, LEGEND_PT, CBAR_PT, SCALE_PT = 8, 6, 7, 5.5, 7, 6
WINDOW_2X_PX = 800
LINE_HALF_MM, LINE_WIDTH, DOT_SIZE = 0.3, 1.5, 3
SCALE_BAR_MM, SCALE_BAR_LABEL = 0.5, "500 µm"
SCALE_BAR_XY = (0.05, 0.06)           # axes fraction of the bar's left end
SPACING_PCT = (5, 95)
SPACING_CMAP, ORIENT_CMAP, IMAGE_CMAP = "viridis", "hsv", "gray"
ROSE_BIN_DEG = 10
ROSE_ALPHA, ROSE_RLABEL_DEG = 0.7, 225
HIST_CLIP_FACTOR, HIST_CLIP_PCT = 1.3, 98
HIST_BINS_MIN, HIST_BINS_MAX, HIST_BINS_DIVISOR = 10, 25, 3
BAND_ALPHA, HIST_ALPHA = 0.15, 0.8
STATS_BOX_XY = (0.02, 0.02)           # axes fraction, lower left
MEAN_LINE_FACTOR = 1.15
EMPTY_NOTE = "no grains analyzed"
P_FLOOR = 0.001


def _fmt_p(p: float) -> str:
    return f"p < {P_FLOOR}" if p < P_FLOOR else f"p = {p:.3f}"


def _empty(ax, text: str = EMPTY_NOTE) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=LABEL_PT)
    ax.set_xticks([])
    ax.set_yticks([])


def _panel_a(fig, ax, run: Run, recs: list[dict]) -> None:
    if not recs:
        _empty(ax)
        return
    f = run.geometry.factor
    ps = run.geometry.pixel_size_um_1x
    bse = run.bse
    h, w = bse.shape
    rc = float(np.median([r["centroid_row"] for r in recs])) / f
    cc = float(np.median([r["centroid_col"] for r in recs])) / f
    r0 = int(np.clip(rc - WINDOW_2X_PX / 2, 0, max(h - WINDOW_2X_PX, 0)))
    c0 = int(np.clip(cc - WINDOW_2X_PX / 2, 0, max(w - WINDOW_2X_PX, 0)))
    r1, c1 = min(h, r0 + WINDOW_2X_PX), min(w, c0 + WINDOW_2X_PX)
    mm = f * ps / 1000
    extent = [c0 * mm, c1 * mm, r1 * mm, r0 * mm]
    ax.imshow(np.clip(bse[r0:r1, c0:c1] / C.BSE_DISPLAY_MAX, 0, 1), cmap=IMAGE_CMAP,
              interpolation="bilinear", extent=extent)
    spac = np.array([r["spacing_um"] for r in recs])
    norm = Normalize(*np.percentile(spac, SPACING_PCT))
    cmap = plt.get_cmap(SPACING_CMAP)
    for r in recs:
        x, y = r["centroid_col"] * ps / 1000, r["centroid_row"] * ps / 1000
        if not (extent[0] <= x <= extent[1] and extent[3] <= y <= extent[2]):
            continue
        th = np.radians(r["orientation_deg"])
        # orientation is measured from the row (image y) axis toward the column axis
        dx, dy = LINE_HALF_MM * np.sin(th), LINE_HALF_MM * np.cos(th)
        colr = cmap(norm(r["spacing_um"]))
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=colr, lw=LINE_WIDTH)
        ax.plot(x, y, "o", ms=DOT_SIZE, color=colr, mec="white", mew=0.3)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Spacing (µm)", fontsize=CBAR_PT)
    cb.ax.tick_params(labelsize=TICK_PT)
    ax.set_xlabel("x (mm)", fontsize=LABEL_PT)
    ax.set_ylabel("y (mm)", fontsize=LABEL_PT)
    ax.tick_params(labelsize=TICK_PT)
    x0 = extent[0] + SCALE_BAR_XY[0] * (extent[1] - extent[0])
    y0 = extent[2] + SCALE_BAR_XY[1] * (extent[3] - extent[2])
    ax.plot([x0, x0 + SCALE_BAR_MM], [y0, y0], color="white", lw=2)
    ax.text(x0 + SCALE_BAR_MM / 2, y0, SCALE_BAR_LABEL, color="white", ha="center", va="bottom",
            fontsize=SCALE_PT, fontweight="bold")


def _panel_b(fig, ax, summary: dict, recs: list[dict]):
    ax.set_visible(False)
    pax = fig.add_axes(ax.get_position(), projection="polar")
    _fill_rose(pax, summary, recs)
    return pax


def _fill_rose(pax, summary: dict, recs: list[dict]) -> None:
    n = len(recs)
    pax.set_title(f"n = {n}", fontsize=TITLE_PT, pad=10)
    pax.tick_params(labelsize=TICK_PT)
    if not recs:
        pax.set_xticks([])
        pax.set_yticks([])
        pax.text(0.5, 0.5, EMPTY_NOTE, ha="center", va="center", transform=pax.transAxes, fontsize=LABEL_PT)
        return
    orient = np.array([r["orientation_deg"] for r in recs])
    area = np.array([r["area_um2"] for r in recs])
    edges = np.arange(0, 180 + ROSE_BIN_DEG, ROSE_BIN_DEG)
    hist, _ = np.histogram(orient, bins=edges, weights=area)
    hist = hist / hist.sum()
    centers = np.radians((edges[:-1] + edges[1:]) / 2)
    width = np.radians(ROSE_BIN_DEG)
    for off in (0, np.pi):
        pax.bar(centers + off, hist, width=width, color=C.PALETTE[PIGEONITE_KEY], alpha=ROSE_ALPHA,
                edgecolor="white", linewidth=0.3)
    pax.set_theta_zero_location("N")
    pax.set_theta_direction(-1)
    pax.set_rlabel_position(ROSE_RLABEL_DEG)
    mu = np.radians(summary["orientation"]["circular_mean_deg"])
    for off in (0, np.pi):
        pax.plot([mu + off, mu + off], [0, MEAN_LINE_FACTOR * hist.max()], color=C.PALETTE[AUGITE_KEY], ls="--", lw=1.2)
    o = summary["orientation"]
    txt = f"R̄ = {o['rayleigh_R_bar']:.2f}\n{_fmt_p(o['rayleigh_p_value'])}\nμ = {o['circular_mean_deg']:.0f}°"
    pax.text(*STATS_BOX_XY, txt, transform=pax.transAxes,
             fontsize=LEGEND_PT, va="bottom", ha="left",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))


def _panel_c(ax, summary: dict, recs: list[dict]) -> None:
    ax.set_xlabel("Spacing (µm)", fontsize=LABEL_PT)
    ax.set_ylabel("Grains", fontsize=LABEL_PT)
    ax.tick_params(labelsize=TICK_PT)
    if not recs:
        _empty(ax)
        return
    spac = np.array([r["spacing_um"] for r in recs])
    n = len(spac)
    cap = HIST_CLIP_FACTOR * np.percentile(spac, HIST_CLIP_PCT)
    shown = spac[spac <= cap]
    n_bins = min(HIST_BINS_MAX, max(HIST_BINS_MIN, n // HIST_BINS_DIVISOR))
    ax.hist(shown, bins=n_bins, color=C.PALETTE[PIGEONITE_KEY], alpha=HIST_ALPHA, edgecolor="white")
    s = summary["spacing"]
    nyq = summary["nyquist_limit_um"]
    ax.axvline(nyq, color=C.NYQUIST_COLOR, ls="--", lw=1, label=f"Nyquist ({nyq:.0f} µm)")
    ax.axvline(s["median_um"], color=C.PALETTE[AUGITE_KEY], lw=1.2, label=f"Median ({s['median_um']:.0f} µm)")
    ax.axvspan(s["q25_um"], s["q75_um"], color=C.PALETTE[AUGITE_KEY], alpha=BAND_ALPHA,
               label=f"IQR ({s['q25_um']:.0f}–{s['q75_um']:.0f} µm)")
    ax.legend(fontsize=LEGEND_PT, frameon=False)
    n_clip = int((spac > cap).sum())
    if n_clip:
        ax.text(0.98, 0.5, f"{n_clip} grain(s) > {cap:.0f} µm not shown", transform=ax.transAxes,
                ha="right", fontsize=LEGEND_PT)


def _panel_d(fig, ax, summary: dict, recs: list[dict]) -> None:
    ax.set_xlabel("Grain area (µm²)", fontsize=LABEL_PT)
    ax.set_ylabel("Lamellae spacing (µm)", fontsize=LABEL_PT)
    ax.tick_params(labelsize=TICK_PT)
    if not recs:
        _empty(ax)
        return
    spac = np.array([r["spacing_um"] for r in recs])
    area = np.array([r["area_um2"] for r in recs])
    iqr = np.array([r["spacing_iqr_um"] for r in recs])
    orient = np.array([r["orientation_deg"] for r in recs])
    norm = Normalize(0, 180)
    cmap = plt.get_cmap(ORIENT_CMAP)
    ax.errorbar(area, spac, yerr=iqr / 2, fmt="none", ecolor="0.7", elinewidth=0.5, zorder=1)
    ax.scatter(area, spac, c=orient, cmap=cmap, norm=norm, s=10, edgecolors="none", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Orientation (°)", fontsize=CBAR_PT)
    cb.ax.tick_params(labelsize=TICK_PT)
    g = summary["grain_size_vs_spacing"]
    ax.text(0.04, 0.96, f"ρ = {g['spearman_rho']:.2f}, {_fmt_p(g['spearman_p'])}",
            transform=ax.transAxes, va="top", fontsize=LEGEND_PT)


def _draw(run: Run, summary: dict, recs: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * FIG_HEIGHT_RATIO),
                             gridspec_kw=GRID_KW)
    fig.set_facecolor("white")
    _panel_a(fig, axes[0, 0], run, recs)
    rose = _panel_b(fig, axes[0, 1], summary, recs)
    _panel_c(axes[1, 0], summary, recs)
    _panel_d(fig, axes[1, 1], summary, recs)
    for ax, letter in zip((axes[0, 0], rose, axes[1, 0], axes[1, 1]), "abcd"):
        ax.text(*PANEL_LABEL_XY, f"({letter})", transform=ax.transAxes, fontsize=PANEL_LABEL_PT,
                fontweight="bold", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(run: Run, out: Path, data: DataPaths, *, settings: C.LamellaeSettings = C.LAMELLAE) -> dict:
    # Two benign warnings come from the ported analysis. numpy warns about a 0/0 in the
    # coherence and spacing arithmetic (the NaN is replaced by 0 by the old code), and
    # skimage warns "Only one label was provided" only when the map holds a single grain.
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Only one label was provided", category=UserWarning)
        summary, records = lamellae_core.analyze(
            run.hires_labels, run.phase(AUGITE_KEY), run.phase(PIGEONITE_KEY),
            run.geometry.pixel_size_um_1x, settings)
    out = Path(out)
    figs, tables = out / "figures", out / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / TABLE_NAME, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(records)
    (tables / SUMMARY_NAME).write_text(json.dumps(N._nan_to_none(summary), indent=2, sort_keys=True))
    _draw(run, summary, records, figs / FIG_NAME)
    return summary
