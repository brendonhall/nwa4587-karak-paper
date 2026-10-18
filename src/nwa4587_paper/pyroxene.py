"""Fig 7 and the section 3.4 numbers: the 1x refinement of the pyroxene split."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from . import constants as C  # noqa: E402
from . import numbers as N  # noqa: E402
from . import render  # noqa: E402
from .io import DataPaths, Run, load_bse_1x, load_channel_1x, require  # noqa: E402

N.register({
    "pyroxene.n_pixels_2x": N.Spec("int"),
    "pyroxene.pct_of_mineral_2x": N.Spec("float", tol=0.005, fmt="{:.1f}"),
    "pyroxene.n_pixels_1x.*": N.Spec("int"),
    "pyroxene.pct_1x.*": N.Spec("float", tol=0.05, fmt="{:.1f}"),
    "pyroxene.gmm_mean.*": N.Spec("float", tol=0.0005, fmt="{:.2f}"),
    "pyroxene.ratio_mean.*": N.Spec("float", tol=0.002, fmt="{:.2f}"),
    "pyroxene.n_undefined_1x": N.Spec("int"),
})

SPLIT_NODE = "pyx"
HIGH_KEY = C.FIG5_KARAK_HIGH_KEY      # Augite
LOW_KEY = C.FIG5_KARAK_LOW_KEY        # Pigeonite
CA_ELEMENT, MG_ELEMENT = "Ca", "Mg"
FIG_NAME = "fig7_pyroxene_refinement"

FIG_SIZE_IN = (6.693, 5.689)
GRID_KW = dict(hspace=0.28, wspace=0.32, top=0.95, bottom=0.08, left=0.08, right=0.97)
UNDIFF_BLEND = (0.5, 0.5)             # weights of (grey, undifferentiated colour) in panel (a)
UNDIFF_LABEL = "Pyroxene (undiff.)"
TITLE_A, TITLE_B = "2x downsample", "1x native resolution"
TITLE_PT, PANEL_LABEL_PT, TICK_PT = 8, 9, 6
ALL_LABEL, ALL_COLOR = "All pyroxene", "black"
HIST_BINS = (0.10, 0.75, 110)
HIST_ALPHA, ALL_LINEWIDTH = 0.5, 0.7
KDE_BW = 0.06
KDE_GRID_N = 150
KDE_MG_RANGE, KDE_CA_RANGE = (0.06, 0.36), (0.04, 0.42)
KDE_LEVELS_FRAC = (0.05, 0.95, 5)
KDE_ALPHAS = (0.08, 0.40, 5)
KDE_LINEWIDTH = 0.7
PANEL_LABEL_XY = (-0.06, 1.04)


def _ratio(ca: np.ndarray, mg: np.ndarray) -> np.ndarray:
    tot = ca + mg
    return np.divide(ca, tot, out=np.full(ca.shape, np.nan, np.float32), where=tot > 0)


def load_1x(run: Run, data: DataPaths) -> dict:
    """Labels, BSE, Ca and Mg at 1x on a common shape, the class masks, ratios and subsamples."""
    eds = require(data.eds_dir, "elemental PNG directory", "data --from PATH")
    trim = run.geometry.header_trim_px
    labels = run.labels_1x
    bse = load_bse_1x(eds, trim)
    ca = load_channel_1x(eds, CA_ELEMENT, trim)
    mg = load_channel_1x(eds, MG_ELEMENT, trim)
    h = min(a.shape[0] for a in (labels, bse, ca, mg))
    w = min(a.shape[1] for a in (labels, bse, ca, mg))
    labels, bse, ca, mg = (a[:h, :w] for a in (labels, bse, ca, mg))
    aug_mask = labels == run.phase(HIGH_KEY)
    pig_mask = labels == run.phase(LOW_KEY)
    ratio = _ratio(ca, mg)
    ratio_aug = ratio[aug_mask & np.isfinite(ratio)]
    ratio_pig = ratio[pig_mask & np.isfinite(ratio)]
    rng = np.random.default_rng(C.FIG7_SUBSAMPLE_SEED)
    out = {"labels": labels, "bse": bse, "ca": ca, "mg": mg, "aug_mask": aug_mask, "pig_mask": pig_mask,
           "ratio_aug": ratio_aug, "ratio_pig": ratio_pig,
           "ratio_all": np.concatenate([ratio_aug, ratio_pig])}
    for tag, mask in (("aug", aug_mask), ("pig", pig_mask)):    # augite drawn first
        c, m = ca[mask], mg[mask]
        idx = rng.choice(c.size, size=min(C.FIG7_SUBSAMPLE_N, c.size), replace=False)
        out[f"ca_{tag}"], out[f"mg_{tag}"] = c[idx], m[idx]
    return out


def _crops(run: Run, d: dict) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """(1x crop, 2x crop): the fixed window clipped to the map, else the pyroxene bounding box."""
    h, w = d["labels"].shape
    r0, r1, c0, c1 = C.FIG7_CROP_1X
    r0, r1, c0, c1 = min(r0, h), min(r1, h), min(c0, w), min(c1, w)
    if r1 <= r0 or c1 <= c0:
        rows, cols = np.nonzero(d["aug_mask"] | d["pig_mask"])
        r0, r1, c0, c1 = int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1
    f = run.geometry.factor
    h2, w2 = run.geometry.shape_2x
    a0, a1, b0, b1 = r0 // f, min(max(r1 // f, r0 // f + 1), h2), c0 // f, min(max(c1 // f, c0 // f + 1), w2)
    return (r0, r1, c0, c1), (a0, a1, b0, b1)


def _kde_contours(ax, mg: np.ndarray, ca: np.ndarray, color) -> None:
    gx = np.linspace(*KDE_MG_RANGE, KDE_GRID_N)
    gy = np.linspace(*KDE_CA_RANGE, KDE_GRID_N)
    xx, yy = np.meshgrid(gx, gy)
    try:
        z = gaussian_kde(np.vstack([mg, ca]), bw_method=KDE_BW)(np.vstack([xx.ravel(), yy.ravel()]))
    except np.linalg.LinAlgError:            # a degenerate (constant) class has no density
        z = None
    if z is not None and z.max() > 0:
        z = z.reshape(xx.shape)
        levels = z.max() * np.linspace(*KDE_LEVELS_FRAC)
        alphas = np.linspace(*KDE_ALPHAS)
        for lo, hi, a in zip(levels, list(levels[1:]) + [z.max() * (1 + 1e-9)], alphas):
            ax.contourf(xx, yy, z, levels=[lo, hi], colors=[color], alpha=a)
        ax.contour(xx, yy, z, levels=levels[1::2], colors=[color], linewidths=KDE_LINEWIDTH)
    ax.plot(mg.mean(), ca.mean(), "+", color="white", markersize=9, markeredgewidth=2.2)
    ax.plot(mg.mean(), ca.mean(), "+", color=color, markersize=8, markeredgewidth=1.2)


def _draw(run: Run, d: dict, path: Path) -> None:
    rec = run.split_record(SPLIT_NODE)
    (r0, r1, c0, c1), (a0, a1, b0, b1) = _crops(run, d)
    col = {HIGH_KEY: to_rgb(C.PALETTE[HIGH_KEY]), LOW_KEY: to_rgb(C.PALETTE[LOW_KEY])}
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_IN, gridspec_kw=GRID_KW)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) the 2x label image before the pyroxene split
    pre = run.labels_before(SPLIT_NODE)[a0:a1, b0:b1]
    img = render.bse_gray(run.bse[a0:a1, b0:b1])
    m = pre == int(rec["parent"])
    wg, wc = UNDIFF_BLEND
    img[m] = wg * img[m] + wc * np.array(to_rgb(C.PYROXENE_UNDIFF_COLOR))
    ax_a.imshow(img, interpolation="nearest")
    ax_a.set_title(TITLE_A, fontsize=TITLE_PT, fontweight="bold")
    ax_a.legend(handles=[Patch(color=C.PYROXENE_UNDIFF_COLOR, label=UNDIFF_LABEL)],
                fontsize=TICK_PT, loc="lower right", framealpha=0.9)

    # (b) the 1x crop with Augite and Pigeonite
    lab = d["labels"][r0:r1, c0:c1]
    img = render.bse_gray(d["bse"][r0:r1, c0:c1])
    m = (d["aug_mask"] | d["pig_mask"])[r0:r1, c0:c1]
    img[m] = (1 - C.OVERLAY_ALPHA) * img[m] + C.OVERLAY_ALPHA * render.colorize(lab, run.keys)[m]
    ax_b.imshow(img, interpolation="nearest")
    ax_b.set_title(TITLE_B, fontsize=TITLE_PT, fontweight="bold")
    ax_b.legend(handles=[Patch(color=col[k], label=C.DISPLAY_NAMES[k]) for k in (HIGH_KEY, LOW_KEY)],
                fontsize=TICK_PT, loc="lower right", framealpha=0.9)
    for ax in (ax_a, ax_b):
        ax.set_xticks([])
        ax.set_yticks([])

    # (c) per-pixel Ca/(Ca+Mg)
    bins = np.linspace(*HIST_BINS)
    for k, key in ((LOW_KEY, "ratio_pig"), (HIGH_KEY, "ratio_aug")):
        ax_c.hist(d[key], bins=bins, density=True, color=col[k], alpha=HIST_ALPHA, label=C.DISPLAY_NAMES[k])
    ax_c.hist(d["ratio_all"], bins=bins, density=True, histtype="step", color=ALL_COLOR,
              linewidth=ALL_LINEWIDTH, label=ALL_LABEL)
    for k, key in ((LOW_KEY, "ratio_pig"), (HIGH_KEY, "ratio_aug")):
        ax_c.axvline(float(d[key].mean()), color=col[k], linestyle="--", linewidth=ALL_LINEWIDTH)
    ax_c.set_xlabel("Ca / (Ca + Mg)", fontsize=TICK_PT + 1)
    ax_c.set_ylabel("Density", fontsize=TICK_PT + 1)
    ax_c.legend(fontsize=TICK_PT, frameon=False)
    ax_c.spines[["top", "right"]].set_visible(False)

    # (d) Mg-Ca density
    for k, tag in ((LOW_KEY, "pig"), (HIGH_KEY, "aug")):
        _kde_contours(ax_d, d[f"mg_{tag}"], d[f"ca_{tag}"], col[k])
    ax_d.set_xlim(*KDE_MG_RANGE)
    ax_d.set_ylim(*KDE_CA_RANGE)
    ax_d.set_xlabel("Mg intensity", fontsize=TICK_PT + 1)
    ax_d.set_ylabel("Ca intensity", fontsize=TICK_PT + 1)
    ax_d.spines[["top", "right"]].set_visible(False)

    for ax, tag in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        ax.text(*PANEL_LABEL_XY, f"({tag})", transform=ax.transAxes, fontsize=PANEL_LABEL_PT,
                fontweight="bold", va="bottom")
        ax.tick_params(labelsize=TICK_PT)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(run: Run, out: Path, data: DataPaths) -> dict:
    require(data.eds_dir, "elemental PNG directory", "data --from PATH")
    d = load_1x(run, data)
    figs = Path(out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    _draw(run, d, figs / FIG_NAME)

    rec = run.split_record(SPLIT_NODE)
    labs = {HIGH_KEY: run.phase(HIGH_KEY), LOW_KEY: run.phase(LOW_KEY)}
    n2 = sum(run.counts[lab] for lab in labs.values())
    n1 = {k: int(rec["n_pixels_hires"][str(lab)]) for k, lab in labs.items()}
    return {
        "n_pixels_2x": int(n2),
        "pct_of_mineral_2x": 100.0 * n2 / run.n_mineral_pixels,
        "n_pixels_1x": n1,
        "pct_1x": {k: 100.0 * n / sum(n1.values()) for k, n in n1.items()},
        "gmm_mean": {k: float(next(iter(rec["component_means"][str(lab)].values())))
                     for k, lab in labs.items()},
        "ratio_mean": {HIGH_KEY: float(d["ratio_aug"].mean()), LOW_KEY: float(d["ratio_pig"].mean())},
        "n_undefined_1x": int(rec["n_undefined"]),
    }
