"""Figs 3 to 5 and Tables 3 and 4: the pipeline against the TIMA raster."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import constants as C  # noqa: E402
from . import numbers as N  # noqa: E402
from . import tima_core as T  # noqa: E402
from .io import DataPaths, MissingInput, Run  # noqa: E402
from .tables import _tex_escape  # noqa: E402

N.register({
    "tima.alignment.dx": N.Spec("int"), "tima.alignment.dy": N.Spec("int"),
    "tima.alignment.shape_hw": N.Spec("int"),
    "tima.*.overall_accuracy": N.Spec("float", tol=0.0005, fmt="{:.4f}"),
    "tima.*.spearman_r": N.Spec("float", tol=0.005, fmt="{:.3f}"),
    "tima.*.spearman_p": N.Spec("float", tol=0.5, rel=True, fmt="{:.2e}"),
    "tima.*.n_included": N.Spec("int"),
    "tima.*.per_phase.*.precision": N.Spec("float", tol=0.005, fmt="{:.2f}"),
    "tima.*.per_phase.*.recall": N.Spec("float", tol=0.005, fmt="{:.2f}"),
    "tima.*.per_phase.*.f1": N.Spec("float", tol=0.005, fmt="{:.2f}"),
    "tima.*.per_phase.*.pixel_count": N.Spec("int"),
    "tima.*.per_phase.*.tp": N.Spec("int"), "tima.*.per_phase.*.fp": N.Spec("int"),
    "tima.*.per_phase.*.fn": N.Spec("int"),
    "tima.*.abundance.*.pipeline_pct": N.Spec("float", tol=0.005, fmt="{:.1f}"),
    "tima.*.abundance.*.tima_pct": N.Spec("float", tol=0.005, fmt="{:.1f}"),
    "tima.rare.*.top_tima_name": N.Spec("str"),
    "tima.rare.*.top_pct": N.Spec("float", tol=0.05, fmt="{:.1f}"),
    "tima.fig5_window.x0": N.Spec("int"), "tima.fig5_window.y0": N.Spec("int"),
    "tima.fig5_window.size": N.Spec("int"), "tima.fig5_window.n_both_pyroxene": N.Spec("int"),
})

RC = {"font.family": "sans-serif", "font.size": 8.5, "pdf.fonttype": 42}
FIG_W = 6.69
FIG3_PANEL_W = 3.15
FIG3_LEGEND_H = 0.85
FIG3_MARGIN_H = 0.35
FIG4_IMG_FRAC = 0.65
FIG4_EXTRA_H = 0.5
FIG5_GAP = 0.1
FIG5_EXTRA_H = 0.7
TITLE_PT = 9
LEGEND_PT = 7


def _save(fig, base: Path) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(base.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pyx(k, run: Run, t) -> tuple[np.ndarray, np.ndarray]:
    high = k == run.phase(C.FIG5_KARAK_HIGH_KEY)
    low = k == run.phase(C.FIG5_KARAK_LOW_KEY)
    return (high | low), (t == C.FIG5_TIMA_HIGH) | (t == C.FIG5_TIMA_LOW)


def find_densest_window(k, t, run: Run, size: int, stride: int) -> tuple[int, int, int]:
    """Window of most pixels that are pyroxene in both maps; returns x0, y0, count."""
    kp, tp = _pyx(k, run, t)
    both = (kp & tp).astype(np.int64)
    h, w = both.shape
    size = min(size, h, w)
    ii = np.zeros((h + 1, w + 1), np.int64)
    ii[1:, 1:] = both.cumsum(0).cumsum(1)
    best = (-1, 0, 0)
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            n = int(ii[y + size, x + size] - ii[y, x + size] - ii[y + size, x] + ii[y, x])
            if n > best[0]:
                best = (n, x, y)
    return best[1], best[2], best[0]


def fig5_window(k, t, run: Run) -> tuple[int, int, int, int]:
    h, w = k.shape
    size = min(C.FIG5_WINDOW_SIZE, h, w)
    if C.FIG5_WINDOW_XY is not None:
        x0 = int(np.clip(C.FIG5_WINDOW_XY[0], 0, w - size))
        y0 = int(np.clip(C.FIG5_WINDOW_XY[1], 0, h - size))
    else:
        x0, y0, _ = find_densest_window(k, t, run, size, C.FIG5_WINDOW_STRIDE)
    kp, tp = _pyx(k[y0:y0 + size, x0:x0 + size], run, t[y0:y0 + size, x0:x0 + size])
    return x0, y0, size, int((kp & tp).sum())


def _key_colors() -> dict[str, tuple]:
    return {**C.MATCHED_COLORS, **C.MATCHED_COLORS_SPLIT_OVERRIDE}


def _fig3_images(k, t, mapping) -> tuple[np.ndarray, np.ndarray, dict]:
    colors = _key_colors()
    ka = np.empty(k.shape + (3,), float)
    ta = np.empty(t.shape + (3,), float)
    ka[:] = ta[:] = C.TIMA_BACKGROUND
    tima_color: dict[int, tuple] = {}
    for key in C.LEGEND_ORDER_SPLIT:
        for lbl, info in mapping.items():
            if info["key"] != key:
                continue
            ka[k == lbl] = colors[key]
            for tl in info["tima_labels"]:
                tima_color.setdefault(tl, colors[key])
    for tl, col in tima_color.items():
        ta[t == tl] = col
    return ka, ta, colors


def _legend_entries(mapping, colors) -> list[Patch]:
    entries = []
    for key in C.LEGEND_ORDER_SPLIT:
        for info in mapping.values():
            if info["key"] != key:
                continue
            names = "/".join(C.TIMA_NAMES[i] for i in info["tima_labels"])
            label = info["name"] if not names or names == info["name"] else f"{info['name']} [{names}]"
            entries.append(Patch(facecolor=colors[key], label=label))
    entries.append(Patch(facecolor=C.TIMA_BACKGROUND, edgecolor=C.FIG_LEGEND_EDGE_COLOR,
                         label=C.FIG3_BACKGROUND_LABEL))
    return entries


def _fig3(k, t, mapping, path: Path) -> None:
    h, w = k.shape
    ka, ta, colors = _fig3_images(k, t, mapping)
    panel_h = FIG3_PANEL_W * h / w
    fig_h = panel_h + FIG3_LEGEND_H + FIG3_MARGIN_H
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, fig_h), gridspec_kw={"wspace": 0.04})
    for ax, img, title in zip(axes, (ka, ta), (C.FIG3_TITLE_A, C.FIG3_TITLE_B)):
        ax.imshow(img, interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(title, fontsize=TITLE_PT, fontweight="bold", loc="left", pad=4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=FIG3_LEGEND_H / fig_h)
    fig.legend(handles=_legend_entries(mapping, colors), loc="lower center", ncol=3, fontsize=LEGEND_PT,
               framealpha=0.95, edgecolor=C.FIG_LEGEND_EDGE_COLOR, borderaxespad=0.3,
               handlelength=1.2, handletextpad=0.5, columnspacing=1.0)
    _save(fig, path)


def _agreement_rgb(agree, included) -> np.ndarray:
    rgb = np.empty(agree.shape + (3,), float)
    rgb[:] = C.EXCLUDED_COLOR
    rgb[included] = C.DISAGREE_COLOR
    rgb[agree] = C.AGREE_COLOR
    return rgb


def _fig4_image(ax, agree, included, acc: float) -> None:
    n_agree, n_inc = int(agree.sum()), int(included.sum())
    pct = n_agree / n_inc * 100 if n_inc else 0.0
    ax.imshow(_agreement_rgb(agree, included), interpolation="nearest")
    ax.set_axis_off()
    ax.text(0.98, 0.98, f"{C.FIG4_ANNOTATION}\n{acc * 100:.1f}%", transform=ax.transAxes, ha="right",
            va="top", fontsize=TITLE_PT, fontweight="bold",
            bbox={"boxstyle": "round", "facecolor": C.FIG_BOX_COLOR, "alpha": 0.9})
    ax.legend(handles=[Patch(facecolor=C.AGREE_COLOR, label=f"Agree ({n_agree:,} px, {pct:.1f}%)"),
                       Patch(facecolor=C.DISAGREE_COLOR,
                             label=f"Disagree ({n_inc - n_agree:,} px, {100 - pct:.1f}%)"),
                       Patch(facecolor=C.EXCLUDED_COLOR, label=C.FIG4_EXCLUDED_LABEL)],
              loc="lower left", fontsize=LEGEND_PT)


def _fig4_bars(ax, k, agree, included, mapping) -> None:
    rows = []
    for lbl, info in mapping.items():
        sel = included & (k == lbl)
        total = int(sel.sum())
        if total:
            rows.append((C.FIG4_SHORT_NAMES.get(info["key"], info["key"]), int((sel & agree).sum()), total))
    rows.sort(key=lambda r: -r[2])
    y = np.arange(len(rows))
    ag = np.array([r[1] for r in rows])
    tot = np.array([r[2] for r in rows])
    ax.barh(y, ag, height=0.65, color=C.AGREE_COLOR)
    ax.barh(y, tot - ag, left=ag, height=0.65, color=C.DISAGREE_COLOR)
    for yi, (_, a, n) in zip(y, rows):
        ax.text(n * 1.02, yi, f"{a / n * 100:.0f}%", va="center", fontsize=6, color=C.FIG_LABEL_GREY)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, (tot.max() if len(tot) else 1) * 1.18)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_title(C.FIG4_TITLE_B, fontsize=TITLE_PT, fontweight="bold", loc="left")


def _fig4(k, t, mapping, ek, et, acc: float, path: Path) -> None:
    h, w = k.shape
    agree, included = T.agreement_masks(k, t, mapping, ek, et)
    img_w = FIG_W * FIG4_IMG_FRAC
    fig = plt.figure(figsize=(FIG_W, img_w * h / w + FIG4_EXTRA_H))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.08, left=0.01, right=0.98,
                          top=0.95, bottom=0.06)
    _fig4_image(fig.add_subplot(gs[0]), agree, included, acc)
    _fig4_bars(fig.add_subplot(gs[1]), k, agree, included, mapping)
    _save(fig, path)


def _fig5(k, t, run: Run, win, path: Path) -> None:
    x0, y0, size, _ = win
    ks, ts = k[y0:y0 + size, x0:x0 + size], t[y0:y0 + size, x0:x0 + size]
    kimg = np.empty(ks.shape + (3,), float)
    timg = np.empty(ts.shape + (3,), float)
    kimg[:] = timg[:] = C.FIG5_NON_PYX
    kimg[ks == run.phase(C.FIG5_KARAK_HIGH_KEY)] = C.FIG5_HIGH_CA
    kimg[ks == run.phase(C.FIG5_KARAK_LOW_KEY)] = C.FIG5_LOW_CA
    timg[ts == C.FIG5_TIMA_HIGH] = C.FIG5_HIGH_CA
    timg[ts == C.FIG5_TIMA_LOW] = C.FIG5_LOW_CA
    panel_w = (FIG_W - FIG5_GAP) / 2
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, panel_w + FIG5_EXTRA_H), gridspec_kw={"wspace": 0.03})
    for ax, img, title in zip(axes, (kimg, timg), (C.FIG5_TITLE_A, C.FIG5_TITLE_B)):
        ax.imshow(img, interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(title, fontsize=TITLE_PT, fontweight="bold", loc="left")
    cols = (C.FIG5_HIGH_CA, C.FIG5_LOW_CA, C.FIG5_NON_PYX)
    fig.legend(handles=[Patch(facecolor=c, edgecolor=C.FIG_LEGEND_EDGE_COLOR, label=lab)
                        for c, lab in zip(cols, C.FIG5_LEGEND_LABELS)],
               loc="lower center", ncol=3, fontsize=7.5)
    _save(fig, path)


def _metrics(run: Run, k, t, mode: str):
    mapping = T.build_mapping(run, mode)
    ek, et = T.exclusions(run, mode)
    conf = T.confusion(k, t, mapping, ek, et)
    table, r, p = T.abundance(k, t, mapping)
    return mapping, ek, et, conf, table, r, p


def _summary(conf, table, r, p) -> dict:
    return {
        "overall_accuracy": float(conf.overall_accuracy), "spearman_r": float(r), "spearman_p": float(p),
        "n_included": int(conf.n_included),
        "per_phase": {d["key"]: {m: (float(d[m]) if m in ("precision", "recall", "f1") else int(d[m]))
                                 for m in ("precision", "recall", "f1", "pixel_count", "tp", "fp", "fn")}
                      for d in conf.per_phase},
        "abundance": {d["key"]: {"pipeline_pct": float(d["pipeline_pct"]), "tima_pct": float(d["tima_pct"])}
                      for d in table},
    }


def _write_confusion(path: Path, run: Run, conf, mapping) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        names = [C.TIMA_NAMES.get(i, f"TIMA {i}") for i in conf.tima_ids]
        w.writerow(["pipeline", *names])
        for lbl, row in zip(conf.pipeline_ids, conf.cm):
            key = mapping.get(lbl, {}).get("key", run.keys.get(lbl, f"label_{lbl}"))
            w.writerow([key, *[int(v) for v in row]])


def write_table3(path: Path, run: Run) -> None:
    lines = [r"\begin{tabular}{lll}", r"\toprule", r"Pipeline phase & TIMA phase & Note \\", r"\midrule"]
    for key, tima_name, note in C.TABLE3_ROWS:
        if run.has(key):
            lines.append(f"{_tex_escape(C.DISPLAY_NAMES.get(key, key))} & {_tex_escape(tima_name)} "
                         f"& {_tex_escape(note)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def write_table4(d: Path, per_phase: dict) -> None:
    rows = sorted(per_phase.items(), key=lambda kv: -kv[1]["pixel_count"])
    rows = [(C.TABLE4_NAMES.get(k, k), v) for k, v in rows]
    with open(d / "table4_concordance.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase", "precision", "recall", "f1", "pixels"])
        for name, v in rows:
            w.writerow([name, f"{v['precision']:.2f}", f"{v['recall']:.2f}", f"{v['f1']:.2f}",
                        v["pixel_count"]])
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Phase & Precision & Recall & F1 & Pixels \\",
             r"\midrule"]
    for name, v in rows:
        px = f"{v['pixel_count']:,}".replace(",", "{,}")
        lines.append(f"{_tex_escape(name)} & {v['precision']:.2f} & {v['recall']:.2f} & {v['f1']:.2f} "
                     f"& {px} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (d / "table4_concordance.tex").write_text("\n".join(lines))


def _rare(kc, tc, run: Run) -> tuple[dict, dict]:
    full = T.rare_phases(kc, tc, run, combined=True)
    top = {e["name"]: {"top_tima_name": e["top_destinations"][0]["tgt_name"],
                       "top_pct": float(e["top_destinations"][0]["pct_of_src"])}
           for e in full["karak_rare"] if e["top_destinations"]}
    return full, top


def run(run: Run, out: Path, data: DataPaths, dx: int = C.TIMA_DX, dy: int = C.TIMA_DY) -> dict:
    if data.tima_png is None:
        raise MissingInput("no TIMA raster under data/tima")
    figs, tabs = Path(out) / "figures", Path(out) / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    k, t, info = T.align(run.labels_1x, T.decode_tima(data.tima_png), dx, dy)
    sp = _metrics(run, k, t, "split")
    kc, tc = T.combine_pyroxene(run.phase, k, t)
    cb = _metrics(run, kc, tc, "combined")
    full_rare, rare = _rare(kc, tc, run)
    win = fig5_window(k, t, run)
    print(f"tima: fig5 window x0={win[0]} y0={win[1]} size={win[2]}")
    with plt.rc_context(RC):
        _fig3(k, t, sp[0], figs / "fig3_phase_comparison")
        _fig4(k, t, sp[0], sp[1], sp[2], sp[3].overall_accuracy, figs / "fig4_agreement_map")
        _fig5(k, t, run, win, figs / "fig5_pyroxene_tima")
    summaries = {"split": _summary(sp[3], sp[4], sp[5], sp[6]),
                 "combined": _summary(cb[3], cb[4], cb[5], cb[6])}
    write_table3(tabs / "table3_correspondence.tex", run)
    write_table4(tabs, summaries["split"]["per_phase"])
    _write_confusion(tabs / "tima_confusion_split.csv", run, sp[3], sp[0])
    _write_confusion(tabs / "tima_confusion_combined.csv", run, cb[3], cb[0])
    (tabs / "tima_rare_phases.json").write_text(json.dumps(full_rare, indent=2))
    return {"alignment": {"dx": info["dx"], "dy": info["dy"], "shape_hw": list(info["shape_hw"])},
            **summaries, "rare": rare,
            "fig5_window": {"x0": win[0], "y0": win[1], "size": win[2], "n_both_pyroxene": win[3]}}
