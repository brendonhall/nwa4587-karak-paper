"""TIMA raster decode, registration, pyroxene merging, confusion metrics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from . import constants as C
from .io import Run

Mode = Literal["split", "combined"]
_TIMA_RGB = np.array([p.rgb for p in C.TIMA_PHASES], dtype=np.float32)


def read_rgb8(png: Path) -> np.ndarray:
    """RGB uint8 (H, W, 3) from an 8-bit or 16-bit PNG.

A 16-bit RGB PNG is read through Pillow, which keeps the high byte (v >> 8).
This matches the old script's Image.convert("RGB").
"""
    import imageio.v3 as iio

    arr = iio.imread(png)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, 2)
    arr = arr[:, :, :3]
    if arr.dtype == np.uint16:
        # Runs only when a reader returns uint16 (for example 16-bit greyscale).
        arr = (arr >> 8).astype(np.uint8)
    return np.ascontiguousarray(arr.astype(np.uint8))


def decode_tima(png: Path, legend_trim: int = C.TIMA_LEGEND_TRIM_ROWS,
                distance_max: float = C.TIMA_COLOR_DISTANCE_MAX) -> np.ndarray:
    arr = read_rgb8(png)
    if legend_trim > 0:
        arr = arr[:-legend_trim, :, :]
    H, W, _ = arr.shape
    arr_f = arr.astype(np.float32)
    channel_range = arr_f.max(axis=2) - arr_f.min(axis=2)
    achromatic = (channel_range < C.TIMA_ACHROMATIC_RANGE) & (arr_f.max(axis=2) > C.TIMA_ACHROMATIC_MIN)
    flat = arr_f.reshape(-1, 3)
    achromatic_flat = achromatic.reshape(-1)
    n = flat.shape[0]
    out = np.empty(n, dtype=np.int8)
    chunk = C.TIMA_DECODE_CHUNK_PX
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = flat[start:end, None, :] - _TIMA_RGB[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        best = dists.argmin(axis=1)
        best_dist = dists[np.arange(end - start), best]
        seg = best.astype(np.int8)
        seg[best_dist > distance_max] = -1
        seg[achromatic_flat[start:end]] = -1
        out[start:end] = seg
    return out.reshape(H, W)


def align(karak: np.ndarray, tima: np.ndarray, dx: int = C.TIMA_DX, dy: int = C.TIMA_DY):
    h_k, w_k = karak.shape
    h_t, w_t = tima.shape
    x0_t, y0_t = max(0, dx), max(0, dy)
    x1_t, y1_t = min(w_t, dx + w_k), min(h_t, dy + h_k)
    if x1_t <= x0_t or y1_t <= y0_t:
        raise ValueError("Translation produces no overlap")
    x0_k, y0_k, x1_k, y1_k = x0_t - dx, y0_t - dy, x1_t - dx, y1_t - dy
    karak_crop = karak[y0_k:y1_k, x0_k:x1_k]
    tima_crop = tima[y0_t:y1_t, x0_t:x1_t]
    assert karak_crop.shape == tima_crop.shape
    info = {"dx": int(dx), "dy": int(dy),
            "tima_window_xyxy": (int(x0_t), int(y0_t), int(x1_t), int(y1_t)),
            "karak_window_xyxy": (int(x0_k), int(y0_k), int(x1_k), int(y1_k)),
            "shape_hw": (int(karak_crop.shape[0]), int(karak_crop.shape[1]))}
    return karak_crop, tima_crop, info


def combine_pyroxene(run_phase: Callable[[str], int], karak: np.ndarray, tima: np.ndarray):
    k, t = karak.copy(), tima.copy()
    k[k == run_phase(C.KARAK_COMBINE_FROM_KEY)] = run_phase(C.KARAK_COMBINE_INTO_KEY)
    for src in C.TIMA_COMBINE_FROM:
        t[t == src] = C.TIMA_COMBINE_INTO
    return k, t


def build_mapping(run: Run, mode: Mode) -> dict[int, dict]:
    order = C.LEGEND_ORDER_SPLIT if mode == "split" else C.LEGEND_ORDER_COMBINED
    table = C.TIMA_MAPPING_SPLIT if mode == "split" else C.TIMA_MAPPING_COMBINED
    out: dict[int, dict] = {}
    for key in order:
        if not run.has(key):
            continue
        name, tima_labels, notes = C.MAPPING_NAMES.get(key, key), list(table[key]), C.TIMA_MAPPING_NOTES[key]
        if mode == "combined" and key == C.KARAK_COMBINE_INTO_KEY:
            name, tima_labels, notes = C.TIMA_COMBINED_NAME, [C.TIMA_COMBINE_INTO], C.COMBINED_PYROXENE_NOTE
        out[run.phase(key)] = {"key": key, "name": name, "tima_labels": tima_labels, "notes": notes}
    return out


def exclusions(run: Run, mode: Mode) -> tuple[set[int], set[int]]:
    karak = {-1} | {run.phase(k) for k in C.EXCLUDED_KARAK_KEYS if run.has(k)}
    tima = {-1} | set(C.EXCLUDED_TIMA_SPLIT if mode == "split" else C.EXCLUDED_TIMA_COMBINED)
    return karak, tima


@dataclass
class Confusion:
    cm: np.ndarray
    per_phase: list[dict]
    overall_accuracy: float
    pipeline_ids: list[int]
    tima_ids: list[int]
    n_included: int


def confusion(pipeline: np.ndarray, tima: np.ndarray, mapping: dict[int, dict],
              excluded_pipeline: set[int], excluded_tima: set[int]) -> Confusion:
    if pipeline.shape != tima.shape:
        raise ValueError(f"Shape mismatch: pipeline {pipeline.shape} vs TIMA {tima.shape}")
    mask = np.ones(pipeline.shape, dtype=bool)
    for lbl in excluded_pipeline:
        mask &= pipeline != lbl
    for lbl in excluded_tima:
        mask &= tima != lbl
    pl_flat, t_flat = pipeline[mask], tima[mask]
    n_included = int(mask.sum())
    pipeline_ids = sorted(int(x) for x in np.unique(pl_flat))
    tima_ids = sorted(int(x) for x in np.unique(t_flat))
    pl_idx = {lbl: i for i, lbl in enumerate(pipeline_ids)}
    t_idx = {lbl: i for i, lbl in enumerate(tima_ids)}
    cm = np.zeros((len(pipeline_ids), len(tima_ids)), dtype=np.int64)
    if n_included:
        pl_lut = np.full(max(pipeline_ids) + 2, -1, np.int32)
        t_lut = np.full(max(tima_ids) + 2, -1, np.int32)
        for lbl, i in pl_idx.items():
            pl_lut[lbl] = i
        for lbl, i in t_idx.items():
            t_lut[lbl] = i
        np.add.at(cm, (pl_lut[pl_flat.astype(np.intp)], t_lut[t_flat.astype(np.intp)]), 1)
    per_phase = []
    for pl_lbl in pipeline_ids:
        row_i = pl_idx[pl_lbl]
        total = int(cm[row_i, :].sum())
        mapped = mapping.get(pl_lbl, {}).get("tima_labels", [])
        tp = sum(int(cm[row_i, t_idx[t]]) for t in mapped if t in t_idx)
        fp = total - tp
        fn = 0
        for t in mapped:
            if t in t_idx:
                col_j = t_idx[t]
                col_total = int(cm[:, col_j].sum())
                own = int(cm[row_i, col_j])
                other = sum(int(cm[pl_idx[o], col_j]) for o, info in mapping.items()
                            if o != pl_lbl and t in info.get("tima_labels", []) and o in pl_idx)
                fn += col_total - own - other
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        info = mapping.get(pl_lbl, {})
        per_phase.append({"pipeline_label": pl_lbl, "key": info.get("key", f"label_{pl_lbl}"),
                          "name": info.get("name", f"Label {pl_lbl}"), "tp": tp, "fp": fp, "fn": fn,
                          "precision": precision, "recall": recall, "f1": f1, "pixel_count": total})
    n_agree = sum(int(cm[pl_idx[p], t_idx[t]]) for p in pipeline_ids
                  for t in mapping.get(p, {}).get("tima_labels", []) if t in t_idx)
    acc = n_agree / n_included if n_included > 0 else 0.0
    return Confusion(cm, per_phase, acc, pipeline_ids, tima_ids, n_included)


def abundance(pipeline: np.ndarray, tima: np.ndarray, mapping: dict[int, dict]):
    from scipy.stats import spearmanr

    if pipeline.shape != tima.shape:
        raise ValueError("Shape mismatch")
    overlap = (pipeline >= 0) & (tima >= 0)
    n = int(overlap.sum())
    pl_flat, t_flat = pipeline[overlap], tima[overlap]
    table, p_pcts, t_pcts = [], [], []
    for pl_lbl, info in sorted(mapping.items(), key=lambda kv: kv[0]):
        pl_count = int((pl_flat == pl_lbl).sum())
        t_count = sum(int((t_flat == t).sum()) for t in info["tima_labels"])
        pl_pct = pl_count / n * 100 if n else 0.0
        t_pct = t_count / n * 100 if n else 0.0
        table.append({"pipeline_label": pl_lbl, "key": info["key"], "name": info["name"],
                      "pipeline_count": pl_count, "pipeline_pct": pl_pct,
                      "tima_count": t_count, "tima_pct": t_pct, "notes": info.get("notes", "")})
        p_pcts.append(pl_pct)
        t_pcts.append(t_pct)
    if len(p_pcts) >= 3:
        res = spearmanr(p_pcts, t_pcts)
        return table, float(res.statistic), float(res.pvalue)
    return table, float("nan"), float("nan")


def agreement_masks(pipeline, tima, mapping, excluded_pipeline, excluded_tima):
    included = ~np.isin(pipeline, list(excluded_pipeline)) & ~np.isin(tima, list(excluded_tima))
    agree = np.zeros(pipeline.shape, bool)
    for p, info in mapping.items():
        agree |= (pipeline == p) & np.isin(tima, info["tima_labels"])
    return agree & included, included


def _top_destinations(src, tgt, src_lbl, name_fn, top_n):
    m = src == src_lbl
    n_src = int(m.sum())
    if n_src == 0:
        return []
    vals, counts = np.unique(tgt[m], return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return [{"tgt_label": int(vals[i]), "tgt_name": name_fn(int(vals[i])), "count": int(counts[i]),
             "pct_of_src": counts[i] / n_src * 100} for i in order[:top_n]]


def rare_phases(pipeline, tima, run: Run, threshold: int = C.RARE_PHASE_THRESHOLD_PX,
                top_n: int = C.RARE_TOP_N, combined: bool = False) -> dict:
    def kname(lbl):
        if lbl == -1:
            return C.RARE_BACKGROUND_NAME
        if combined and lbl == run.phase(C.KARAK_COMBINE_INTO_KEY):
            return C.RARE_COMBINED_PYROXENE_NAME
        return run.keys.get(lbl, f"karak {lbl}")

    def tname(lbl):
        if lbl == -1:
            return C.RARE_BACKGROUND_NAME
        if combined and lbl == C.TIMA_COMBINE_INTO:
            return C.RARE_COMBINED_PYROXENE_NAME
        return C.TIMA_NAMES.get(lbl, f"TIMA {lbl}")

    def counts(a):
        v, c = np.unique(a, return_counts=True)
        return {int(x): int(y) for x, y in zip(v, c)}

    def table(src_counts, name_fn, src, other, other_name_fn):
        n_mineral = sum(c for l, c in src_counts.items() if l != -1)
        rare = sorted(((l, c) for l, c in src_counts.items() if l != -1 and 0 < c < threshold),
                      key=lambda lc: -lc[1])
        return [{"label": l, "name": name_fn(l), "pixel_count": c, "pct_of_mineral": c / n_mineral * 100,
                 "top_destinations": _top_destinations(src, other, l, other_name_fn, top_n)} for l, c in rare]

    kc, tc = counts(pipeline), counts(tima)
    return {"threshold": threshold, "combine_pyroxene": combined,
            "karak_phase_counts": {str(k): v for k, v in kc.items()},
            "tima_phase_counts": {str(k): v for k, v in tc.items()},
            "karak_rare": table(kc, kname, pipeline, tima, tname),
            "tima_rare": table(tc, tname, tima, pipeline, kname)}
