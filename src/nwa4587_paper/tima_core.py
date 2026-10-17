"""TIMA raster decode, registration, pyroxene merging, confusion metrics."""
from __future__ import annotations

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
