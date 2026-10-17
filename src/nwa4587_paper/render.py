"""Shared image rendering: phase colours and the BSE grey."""
from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb

from . import constants as C


def colorize(labels: np.ndarray, keys: dict[int, str]) -> np.ndarray:
    """Float RGB (H, W, 3): the palette colour per label, grey for a label without a palette entry,
    the unassigned colour for -1."""
    out = np.empty(labels.shape + (3,), np.float64)
    out[...] = C.PALETTE_FALLBACK_COLOR
    out[labels == -1] = to_rgb(C.UNASSIGNED_COLOR)
    for lab in np.unique(labels):
        color = C.PALETTE.get(keys.get(int(lab), ""))
        if lab != -1 and color is not None:
            out[labels == lab] = to_rgb(color)
    return out


def bse_gray(bse: np.ndarray) -> np.ndarray:
    """BSE as grey RGB (H, W, 3), clipped to [0, 1] at C.BSE_DISPLAY_MAX."""
    g = np.clip(np.asarray(bse, np.float64) / C.BSE_DISPLAY_MAX, 0, 1)
    return np.repeat(g[:, :, None], 3, axis=2)
