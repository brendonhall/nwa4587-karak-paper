import numpy as np
from matplotlib.colors import to_rgb

from nwa4587_paper import constants as C, render


def test_colorize_palette_unassigned_and_unknown():
    labels = np.array([[3, -1], [99, 3]])
    rgb = render.colorize(labels, {3: "Plagioclase"})
    assert rgb.shape == (2, 2, 3) and rgb.dtype.kind == "f"
    np.testing.assert_allclose(rgb[0, 0], to_rgb(C.PALETTE["Plagioclase"]))
    np.testing.assert_allclose(rgb[1, 1], to_rgb(C.PALETTE["Plagioclase"]))
    np.testing.assert_allclose(rgb[0, 1], to_rgb(C.UNASSIGNED_COLOR))
    np.testing.assert_allclose(rgb[1, 0], C.PALETTE_FALLBACK_COLOR)


def test_colorize_key_without_palette_entry_falls_back():
    rgb = render.colorize(np.array([[5]]), {5: "No Such Phase"})
    np.testing.assert_allclose(rgb[0, 0], C.PALETTE_FALLBACK_COLOR)


def test_bse_gray_clips_and_repeats():
    bse = np.array([[-0.2, 0.25, 0.9]], np.float32)
    g = render.bse_gray(bse)
    assert g.shape == (1, 3, 3)
    np.testing.assert_allclose(g[0, :, 0], [0.0, 0.25 / C.BSE_DISPLAY_MAX, 1.0])
    np.testing.assert_array_equal(g[..., 0], g[..., 1])
    np.testing.assert_array_equal(g[..., 0], g[..., 2])
