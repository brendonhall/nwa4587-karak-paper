import math
import numpy as np
import pytest

from nwa4587_paper import lamellae_core as L

# The verbatim code emits two warnings on synthetic data: np.where evaluates the
# coherence division where lam_max + lam_min == 0 (empty background), and skimage
# warns when the labelled image holds a single grain. Real maps hold many grains.
pytestmark = [
    pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:Only one label was provided:UserWarning"),
]


def striped_grain(h=300, w=300, period=20, angle_deg=30.0):
    """A disc of radius 120 filled with augite/pigeonite stripes of the given period and angle."""
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < 120 ** 2
    th = math.radians(angle_deg)
    coord = xx * math.sin(th) - yy * math.cos(th)         # distance across the stripes
    aug = ((coord // (period / 2)) % 2 == 0)
    hires = np.full((h, w), -1, np.int16)
    hires[disc & aug] = 14
    hires[disc & ~aug] = 13
    return hires


def test_binary_map_and_segmentation():
    hires = striped_grain()
    b = L.binary_map(hires, augite=14, pigeonite=13)
    assert set(np.unique(b)) == {-1, 0, 1}
    grains, n = L.segment_grains(b, min_size=500, settings=L.C.LAMELLAE)
    assert n == 1


def test_orientation_and_spacing_recovered():
    hires = striped_grain(period=20, angle_deg=30.0)
    b = L.binary_map(hires, 14, 13)
    grain = b >= 0
    orient, coh = L.measure_orientation(b == 1, grain, sigma=1.5, settings=L.C.LAMELLAE)
    assert coh > 0.2
    # The tensor axes are (row, col): the returned angle is the lamellae direction measured from the
    # row axis toward the column axis, mod 180. A 30 degree stripe normal gives 150 as gradient, +90 = 60.
    assert abs(orient - 60.0) < 6.0


def test_spacing_recovered_at_45_degrees():
    # At 45 degrees the row/col conventions of orientation and spacing coincide (see the task report).
    hires = striped_grain(period=20, angle_deg=45.0)
    b = L.binary_map(hires, 14, 13)
    grain = b >= 0
    orient, coh = L.measure_orientation(b == 1, grain, sigma=1.5, settings=L.C.LAMELLAE)
    assert abs(orient - 45.0) < 6.0
    spacing_px, iqr, n = L.measure_spacing(b == 1, grain, orient, n_profiles=20, settings=L.C.LAMELLAE)
    assert n >= 2 and abs(spacing_px - 20.0) <= 2.0


def test_rayleigh_uniform_and_concentrated():
    r_u, p_u = L.rayleigh_test(np.linspace(0, 180, 36, endpoint=False))
    r_c, p_c = L.rayleigh_test(np.full(30, 40.0))
    assert p_u > 0.5 and r_c > 0.99 and p_c < 1e-6


def test_analyze_summary_keys_and_values():
    summary, grains = L.analyze(striped_grain(angle_deg=45.0), augite=14, pigeonite=13, pixel_size_um=3.0)
    assert summary["n_grains_total"] == 1 and summary["n_grains_analyzed"] == 1
    assert summary["nyquist_limit_um"] == 6.0
    assert set(summary) == {"n_grains_total", "n_grains_analyzed", "pixel_size_um", "nyquist_limit_um",
                            "orientation", "spacing", "grain_size_vs_spacing", "coherence"}
    assert abs(summary["spacing"]["median_um"] - 60.0) <= 6.0
    assert grains[0]["near_nyquist"] is False and grains[0]["area_px"] > 40000
    assert set(grains[0]) == {"grain_id", "area_px", "area_um2", "equiv_diameter_um", "centroid_row",
                              "centroid_col", "bbox", "augite_fraction", "orientation_deg", "coherence",
                              "spacing_px", "spacing_um", "spacing_iqr_um", "n_profiles", "near_nyquist"}


def test_lamellae_no_grains():
    hires = np.full((50, 50), -1, np.int16)
    hires[10:14, 10:14] = 14                                  # 16 px: below min_grain_px
    summary, grains = L.analyze(hires, 14, 13, pixel_size_um=3.0)
    assert summary["n_grains_total"] == 0 and summary["n_grains_analyzed"] == 0 and grains == []
    assert math.isnan(summary["spacing"]["median_um"]) and summary["orientation"]["uniform"] is None


def test_lamellae_grain_below_analysis_size():
    # A grain that passes segmentation but not min_analysis_px: zero analyzed, NaN stats, no error.
    hires = np.full((100, 100), -1, np.int16)
    hires[10:50, 10:50] = 14
    hires[10:50, 30:50] = 13                                  # 1600 px < 2000
    summary, grains = L.analyze(hires, 14, 13, pixel_size_um=3.0)
    assert summary["n_grains_total"] == 1 and summary["n_grains_analyzed"] == 0 and grains == []
    assert math.isnan(summary["orientation"]["circular_mean_deg"])
    assert math.isnan(summary["coherence"]["mean"]) and summary["orientation"]["uniform"] is None
