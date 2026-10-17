import numpy as np
import pytest
import imageio.v3 as iio

from nwa4587_paper import constants as C, synth, tima_core as T
from nwa4587_paper.io import Run


def _png(tmp_path, rgb8, bits=8):
    p = tmp_path / f"t{bits}.png"
    if bits == 16:
        synth.write_png16_rgb(p, rgb8.astype(np.uint16) * 257)
    else:
        iio.imwrite(p, rgb8.astype(np.uint8))
    return p


def test_decode_exact_near_far_grey_and_trim(tmp_path):
    img = np.zeros((6, 4, 3), np.uint8)
    img[0, 0] = C.TIMA_PHASES[0].rgb                       # exact -> 0
    img[0, 1] = np.array(C.TIMA_PHASES[7].rgb) - 10        # within 50 -> 7
    img[0, 2] = (200, 60, 60)                              # far from every colour -> -1
    img[0, 3] = (120, 125, 118)                            # achromatic -> -1
    img[1, 0] = (10, 10, 10)                               # dark, max <= 20: not achromatic rule; nearest wins or -1 by distance
    img[5, :] = C.TIMA_PHASES[0].rgb                       # legend row, trimmed away
    for bits in (8, 16):
        out = T.decode_tima(_png(tmp_path, img, bits), legend_trim=1)
        assert out.shape == (5, 4) and out.dtype == np.int8
        assert out[0, 0] == 0 and out[0, 1] == 7 and out[0, 2] == -1 and out[0, 3] == -1
        assert out[1, 0] == -1                              # (10,10,10) is 30+ from (0,85,0): over the limit


def test_align_windows():
    karak = np.zeros((10, 8), np.int16)
    tima = np.zeros((30, 30), np.int8)
    k, t, info = T.align(karak, tima, dx=5, dy=7)
    assert k.shape == t.shape == (10, 8)
    assert info["tima_window_xyxy"] == (5, 7, 13, 17)
    assert info["karak_window_xyxy"] == (0, 0, 8, 10)


def test_align_crops_karak_when_tima_is_short():
    karak = np.zeros((10, 8), np.int16)
    tima = np.zeros((12, 30), np.int8)
    k, t, info = T.align(karak, tima, dx=5, dy=7)
    assert k.shape == (5, 8) and info["shape_hw"] == (5, 8)


def test_align_no_overlap():
    with pytest.raises(ValueError, match="^Translation produces no overlap$"):
        T.align(np.zeros((4, 4)), np.zeros((4, 4)), dx=10, dy=10)


def test_combine_and_mapping(synth_h5):
    with Run.open(synth_h5) as run:
        karak = np.array([[run.phase("Augite"), run.phase("Pigeonite"), -1]], np.int16)
        tima = np.array([[2, 5, 1]], np.int8)
        k, t = T.combine_pyroxene(run.phase, karak, tima)
        assert list(k[0]) == [run.phase("Pigeonite")] * 2 + [-1]
        assert list(t[0]) == [1, 1, 1]
        split = T.build_mapping(run, "split")
        assert split[run.phase("Augite")]["tima_labels"] == [2]
        assert "Calcite" not in {v["key"] for v in split.values()}    # absent from the synthetic run
        comb = T.build_mapping(run, "combined")
        assert run.phase("Augite") not in comb
        assert comb[run.phase("Pigeonite")]["name"] == "Pyroxene"
        ek, et = T.exclusions(run, "split")
        assert ek == {-1, run.phase("Xenotime"), run.phase("Weathering Assemblage")}
        assert et == {-1, 5, 9, 10, 11, 12}
