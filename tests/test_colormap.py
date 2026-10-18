import json

import numpy as np
import pytest

from nwa4587_paper import cli, colormap as CM, constants as C, numbers as N


def test_roundtrip_small_lut_is_accurate():
    settings = C.ColormapSettings(lut_n=256, n_test=2000, cmaps=("jet",))
    nums = CM.run(None, None, None, settings=settings)
    N.validate("colormap", nums)
    assert nums["jet"]["mae"] < 0.01 and nums["jet"]["max_ae"] < 0.05
    assert set(nums["jet_regions"]) == {"dark blue", "cyan", "green", "yellow", "orange-red", "dark red"}
    assert nums["lut_step"] == 1 / 255


def test_regions_come_from_constants():
    assert [CM.classify_jet_region(v) for v in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)] == [
        n for _, n in C.COLORMAP_REGION_BOUNDS
    ]


def test_cli_colormap_writes_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "COLORMAP", C.ColormapSettings(lut_n=256, n_test=500, cmaps=("jet",)))
    assert cli.main(["colormap", "--out", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "numbers" / "colormap.json").read_text())
    assert "jet" in data and "lut_step" in data


@pytest.mark.slow
def test_lut_matches_karak_on_samples():
    from karak.io.loaders import get_full_lut
    ours = CM.build_lut("jet", 4096)
    theirs = get_full_lut("cmap:jet")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 256, size=(1000, 3))
    assert np.allclose(ours[idx[:, 0], idx[:, 1], idx[:, 2]], theirs[idx[:, 0], idx[:, 1], idx[:, 2]], atol=1e-6)
