import json

import numpy as np
import pytest

from nwa4587_paper import cli, numbers as N, xenotime
from nwa4587_paper.io import DataPaths, MissingInput, Run


def test_region_is_square_and_padded():
    img = np.full((100, 100), -1)
    img[40:44, 50:60] = 10
    img[90, 90] = 10
    info, rs, cs = xenotime.region(img, 10, pad=5)
    assert info["n_components"] == 2 and info["largest_component_px"] == 40
    assert (rs.stop - rs.start) == (cs.stop - cs.start)
    assert rs.start <= 35 and cs.start <= 45 and cs.stop >= 65


def test_xenotime_stage(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        nums = xenotime.run(run, tmp_path, DataPaths())
    N.validate("xenotime", nums)
    assert nums["n_pixels_2x"] == 16 and nums["n_components"] == 1
    assert nums["largest_component_px"] == 16
    assert len(nums["bbox_2x"]) == 4 and len(nums["roi_2x"]) == 4 and len(nums["centroid_2x"]) == 2
    r0, c0, r1, c1 = nums["bbox_2x"]
    assert (r1 - r0, c1 - c0) == (4, 4)
    assert nums["centroid_2x"] == [(r0 + r1 - 1) / 2, (c0 + c1 - 1) / 2]
    for ext in ("png", "pdf"):
        assert (tmp_path / "figures" / f"fig6_xenotime_callout.{ext}").exists()


def test_missing_channel_names_it(synth_h5, tmp_path, monkeypatch):
    with Run.open(synth_h5) as run:
        monkeypatch.setattr(type(run), "element_order", ["Ca", "Mg"])
        with pytest.raises(MissingInput, match=r"channel Y not in denoised cube \(elements: Ca, Mg\)"):
            xenotime.run(run, tmp_path, DataPaths())


def test_cli_xenotime(synth_h5, synth_eds, tmp_path):
    rc = cli.main(["xenotime", "--h5", str(synth_h5), "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "figures" / "fig6_xenotime_callout.png").exists()
    assert json.loads((tmp_path / "numbers" / "xenotime.json").read_text())["n_pixels_2x"] == 16
