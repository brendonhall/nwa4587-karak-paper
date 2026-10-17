import json

import numpy as np
import pytest

from nwa4587_paper import constants as C
from nwa4587_paper import numbers as N
from nwa4587_paper import synth, tima
from nwa4587_paper.io import DataPaths, MissingInput, Run

OFFSET = {"dx": synth.SYNTH_TIMA_DX, "dy": synth.SYNTH_TIMA_DY}


def test_tima_stage_on_synthetic(synth_h5, synth_tima, tmp_path):
    with Run.open(synth_h5) as run:
        nums = tima.run(run, tmp_path, DataPaths(tima_png=synth_tima), **OFFSET)
    N.validate("tima", nums)
    for f in ("fig3_phase_comparison", "fig4_agreement_map", "fig5_pyroxene_tima"):
        for ext in ("pdf", "png"):
            assert (tmp_path / "figures" / f"{f}.{ext}").stat().st_size > 0
    for f in ("table3_correspondence.tex", "table4_concordance.tex", "table4_concordance.csv",
              "tima_confusion_split.csv", "tima_confusion_combined.csv", "tima_rare_phases.json"):
        assert (tmp_path / "tables" / f).stat().st_size > 0
    assert nums["split"]["overall_accuracy"] == pytest.approx(1.0)
    assert nums["combined"]["overall_accuracy"] == pytest.approx(1.0)
    assert nums["split"]["per_phase"]["Augite"]["f1"] == pytest.approx(1.0)
    assert set(nums["split"]["abundance"]) and set(nums["combined"]["per_phase"])
    assert nums["alignment"]["shape_hw"] == [128, 96]
    assert nums["alignment"]["dx"] == synth.SYNTH_TIMA_DX
    assert nums["fig5_window"]["size"] <= 96
    json.dumps(nums)


def test_tima_tables_format(synth_h5, synth_tima, tmp_path):
    with Run.open(synth_h5) as run:
        tima.run(run, tmp_path, DataPaths(tima_png=synth_tima), **OFFSET)
    t3 = (tmp_path / "tables" / "table3_correspondence.tex").read_text()
    assert t3.startswith(r"\begin{tabular}") and "Plagioclase" in t3
    csv_lines = (tmp_path / "tables" / "table4_concordance.csv").read_text().splitlines()
    counts = [int(l.split(",")[-1]) for l in csv_lines[1:]]
    assert counts == sorted(counts, reverse=True)
    conf = (tmp_path / "tables" / "tima_confusion_split.csv").read_text().splitlines()
    assert "Plagioclase" in conf[0] or "Plagioclase" in conf[1]


def test_tima_missing_raster(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        with pytest.raises(MissingInput):
            tima.run(run, tmp_path, DataPaths(tima_png=None))


def test_tima_window_search_is_capped_to_map(synth_h5):
    with Run.open(synth_h5) as run:
        k = run.labels_1x
        t = np.where(k >= 0, 1, -1).astype(np.int8)
        x0, y0, n = tima.find_densest_window(k, t, run, size=40, stride=10)
        assert 0 <= x0 <= k.shape[1] - 40 and 0 <= y0 <= k.shape[0] - 40


def test_fig5_window_clamps_frozen_position(synth_h5, monkeypatch):
    monkeypatch.setattr(C, "FIG5_WINDOW_XY", (5000, 7000))
    with Run.open(synth_h5) as run:
        k = run.labels_1x
        x0, y0, size, _ = tima.fig5_window(k, k.copy(), run)
    h, w = k.shape
    assert size == min(C.FIG5_WINDOW_SIZE, h, w)
    assert (x0, y0) == (w - size, h - size)
