import json

from nwa4587_paper import cli, fingerprints, numbers as N
from nwa4587_paper.io import Run, DataPaths


def test_fingerprints_excludes_and_orders(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        nums = fingerprints.run(run, tmp_path, DataPaths())
    N.validate("fingerprints", nums)
    assert "Epoxy" not in nums["phases"] and "Weathering Assemblage" not in nums["phases"]
    assert "Augite" in nums["phases"] and "Xenotime" in nums["phases"]
    assert set(nums["element_order"]) == {"Ca", "Mg", "Y", "P"}
    # Variance of the per-phase means over the 7 kept phases: Y = P (0.9 in Xenotime only,
    # ~0.095) > Mg (0.8 and 0.4, ~0.085) > Ca (0.3, 0.2, 0.6, ~0.045); the Y/P tie keeps cube order.
    assert nums["element_order"] == ["Y", "P", "Mg", "Ca"]
    assert abs(nums["phases"]["Augite"]["mean"]["Ca"] - 0.6) < 1e-6
    assert (tmp_path / "figures" / "fig2_fingerprints.pdf").exists()
    assert (tmp_path / "figures" / "fig2_fingerprints.png").exists()


def test_area_pct_uses_all_mineral_pixels(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        nums = fingerprints.run(run, tmp_path, DataPaths())
        total = run.n_mineral_pixels
        assert abs(nums["phases"]["Xenotime"]["area_pct"] - 100 * 16 / total) < 1e-9


def test_cli_fingerprints_writes_figure_and_numbers(synth_h5, tmp_path):
    rc = cli.main(["fingerprints", "--h5", str(synth_h5), "--out", str(tmp_path),
                   "--data", str(tmp_path / "nodata")])
    assert rc == 0
    assert (tmp_path / "figures" / "fig2_fingerprints.png").exists()
    with open(tmp_path / "numbers" / "fingerprints.json") as fh:
        assert "phases" in json.load(fh)
