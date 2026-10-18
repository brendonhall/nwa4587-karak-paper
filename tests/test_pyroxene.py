import pytest

from nwa4587_paper import cli, numbers as N, pyroxene
from nwa4587_paper.io import DataPaths, MissingInput, Run


def test_pyroxene_numbers_and_figure(synth_h5, synth_eds, tmp_path):
    with Run.open(synth_h5) as run:
        nums = pyroxene.run(run, tmp_path, DataPaths(eds_dir=synth_eds))
        rec = run.split_record("pyx")
        aug, pig = run.phase("Augite"), run.phase("Pigeonite")
        assert nums["n_pixels_2x"] == run.counts[aug] + run.counts[pig]
        assert nums["n_pixels_1x"]["Augite"] == rec["n_pixels_hires"][str(aug)]
        assert nums["gmm_mean"] == {"Augite": 0.53, "Pigeonite": 0.31}
        assert nums["ratio_mean"]["Augite"] == pytest.approx(0.6, abs=0.03)
        assert nums["ratio_mean"]["Pigeonite"] == pytest.approx(0.2, abs=0.03)
        n1 = nums["n_pixels_1x"]
        assert nums["pct_1x"]["Augite"] == pytest.approx(100.0 * n1["Augite"] / sum(n1.values()))
        assert nums["pct_1x"]["Augite"] == pytest.approx(100 * 960 / 1916)
    assert sum(nums["pct_1x"].values()) == pytest.approx(100.0)
    N.validate("pyroxene", nums)
    assert (tmp_path / "figures" / "fig7_pyroxene_refinement.pdf").exists()
    assert (tmp_path / "figures" / "fig7_pyroxene_refinement.png").exists()


def test_load_1x_arrays(synth_h5, synth_eds):
    with Run.open(synth_h5) as run:
        d = pyroxene.load_1x(run, DataPaths(eds_dir=synth_eds))
    assert d["labels"].shape == d["bse"].shape == d["ca"].shape == d["mg"].shape
    assert d["aug_mask"].sum() > 0 and d["pig_mask"].sum() > 0
    assert len(d["ca_aug"]) == len(d["mg_aug"]) <= d["aug_mask"].sum()


def test_pyroxene_needs_eds(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        with pytest.raises(MissingInput):
            pyroxene.run(run, tmp_path, DataPaths(eds_dir=None))


def test_cli_pyroxene(synth_h5, synth_eds, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "eds").symlink_to(synth_eds)
    out = tmp_path / "out"
    rc = cli.main(["pyroxene", "--h5", str(synth_h5), "--out", str(out), "--data", str(data)])
    assert rc == 0
    assert (out / "figures" / "fig7_pyroxene_refinement.png").exists()
    assert (out / "numbers" / "pyroxene.json").exists()


def test_cli_pyroxene_without_eds_exits_2(synth_h5, tmp_path, capsys):
    args = ["pyroxene", "--h5", str(synth_h5), "--out", str(tmp_path), "--data", str(tmp_path / "none")]
    assert cli.main(args) == 2
    assert "elemental PNG directory" in capsys.readouterr().err
