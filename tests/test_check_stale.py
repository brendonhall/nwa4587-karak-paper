"""`--check` sees stage files that are missing or stale."""
import json

from nwa4587_paper import cli


def test_numbers_check_lists_expected_stage_the_run_lacks(synth_h5, tmp_path, capsys):
    cli.main(["tables", "--h5", str(synth_h5), "--out", str(tmp_path)])
    tables = json.loads((tmp_path / "numbers" / "tables.json").read_text())
    exp = tmp_path / "expected.json"
    exp.write_text(json.dumps({"tables": tables, "xenotime": {"n_pixels_2x": 16}}))
    data = tmp_path / "nodata"
    data.mkdir()
    args = ["numbers", "--out", str(tmp_path), "--data", str(data), "--check", "--expected", str(exp)]
    assert cli.main(args) == 1
    assert "xenotime: no stage file" in capsys.readouterr().out


def test_numbers_check_skips_tima_without_raster(synth_h5, tmp_path, capsys):
    cli.main(["tables", "--h5", str(synth_h5), "--out", str(tmp_path)])
    tables = json.loads((tmp_path / "numbers" / "tables.json").read_text())
    exp = tmp_path / "expected.json"
    exp.write_text(json.dumps({"tables": tables, "tima": {"alignment": {"dx": 19}}}))
    data = tmp_path / "nodata"
    data.mkdir()
    args = ["numbers", "--out", str(tmp_path), "--data", str(data), "--check", "--expected", str(exp)]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "tima: not compared (no TIMA raster)" in out and "all numbers match" in out


def test_figures_removes_stale_file_of_failed_stage(synth_h5, synth_eds, tmp_path, monkeypatch, capsys,
                                                     small_colormap):
    from nwa4587_paper.io import MissingInput

    stale = tmp_path / "numbers" / "xenotime.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"n_pixels_2x": 16}))

    def boom(run, out, data):
        raise MissingInput("simulated crash")

    monkeypatch.setitem(cli.STAGES, "xenotime", boom)
    data = tmp_path / "data"
    data.mkdir()
    (data / "eds").symlink_to(synth_eds)
    assert cli.main(["figures", "--h5", str(synth_h5), "--out", str(tmp_path), "--data", str(data)]) == 2
    assert "xenotime: simulated crash" in capsys.readouterr().err
    assert not stale.exists()
