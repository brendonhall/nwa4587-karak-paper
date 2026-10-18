import json
import subprocess
import sys

from nwa4587_paper import cli, synth


def test_tables_subcommand_writes_numbers(synth_h5, tmp_path):
    rc = cli.main(["tables", "--h5", str(synth_h5), "--out", str(tmp_path)])
    assert rc == 0
    with open(tmp_path / "numbers" / "tables.json") as fh:
        assert "phases" in json.load(fh)


def test_check_fails_on_mismatch(synth_h5, tmp_path, capsys):
    cli.main(["tables", "--h5", str(synth_h5), "--out", str(tmp_path)])
    with open(tmp_path / "numbers" / "tables.json") as fh:
        exp = {"tables": json.load(fh)}
    exp["tables"]["n_mineral_pixels"] += 1
    p = tmp_path / "expected.json"
    p.write_text(json.dumps(exp))
    args = ["tables", "--h5", str(synth_h5), "--out", str(tmp_path), "--check", "--expected", str(p)]
    assert cli.main(args) == 1
    exp["tables"]["n_mineral_pixels"] -= 1
    p.write_text(json.dumps(exp))
    assert cli.main(args) == 0


def test_missing_h5_exits_2(tmp_path, capsys):
    assert cli.main(["tables", "--h5", str(tmp_path / "none.h5"), "--out", str(tmp_path)]) == 2
    assert "none.h5" in capsys.readouterr().err


def test_missing_phase_exits_3(synth_h5, tmp_path, monkeypatch, capsys):
    from nwa4587_paper.io import MissingPhase

    def boom(run, out, data):
        raise MissingPhase("no phase with key 'X'")

    monkeypatch.setitem(cli.STAGES, "tables", boom)
    assert cli.main(["tables", "--h5", str(synth_h5), "--out", str(tmp_path)]) == 3
    assert "no phase" in capsys.readouterr().err


def test_stage_kwargs_pass_through(synth_h5, tmp_path, monkeypatch):
    seen = {}

    def fake(run, out, data, **kw):
        seen.update(kw)
        return {}

    monkeypatch.setitem(cli.STAGES, "tables", fake)
    args = cli.build_parser().parse_args(["tables", "--h5", str(synth_h5), "--out", str(tmp_path)])
    assert cli.run_stage("tables", args, offset=3) == 0
    assert seen == {"offset": 3}


def test_synth_subcommand(tmp_path):
    assert cli.main(["synth", str(tmp_path)]) == 0
    assert (tmp_path / "results" / "paperflow.h5").exists()
    assert (tmp_path / "data" / "tima" / "phases.png").exists()


def test_figures_runs_present_stages(synth_h5, synth_eds, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "eds").symlink_to(synth_eds)
    assert cli.main(["figures", "--h5", str(synth_h5), "--out", str(tmp_path), "--data", str(data)]) == 0
    assert (tmp_path / "numbers" / "tables.json").exists()


def test_console_script_runs():
    out = subprocess.run([sys.executable, "-m", "nwa4587_paper.cli", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "tables" in out.stdout


def test_figures_then_numbers_on_synthetic(tmp_path):
    assert cli.main(["synth", str(tmp_path)]) == 0
    common = ["--h5", str(tmp_path / "results" / "paperflow.h5"), "--out", str(tmp_path / "results"),
              "--data", str(tmp_path / "data")]
    offset = ["--tima-offset", f"{synth.SYNTH_TIMA_DX},{synth.SYNTH_TIMA_DY}"]
    assert cli.main(["figures", *common, *offset]) == 0
    assert cli.main(["numbers", *common]) == 0
    merged = json.loads((tmp_path / "results" / "numbers.json").read_text())
    assert {"tables", "fingerprints", "tima"} <= set(merged)
    tex = (tmp_path / "results" / "numbers.tex").read_text()
    assert r"\nwaTablesNMineralPixels" in tex


def test_missing_tima_raster_skips_with_exit_0(synth_h5, synth_eds, tmp_path, capsys):
    data = tmp_path / "nodata"
    data.mkdir()
    (data / "eds").symlink_to(synth_eds)
    common = ["--h5", str(synth_h5), "--out", str(tmp_path), "--data", str(data)]
    assert cli.main(["tima", *common]) == 0
    assert "tima: skipped (no TIMA raster under data/tima)" in capsys.readouterr().out
    assert not (tmp_path / "numbers" / "tima.json").exists()
    assert cli.main(["figures", *common]) == 0
    assert "tima: skipped" in capsys.readouterr().out


def test_tima_offset_option_reaches_stage(synth_h5, synth_tima, tmp_path, monkeypatch):
    seen = {}

    def fake(run, out, data, **kw):
        seen.update(kw)
        return {}

    monkeypatch.setitem(cli.STAGES, "tima", fake)
    data = tmp_path / "data"
    (data / "tima").mkdir(parents=True)
    (data / "tima" / "phases.png").write_bytes(synth_tima.read_bytes())
    args = ["tima", "--h5", str(synth_h5), "--out", str(tmp_path), "--data", str(data)]
    assert cli.main(args) == 0 and seen == {}
    assert cli.main([*args, "--tima-offset", "19,51"]) == 0
    assert seen == {"dx": 19, "dy": 51}


def test_synth_prints_offset_hint(tmp_path, capsys):
    cli.main(["synth", str(tmp_path)])
    assert "use: --tima-offset 19,51" in capsys.readouterr().out


def test_workflow_subcommand_writes_figure_and_numbers(synth_h5, synth_eds, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "eds").symlink_to(synth_eds)
    assert cli.main(["workflow", "--h5", str(synth_h5), "--out", str(tmp_path), "--data", str(data)]) == 0
    assert (tmp_path / "figures" / "fig1_workflow.png").exists()
    assert (tmp_path / "numbers" / "workflow.json").exists()
