import json
import subprocess
import sys

from nwa4587_paper import cli


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


def test_figures_runs_present_stages(synth_h5, tmp_path):
    assert cli.main(["figures", "--h5", str(synth_h5), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "numbers" / "tables.json").exists()


def test_console_script_runs():
    out = subprocess.run([sys.executable, "-m", "nwa4587_paper.cli", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "tables" in out.stdout


def test_figures_then_numbers_on_synthetic(tmp_path):
    assert cli.main(["synth", str(tmp_path)]) == 0
    common = ["--h5", str(tmp_path / "results" / "paperflow.h5"), "--out", str(tmp_path / "results"),
              "--data", str(tmp_path / "data")]
    assert cli.main(["figures", *common]) == 0
    assert cli.main(["numbers", *common]) == 0
    merged = json.loads((tmp_path / "results" / "numbers.json").read_text())
    assert {"tables", "fingerprints"} <= set(merged)
    tex = (tmp_path / "results" / "numbers.tex").read_text()
    assert r"\nwaTablesNMineralPixels" in tex
