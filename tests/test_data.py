from unittest import mock

import pytest

from nwa4587_paper import cli, data, synth
from nwa4587_paper.io import MissingInput


def _make_src(tmp_path, synth_eds, tima_names=("X-Phases.png",)):
    src = tmp_path / "src"
    src.mkdir()
    for p in synth_eds.glob("*.png"):
        (src / p.name).write_bytes(p.read_bytes())
    (src / "mask").mkdir()
    (src / "mask" / "Valid_mask.csv").write_text("index,shape-type,vertex-index,axis-0,axis-1\n")
    (src / "TIMA results").mkdir()
    for n in tima_names:
        (src / "TIMA results" / n).write_bytes(b"\x89PNG\r\n\x1a\n")
    return src


def test_copy_inputs(tmp_path, synth_eds):
    src = _make_src(tmp_path, synth_eds)
    info = data.copy_inputs(src, tmp_path / "data")
    assert info == {"n_png": 3, "mask": True, "tima": True}
    assert (tmp_path / "data" / "eds" / "SYN-Ca.png").exists()
    assert (tmp_path / "data" / "eds" / "mask" / "Valid_mask.csv").exists()
    assert (tmp_path / "data" / "tima" / "phases.png").exists()


def test_copy_inputs_skips_appledouble(tmp_path, synth_eds):
    src = _make_src(tmp_path, synth_eds, ("._X-Phases.png", "X-Phases.png"))
    (src / "._SYN-Ca.png").write_bytes(b"junk")
    info = data.copy_inputs(src, tmp_path / "data")
    assert info["n_png"] == 3 and info["tima"] is True
    assert not (tmp_path / "data" / "eds" / "._SYN-Ca.png").exists()
    assert (tmp_path / "data" / "tima" / "phases.png").read_bytes() == b"\x89PNG\r\n\x1a\n"


def test_copy_inputs_prefers_phases_png(tmp_path, synth_eds):
    src = _make_src(tmp_path, synth_eds, ("A-Other.png", "B-Phases.png"))
    (src / "TIMA results" / "B-Phases.png").write_bytes(b"phases")
    data.copy_inputs(src, tmp_path / "data")
    assert (tmp_path / "data" / "tima" / "phases.png").read_bytes() == b"phases"


def test_copy_inputs_ambiguous_tima(tmp_path, synth_eds):
    src = _make_src(tmp_path, synth_eds, ("A.png", "B.png"))
    with pytest.raises(MissingInput, match="A.png.*B.png"):
        data.copy_inputs(src, tmp_path / "data")


def test_copy_inputs_keeps_existing_unless_force(tmp_path, synth_eds):
    src = _make_src(tmp_path, synth_eds)
    dst = tmp_path / "data"
    data.copy_inputs(src, dst)
    (dst / "eds" / "SYN-Ca.png").write_bytes(b"mine")
    data.copy_inputs(src, dst)
    assert (dst / "eds" / "SYN-Ca.png").read_bytes() == b"mine"
    data.copy_inputs(src, dst, force=True)
    assert (dst / "eds" / "SYN-Ca.png").read_bytes() != b"mine"


def test_zenodo_stub_exits_2(capsys):
    assert data.zenodo_stub("10.5281/zenodo.0000000") == 2
    assert "not available yet" in capsys.readouterr().out


def test_run_karak_builds_the_command(tmp_path):
    with mock.patch("subprocess.run") as run:
        run.return_value.returncode = 0
        assert data.run_karak(tmp_path / "data", tmp_path / "results", workers=4) == 0
        argv = run.call_args.args[0]
        assert "--builtin" in argv and "paper" in argv and "--workers" in argv and "4" in argv
        assert str(tmp_path / "results" / "paperflow") in argv
        assert str(tmp_path / "data" / "eds") in argv and "--plain" in argv


def test_run_karak_passes_extra_and_code(tmp_path):
    with mock.patch("subprocess.run") as run:
        run.return_value.returncode = 3
        assert data.run_karak(tmp_path, tmp_path, extra=["--no-qc"]) == 3
        assert run.call_args.args[0][-1] == "--no-qc"


def test_cli_data_from(tmp_path, synth_eds):
    assert cli.main(["data", "--from", str(synth_eds), "--data", str(tmp_path / "d")]) == 0
    assert cli.main(["data", "--from", "zenodo:10.5281/zenodo.1", "--data", str(tmp_path / "d")]) == 2


def _synth_root(tmp_path):
    cli.main(["synth", str(tmp_path)])
    return ["--h5", str(tmp_path / "results" / "paperflow.h5"), "--out", str(tmp_path / "results"),
            "--data", str(tmp_path / "data"), "--tima-offset", f"{synth.SYNTH_TIMA_DX},{synth.SYNTH_TIMA_DY}"]


def test_cli_all_runs_karak_figures_numbers(tmp_path, small_colormap, capsys):
    common = _synth_root(tmp_path)
    with mock.patch.object(data, "run_karak", return_value=0) as rk:
        assert cli.main(["all", *common]) == 0
    rk.assert_called_once()
    assert (tmp_path / "results" / "numbers.json").exists()


def test_cli_all_stops_when_karak_fails(tmp_path, small_colormap):
    common = _synth_root(tmp_path)
    with mock.patch.object(data, "run_karak", return_value=1):
        assert cli.main(["all", *common]) == 1
    assert not (tmp_path / "results" / "numbers.json").exists()


def test_cli_karak_subcommand(tmp_path):
    with mock.patch.object(data, "run_karak", return_value=0) as rk:
        assert cli.main(["karak", "--data", str(tmp_path), "--out", str(tmp_path / "r"), "--workers", "2", "--", "--no-qc"]) == 0
    assert rk.call_args.kwargs["workers"] == 2 and list(rk.call_args.kwargs["extra"]) == ["--no-qc"]
