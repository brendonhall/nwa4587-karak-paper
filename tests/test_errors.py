"""Wrong inputs reach the user as one line with the right exit code, never a traceback."""
import json
import shutil

import h5py
import pytest
import yaml

from nwa4587_paper import cli, synth


def _variant(synth_h5, tmp_path, edit):
    path = tmp_path / "variant.h5"
    shutil.copy(synth_h5, path)
    with h5py.File(path, "a") as f:
        edit(f)
    return path


def _no_pyx(f):
    cfg = yaml.safe_load(f["clusters"].attrs["cluster_config"])
    for node in cfg["refinement_nodes"]:
        if node["id"] == "pyx":
            node["id"] = "pyx_renamed"
    f["clusters"].attrs["cluster_config"] = yaml.safe_dump(cfg)


@pytest.mark.parametrize("stage", ["pyroxene", "workflow"])
def test_missing_split_node_exits_3(synth_h5, synth_eds, tmp_path, capsys, stage):
    h5 = _variant(synth_h5, tmp_path, _no_pyx)
    data = tmp_path / "data"
    data.mkdir()
    (data / "eds").symlink_to(synth_eds)
    assert cli.main([stage, "--h5", str(h5), "--out", str(tmp_path), "--data", str(data)]) == 3
    err = capsys.readouterr().err
    assert "'pyx'" in err and "pyx_renamed" in err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1


def test_missing_hires_group_exits_2(synth_h5, tmp_path, capsys):
    h5 = _variant(synth_h5, tmp_path, lambda f: f["clusters"].__delitem__("hires"))
    assert cli.main(["xenotime", "--h5", str(h5), "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "clusters/hires/labels" in err and "variant.h5" in err


def test_missing_names_attr_exits_2(synth_h5, tmp_path, capsys):
    h5 = _variant(synth_h5, tmp_path, lambda f: f["clusters"].attrs.__delitem__("mineral_names"))
    assert cli.main(["tables", "--h5", str(h5), "--out", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "clusters" in err and "mineral_names" in err and "variant.h5" in err


@pytest.mark.parametrize("bad", ["19;51", "19", "a,b", "1,2,3"])
def test_malformed_tima_offset_is_a_usage_error(synth_h5, tmp_path, capsys, bad):
    with pytest.raises(SystemExit) as e:
        cli.main(["tima", "--h5", str(synth_h5), "--out", str(tmp_path), "--tima-offset", bad])
    assert e.value.code == 2
    assert "DX,DY" in capsys.readouterr().err


def test_tima_no_overlap_exits_2(synth_h5, synth_tima, tmp_path, capsys):
    data = tmp_path / "data"
    (data / "tima").mkdir(parents=True)
    (data / "tima" / "phases.png").write_bytes(synth_tima.read_bytes())
    args = ["tima", "--h5", str(synth_h5), "--out", str(tmp_path), "--data", str(data),
            "--tima-offset", "100000,100000"]
    assert cli.main(args) == 2
    err = capsys.readouterr().err
    assert "Translation produces no overlap" in err and len(err.strip().splitlines()) == 1


def test_missing_expected_file_exits_2(synth_h5, tmp_path, capsys):
    missing = tmp_path / "nope.json"
    args = ["tables", "--h5", str(synth_h5), "--out", str(tmp_path), "--check", "--expected", str(missing)]
    assert cli.main(args) == 2
    assert "nope.json" in capsys.readouterr().err
    assert cli.main(["numbers", "--out", str(tmp_path), "--check", "--expected", str(missing)]) == 2
    assert "nope.json" in capsys.readouterr().err


def test_unexpected_error_is_one_line_exit_5(synth_h5, tmp_path, monkeypatch, capsys):
    def boom(run, out, data):
        raise RuntimeError("boom")

    monkeypatch.setitem(cli.STAGES, "tables", boom)
    monkeypatch.delenv("PAPER_DEBUG", raising=False)
    assert cli.main(["tables", "--h5", str(synth_h5), "--out", str(tmp_path)]) == 5
    err = capsys.readouterr().err
    assert err.strip() == "paper: RuntimeError: boom"
    monkeypatch.setenv("PAPER_DEBUG", "1")
    assert cli.main(["tables", "--h5", str(synth_h5), "--out", str(tmp_path)]) == 5
    err = capsys.readouterr().err
    assert "Traceback" in err and err.strip().endswith("paper: RuntimeError: boom")


def test_figures_continues_after_a_failed_stage(synth_h5, synth_eds, tmp_path, monkeypatch, capsys,
                                                 small_colormap):
    def boom(run, out, data):
        raise RuntimeError("boom")

    monkeypatch.setitem(cli.STAGES, "tables", boom)
    data = tmp_path / "data"
    data.mkdir()
    (data / "eds").symlink_to(synth_eds)
    assert cli.main(["figures", "--h5", str(synth_h5), "--out", str(tmp_path), "--data", str(data)]) == 5
    assert "tables: RuntimeError: boom" in capsys.readouterr().err
    assert (tmp_path / "numbers" / "xenotime.json").exists()
