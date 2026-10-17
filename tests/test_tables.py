import csv

from nwa4587_paper import numbers as N
from nwa4587_paper import tables
from nwa4587_paper.io import DataPaths, Run


def test_tables_writes_files_and_numbers(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        nums = tables.run(run, tmp_path, DataPaths())
        noise = run.counts.get(-1, 0)
    N.validate("tables", nums)
    assert (tmp_path / "tables" / "table1_params.tex").exists()
    tex = (tmp_path / "tables" / "table2_phases.tex").read_text()
    assert "Augite" in tex and r"\toprule" in tex
    with open(tmp_path / "tables" / "table2_phases.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["phase"] == "Plagioclase"  # sorted by area, descending
    assert all(int(r["label"]) >= 0 for r in rows)  # noise label -1 is not a phase
    assert sum(int(r["pixels"]) for r in rows) == nums["n_mineral_pixels"] - noise
    assert abs(sum(float(r["pct"]) for r in rows) - 100.0 * (1 - noise / nums["n_mineral_pixels"])) < 0.01
    assert nums["phases"]["Xenotime"]["pixels"] == 16
    assert nums["params"]["Tile size (px)"] == "1024"
    assert nums["n_phases_with_pixels"] == 9


def test_tables_is_deterministic(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        a = tables.run(run, tmp_path / "a", DataPaths())
        b = tables.run(run, tmp_path / "b", DataPaths())
    assert a == b
