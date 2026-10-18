"""The lamellae stage (Fig 8). No warning filters here: the stage handles its own."""
import csv
import dataclasses
import math
import shutil

import h5py
import numpy as np
import pytest

from nwa4587_paper import constants as C
from nwa4587_paper import lamellae, numbers as N
from nwa4587_paper.io import DataPaths, Run


def _check_common(nums, out):
    N.validate("lamellae", nums)
    assert nums["pixel_size_um"] == 3.0 and nums["nyquist_limit_um"] == 6.0
    assert (out / "figures" / "fig8_lamellae.pdf").exists()
    assert (out / "figures" / "fig8_lamellae.png").exists()
    assert (out / "tables" / "lamellae_grain_table.csv").exists()


def test_stage_default_settings_analyzes_nothing(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        nums = lamellae.run(run, tmp_path, DataPaths())
    _check_common(nums, tmp_path)
    assert nums["n_grains_analyzed"] == 0
    assert math.isnan(nums["spacing"]["median_um"])
    N.write_stage(tmp_path, "lamellae", nums)
    assert "NaN" not in (tmp_path / "numbers" / "lamellae.json").read_text()


@pytest.fixture
def diagonal_h5(synth_h5, tmp_path):
    """The synthetic file with 45 degree stripes in the pyroxene region. The ported spacing code
    measures orientation from the row axis and samples from the column axis, so horizontal
    stripes give flat profiles; at 45 degrees both conventions agree and the grain is analyzed."""
    path = tmp_path / "diagonal.h5"
    shutil.copy(synth_h5, path)
    with h5py.File(path, "r+") as f:
        ds = f["clusters/hires/labels"]
        hires = ds[:]
        rows, cols = np.indices(hires.shape)
        region = hires >= 13
        stripe = ((rows + cols) // 4) % 2 == 0
        hires[region & stripe] = 14
        hires[region & ~stripe] = 13
        ds[:] = hires
    return path


def test_stage_lowered_threshold_analyzes_a_grain(diagonal_h5, tmp_path):
    settings = dataclasses.replace(C.LAMELLAE, min_analysis_px=500)
    with Run.open(diagonal_h5) as run:
        nums = lamellae.run(run, tmp_path, DataPaths(), settings=settings)
    _check_common(nums, tmp_path)
    assert nums["n_grains_analyzed"] >= 1
    with open(tmp_path / "tables" / "lamellae_grain_table.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == nums["n_grains_analyzed"]
    assert "bbox" not in rows[0] and "spacing_um" in rows[0]
