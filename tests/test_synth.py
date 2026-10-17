import json
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
from PIL import Image

from nwa4587_paper import constants as C
from nwa4587_paper import synth

LAYOUT = Path(__file__).resolve().parent.parent / "expected" / "h5_layout.json"


def test_layout_matches_reference_listing(synth_h5):
    layout = json.loads(LAYOUT.read_text())
    with h5py.File(synth_h5, "r") as f:
        for name, spec in layout["datasets"].items():
            if any(name.startswith(g) for g in layout["optional_groups"]):
                continue
            assert name in f, name
            assert f[name].ndim == spec["ndim"], name
        for name, keys in layout["attrs"].items():
            if name != "/" and any(name.startswith(g) for g in layout["optional_groups"]):
                continue
            obj = f if name == "/" else f[name]
            assert set(keys) <= set(obj.attrs), (name, set(keys) - set(obj.attrs))


def test_synthetic_labels_and_history(synth_h5):
    with h5py.File(synth_h5, "r") as f:
        labels = f["clusters/cleaned_labels"][:]
        names = json.loads(f["clusters"].attrs["mineral_names"])
        stats = json.loads(f["clusters"].attrs["cluster_stats"])
        history = json.loads(f["clusters/subclustering"].attrs["history"])
        hires = f["clusters/hires/labels"]
        assert set(np.unique(labels)) - {-1} == {3, 4, 10, 11, 12, 13, 14, 15, 16}
        assert 2 not in labels and 7 not in labels
        assert set(map(int, names)) == {3, 4, 10, 11, 12, 13, 14, 15, 16}
        assert set(map(int, stats["clusters"])) == set(map(int, names))
        assert names["14"].startswith("Augite") and names["13"].startswith("Pigeonite")
        assert [h["stage"] for h in history] == ["split_threshold", "split_gmm", "split_hires", "split_gmm"]
        assert all("node" not in h and "node_id" not in h for h in history)
        assert hires.shape == (128, 96) and hires.attrs["ratio"] == 2
        for key in ("names", "downsample_factor", "header_trim_px", "left_trim_px"):
            assert key in hires.attrs
        assert set(np.unique(hires[:])) == {-1, 13, 14}


def test_final_labels_have_no_parent_labels():
    img, mask, hires, _ = synth.final_labels()
    assert 2 not in img and 7 not in img


def test_synthetic_pngs_and_tima(synth_eds, synth_tima):
    with Image.open(synth_eds / "SYN-Ca.png") as im:
        ca = np.asarray(im)
    assert ca.shape == (228, 96, 3)
    assert (synth_eds / "mask" / "Valid_mask.csv").exists()
    with Image.open(synth_tima) as tima:
        assert tima.size == (200, 1200)


def test_png16_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    rgb8 = rng.integers(0, 256, (7, 5, 3)).astype(np.uint16)
    rgb16 = rgb8 * 257
    path = synth.write_png16_rgb(tmp_path / "x.png", rgb16)
    back = iio.imread(path)
    assert back.shape == (7, 5, 3)
    expected = np.rint(rgb16 / 257)
    assert np.abs(back.astype(int) - expected.astype(int)).max() <= 1


def test_tima_offsets():
    assert synth.SYNTH_TIMA_DX == C.TIMA_DX - 1900
    assert synth.SYNTH_TIMA_DY == C.TIMA_DY - 1000
