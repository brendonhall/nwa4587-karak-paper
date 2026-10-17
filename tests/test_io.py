import numpy as np
import pytest

from nwa4587_paper.io import Run, phase_key, MissingPhase, load_channel_1x, find_png


def test_phase_key():
    assert phase_key("Augite (high-Ca pyroxene)") == "Augite"
    assert phase_key("Zn-bearing Phase") == "Zn-bearing Phase"
    assert phase_key("Apatite (Ca₅(PO₄)₃(OH,Cl,F))") == "Apatite"


def test_phase_lookup(synth_h5):
    with Run.open(synth_h5) as run:
        assert run.phase("Augite") == 14
        assert run.phase("Pigeonite") == 13
        assert run.has("Epoxy") and not run.has("Calcite")


def test_phase_unknown_key(synth_h5):
    with Run.open(synth_h5) as run:
        with pytest.raises(MissingPhase) as e:
            run.phase("Calcite")
        assert "Calcite" in str(e.value) and "Augite" in str(e.value)


def test_labels_and_counts(synth_h5):
    with Run.open(synth_h5) as run:
        assert run.labels.shape == (64, 48)
        assert run.labels[0, 0] == -1
        assert sum(run.counts.values()) == run.n_mineral_pixels
        assert run.counts[run.phase("Xenotime")] == 16


def test_labels_before_undoes_splits(synth_h5):
    with Run.open(synth_h5) as run:
        assert run.split_node_ids == ["oliv", "weath", "pyx", "phos"]
        before_pyx = run.labels_before("pyx")
        assert 13 not in before_pyx and 14 not in before_pyx and 2 in before_pyx
        assert 7 in before_pyx and 15 not in before_pyx and 16 not in before_pyx   # the pyx split and every later one are undone
        assert 11 in before_pyx and 12 in before_pyx
        before_oliv = run.labels_before("oliv")
        assert set(np.unique(before_oliv)) == {-1, 2, 3, 4, 7, 10}


def test_labels_1x_overlays_hires(synth_h5):
    with Run.open(synth_h5) as run:
        l1 = run.labels_1x
        assert l1.shape == run.hires_labels.shape == (128, 96)
        assert l1[12, 12] == run.phase("Plagioclase")           # repeated 2x label
        assert set(np.unique(l1[60:116, 12:60])) <= {13, 14, 2, -1, 11, 12}
        assert l1[60, 12] == -1 or l1[60, 12] in (13, 14)       # undefined corner stays what the 2x map says
        assert (l1 == run.phase("Epoxy")).sum() == 0            # epoxy becomes -1 at 1x


def test_labels_1x_clips(synth_h5, tmp_path):
    import h5py, shutil
    p = shutil.copy(synth_h5, tmp_path / "odd.h5")
    with h5py.File(p, "a") as f:
        data = f["clusters/hires/labels"][:-1, :-1]
        attrs = dict(f["clusters/hires/labels"].attrs)
        del f["clusters/hires/labels"]
        d = f["clusters/hires"].create_dataset("labels", data=data)
        d.attrs.update(attrs)
    with Run.open(p) as run:
        assert run.labels_1x.shape == (127, 95)


def test_geometry_and_flow(synth_h5):
    with Run.open(synth_h5) as run:
        g = run.geometry
        assert (g.factor, g.ratio, g.header_trim_px) == (2, 2, 100)
        assert g.shape_2x == (64, 48) and g.shape_1x == (128, 96)
        assert g.pixel_size_um_2x == 6.0
        assert run.flow["name"] == "paper"
        assert run.flow_param("hdb", "tile_size") == 1024
        assert run.split_record("pyx")["component_means"]["14"]["Ca/(Ca+Mg)"] == 0.53


def test_load_channel_1x(synth_eds):
    ca = load_channel_1x(synth_eds, "Ca", header_trim_px=100)
    assert ca.shape == (128, 96) and ca.dtype == np.float32
    assert abs(float(ca[12, 12]) - 0.3) < 0.02                  # plagioclase block
    assert find_png(synth_eds, "SEM").name == "SYN-SEM.png"
