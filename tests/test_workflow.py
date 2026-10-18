import matplotlib.pyplot as plt

from nwa4587_paper import numbers as N, workflow
from nwa4587_paper.io import DataPaths, Run


def test_workflow_figure(synth_h5, synth_eds, tmp_path):
    with Run.open(synth_h5) as run:
        nums = workflow.run(run, tmp_path, DataPaths(eds_dir=synth_eds))
        n_final = len(run.keys)
    N.validate("workflow", nums)
    assert nums["n_clusters_before_pyx"] == 7
    assert nums["n_phases_final"] == n_final
    for ext in ("pdf", "png"):
        assert (tmp_path / "figures" / f"fig1_workflow.{ext}").stat().st_size > 0
    assert plt.get_fignums() == []


def test_legend_lists_every_final_key(synth_h5):
    with Run.open(synth_h5) as run:
        entries = workflow._legend_entries(run)
        assert len(entries) == len(run.keys)
        assert len({name for name, _ in entries}) == len(run.keys)


def test_panel_c_uses_panel_b_vmax(synth_h5, synth_eds):
    import numpy as np

    with Run.open(synth_h5) as run:
        imgs, _ = workflow._panels(run, synth_eds)
        s = workflow.C.FIG1_H5_STRIDE
        raw = run.raw_channel("Ca")[::s, ::s]
        vmax = np.percentile(raw[raw > 0], workflow.SCALAR_PERCENTILE)
        ca = run.channel("Ca")[::s, ::s]
        m = run.mask[::s, ::s]
        assert np.allclose(imgs[2][m][:, 0], np.clip(ca[m] / vmax, 0, 1))
        assert np.allclose(imgs[2][~m], workflow.MASKED_GREY)
        for img, lab in ((imgs[3], run.labels_before("pyx")[::s, ::s]), (imgs[4], run.labels[::s, ::s])):
            assert np.allclose(img[lab == -1], workflow.MASKED_GREY)
