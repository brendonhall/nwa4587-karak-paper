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
