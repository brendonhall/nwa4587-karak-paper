import pytest

from nwa4587_paper import synth


@pytest.fixture(scope="session")
def synth_h5(tmp_path_factory):
    return synth.write_synthetic_h5(tmp_path_factory.mktemp("h5") / "paperflow.h5")


@pytest.fixture(scope="session")
def synth_eds(tmp_path_factory):
    return synth.write_synthetic_eds(tmp_path_factory.mktemp("eds"))


@pytest.fixture(scope="session")
def synth_tima(tmp_path_factory):
    return synth.write_synthetic_tima(tmp_path_factory.mktemp("tima") / "phases.png")


@pytest.fixture
def small_colormap(monkeypatch):
    """`paper figures` also runs the colormap stage; keep its default 3-minute LUT build out of the tests."""
    from nwa4587_paper import constants as C

    monkeypatch.setattr(C, "COLORMAP", C.ColormapSettings(lut_n=256, n_test=500, cmaps=("jet",)))
