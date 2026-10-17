import matplotlib


def test_package_imports_and_sets_agg():
    import nwa4587_paper

    assert nwa4587_paper.__version__ == "0.1.0"
    assert matplotlib.get_backend().lower() == "agg"
