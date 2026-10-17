import json
import math

import pytest

from nwa4587_paper import numbers as N


@pytest.fixture
def reg(monkeypatch):
    """Register specs for one test only; monkeypatch undoes each setitem."""

    def add(entries):
        for key, spec in entries.items():
            monkeypatch.setitem(N.REGISTRY, key, spec)

    return add


def test_flatten_and_spec_lookup(reg):
    reg({"demo.total": N.Spec("int"),
         "demo.phases.*.pct": N.Spec("float", tol=0.001, fmt="{:.3f}")})
    flat = N.flatten({"demo": {"total": 3, "phases": {"Augite": {"pct": 1.5}}}})
    assert flat == {"demo.total": 3, "demo.phases.Augite.pct": 1.5}
    assert N.spec_for("demo.phases.Augite.pct").tol == 0.001
    with pytest.raises(KeyError):
        N.spec_for("demo.unknown")


def test_validate_rejects_unlisted_key(reg):
    reg({"demo.total": N.Spec("int")})
    with pytest.raises(KeyError) as e:
        N.validate("demo", {"total": 1, "extra": 2})
    assert "demo.extra" in str(e.value)
    assert "demo.total" not in str(e.value)


def test_write_merge_render(reg, tmp_path):
    reg({"demo.total": N.Spec("int"),
         "demo.phases.*.pct": N.Spec("float", tol=0.001, fmt="{:.3f}")})
    N.write_stage(tmp_path, "demo", {"total": 3, "phases": {"Fe-Ti-Cr Spinel": {"pct": 1.5}}})
    merged = N.merge(tmp_path)
    assert merged["demo"]["total"] == 3
    assert json.loads((tmp_path / "numbers.json").read_text()) == merged
    tex = N.render_tex(merged)
    assert r"\newcommand{\nwaDemoTotal}{3}" in tex
    assert r"\newcommand{\nwaDemoPhasesFeTiCrSpinelPct}{1.500}" in tex


def test_macro_name_digits():
    assert N.macro_name("pyroxene.n_pixels_2x") == r"\nwaPyroxeneNPixelsTwoX"


def test_check_tolerances(reg):
    reg({"demo.f": N.Spec("float", tol=0.01),
         "demo.r": N.Spec("float", tol=0.01, rel=True),
         "demo.n": N.Spec("int")})
    actual = {"demo": {"f": 1.005, "r": 1010.0, "n": 5}}
    expected = {"demo": {"f": 1.0, "r": 1000.0, "n": 5}}
    assert N.check(actual, expected) == []
    expected["demo"]["f"] = 1.02
    misses = N.check(actual, expected)
    assert [m.key for m in misses] == ["demo.f"]


def test_check_int_exact_and_relative(reg):
    reg({"demo.exact": N.Spec("int"),
         "demo.big": N.Spec("int", tol=0.001, rel=True)})
    assert N.check({"demo": {"exact": 5}}, {"demo": {"exact": 6}})[0].key == "demo.exact"
    assert N.check({"demo": {"big": 100050}}, {"demo": {"big": 100000}}) == []
    misses = N.check({"demo": {"big": 100200}}, {"demo": {"big": 100000}})
    assert [m.key for m in misses] == ["demo.big"]


def test_check_missing_key(reg):
    reg({"demo.n": N.Spec("int"), "demo.m": N.Spec("int")})
    misses = N.check({"demo": {"n": 1}}, {"demo": {"m": 1}})
    by_key = {m.key: m for m in misses}
    assert sorted(by_key) == ["demo.m", "demo.n"]
    # expected has m, fresh run lacks it
    assert by_key["demo.m"].expected == 1 and by_key["demo.m"].actual is None
    # fresh run has n, expected file lacks it
    assert by_key["demo.n"].actual == 1 and by_key["demo.n"].expected is None
    assert "demo.m" in N.format_misses(misses)


def test_check_nan_equals_nan(reg):
    reg({"demo.f": N.Spec("float", tol=0.01)})
    assert N.check({"demo": {"f": math.nan}}, {"demo": {"f": math.nan}}) == []
