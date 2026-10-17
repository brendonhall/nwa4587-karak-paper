from nwa4587_paper import constants as C

NAMES = [
    "Ilmenite (FeTiO₃)", "Silica polymorph (SiO₂)", "Pyroxene (Ca,Mg,Fe)SiO₃",
    "Plagioclase (CaAl₂Si₂O₈)", "Epoxy", "Fe-Ti-Cr Spinel (Fe₂TiO₄)", "Calcite (CaCO₃)",
    "Apatite (Ca₅(PO₄)₃(OH,Cl,F))", "Fe Oxyhydroxide (FeOOH)", "Zn-bearing Phase",
    "Xenotime (YPO₄)", "Ferroan Olivine ((Fe,Mg)₂SiO₄)", "Weathering Assemblage",
    "Pigeonite (low-Ca pyroxene)", "Augite (high-Ca pyroxene)",
    "Merrillite (Ca₉NaMg(PO₄)₇)", "Chlorapatite (Ca₅(PO₄)₃(Cl,F,OH))",
]
KEYS = {n.split(" (")[0] for n in NAMES}


def test_every_palette_key_is_a_phase_key():
    assert set(C.PALETTE) <= KEYS


def test_every_mapping_key_is_a_phase_key_and_index_exists():
    indices = {p.index for p in C.TIMA_PHASES}
    for mapping in (C.TIMA_MAPPING_SPLIT, C.TIMA_MAPPING_COMBINED):
        assert set(mapping) <= KEYS
        for tima in mapping.values():
            assert set(tima) <= indices


def test_display_and_exclusion_keys():
    assert set(C.DISPLAY_NAMES) <= KEYS
    assert set(C.FIG4_SHORT_NAMES) <= KEYS
    assert set(C.EXCLUDED_KARAK_KEYS) <= KEYS
    assert set(C.FINGERPRINT_EXCLUDED_KEYS) <= KEYS
    assert set(C.LEGEND_ORDER_SPLIT) <= KEYS and set(C.LEGEND_ORDER_COMBINED) <= KEYS


def test_tima_table_has_14_unique_colours():
    assert len(C.TIMA_PHASES) == 14
    assert len({p.rgb for p in C.TIMA_PHASES}) == 14
    assert [p.index for p in C.TIMA_PHASES] == list(range(14))


def test_offsets_and_geometry():
    assert (C.TIMA_DX, C.TIMA_DY) == (1919, 1051)
    assert C.PIXEL_SIZE_UM_1X == 3.0
    assert C.FIG7_CROP_1X == (6061, 6861, 3570, 4370)
