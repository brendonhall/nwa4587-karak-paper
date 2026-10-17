"""Every colour, name, mapping, offset, window and threshold of the paper.

Keyed by phase key: the HDF5 phase name before the first " (".
Source: the paper repository's poster/make_phase_map_1x.py (palette),
analysis/tima_comparison_1x.py and tima_comparison.py (TIMA), the figure
scripts (windows, settings) and data/sample.yaml (pixel size).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

EPOXY_KEY = "Epoxy"

# ----- palette (poster/make_phase_map_1x.py L39-58) -----
PALETTE: dict[str, str] = {
    "Ilmenite": "#8B4513",
    "Silica polymorph": "#F0E442",
    "Plagioclase": "#56B4E9",
    "Epoxy": "#999999",
    "Fe-Ti-Cr Spinel": "#D55E00",
    "Calcite": "#CC79A7",
    "Apatite": "#0072B2",          # pre-split panels only (Fig 1 d)
    "Fe Oxyhydroxide": "#BBBBBB",
    "Zn-bearing Phase": "#882255",
    "Xenotime": "#00FF00",
    "Ferroan Olivine": "#44AA99",
    "Weathering Assemblage": "#332288",
    "Augite": "#E6AB02",
    "Pigeonite": "#2B83BA",
    "Merrillite": "#1B9E77",
    "Chlorapatite": "#E7298A",
    "Pyroxene": "#666666",         # unresolved pyroxene, pre-split panels
}
UNASSIGNED_COLOR = "#000000"
PALETTE_FALLBACK_COLOR = (0.85, 0.85, 0.85)   # a label with no palette entry
BSE_DISPLAY_MAX = 0.5   # BSE intensity shown as white
OVERLAY_ALPHA = 0.55   # the Fig 7 BSE overlay alpha
PYROXENE_UNDIFF_COLOR = "#66C266"   # Fig 7 panel (a)
NYQUIST_COLOR = "#CC3333"

# ----- display names (make_fig1_workflow.py L76-92, make_fig5 _SHORT_NAMES) -----
DISPLAY_NAMES: dict[str, str] = {
    "Ilmenite": "Ilmenite",
    "Silica polymorph": "Silica polymorph",
    "Plagioclase": "Plagioclase",
    "Epoxy": "Epoxy",
    "Fe-Ti-Cr Spinel": "Fe–Ti–Cr spinel",
    "Calcite": "Calcite",
    "Apatite": "Apatite",
    "Fe Oxyhydroxide": "Fe oxyhydroxide",
    "Zn-bearing Phase": "Zn-bearing phase",
    "Xenotime": "Xenotime",
    "Ferroan Olivine": "Ferroan olivine",
    "Weathering Assemblage": "Weathering assemblage",
    "Augite": "Augite (high-Ca px)",
    "Pigeonite": "Pigeonite (low-Ca px)",
    "Merrillite": "Merrillite",
    "Chlorapatite": "Chlorapatite",
    "Pyroxene": "Pyroxene (unresolved)",
}
FIG4_SHORT_NAMES: dict[str, str] = {
    "Fe-Ti-Cr Spinel": "Spinel",
    "Ferroan Olivine": "Olivine",
    "Silica polymorph": "Silica",
    "Fe Oxyhydroxide": "Fe-OOH",
}
TABLE4_NAMES: dict[str, str] = {"Pigeonite": "Pigeonite (low-Ca)", "Augite": "Augite (high-Ca)"}

# ----- TIMA colour table (analysis/tima_comparison.py L53-68) -----
class TimaPhase(NamedTuple):
    index: int
    name: str
    rgb: tuple[int, int, int]


TIMA_PHASES: tuple[TimaPhase, ...] = (
    TimaPhase(0, "Plagioclase", (170, 0, 127)),
    TimaPhase(1, "Pyroxene low-Ca", (0, 255, 255)),
    TimaPhase(2, "Pyroxene high-Ca", (103, 70, 156)),
    TimaPhase(3, "Ilmenite Cr", (128, 170, 10)),
    TimaPhase(4, "Olivine+Ilmenite", (0, 85, 0)),
    TimaPhase(5, "Pyroxene Bi", (170, 85, 255)),
    TimaPhase(6, "Apatite", (170, 170, 0)),
    TimaPhase(7, "Calcite", (255, 170, 255)),
    TimaPhase(8, "Quartz", (85, 85, 255)),
    TimaPhase(9, "Unknown1", (85, 0, 255)),
    TimaPhase(10, "Mixture", (100, 100, 150)),
    TimaPhase(11, "Troilite", (255, 0, 0)),
    TimaPhase(12, "Unknown2", (190, 255, 242)),
    TimaPhase(13, "Hematite/Magnetite", (0, 85, 255)),
)
TIMA_NAMES: dict[int, str] = {p.index: p.name for p in TIMA_PHASES}
TIMA_COMBINED_NAME = "Pyroxene"          # TIMA 1 after combining 1, 2 and 5

# ----- TIMA to karak mapping (analysis/tima_comparison_1x.py L78-118) -----
TIMA_MAPPING_SPLIT: dict[str, tuple[int, ...]] = {
    "Plagioclase": (0,),
    "Augite": (2,),
    "Pigeonite": (1,),
    "Ferroan Olivine": (4,),
    "Ilmenite": (3, 4),
    "Fe-Ti-Cr Spinel": (3,),
    "Merrillite": (6,),
    "Chlorapatite": (6,),
    "Calcite": (7,),
    "Silica polymorph": (8,),
    "Fe Oxyhydroxide": (13,),
}
TIMA_MAPPING_COMBINED: dict[str, tuple[int, ...]] = {
    k: v for k, v in TIMA_MAPPING_SPLIT.items() if k != "Augite"
}
MAPPING_NAMES: dict[str, str] = {
    **{k: k for k in TIMA_MAPPING_SPLIT},
    "Augite": "Augite (high-Ca px)",
    "Pigeonite": "Pigeonite (low-Ca px)",
}
TIMA_MAPPING_NOTES: dict[str, str] = {
    "Plagioclase": "Direct match",
    "Augite": "Direct match",
    "Pigeonite": "Direct match",
    "Ferroan Olivine": "TIMA 4 = Olivine+Ilmenite (shared color)",
    "Ilmenite": "TIMA splits: Ilmenite Cr (3) + Olivine+Ilmenite (4)",
    "Fe-Ti-Cr Spinel": "Possible overlap with TIMA Ilmenite Cr",
    "Merrillite": "TIMA lumps phosphates into Apatite",
    "Chlorapatite": "TIMA lumps phosphates into Apatite",
    "Calcite": "Direct match",
    "Silica polymorph": "Direct match (Quartz)",
    "Fe Oxyhydroxide": "Possible match with Hematite/Magnetite",
}
COMBINED_PYROXENE_NOTE = "Augite + Pigeonite merged; TIMA low-Ca + high-Ca + Bi merged"
# in combined mode: karak Augite -> Pigeonite's label; TIMA 2 and 5 -> 1
TIMA_COMBINE_INTO = 1
KARAK_COMBINE_FROM_KEY = "Augite"
KARAK_COMBINE_INTO_KEY = "Pigeonite"
TIMA_COMBINE_FROM = (2, 5)

EXCLUDED_KARAK_KEYS: tuple[str, ...] = ("Zn-bearing Phase", "Xenotime", "Weathering Assemblage")
EXCLUDED_TIMA_SPLIT: tuple[int, ...] = (5, 9, 10, 11, 12)
EXCLUDED_TIMA_COMBINED: tuple[int, ...] = (9, 10, 11, 12)
# -1 (unassigned / background) is always excluded in code.

# ----- registration (tima_comparison_1x.py L67-71) -----
TIMA_DX = 1919
TIMA_DY = 1051
TIMA_LEGEND_TRIM_ROWS = 1000
TIMA_COLOR_DISTANCE_MAX = 50.0
TIMA_DECODE_CHUNK_PX = 8_000_000
TIMA_ACHROMATIC_RANGE = 30
TIMA_ACHROMATIC_MIN = 20

# ----- Fig 3 and Fig 4 colours (make_fig3_fig4.py L80-91, L413-414) -----
MATCHED_COLORS: dict[str, tuple[float, float, float]] = {
    "Plagioclase": (0.698, 0.133, 0.133),
    "Pigeonite": (0.357, 0.510, 0.671),      # combined mode: the merged pyroxene
    "Ferroan Olivine": (0.467, 0.533, 0.600),
    "Ilmenite": (0.133, 0.133, 0.400),
    "Fe-Ti-Cr Spinel": (0.400, 0.400, 0.133),
    "Merrillite": (0.106, 0.620, 0.467),
    "Chlorapatite": (0.906, 0.161, 0.541),
    "Calcite": (0.600, 0.800, 0.400),
    "Silica polymorph": (0.800, 0.600, 0.200),
    "Fe Oxyhydroxide": (0.200, 0.200, 0.200),
}
MATCHED_COLORS_SPLIT_OVERRIDE: dict[str, tuple[float, float, float]] = {
    "Augite": (0.902, 0.671, 0.008),
    "Pigeonite": (0.169, 0.514, 0.729),
}
LEGEND_ORDER_SPLIT: tuple[str, ...] = (
    "Plagioclase", "Augite", "Pigeonite", "Ilmenite", "Fe-Ti-Cr Spinel", "Ferroan Olivine",
    "Merrillite", "Chlorapatite", "Calcite", "Silica polymorph", "Fe Oxyhydroxide",
)
LEGEND_ORDER_COMBINED: tuple[str, ...] = tuple(k for k in LEGEND_ORDER_SPLIT if k != "Augite")
TIMA_BACKGROUND = (0.92, 0.92, 0.92)
AGREE_COLOR = (0.25, 0.68, 0.25)
DISAGREE_COLOR = (0.75, 0.20, 0.20)
EXCLUDED_COLOR = (0.88, 0.88, 0.88)
FIG5_HIGH_CA = (0.902, 0.671, 0.008)
FIG5_LOW_CA = (0.169, 0.514, 0.729)
FIG5_NON_PYX = (0.93, 0.93, 0.93)

# ----- geometry -----
PIXEL_SIZE_UM_1X = 3.0          # data/sample.yaml acquisition.pixel_size_um

# ----- fixed windows -----
FIG7_CROP_1X = (6061, 6861, 3570, 4370)     # rows r0, r1, cols c0, c1 at 1x
FIG5_WINDOW_SIZE = 520
FIG5_WINDOW_STRIDE = 100
FIG5_WINDOW_XY: tuple[int, int] | None = None   # None: search and record; Task 6.2 freezes it
XENOTIME_PAD_PX_2X = 80
XENOTIME_CONTOUR_COLOR = "#FFD700"      # the xenotime outline on the BSE panel
FIG7_SUBSAMPLE_N = 80_000
FIG7_SUBSAMPLE_SEED = 42
FIG1_PNG_STRIDE = 16
FIG1_H5_STRIDE = 8
FIG1_ELEMENT = "Ca"


@dataclass(frozen=True)
class LamellaeSettings:
    min_grain_px: int = 500
    min_analysis_px: int = 2000
    closing_iterations: int = 2
    tensor_sigma: float = 1.5
    coherence_min: float = 0.2
    n_profiles: int = 20
    nyquist_px: float = 4.0


LAMELLAE = LamellaeSettings()


@dataclass(frozen=True)
class ColormapSettings:
    lut_n: int = 4096
    n_test: int = 100_000
    cmaps: tuple[str, ...] = ("jet", "viridis")


COLORMAP = ColormapSettings()

RARE_PHASE_THRESHOLD_PX = 500_000
FINGERPRINT_EXCLUDED_KEYS: tuple[str, ...] = ("Epoxy", "Weathering Assemblage")

# ----- Fig 2 -----
FIG2_BAR_COLOR = "#4878a8"
FIG2_BAR_EDGE_COLOR = "#345a80"
FIG2_ERROR_COLOR = "#444444"

# ----- Table 1: flow node params shown (label, node id, param, format) -----
TABLE1_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("Header trim (px)", "src", "header_trim_px", "{}"),
    ("Downsample factor", "src", "downsample_factor", "{}"),
    ("Minimum object size (px)", "msk", "min_object_size", "{}"),
    ("Bilateral sigma spatial (px)", "dn", "sigma_spatial", "{}"),
    ("Bilateral sigma colour", "dn", "sigma_color", "{}"),
    ("Normalisation", "nrm", "method", "{}"),
    ("PCA variance kept", "pca", "variance_threshold", "{}"),
    ("Tile size (px)", "hdb", "tile_size", "{}"),
    ("Tile merge cosine threshold", "hdb", "merge_threshold", "{}"),
    ("Minimum clusters per tile", "hdb", "min_clusters_per_tile", "{}"),
    ("HDBSCAN min cluster size", "hdb", "min_cluster_size", "{}"),
    ("HDBSCAN min samples", "hdb", "min_samples", "{}"),
    ("Rare-phase min cluster size", "rare", "min_cluster_size", "{}"),
    ("Rare-phase subsample", "rare", "subsample_n", "{:,}"),
    ("kNN neighbours", "knn", "k", "{}"),
    ("Olivine rule", "oliv", "rule", "{}"),
    ("Weathering GMM features", "weath", "features", "{}"),
    ("Pyroxene split feature (1x)", "pyx", "feature", "{}"),
    ("Phosphate GMM features", "phos", "features", "{}"),
)

# ----- Table 3 (main.tex tbl:correspondence) -----
TABLE3_ROWS: tuple[tuple[str, str, str], ...] = (
    ("Plagioclase", "Plagioclase", "Direct match"),
    ("Augite", "Pyroxene high-Ca", "Differing lamellae cut-off"),
    ("Pigeonite", "Pyroxene low-Ca", "Differing lamellae cut-off"),
    ("Ferroan Olivine", "Olivine+Ilmenite", "Shared TIMA colour"),
    ("Ilmenite", "Ilmenite Cr + Olivine+Ilmenite", "Pipeline merges"),
    ("Fe-Ti-Cr Spinel", "Ilmenite Cr?", "Possible overlap"),
    ("Merrillite", "Apatite", "TIMA lumps phosphates"),
    ("Chlorapatite", "Apatite", "TIMA lumps phosphates"),
    ("Calcite", "Calcite", "Direct match"),
    ("Silica polymorph", "Quartz", "Direct match"),
    ("Xenotime", "---", "Not detected by TIMA"),
    ("Zn-bearing Phase", "---", "Not detected by TIMA"),
    ("Fe Oxyhydroxide", "Hematite/Magnetite?", "Uncertain"),
)
RARE_BACKGROUND_NAME = "Background"
RARE_COMBINED_PYROXENE_NAME = "Pyroxene (combined)"
RARE_TOP_N = 5

# ----- Figs 3 to 5 text and neutral colours (make_fig3_fig4.py, tima_pyroxene_zoom.py) -----
FIG3_TITLE_A = "(a) Pipeline"
FIG3_TITLE_B = "(b) TIMA (aligned)"
FIG3_BACKGROUND_LABEL = "Background / excluded"
FIG4_TITLE_B = "(b) Per-phase breakdown"
FIG4_EXCLUDED_LABEL = "Excluded"
FIG4_ANNOTATION = "Overall agreement"
FIG5_TITLE_A = "(a) Pipeline (unsupervised)"
FIG5_TITLE_B = "(b) TIMA"
FIG5_LEGEND_LABELS: tuple[str, str, str] = (
    "Augite / pyroxene high-Ca", "Pigeonite / pyroxene low-Ca", "Other phase / background")
FIG5_KARAK_HIGH_KEY = "Augite"
FIG5_KARAK_LOW_KEY = "Pigeonite"
FIG5_TIMA_HIGH = 2
FIG5_TIMA_LOW = 1
FIG_LEGEND_EDGE_COLOR = "0.7"
FIG_LABEL_GREY = "0.3"
FIG_BOX_COLOR = "white"
