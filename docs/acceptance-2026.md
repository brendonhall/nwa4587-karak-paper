# Real-data acceptance, 2026-10-17

PR 6 of 6. Base: `feat/lamellae-colormap` (PR 5). PRs 1 to 5 are open; the owner merges the stack in order.

## What this PR adds

- `expected/numbers.json`: every number of the acceptance run below (875 LaTeX macros plus the list values). `paper numbers --check` compares a fresh run with it.
- `src/nwa4587_paper/constants.py`: `FIG5_WINDOW_XY = (1700, 3500)`, the Fig 5 window that the search chose on the real data. `tests/test_constants.py` asserts that it is set.
- `README.md`: the full README (outputs, requirements, quick start, data, numbers bridge, known differences, licensing, citation).
- `CITATION.cff`: the repository and the paper as the preferred citation ("in revision, 2026"; the journal reference follows on acceptance). `cffconvert --validate` passes.
- `docs/acceptance-2026.md`: this file.

No algorithm and no `REGISTRY` tolerance changes.

## How the acceptance ran

Machine: AMD Ryzen 7 5800X3D (8 cores, 16 threads), 31 GB RAM, Arch Linux, Python 3.13.15 in the uv environment, karak at `7ccd7a7`, scikit-image 0.26.0, SciPy 1.18.1.

Inputs:

- `uv run paper data --from <raw maps>`: 21 element PNGs and `mask/Valid_mask.csv` into `data/eds`.
- The TIMA phase raster copied to `data/tima/phases.png`.
- `results/paperflow.h5`: a copy of the 2026-10-07 karak reference run (`karak run --builtin paper`, 2.2 GB). `paper karak` did not run again. A second acceptance from a fresh `paper karak` run is a follow-up once karak has a release.

Each stage ran alone as `uv run paper <stage>`. A wrapper recorded the wall time and the peak resident memory of the child process.

| Stage | Wall time | Peak RSS |
|---|---|---|
| tables | 0.9 s | 0.26 GB |
| fingerprints | 7.7 s | 3.54 GB |
| tima | 113.9 s (114.6 s with the frozen window) | 16.41 GB |
| pyroxene | 59.8 s | 3.08 GB |
| xenotime | 9.0 s | 0.83 GB |
| lamellae | 9.8 s | 2.00 GB |
| colormap | 196.5 s | 1.34 GB |
| workflow | 7.4 s | 1.69 GB |
| numbers | 0.7 s | 0.15 GB |
| total | 405.7 s (6.8 min) | |

Then:

- The TIMA stage logged `fig5 window x0=1700 y0=3500 size=520`. With `FIG5_WINDOW_XY = (1700, 3500)` frozen, a second `paper tima` wrote a `numbers/tima.json` identical to the first one (same window, `n_both_pyroxene` 243,156).
- `cp results/numbers.json expected/numbers.json`, then `uv run paper numbers --check`: "all numbers match", exit 0.
- `uv run pytest -q -W error tests/test_constants.py tests/test_tima.py`: 10 passed. The synthetic Fig 5 test passes because the window clamps to the synthetic raster.
- `uv run pytest -q -W error`: 112 passed in 178 s.

## Numbers

Fresh is this run. Expected is the reference run and the paper, as the task brief gives them.

| Key | Fresh | Expected or published | Status |
|---|---|---|---|
| `tables.n_mineral_pixels` | 12,495,787 | 12,495,787 | OK |
| `tables.n_phases_with_pixels` | 15 | 15 | OK |
| `tables.phases.Plagioclase.pct` | 28.923 | 28.923 | OK |
| `tables.phases.Pigeonite.pct` | 19.320 | 19.320 | OK |
| `tables.phases.Augite.pct` | 17.760 | 17.760 | OK |
| `tables.phases.Weathering Assemblage.pct` | 14.140 | 14.140 | OK |
| `tables.phases.Epoxy.pct` | 8.725 | 8.725 | OK |
| `tables.phases.Xenotime.pixels` | 213 | 213 | OK |
| `pyroxene.n_pixels_2x` | 4,633,456 | 4,633,456 | OK |
| `pyroxene.n_pixels_1x.Augite` | 8,904,186 | 8,904,186 | OK |
| `pyroxene.n_pixels_1x.Pigeonite` | 9,629,508 | 9,629,508 | OK |
| `pyroxene.pct_1x.Augite` | 48.04 | 48.0 | OK |
| `pyroxene.pct_1x.Pigeonite` | 51.96 | 52.0 | OK |
| `pyroxene.gmm_mean.Augite` | 0.5336 | 0.5336 | OK |
| `pyroxene.gmm_mean.Pigeonite` | 0.3056 | 0.3056 | OK |
| `lamellae.n_grains_total` | 274 | 272 | MISMATCH |
| `lamellae.n_grains_analyzed` | 154 | 154 | OK |
| `lamellae.spacing.median_um` | 45.0 | 45.0 | OK |
| `lamellae.orientation.rayleigh_p_value` | 0.7994 | 0.80 (within 0.01) | OK |
| `xenotime.n_pixels_2x` | 213 | 213 | OK |
| `xenotime.largest_component_px` | 200 | 200 | OK |
| `colormap.jet.mae` | 0.001385 | about 0.0014 | OK |
| `colormap.jet.max_ae` | 0.01559 | about 0.016 | OK |
| `tima.alignment.shape_hw` | [12685, 7980] | [12685, 7980] | OK |
| `tima.split.overall_accuracy` | 0.8412 | 0.8407 published | recorded (+0.0005) |
| `tima.combined.overall_accuracy` | 0.9909 | 0.9910 published | recorded (-0.0001) |
| `tima.split.n_included` | 37,730,564 | 37,640,743 published | recorded (+89,821, +0.24 %) |

Other TIMA values from this run: `tima.combined.n_included` 37,770,693; split Spearman r 0.952 (p 6.3e-6); combined Spearman r 0.936 (p 6.7e-5); agree 31,740,698 px and disagree 5,989,866 px (published 31,646,381 and 5,994,362).

Other lamellae values: circular mean 33.7°, R̄ 0.038, orientations uniform; spacing IQR 36 to 63 µm, mean 55.0 µm; grain size against spacing Spearman ρ 0.322 (p 4.6e-5); 0 grains near Nyquist.

### The lamellae grain count

`lamellae.n_grains_total` is 274 here and 272 in the karak reference analysis. The cause is the small-object threshold in scikit-image 0.26.0, not the input:

- The closed pyroxene mask has 4,436 components. 274 have at least 500 pixels, and two of those have exactly 500 pixels.
- The port calls `remove_small_objects(..., max_size=min_size - 1)`. That keeps objects of 500 pixels or more, the documented meaning of the old `min_size=500`: 274 grains.
- The reference script calls `remove_small_objects(..., min_size=500)` under scikit-image 0.26.0. The deprecation wrapper passes the value on as `max_size=500`, which removes objects of 500 pixels or fewer: 272 grains. The 0.26.0 source has the exclusive `< min_size` branch, but the wrapper never reaches it.

Both 500-pixel grains are below the 2,000-pixel analysis limit, so the 154 analyzed grains and every statistic match. The code is unchanged. The decision is open: keep 274 (the documented threshold) or change the call to `max_size=min_size` to match 272.

## Figure comparison

Each fresh PNG under `results/figures/` was compared with the published PDF.

**Fig 1, workflow** (published `fig1_workflow`). Panels (a) to (c) are the same. Panel (d), 13 clusters, uses the phase palette, so the clusters show plagioclase blue and epoxy grey; the published panel uses a separate cluster palette (orange, green, pink). Panel (e) and the 15-key legend are the same. The fresh panel frames are black; the published frames are grey. The published panel (e) has a dark triangle at the bottom tip that the fresh panel does not have.

**Fig 2, fingerprints** (published `fig5_fingerprints`). The same 13 phases and bar heights. Differences: the y axis scales per panel (published: 0 to 1 on every panel, with grid lines); titles drop the % sign and the Merrillite and Chlorapatite formulas; the element order puts Zn before Na (published: Mn, Na, Zn). Abundances in the titles follow the new inventory: Plagioclase 28.9 (28.8), Augite 17.8 (17.9), Calcite 0.18 (0.15), Zn-bearing phase 0.023 (<0.01). The Zn-bearing phase fingerprint is different: Zn 0.21 and O 0.15 against Zn 0.17 and O 0.14, with larger error bars, because the class holds 0.023 % of the pixels against 0.006 %.

**Fig 3, phase comparison** (published `fig3_phase_comparison`). Same layout, legend and TIMA panel. In panel (a) the pyroxene grains read olive-green at page scale, where the published panel reads blue and yellow. The 1x augite and pigeonite stripes now come from `clusters/hires/labels` and blend when the map is drawn small. The published panel (a) has a flat cut at the bottom tip; the fresh one does not.

**Fig 4, agreement map** (published `fig4_agreement_map`). The map is the same. Counts: agree 31,740,698 (31,646,381), disagree 5,989,866 (5,994,362), overall 84.1 % in both. Per-phase bars: Olivine 94 % (93 %), Calcite 92 % (93 %), the rest the same. Layout differences: no "(a) Agreement map" title, no "Pixel count" axis label, no Agree and Disagree legend in panel (b), short y labels "Pigeonite" and "Augite", and the overall-agreement box overlaps the Plagioclase label.

**Fig 5, pyroxene against TIMA** (published `fig_pyroxene_tima`). The frozen window (1700, 3500) shows the same area as the published figure, and the stripes match. The fresh panels have no frame, and a blank band sits between the panels and the legend.

**Fig 6, xenotime call-out** (published `fig7_xenotime_callout`). The same grain, outline and intensity panels. Panel (b) uses the canonical palette (xenotime bright green, plagioclase blue, weathering assemblage dark blue), where the published panel used a separate palette (xenotime grey, plagioclase green). The legend lists 14 fresh phases with Merrillite and Chlorapatite (published: one "Apatite" entry, 10 items). The BSE panel is brighter and shows a vertical tile seam. The panel letters are smaller.

**Fig 7, pyroxene refinement** (published `fig6_pyroxene_refinement`). The same window, maps, histogram and density contours. The augite and pigeonite colours follow the names, not labels 13 and 14. Legends: full names "Augite (high-Ca px)" and "Pigeonite (low-Ca px)"; panel (c) drops the pixel counts (9.6M, 8.9M) and the legend frame; panel (d) has no legend. The panel (c) x axis starts at about 0.07 (published 0.1).

**Fig 8, lamellae** (published `fig8_lamellae`). Expected differences, from the factor-2 correction: n = 154 (59), median 45 µm (51), IQR 36 to 63 µm (28 to 73), R̄ 0.04 (0.28), p 0.799 (0.008), mean direction 34° (94°), ρ 0.32 with p < 0.001 (0.37, p 0.004). Panel (a) now shows the correct extent, x 11 to 16 mm and y 17.5 to 22 mm at factor 2 (published: 14 to 23 mm and 24 to 33 mm from the factor-4 upscale). Panel (c) clips at 167 µm and notes the one grain beyond it (published axis to 900 µm). Layout problems: panel (d) y tick labels mix large "3 × 10²" minor labels with a small "10²" major label; the panel (b) radial labels run past the bars and the statistics box overlaps the circle.

## Known differences and open findings

- **Lamellae spacing-direction bias.** The verbatim lamellae method has a confirmed bias, found during the port: the measured spacing is period / |sin 2θ|, exact only at 45°, and grains with lamellae near 0° and 90° drop out. Version 1 keeps the method verbatim; the author decides whether to change it.
- **Rose diagram rotation sense.** The Fig 8 rose diagram turns in the mirror sense of the image. Version 1 keeps the rotation sense as the source has it.
- **Lamellae grain count 274 against 272.** See "The lamellae grain count" above.
- **TIMA metrics.** They now come from `clusters/hires/labels`. Split accuracy 0.8412 (published 0.8407), combined 0.9909 (0.9910), 37,730,564 pixels compared (37,640,743).
- **Inventory.** 15 phases carry pixels against the published 16; the published 92-pixel "Pyroxene (unresolved)" resolves at 1x.
- **TIMA memory.** The TIMA stage peaks at 16.4 GB, not the 4 GB the plan assumed. The README says 17 GB.

## Follow-ups

- Rerun the acceptance from a fresh `paper karak` run once karak has a release.
- Decide the lamellae grain threshold (274 or 272) and the spacing-direction bias.
- `REGISTRY` tolerances: these specs compare exactly although a fresh `paper karak` run can move them. Consider a relative tolerance for each:
  - `tables.n_mineral_pixels`, `tables.phases.*.pixels`
  - `fingerprints.phases.*.n_pixels`
  - `pyroxene.n_pixels_2x`, `pyroxene.n_pixels_1x.*`, `pyroxene.n_undefined_1x`
  - `tima.*.n_included`, `tima.*.per_phase.*.pixel_count`, `.tp`, `.fp`, `.fn`
  - `lamellae.n_grains_total`, `lamellae.n_grains_analyzed`
  - `xenotime.n_pixels_2x`, `xenotime.largest_component_px`, `xenotime.n_components`, `xenotime.bbox_2x`, `xenotime.roi_2x`

  Every float spec has a tolerance. `tables.phases.*.pct` and `fingerprints.phases.*.area_pct` (0.0005 pp) and `pyroxene.gmm_mean.*` (0.0005) are tight for a fresh clustering run.
- Pillow warns `DecompressionBombWarning` for the TIMA raster (159 Mpx) and one EDS map (104 Mpx). Set `Image.MAX_IMAGE_PIXELS` for these known inputs.
- Figure layout: the Fig 4 box overlap and missing labels, the Fig 8 panel (d) tick labels and panel (b) radial labels, the Fig 2 fixed y scale, the Fig 7 legends.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WyX19sxPcuhKUG3Jw8YCFn
