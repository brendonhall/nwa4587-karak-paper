# nwa4587-karak-paper

This repository regenerates the figures, tables and numbers of the NWA 4587 paper
"Unsupervised mineral phase mapping from SEM-EDS element maps: a density-based
clustering pipeline with multi-resolution refinement" (Izawa, Hall et al., in revision, 2026).
It reads the output of [karak](https://github.com/brendonhall/karak), pinned at commit `7ccd7a7`.

## What it regenerates

Every output goes under `results/`. The numbering follows the revised paper.

| Paper item | Content | Output file |
|---|---|---|
| Table 1 | pipeline parameters | `tables/table1_params.tex` |
| Table 2 | phase inventory and abundances | `tables/table2_phases.tex`, `.csv` |
| Table 3 | karak to TIMA phase correspondence | `tables/table3_correspondence.tex` |
| Table 4 | TIMA concordance per phase | `tables/table4_concordance.tex`, `.csv` |
| Fig 1 | workflow, raw map to named phases | `figures/fig1_workflow.pdf` |
| Fig 2 | chemical fingerprints | `figures/fig2_fingerprints.pdf` |
| Fig 3 | karak and TIMA phase maps | `figures/fig3_phase_comparison.pdf` |
| Fig 4 | agreement map and per-phase breakdown | `figures/fig4_agreement_map.pdf` |
| Fig 5 | pyroxene window, karak and TIMA | `figures/fig5_pyroxene_tima.pdf` |
| Fig 6 | xenotime call-out | `figures/fig6_xenotime_callout.pdf` |
| Fig 7 | pyroxene refinement, 2x to 1x | `figures/fig7_pyroxene_refinement.pdf` |
| Fig 8 | pyroxene exsolution lamellae | `figures/fig8_lamellae.pdf` |

Each figure also comes as a PNG. The numbers that the text quotes go to
`results/numbers.json` and `results/numbers.tex`.

## Requirements

- Python 3.12 or later and [uv](https://docs.astral.sh/uv/).
- About 17 GB of RAM for the TIMA stage. It decodes a 10,800 by 14,736 pixel raster.
  The other stages need less than 4 GB.
- The karak output `paperflow.h5` (2.2 GB) and the raw element maps.

## Quick start

```bash
uv sync
uv run paper data --from PATH            # PATH holds the 21 element PNGs and mask/Valid_mask.csv
uv run paper karak                       # about 45 min on a CPU; writes results/paperflow.h5
# or: copy the deposited paperflow.h5 into results/ and skip `paper karak`
uv run paper figures                     # every stage: tables, figures, per-stage numbers
uv run paper numbers --check             # merge the numbers and compare with expected/numbers.json
```

`paper figures` takes about 7 minutes on an 8-core AMD Ryzen 7 5800X3D.
The colormap stage (3.3 min) and the TIMA stage (1.9 min) take most of it.
Each stage also runs alone: `paper tables`, `fingerprints`, `tima`, `pyroxene`,
`xenotime`, `lamellae`, `colormap` and `workflow`.
`paper all --from PATH` runs data, karak, figures and numbers in one command.

## The data

The data is not in this repository. Until the Zenodo deposit is public, ask the authors
for the raw element PNGs and `paperflow.h5`. `paper data --from zenodo:DOI` is a stub
until the deposit exists; it prints where to put the files by hand.

The TIMA phase raster comes from TESCAN under a separate agreement. It is optional.
Put it at `data/tima/phases.png` to make Figs 3 to 5 and Tables 3 and 4.
Without it, `paper figures` skips the TIMA stage and the other stages run.

## The numbers bridge

Each stage writes `results/numbers/<stage>.json`. `paper numbers` merges them into
`results/numbers.json` and writes `results/numbers.tex`, one LaTeX macro per number
(for example `\nwaTablesNMineralPixels`). The paper source reads these macros.

`expected/numbers.json` holds the values of the 2026-10-17 acceptance run.
`paper numbers --check` compares a fresh run with that file. The tolerance for each key
is in the `REGISTRY` of `src/nwa4587_paper/numbers.py`: counts compare exactly unless a
spec says otherwise, and floats compare within an absolute or a relative tolerance.
The check is exact for the reference `paperflow.h5`. A fresh `paper karak` run can miss
on large counts; a follow-up widens those tolerances.
Each stage deletes its old `results/numbers/<stage>.json` before it runs, so a stage
that fails leaves no file. `--check` reports every expected stage without a file as a
miss. The one exception is `tima` when `data/tima` holds no raster: `--check` prints
`tima: not compared (no TIMA raster)` and goes on.

## Known differences from the published paper

- 15 phases carry pixels, against 16 in the published inventory. The published
  92-pixel "Pyroxene (unresolved)" class resolves into augite and pigeonite at 1x.
  The section has 12,495,787 mineral pixels at 2x.
- The published lamellae statistics are wrong. They came from a 4x upscale of the
  2x pyroxene mask: the analysis script read a downsample factor of 4 from a config
  file, but the HDF5 is at factor 2. The mask therefore covered the top-left quarter
  of the section, stretched over the Ca and Mg maps.

  | Lamellae value | Published | This repository |
  |---|---|---|
  | grains segmented | 128 | 272 |
  | grains analyzed | 59 | 154 |
  | median spacing | 51 µm | 45 µm |
  | Rayleigh p | 0.008 (not uniform) | 0.80 (uniform) |

  The grain filter removes components of 500 pixels or fewer, as the lamellae
  reference script did under scikit-image 0.26.
- The lamellae spacing method has a direction bias. It measures spacing as
  period / |sin 2θ|, which is exact only at 45 degrees, and grains with lamellae near
  0 and 90 degrees drop out. The port found the bias. Version 1 keeps the method
  verbatim, and the author has not yet decided whether to change it. See
  `docs/acceptance-2026.md`.
- The fingerprints accumulate in float32 on purpose. The published fingerprint values
  came from float32 accumulation, and float32 reproduces them.
- The TIMA metrics now come from karak's full-resolution labels
  (`clusters/hires/labels`). They move slightly:

  | TIMA value | Published | This repository |
  |---|---|---|
  | split overall accuracy | 0.8407 | 0.8412 |
  | combined overall accuracy | 0.9910 | 0.9909 |
  | pixels compared (split) | 37,640,743 | 37,730,564 |

`docs/acceptance-2026.md` lists every number and every figure difference.

## Licensing

The code is under the MIT license (`LICENSE`). Everything that the code writes
under `results/` is under CC BY 4.0 (`LICENSE-FIGURES`).

## How to cite

Cite the paper and this repository. `CITATION.cff` has both entries; GitHub shows them
under "Cite this repository". The journal reference follows when the paper is accepted.

## Development

The code arrives as a stack of six pull requests (PR 1 to PR 6). Each PR uses the one
before it as its base. The repository owner reviews and merges them in order.
Run the tests with `uv sync --group dev` and `uv run pytest`.

A wrong input stops a stage with one line on stderr and an exit code: 1 for a
`--check` miss, 2 for a missing file, HDF5 path or attribute, 3 for a missing phase
or split node, 4 for a stage that does not exist yet, and 5 for any other error.
Set `PAPER_DEBUG=1` to print the full traceback as well.
