"""The `paper` command."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from . import numbers as N
from .io import DataPaths, MissingInput, MissingPhase, Run

STAGES: dict[str, Callable] = {}
STAGE_ORDER = ["tables", "fingerprints", "tima", "pyroxene", "xenotime", "lamellae", "colormap", "workflow"]
NEEDS_RUN = {"colormap": False}


def _import_stages() -> None:
    """Each stage module is imported explicitly; later tasks add theirs here."""
    from . import fingerprints, lamellae, pyroxene, tables, tima, workflow, xenotime

    STAGES.setdefault("tables", tables.run)
    STAGES.setdefault("fingerprints", fingerprints.run)
    STAGES.setdefault("tima", tima.run)
    STAGES.setdefault("pyroxene", pyroxene.run)
    STAGES.setdefault("xenotime", xenotime.run)
    STAGES.setdefault("lamellae", lamellae.run)
    STAGES.setdefault("workflow", workflow.run)


def data_paths(data_dir: Path) -> DataPaths:
    eds = data_dir / "eds"
    tima_dir = data_dir / "tima"
    tima = None
    if (tima_dir / "phases.png").exists():
        tima = tima_dir / "phases.png"
    elif tima_dir.is_dir():
        pngs = sorted(tima_dir.glob("*.png"))
        tima = pngs[0] if len(pngs) == 1 else None
    return DataPaths(eds_dir=eds if eds.is_dir() else None, tima_png=tima)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--h5", default="results/paperflow.h5")
    p.add_argument("--out", default="results")
    p.add_argument("--data", default="data")
    p.add_argument("--check", action="store_true")
    p.add_argument("--expected", default="expected/numbers.json")
    p.add_argument("--tima-offset", default=None, help=argparse.SUPPRESS)


def run_stage(name: str, args, **stage_kwargs) -> int:
    fn = STAGES[name]
    out = Path(args.out)
    data = data_paths(Path(args.data))
    if name == "tima":
        if data.tima_png is None:
            print("tima: skipped (no TIMA raster under data/tima)")
            return 0
        offset = getattr(args, "tima_offset", None)
        if offset:
            dx, dy = (int(v) for v in offset.split(","))
            stage_kwargs = {**stage_kwargs, "dx": dx, "dy": dy}
    try:
        if NEEDS_RUN.get(name, True):
            with Run.open(args.h5) as run:
                nums = fn(run, out, data, **stage_kwargs)
        else:
            nums = fn(None, out, data, **stage_kwargs)
    except MissingInput as e:
        print(f"{name}: {e}", file=sys.stderr)
        return 2
    except MissingPhase as e:
        print(f"{name}: {e}", file=sys.stderr)
        return 3
    path = N.write_stage(out, name, nums)
    print(f"{name}: wrote {path}")
    if args.check:
        return _check({name: nums}, args.expected)
    return 0


def _check(numbers: dict, expected_path: str) -> int:
    expected = json.loads(Path(expected_path).read_text())
    expected = {k: v for k, v in expected.items() if k in numbers}
    misses = N.check(numbers, expected)
    print(N.format_misses(misses))
    return 1 if misses else 0


def cmd_numbers(args) -> int:
    out = Path(args.out)
    merged = N.merge(out)
    (out / "numbers.tex").write_text(N.render_tex(merged))
    print(f"numbers: wrote {out / 'numbers.json'} and numbers.tex")
    if args.check:
        return _check(merged, args.expected)
    return 0


def cmd_figures(args) -> int:
    worst = 0
    for name in STAGE_ORDER:
        if name in STAGES:
            worst = max(worst, run_stage(name, args))
    return worst


def cmd_synth(args) -> int:
    from . import synth

    root = Path(args.dir)
    (root / "results").mkdir(parents=True, exist_ok=True)
    synth.write_synthetic_h5(root / "results" / "paperflow.h5")
    synth.write_synthetic_eds(root / "data" / "eds")
    (root / "data" / "tima").mkdir(parents=True, exist_ok=True)
    synth.write_synthetic_tima(root / "data" / "tima" / "phases.png")
    print(f"synthetic data under {root}")
    print(f"use: --tima-offset {synth.SYNTH_TIMA_DX},{synth.SYNTH_TIMA_DY}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    _import_stages()
    p = argparse.ArgumentParser(prog="paper", description="Regenerate the NWA 4587 paper results.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in STAGE_ORDER:
        sp = sub.add_parser(name, help=f"run the {name} stage")
        _add_common(sp)
        sp.set_defaults(func=lambda a, n=name: run_stage(n, a))
    sp = sub.add_parser("figures", help="run every stage")
    _add_common(sp)
    sp.set_defaults(func=cmd_figures)
    sp = sub.add_parser("numbers", help="merge numbers, render numbers.tex, optionally --check")
    _add_common(sp)
    sp.set_defaults(func=cmd_numbers)
    sp = sub.add_parser("synth", help="write synthetic data for a smoke run")
    sp.add_argument("dir")
    sp.set_defaults(func=cmd_synth)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd in STAGE_ORDER and args.cmd not in STAGES:
        print(f"{args.cmd}: stage not implemented yet", file=sys.stderr)
        return 4
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
