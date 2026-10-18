"""Place the input data and run the karak paper flow."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .io import MissingInput


def _real(paths) -> list[Path]:
    """Drop macOS AppleDouble files (names that start with ._)."""
    return sorted(p for p in paths if not p.name.startswith("._"))


def _copy(src: Path, dst: Path, force: bool) -> bool:
    if dst.exists() and not force:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def _find_tima(src: Path) -> Path | None:
    cands = _real((src / "TIMA results").glob("*.png")) if (src / "TIMA results").is_dir() else []
    if not cands and (src / "tima").is_dir():
        cands = _real((src / "tima").glob("*.png"))
    if len(cands) > 1:
        named = [c for c in cands if c.name.endswith("Phases.png")]
        if len(named) == 1:
            return named[0]
        raise MissingInput("several TIMA rasters, cannot choose: " + ", ".join(c.name for c in cands))
    return cands[0] if cands else None


def copy_inputs(src: Path, data_dir: Path, force: bool = False) -> dict:
    src, data_dir = Path(src), Path(data_dir)
    pngs = _real(src.glob("*.png"))
    for p in pngs:
        _copy(p, data_dir / "eds" / p.name, force)
    mask = src / "mask" / "Valid_mask.csv"
    has_mask = mask.is_file()
    if has_mask:
        _copy(mask, data_dir / "eds" / "mask" / "Valid_mask.csv", force)
    tima = _find_tima(src)
    if tima is not None:
        _copy(tima, data_dir / "tima" / "phases.png", force)
    return {"n_png": len(pngs), "mask": has_mask, "tima": tima is not None}


def zenodo_stub(doi: str) -> int:
    print(
        "Zenodo download is not available yet; sub-project 5 deposits the data. "
        f"Place the files by hand: {doi}\n"
        "Use `paper data --from PATH` with a directory that holds the elemental PNGs, "
        "mask/Valid_mask.csv and the TIMA results folder."
    )
    return 2


def run_karak(data_dir, out_dir, workers: int = 0, extra: Sequence[str] = ()) -> int:
    exe = shutil.which("karak")
    base = [exe] if exe else [sys.executable, "-m", "karak.flow", "run"]
    if exe:
        base.append("run")
    argv = [
        *base, "--builtin", "paper",
        "--input", str(Path(data_dir) / "eds"),
        "--out", str(Path(out_dir) / "paperflow"),
        "--workers", str(workers), "--plain", *extra,
    ]
    return subprocess.run(argv).returncode
