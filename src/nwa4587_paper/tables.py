"""Table 1 (flow parameters) and Table 2 (phase abundances)."""
from __future__ import annotations

import csv
from pathlib import Path

from . import constants as C
from . import numbers as N
from .io import DataPaths, Run

N.register({
    "tables.n_mineral_pixels": N.Spec("int"),
    "tables.n_phases_with_pixels": N.Spec("int"),
    "tables.phases.*.label": N.Spec("int"),
    "tables.phases.*.pixels": N.Spec("int"),
    "tables.phases.*.pct": N.Spec("float", tol=0.0005, fmt="{:.2f}"),
    "tables.params.*": N.Spec("str"),
})


def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table1_rows(run: Run) -> list[tuple[str, str]]:
    rows = []
    for label, node, param, fmt in C.TABLE1_ROWS:
        value = run.flow_param(node, param)
        rows.append((label, fmt.format("auto" if value is None else value)))
    return rows


def write_table1(path: Path, rows: list[tuple[str, str]]) -> None:
    lines = [r"\begin{tabular}{ll}", r"\toprule", r"Parameter & Value \\", r"\midrule"]
    lines += [f"{_tex_escape(k)} & {_tex_escape(v)} \\\\" for k, v in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def table2_rows(run: Run) -> list[dict]:
    total = run.n_mineral_pixels
    rows = [{"label": lab, "phase": run.keys[lab], "name": run.names[lab],
             "pixels": n, "pct": 100.0 * n / total}
            for lab, n in run.counts.items() if lab >= 0 and lab in run.keys and n > 0]
    rows.sort(key=lambda r: (-r["pixels"], r["label"]))
    return rows


def _pct_text(pct: float) -> str:
    return "<0.01" if pct < 0.01 else f"{pct:.2f}"


def write_table2(dir: Path, rows: list[dict]) -> None:
    with open(dir / "table2_phases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "phase", "name", "pixels", "pct"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "pct": f"{r['pct']:.4f}"})
    lines = [r"\begin{tabular}{rlrr}", r"\toprule", r"Label & Phase & Area \% & Pixels \\", r"\midrule"]
    for r in rows:
        pixels = f"{r['pixels']:,}".replace(",", "{,}")
        lines.append(f"{r['label']} & {_tex_escape(r['phase'])} & {_pct_text(r['pct'])} & {pixels} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (dir / "table2_phases.tex").write_text("\n".join(lines))


def run(run: Run, out: Path, data: DataPaths) -> dict:
    d = Path(out) / "tables"
    d.mkdir(parents=True, exist_ok=True)
    t1 = table1_rows(run)
    write_table1(d / "table1_params.tex", t1)
    t2 = table2_rows(run)
    write_table2(d, t2)
    return {
        "n_mineral_pixels": run.n_mineral_pixels,
        "n_phases_with_pixels": len(t2),
        "phases": {r["phase"]: {"label": r["label"], "pixels": r["pixels"], "pct": r["pct"]} for r in t2},
        "params": {k: v for k, v in t1},
    }
