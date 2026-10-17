"""Synthetic karak-layout data for tests and the CI smoke run."""
from __future__ import annotations

import json
import struct
import zlib
from importlib.resources import files
from pathlib import Path

import h5py
import numpy as np
import yaml
from matplotlib import colormaps
from PIL import Image

from . import constants as C

H2, W2 = 64, 48
RATIO = 2
ELEMENTS = ["Ca", "Mg", "Y", "P"]
SYNTH_NAMES: dict[int, str] = {
    0: "Ilmenite (FeTiO₃)", 1: "Silica polymorph (SiO₂)", 2: "Pyroxene (Ca,Mg,Fe)SiO₃",
    3: "Plagioclase (CaAl₂Si₂O₈)", 4: "Epoxy", 5: "Fe-Ti-Cr Spinel (Fe₂TiO₄)",
    6: "Calcite (CaCO₃)", 7: "Apatite (Ca₅(PO₄)₃(OH,Cl,F))", 8: "Fe Oxyhydroxide (FeOOH)",
    9: "Zn-bearing Phase", 10: "Xenotime (YPO₄)", 11: "Ferroan Olivine ((Fe,Mg)₂SiO₄)",
    12: "Weathering Assemblage", 13: "Pigeonite (low-Ca pyroxene)",
    14: "Augite (high-Ca pyroxene)", 15: "Merrillite (Ca₉NaMg(PO₄)₇)",
    16: "Chlorapatite (Ca₅(PO₄)₃(Cl,F,OH))",
}
SYNTH_TIMA_DX = C.TIMA_DX - 1900
SYNTH_TIMA_DY = C.TIMA_DY - 1000
SYNTH_FLOW_YAML = yaml.safe_dump(
    json.loads(files("nwa4587_paper").joinpath("data/paper_flow.json").read_text()),
    sort_keys=False)


def _base_labels() -> tuple[np.ndarray, np.ndarray]:
    """2x label image before any split (labels 2, 3, 4, 7, 10) and the mineral mask."""
    img = np.full((H2, W2), -1, np.int32)
    mask = np.zeros((H2, W2), bool)
    mask[6:58, 6:42] = True
    img[6:30, 6:42] = 3
    img[30:58, 6:30] = 2
    img[30:58, 30:42] = 4
    img[40:44, 36:40] = 10
    img[8:12, 36:44] = 7
    mask[8:12, 42:44] = True
    img[~mask] = -1
    return img, mask


def _hires_labels(pre: np.ndarray) -> np.ndarray:
    region = np.repeat(np.repeat(pre == 2, RATIO, 0), RATIO, 1)
    hires = np.full(region.shape, -1, np.int16)
    rows = np.arange(region.shape[0])[:, None]
    stripe = ((rows // 4) % 2 == 0)
    hires[region & np.broadcast_to(stripe, region.shape)] = 14      # Augite
    hires[region & ~np.broadcast_to(stripe, region.shape)] = 13     # Pigeonite
    r0, c0 = 60, 12                                                  # top-left of the region at 1x
    hires[r0:r0 + 2, c0:c0 + 2] = -1
    return hires


def final_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Returns (final 2x labels, mineral mask, hires labels, history)."""
    pre, mask = _base_labels()
    img = pre.copy()
    img[54:58, 6:30] = 11
    img[50:54, 6:30] = 12
    hires = _hires_labels(pre)
    hires[np.repeat(np.repeat(img != 2, RATIO, 0), RATIO, 1)] = -1
    pyx = img == 2
    for r in range(H2):
        for c in range(W2):
            if pyx[r, c]:
                block = hires[2 * r:2 * r + 2, 2 * c:2 * c + 2]
                vals = block[block >= 0]
                img[r, c] = -1 if vals.size == 0 else int(np.bincount(vals).argmax())
    img[8:12, 36:40] = 15
    img[8:12, 40:44] = 16
    n = lambda lab, arr=img: int((arr == lab).sum())
    history = [
        {"stage": "split_threshold", "parent": 2, "new_labels": [11], "names": {"11": SYNTH_NAMES[11]},
         "n_pixels": {"11": n(11)}, "method": "Fe-K > 0.6 & Ca < 0.10", "note": "synthetic"},
        {"stage": "split_gmm", "parent": 2, "new_labels": [12], "names": {"12": SYNTH_NAMES[12]},
         "n_pixels": {"12": n(12)}, "method": "GMM 2-component on Ca/(Ca+Mg)+BSE+Fe-K", "note": "synthetic",
         "component_means": {"2": {"Ca/(Ca+Mg)": 0.43}, "12": {"Ca/(Ca+Mg)": 0.62}}},
        {"stage": "split_hires", "parent": 2, "new_labels": [13, 14],
         "names": {"13": SYNTH_NAMES[13], "14": SYNTH_NAMES[14]},
         "n_pixels": {"13": n(13), "14": n(14)}, "method": "GMM 2-component on Ca/(Ca+Mg) at 2x resolution",
         "note": "synthetic", "parents": [2], "ratio": RATIO,
         "n_pixels_hires": {"13": n(13, hires), "14": n(14, hires)}, "n_undefined": 4,
         "component_means": {"14": {"Ca/(Ca+Mg)": 0.53}, "13": {"Ca/(Ca+Mg)": 0.31}}},
        {"stage": "split_gmm", "parent": 7, "new_labels": [15, 16],
         "names": {"15": SYNTH_NAMES[15], "16": SYNTH_NAMES[16]},
         "n_pixels": {"15": n(15), "16": n(16)}, "method": "GMM 2-component on Cl+Na+Mg+F", "note": "synthetic"},
    ]
    return img, mask, hires, history


def _cube(img: np.ndarray) -> np.ndarray:
    cube = np.zeros((H2, W2, len(ELEMENTS)), np.float32)
    ca, mg, y, p = (cube[:, :, i] for i in range(4))
    ca[img == 3] = 0.3
    ca[img == 14] = 0.6
    ca[img == 13] = 0.2
    mg[img == 14] = 0.4
    mg[img == 13] = 0.8
    y[:] = 0.02
    p[:] = 0.02
    y[img == 10] = 0.9
    p[img == 10] = 0.9
    return cube


def write_synthetic_h5(path: Path) -> Path:
    img, mask, hires, history = final_labels()
    rows, cols = np.nonzero(mask)
    labels = img[rows, cols].astype(np.int32)
    cube = _cube(img)
    bse = np.linspace(0, 1, H2, dtype=np.float32)[:, None].repeat(W2, 1)
    present = [int(k) for k in np.unique(labels) if k >= 0 and (labels == k).any()]
    names = {k: SYNTH_NAMES[k] for k in present}
    counts = {str(k): {"n_pixels": int((labels == k).sum()),
                       "pct": float(100 * (labels == k).sum() / labels.size),
                       "mean_prob": 1.0} for k in present}
    flow = yaml.safe_load(SYNTH_FLOW_YAML)
    refinement_types = ("name_phases", "split_threshold", "split_gmm", "split_hires")
    refinement_nodes = [{"id": n["id"], "type": n["type"], "params": n["params"]}
                        for n in flow["nodes"] if n["type"] in refinement_types]
    with h5py.File(path, "w") as f:
        f.attrs["pipeline_config"] = SYNTH_FLOW_YAML
        f.attrs["pipeline_version"] = "synthetic"
        f.attrs["created"] = "2026-01-01T00:00:00+00:00"
        f.attrs["library_versions"] = "{}"
        f.attrs["platform"] = "synthetic"
        f.attrs["python_version"] = "3.12"
        g = f.create_group("raw")
        g.attrs.update({"element_names": json.dumps(ELEMENTS), "height": H2, "width": W2,
                        "n_elements": len(ELEMENTS)})
        for i, el in enumerate(ELEMENTS):
            g.create_dataset(el, data=cube[:, :, i])
        b = f.create_group("bse")
        b.attrs.update({"downsample_factor": RATIO, "original_shape": [H2, W2], "stored_shape": [H2, W2]})
        b.create_dataset("image", data=bse)
        m = f.create_group("masks")
        m.attrs.update({"coverage_of_valid_pct": 100.0, "coverage_pct": float(100 * mask.mean()),
                        "mask_config": "min_object_size: 100\n", "n_connected_components": 1,
                        "n_mineral_pixels": int(mask.sum()), "n_total_pixels": int(mask.size),
                        "n_valid_pixels": int(mask.sum())})
        m.create_dataset("mineral", data=mask)
        m.create_dataset("valid", data=mask)
        d = f.create_group("denoised")
        d.attrs.update({"element_order": json.dumps(ELEMENTS), "method": "bilateral",
                        "sigma_color": "auto", "sigma_spatial": 1.0})
        d.create_dataset("cube", data=cube)
        c = f.create_group("clusters")
        c.attrs.update({
            "mineral_names": json.dumps({str(k): v for k, v in names.items()}),
            "cluster_stats": json.dumps({"n_clusters": len(counts), "n_noise": 0, "noise_pct": 0.0,
                                         "n_total": int(labels.size), "clusters": counts}),
            "cluster_config": yaml.safe_dump({"strategy": "tiled", "refinement_nodes": refinement_nodes},
                                             sort_keys=False),
            "clustering_strategy": "tiled", "n_clusters": len(counts), "n_mineral_pixels": int(labels.size),
            "n_noise": 0, "n_pca_components_used": 3, "noise_pct": 0.0,
        })
        replaced = {rec["parent"] for rec in history}     # the real file has no name attr for replaced parents
        for k, v in SYNTH_NAMES.items():     # the real file stores one attribute per palette entry
            if k not in replaced:
                c.attrs[f"cluster_{k}_name"] = v
        c.create_dataset("cleaned_labels", data=labels)
        c.create_dataset("raw_labels", data=labels)
        c.create_dataset("probabilities", data=np.ones(labels.size, np.float32))
        c.create_dataset("mineral_indices", data=np.stack([rows, cols], 1).astype(np.int32))
        c.create_dataset("pca_variance_ratio", data=np.array([0.6, 0.3, 0.1]))
        s = c.create_group("subclustering")
        s.attrs["history"] = json.dumps(history)
        for i, rec in enumerate(history):
            s.attrs[f"split_{i:02d}"] = json.dumps(rec)
        h = c.create_group("hires").create_dataset("labels", data=hires)
        h.attrs.update({"downsample_factor": 1, "header_trim_px": 100, "left_trim_px": 0,
                        "names": json.dumps({"13": SYNTH_NAMES[13], "14": SYNTH_NAMES[14]}),
                        "ratio": RATIO})
    return path


def _jet_png(values: np.ndarray, path: Path) -> None:
    rgb = (colormaps["jet"](values)[:, :, :3] * 255).astype(np.uint8)
    banner = np.zeros((100, rgb.shape[1], 3), np.uint8)
    Image.fromarray(np.vstack([banner, rgb])).save(path)


def write_synthetic_eds(dir: Path) -> Path:
    img, mask, hires, _ = final_labels()
    ca1 = np.where(hires == 14, 0.6, np.where(hires == 13, 0.2, 0.0)).astype(np.float32)
    ca1[np.repeat(np.repeat(img == 3, RATIO, 0), RATIO, 1)] = 0.3
    mg1 = np.where(hires == 14, 0.4, np.where(hires == 13, 0.8, 0.0)).astype(np.float32)
    bse1 = np.linspace(0, 1, H2 * RATIO, dtype=np.float32)[:, None].repeat(W2 * RATIO, 1)
    dir.mkdir(parents=True, exist_ok=True)
    _jet_png(ca1, dir / "SYN-Ca.png")
    _jet_png(mg1, dir / "SYN-Mg.png")
    sem = np.vstack([np.zeros((100, W2 * RATIO, 3), np.uint8),
                     np.repeat((bse1 * 255).astype(np.uint8)[:, :, None], 3, 2)])
    Image.fromarray(sem).save(dir / "SYN-SEM.png")
    (dir / "mask").mkdir(exist_ok=True)
    # napari polygon CSV: index, shape-type, vertex-index, axis-0, axis-1 (1x pixel coordinates incl. banner)
    r0, r1, c0, c1 = 100 + 12, 100 + 116, 12, 84
    lines = ["index,shape-type,vertex-index,axis-0,axis-1"]
    for i, (r, c) in enumerate([(r0, c0), (r0, c1), (r1, c1), (r1, c0)]):
        lines.append(f"0,polygon,{i},{r},{c}")
    (dir / "mask" / "Valid_mask.csv").write_text("\n".join(lines) + "\n")
    return dir


def write_png16_rgb(path: Path, rgb16: np.ndarray) -> Path:
    """Write a 16-bit RGB PNG with zlib and struct (Pillow 12 cannot)."""
    h, w, _ = rgb16.shape

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    raw = b"".join(b"\x00" + row.astype(">u2").tobytes() for row in rgb16)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)
    return Path(path)


def write_synthetic_tima(path: Path) -> Path:
    """A 16-bit RGB TIMA-style raster: the hires map in TIMA colours at the fixed
    offset minus (1900, 1000), with a legend strip of C.TIMA_LEGEND_TRIM_ROWS rows."""
    img, _, hires, _ = final_labels()
    tima_index = {14: 2, 13: 1}                      # Augite -> high-Ca, Pigeonite -> low-Ca
    lab2 = np.repeat(np.repeat(img, RATIO, 0), RATIO, 1)
    idx = np.full(hires.shape, -1, np.int16)
    idx[lab2 == 3] = 0
    idx[lab2 == 15] = 6
    idx[lab2 == 16] = 6
    idx[lab2 == 4] = -1
    idx[lab2 == 10] = 12
    for k, t in tima_index.items():
        idx[hires == k] = t
    H, W = 200, 200
    rgb = np.full((H + C.TIMA_LEGEND_TRIM_ROWS, W, 3), 40, np.uint8)   # dark grey: achromatic
    dx, dy = SYNTH_TIMA_DX, SYNTH_TIMA_DY
    for p in C.TIMA_PHASES:
        rr, cc = np.nonzero(idx == p.index)
        rgb[rr + dy, cc + dx] = p.rgb
    return write_png16_rgb(path, rgb.astype(np.uint16) * 257)