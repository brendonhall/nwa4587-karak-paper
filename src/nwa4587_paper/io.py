"""The karak HDF5 reader and the 1x PNG loaders."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
import yaml

from . import constants as C


class MissingPhase(KeyError):
    pass


class MissingInput(FileNotFoundError):
    pass


def phase_key(name: str) -> str:
    return name.split(" (")[0].strip()


def require(path: Path | None, what: str, command: str) -> Path:
    if path is None or not Path(path).exists():
        raise MissingInput(f"{what} not found at {path}; run `paper {command}` first")
    return Path(path)


@dataclass(frozen=True)
class Geometry:
    factor: int
    header_trim_px: int
    left_trim_px: int
    shape_2x: tuple[int, int]
    shape_1x: tuple[int, int]
    ratio: int

    @property
    def pixel_size_um_1x(self) -> float:
        return C.PIXEL_SIZE_UM_1X

    @property
    def pixel_size_um_2x(self) -> float:
        return C.PIXEL_SIZE_UM_1X * self.factor


@dataclass(frozen=True)
class DataPaths:
    eds_dir: Path | None = None
    tima_png: Path | None = None


SPLIT_TYPES = ("split_threshold", "split_gmm", "split_hires")


class Run:
    def __init__(self, f: h5py.File):
        self.f = f

    @classmethod
    def open(cls, path: str | Path) -> "Run":
        return cls(h5py.File(require(Path(path), "karak HDF5", "karak"), "r"))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.f.close()

    # ----- names -----
    @cached_property
    def names(self) -> dict[int, str]:
        return {int(k): v for k, v in json.loads(self.f["clusters"].attrs["mineral_names"]).items()}

    @cached_property
    def keys(self) -> dict[int, str]:
        return {k: phase_key(v) for k, v in self.names.items()}

    def phase(self, key: str) -> int:
        for lab, k in self.keys.items():
            if k == key:
                return lab
        raise MissingPhase(f"no phase with key '{key}'; names: {sorted(self.keys.values())}")

    def has(self, key: str) -> bool:
        return key in self.keys.values()

    # ----- 2x labels -----
    @cached_property
    def mineral_indices(self) -> np.ndarray:
        return self.f["clusters/mineral_indices"][:]

    @cached_property
    def cleaned_labels_flat(self) -> np.ndarray:
        return self.f["clusters/cleaned_labels"][:].astype(np.int32)

    @cached_property
    def mask(self) -> np.ndarray:
        return self.f["masks/mineral"][:].astype(bool)

    def _to_image(self, flat: np.ndarray) -> np.ndarray:
        img = np.full(self.geometry.shape_2x, -1, np.int32)
        img[self.mineral_indices[:, 0], self.mineral_indices[:, 1]] = flat
        return img

    @cached_property
    def labels(self) -> np.ndarray:
        return self._to_image(self.cleaned_labels_flat)

    @cached_property
    def counts(self) -> dict[int, int]:
        vals, n = np.unique(self.cleaned_labels_flat, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, n)}

    @property
    def n_mineral_pixels(self) -> int:
        return int(self.cleaned_labels_flat.size)

    # ----- history -----
    @cached_property
    def history(self) -> list[dict]:
        return json.loads(self.f["clusters/subclustering"].attrs["history"])

    @cached_property
    def split_node_ids(self) -> list[str]:
        cfg = yaml.safe_load(self.f["clusters"].attrs["cluster_config"]) or {}
        ids = [n["id"] for n in cfg.get("refinement_nodes", []) if n["type"] in SPLIT_TYPES]
        if len(ids) != len(self.history):
            ids = [f"split_{i:02d}" for i in range(len(self.history))]
        return ids

    def split_record(self, node_id: str) -> dict:
        return self.history[self.split_node_ids.index(node_id)]

    def labels_before(self, node_id: str) -> np.ndarray:
        """The 2x label image before the named split node: later splits and this one undone."""
        flat = self.cleaned_labels_flat.copy()
        start = self.split_node_ids.index(node_id)
        for rec in reversed(self.history[start:]):
            for new in rec["new_labels"]:
                flat[flat == int(new)] = int(rec["parent"])
        return self._to_image(flat)

    # ----- hires -----
    @cached_property
    def hires_labels(self) -> np.ndarray:
        return self.f["clusters/hires/labels"][:].astype(np.int16)

    @cached_property
    def hires_names(self) -> dict[int, str]:
        return {int(k): v for k, v in json.loads(self.f["clusters/hires/labels"].attrs["names"]).items()}

    @cached_property
    def labels_1x(self) -> np.ndarray:
        """2x labels repeated ratio x ratio, clipped to the hires shape, with the hires
        labels on top where defined. Epoxy becomes -1 (the old PNG decode did the same)."""
        r = self.geometry.ratio
        up = np.repeat(np.repeat(self.labels, r, 0), r, 1)
        h, w = self.hires_labels.shape
        out = up[:h, :w].astype(np.int16)
        if self.has(C.EPOXY_KEY):
            out[out == self.phase(C.EPOXY_KEY)] = -1
        defined = self.hires_labels >= 0
        out[defined] = self.hires_labels[defined]
        return out

    # ----- images -----
    @cached_property
    def bse(self) -> np.ndarray:
        return self.f["bse/image"][:].astype(np.float32)

    @cached_property
    def element_order(self) -> list[str]:
        raw = self.f["denoised"].attrs["element_order"]
        return json.loads(raw) if isinstance(raw, str) else [str(x) for x in raw]

    def channel(self, name: str) -> np.ndarray:
        return self.f["denoised/cube"][:, :, self.element_order.index(name)].astype(np.float32)

    def raw_channel(self, name: str) -> np.ndarray:
        return self.f[f"raw/{name}"][:].astype(np.float32)

    # ----- geometry and flow -----
    @cached_property
    def geometry(self) -> Geometry:
        bse = self.f["bse"]
        hires = self.f["clusters/hires/labels"]
        shape_2x = tuple(int(x) for x in self.f["bse/image"].shape)
        return Geometry(
            factor=int(bse.attrs["downsample_factor"]),
            header_trim_px=int(hires.attrs["header_trim_px"]),
            left_trim_px=int(hires.attrs["left_trim_px"]),
            shape_2x=shape_2x,
            shape_1x=tuple(int(x) for x in hires.shape),
            ratio=int(hires.attrs["ratio"]),
        )

    @cached_property
    def flow(self) -> dict:
        raw = self.f.attrs["pipeline_config"]
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return yaml.safe_load(raw)

    def flow_param(self, node_id: str, name: str):
        for node in self.flow["nodes"]:
            if node["id"] == node_id:
                return node["params"][name]
        raise KeyError(f"flow has no node '{node_id}'")


# ----- 1x PNG loaders -----
def find_png(eds_dir: Path, element: str) -> Path:
    eds_dir = require(eds_dir, "elemental PNG directory", "data --from PATH")
    hits = sorted(eds_dir.glob(f"*-{element}.png"))
    if len(hits) != 1:
        raise MissingInput(f"expected one *-{element}.png in {eds_dir}, found {len(hits)}")
    return hits[0]


def load_channel_1x(eds_dir: Path, element: str, header_trim_px: int) -> np.ndarray:
    import imageio.v3 as iio
    from karak.io.loaders import invert_jet_colormap

    rgb = iio.imread(find_png(eds_dir, element))
    return invert_jet_colormap(rgb)[header_trim_px:, :].astype(np.float32)


def load_bse_1x(eds_dir: Path, header_trim_px: int) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    arr = np.asarray(Image.open(find_png(eds_dir, "SEM")).convert("L"), dtype=np.float32) / 255.0
    return arr[header_trim_px:, :]
