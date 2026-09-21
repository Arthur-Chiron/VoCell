"""Build the point-cloud assets: morphological descriptors + thumbnail atlases.

Two outputs, both regenerated per machine (they live under gitignored paths):

  data/cloud/features.npz
      One row per nucleus, CODEX first then RESTORE, in a single global index
      space. Holds the descriptor table plus the mapping back to
      (dataset, local index) that the rest of the app addresses nuclei by.

  src/components/nuclei_cloud/atlas/atlas_NNNN.png (+ atlas.json)
      32x32 greyscale thumbnails packed 32 per row into 1024x1024 PNGs. They
      live inside the component directory because that is the only tree
      Streamlit serves over HTTP to a custom component's iframe, which fetches
      an atlas on demand when the user zooms into a region.

Descriptors are computed in raw pixels, NOT in microns: CODEX is ~0.377 um/px
and RESTORE ~0.15 um/px, and keeping them un-harmonised is deliberate -- the
domain gap between the two acquisitions is part of what the cloud shows.

Usage:
    python scripts/build_cloud.py                      # both datasets
    python scripts/build_cloud.py --codex-limit 20000  # quick smoke run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import CLASS_MAPPING  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODEX_CROPS = os.path.join(ROOT, "data", "CODEX", "crops.npy")
CODEX_META = os.path.join(ROOT, "data", "CODEX", "crop_metadata.csv")
RESTORE_NUCLEI = os.path.join(ROOT, "data", "RESTORE", "nuclei.npy")
OUT_DIR = os.path.join(ROOT, "data", "cloud")
OUT_FEATURES = os.path.join(OUT_DIR, "features.npz")
OUT_ATLAS_DIR = os.path.join(ROOT, "src", "components", "nuclei_cloud", "atlas")

THUMB = 32           # thumbnail edge in pixels
THUMB_CROP = 44      # centre crop fed to the thumbnail, out of 64
# Atlas sheets are deliberately small (8x8 thumbnails, 256x256 px). Access is
# random, not sequential: the points a zoomed view needs are neighbours in the
# projection, which says nothing about where they sit in the global index, so
# a zoomed region touches roughly as many sheets as it has points. Big sheets
# meant fetching and decoding megabytes to show a few dozen nuclei.
PER_ROW = 8
PER_ATLAS = PER_ROW * PER_ROW
BATCH = 2048

# Descriptor keys, in column order. The French display labels live in
# src/cloud.py, next to the code that renders them.
FEATURE_NAMES = [
    "area",
    "elongation",
    "mean_intensity",
    "total_intensity",
    "contrast",
    "sharpness",
    "concentration",
    "fill",
    "cv_intensity",
    "off_center",
]

# Qualitative palette, read on the component's dark background. RESTORE is
# white on purpose: it is a different acquisition, not another cell family,
# and it should be impossible to mistake for one.
PALETTE = [
    "#f2c94c", "#4c9aff", "#b57edc", "#56c596", "#ff8a5b", "#d98cb3",
    "#6fd3e0", "#ef5f6b", "#a3e635", "#8b93a7", "#c98d5a", "#7b61ff",
    "#35b4e8", "#e879c7", "#38d9a9", "#ffffff",
]

_YY, _XX = np.meshgrid(np.arange(64, dtype=np.float32),
                       np.arange(64, dtype=np.float32), indexing="ij")
EPS = np.float32(1e-6)


def describe(batch: np.ndarray) -> np.ndarray:
    """Morphological descriptors for a batch of 2D nuclei.

    `batch` is (b, 64, 64) float32 in [0, 1]; returns (b, len(FEATURE_NAMES)).

    The foreground mask uses a per-image relative threshold (35% of the 99th
    percentile, floored at 0.05) rather than a global one: CODEX and RESTORE
    have very different dynamic ranges, and an absolute cut would end up
    measuring exposure instead of shape.
    """
    b = batch.shape[0]
    flat = batch.reshape(b, -1)

    pct = np.percentile(flat, [2.0, 98.0, 99.0], axis=1).astype(np.float32)
    p2, p98, p99 = pct[0], pct[1], pct[2]
    thr = np.maximum(np.float32(0.35) * p99, np.float32(0.05))
    mask = batch >= thr[:, None, None]
    m = mask.astype(np.float32)

    area = m.sum(axis=(1, 2))
    norm = np.maximum(area, EPS)

    # Mask centroid and second moments -> elongation from the axis ratio.
    cy = (m * _YY).sum(axis=(1, 2)) / norm
    cx = (m * _XX).sum(axis=(1, 2)) / norm
    dy = _YY[None] - cy[:, None, None]
    dx = _XX[None] - cx[:, None, None]
    mu20 = (m * dx * dx).sum(axis=(1, 2)) / norm
    mu02 = (m * dy * dy).sum(axis=(1, 2)) / norm
    mu11 = (m * dx * dy).sum(axis=(1, 2)) / norm
    tr = mu20 + mu02
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - (mu20 * mu02 - mu11 * mu11), 0.0))
    lam1 = tr / 2.0 + disc
    lam2 = np.maximum(tr / 2.0 - disc, EPS)
    elongation = np.sqrt(lam1 / lam2)

    masked = batch * m
    total_intensity = batch.sum(axis=(1, 2))
    mean_intensity = masked.sum(axis=(1, 2)) / norm
    sq = (masked * masked).sum(axis=(1, 2)) / norm
    cv_intensity = np.sqrt(np.maximum(sq - mean_intensity ** 2, 0.0)) / np.maximum(
        mean_intensity, EPS)
    contrast = p98 - p2

    # Mean absolute gradient over the whole image, normalised by mean
    # intensity: a dimensionless "how crisp is the edge" number that does not
    # depend on how bright the acquisition was.
    gy = np.abs(batch[:, 1:, :] - batch[:, :-1, :]).mean(axis=(1, 2))
    gx = np.abs(batch[:, :, 1:] - batch[:, :, :-1]).mean(axis=(1, 2))
    sharpness = (gy + gx) / np.maximum(mean_intensity, np.float32(0.02))

    # Fraction of the signal inside half the equivalent radius: separates a
    # nucleus with a dense core from one with an even, diffuse texture.
    r_eq = np.sqrt(norm / np.pi)
    r2 = dy * dy + dx * dx
    inner = (r2 <= (0.5 * r_eq)[:, None, None] ** 2).astype(np.float32)
    concentration = (batch * inner).sum(axis=(1, 2)) / np.maximum(
        total_intensity, EPS)

    # Mask area over its bounding box: how convex / how ragged the outline is.
    rows_any = mask.any(axis=2)
    cols_any = mask.any(axis=1)
    idx = np.arange(64, dtype=np.float32)
    big = np.float32(1e6)
    y0 = np.where(rows_any, idx[None, :], big).min(axis=1)
    y1 = np.where(rows_any, idx[None, :], -big).max(axis=1)
    x0 = np.where(cols_any, idx[None, :], big).min(axis=1)
    x1 = np.where(cols_any, idx[None, :], -big).max(axis=1)
    bbox = np.maximum((y1 - y0 + 1.0) * (x1 - x0 + 1.0), EPS)
    fill = np.where(area > 0, area / bbox, 0.0)

    off_center = np.sqrt((cy - 31.5) ** 2 + (cx - 31.5) ** 2)

    # A handful of CODEX crops are empty or all but empty. Their shape
    # descriptors are not small, they are meaningless -- a two-pixel mask has
    # an axis ratio in the thousands -- so they are pinned to neutral values
    # instead of being allowed to set the scale of every axis in the cloud.
    valid = area >= 4.0
    elongation = np.where(valid, np.minimum(elongation, 20.0), 1.0)
    sharpness = np.where(valid, sharpness, 0.0)
    concentration = np.where(valid, concentration, 0.0)
    fill = np.where(valid, fill, 0.0)
    cv_intensity = np.where(valid, cv_intensity, 0.0)
    off_center = np.where(valid, off_center, 0.0)

    out = np.stack([area, elongation, mean_intensity, total_intensity,
                    contrast, sharpness, concentration, fill,
                    cv_intensity, off_center], axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _area_matrix(src: int, dst: int) -> np.ndarray:
    """Box-filter resampling matrix, (dst, src). Separable, so one suffices."""
    w = np.zeros((dst, src), dtype=np.float32)
    step = src / dst
    for i in range(dst):
        a, b = i * step, (i + 1) * step
        for j in range(int(np.floor(a)), int(np.ceil(b))):
            w[i, j] = min(b, j + 1) - max(a, j)
    return w / w.sum(axis=1, keepdims=True)


_RESIZE = _area_matrix(THUMB_CROP, THUMB)
_OFF = (64 - THUMB_CROP) // 2


def to_thumbs(batch: np.ndarray) -> np.ndarray:
    """(b, 64, 64) float32 in [0, 1] -> (b, 32, 32) uint8 thumbnails.

    Three things happen, all of them about legibility at 32 px, and all of
    them applied identically to every nucleus so that comparing two
    thumbnails still means something:

    - a fixed 44x44 centre crop. Nuclei sit in the middle of a 64x64 box with
      a lot of empty field around them; the crop keeps 99% of the signal for
      99.5% of nuclei (measured on both datasets) and buys 1.45x of scale.
    - a per-thumbnail percentile stretch (p0.5 / p99.8) rather than min-max,
      so one hot pixel cannot set the ceiling and black out the nucleus.
    - gamma 0.65. CODEX nuclei are dim and thin; linear tone makes them a
      dark smudge, which is faithful and useless.
    """
    crop = batch[:, _OFF:_OFF + THUMB_CROP, _OFF:_OFF + THUMB_CROP]
    small = np.einsum("ij,bjk,lk->bil", _RESIZE, crop, _RESIZE, optimize=True)

    flat = small.reshape(small.shape[0], -1)
    lo, hi = np.percentile(flat, [0.5, 99.8], axis=1).astype(np.float32)
    scaled = (small - lo[:, None, None]) / np.maximum(
        (hi - lo)[:, None, None], EPS)
    scaled = np.clip(scaled, 0.0, 1.0) ** np.float32(0.65)
    return (scaled * 255.0).astype(np.uint8)


class AtlasWriter:
    """Packs thumbnails into fixed-size PNG sheets, in global index order."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        for stale in os.listdir(out_dir):
            if stale.startswith("atlas_") and stale.endswith(".png"):
                os.remove(os.path.join(out_dir, stale))
        self.sheet = np.zeros((PER_ROW * THUMB, PER_ROW * THUMB), dtype=np.uint8)
        self.n = 0

    def add(self, thumbs: np.ndarray) -> None:
        for t in thumbs:
            slot = self.n % PER_ATLAS
            r, c = divmod(slot, PER_ROW)
            self.sheet[r * THUMB:(r + 1) * THUMB,
                       c * THUMB:(c + 1) * THUMB] = t
            self.n += 1
            if self.n % PER_ATLAS == 0:
                self._flush(self.n // PER_ATLAS - 1)

    def _flush(self, sheet_idx: int) -> None:
        path = os.path.join(self.out_dir, f"atlas_{sheet_idx:04d}.png")
        Image.fromarray(self.sheet).save(path, optimize=True)
        self.sheet[:] = 0

    def close(self) -> None:
        if self.n % PER_ATLAS:
            self._flush(self.n // PER_ATLAS)
        with open(os.path.join(self.out_dir, "atlas.json"), "w") as f:
            json.dump({"thumb": THUMB, "per_row": PER_ROW,
                       "per_atlas": PER_ATLAS, "count": self.n}, f)


def codex_classes(n: int) -> tuple[np.ndarray, list[str]]:
    """Per-crop consolidated class index, aligned on the crops.npy row order."""
    raw = ["Inconnu"] * n
    with open(CODEX_META, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                i = int(row["crop_index"])
            except (ValueError, KeyError):
                continue
            if 0 <= i < n:
                raw[i] = CLASS_MAPPING.get(row["classes"], row["classes"])
    names = sorted(set(raw))
    lookup = {name: i for i, name in enumerate(names)}
    return np.array([lookup[r] for r in raw], dtype=np.int16), names


def write_component_meta() -> None:
    """Emit the two files the JS component fetches once, from its own dir.

    Everything here is fixed for a given build, so it travels over plain HTTP
    (cached by the browser) instead of over the websocket on every rerun. Only
    the 2D positions are pushed as component args, and those are quantised.
    """
    d = np.load(OUT_FEATURES)
    class_names = [str(x) for x in d["class_names"]]
    n_codex = int((d["dataset"] == 0).sum())
    n_restore = int((d["dataset"] == 1).sum())

    d["class_idx"].astype(np.uint8).tofile(
        os.path.join(OUT_ATLAS_DIR, "classes.bin"))

    with open(os.path.join(OUT_ATLAS_DIR, "atlas.json")) as f:
        atlas_meta = json.load(f)

    palette = [PALETTE[i % len(PALETTE)] for i in range(len(class_names))]
    if class_names and class_names[-1].startswith("RESTORE"):
        palette[-1] = "#ffffff"

    atlas_meta.update({
        # Streamlit serves component files with Cache-Control: public, so a
        # rebuilt atlas would stay invisible behind the browser's copy. The
        # component fetches this file uncached and hangs this id off every
        # other URL, which makes a rebuild land and keeps the atlases
        # cacheable within a build.
        "build": int(time.time()),
        "n_codex": n_codex,
        "n_restore": n_restore,
        "class_names": class_names,
        "palette": palette,
    })
    with open(os.path.join(OUT_ATLAS_DIR, "meta.json"), "w") as f:
        json.dump(atlas_meta, f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codex-limit", type=int, default=None,
                    help="Only process the first N CODEX crops (smoke test).")
    ap.add_argument("--restore-limit", type=int, default=None)
    ap.add_argument("--meta-only", action="store_true",
                    help="Rewrite the component metadata from an existing "
                         "features.npz, without recomputing anything.")
    args = ap.parse_args()

    if args.meta_only:
        write_component_meta()
        print(f"{OUT_ATLAS_DIR}/meta.json + classes.bin r\u00e9\u00e9crits")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    atlas = AtlasWriter(OUT_ATLAS_DIR)
    feats: list[np.ndarray] = []
    t0 = time.time()

    # --- CODEX: natively 2D, used as-is.
    crops = np.load(CODEX_CROPS, mmap_mode="r")
    n_codex = len(crops) if args.codex_limit is None else min(args.codex_limit,
                                                              len(crops))
    for start in range(0, n_codex, BATCH):
        stop = min(start + BATCH, n_codex)
        b = np.asarray(crops[start:stop], dtype=np.float32) / 255.0
        feats.append(describe(b))
        atlas.add(to_thumbs(b))
        print(f"\rCODEX {stop}/{n_codex}", end="", flush=True)
    print(f"  ({time.time() - t0:.0f}s)")

    # --- RESTORE: a real stack, reduced by max projection along Z. That is
    # what a 2D acquisition of the same nucleus would give, so it is the fair
    # thing to put next to a CODEX crop -- same reasoning as data.get_source_2d.
    nuclei = np.load(RESTORE_NUCLEI, mmap_mode="r")
    n_restore = len(nuclei) if args.restore_limit is None else min(
        args.restore_limit, len(nuclei))
    for start in range(0, n_restore, BATCH):
        stop = min(start + BATCH, n_restore)
        vol = np.asarray(nuclei[start:stop, ..., 0], dtype=np.float32) / 255.0
        b = vol.max(axis=1)
        feats.append(describe(b))
        atlas.add(to_thumbs(b))
        print(f"\rRESTORE {stop}/{n_restore}", end="", flush=True)
    print(f"  ({time.time() - t0:.0f}s)")

    atlas.close()

    values = np.concatenate(feats, axis=0)
    dataset = np.concatenate([np.zeros(n_codex, np.uint8),
                              np.ones(n_restore, np.uint8)])
    local_idx = np.concatenate([np.arange(n_codex, dtype=np.int32),
                                np.arange(n_restore, dtype=np.int32)])

    codex_cls, codex_names = codex_classes(n_codex)
    class_names = codex_names + ["RESTORE (DAPI)"]
    class_idx = np.concatenate(
        [codex_cls, np.full(n_restore, len(codex_names), dtype=np.int16)])

    np.savez_compressed(
        OUT_FEATURES,
        values=values,
        names=np.array(FEATURE_NAMES),
        dataset=dataset,
        local_idx=local_idx,
        class_idx=class_idx,
        class_names=np.array(class_names),
    )

    write_component_meta()

    mb = sum(os.path.getsize(os.path.join(OUT_ATLAS_DIR, f))
             for f in os.listdir(OUT_ATLAS_DIR)) / 1e6
    print(f"\n{values.shape[0]} noyaux · {values.shape[1]} descripteurs")
    print(f"{OUT_FEATURES} ({os.path.getsize(OUT_FEATURES) / 1e6:.1f} Mo)")
    print(f"{OUT_ATLAS_DIR} · {atlas.n // PER_ATLAS + 1} atlas ({mb:.0f} Mo)")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
