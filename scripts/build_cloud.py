"""Build the point-cloud assets: morphological descriptors + thumbnail atlases.

Covers the eleven nucleus sources available locally: the ten harmonised crop
sets of the neighbouring `cellf-supervised` repository, plus VoCell's own
RESTORE volumes reduced to their Z projection. 2.6 M nuclei in one global
index space, CODEX first, RESTORE last.

Outputs, all regenerated per machine and all under gitignored paths:

  data/cloud/features.npz
      The canonical descriptor table, for analysis. The app never reads it.

  src/components/nuclei_cloud/{cols,atlas}/ + meta.json + classes.bin
      What the JS component fetches over HTTP. One uint16 column per
      descriptor (plus the two principal components), thumbnails packed 8x8
      per 256x256 PNG, and the metadata describing both. This directory is the
      only tree Streamlit serves to a custom component's iframe.

Nothing is rescaled to a common pixel size. The sources were never harmonised
either -- `rescale_images()` is commented out in the cellf-supervised build
notebook -- and it shows: median nucleus diameter runs from 13 px on
HelaCytoNuc to 42 px on AitslabBioimaging1. Keeping that is the point: any
axis involving a size separates the datasets, and that spread is the domain
gap a model trained on one and shown another has to cross.

Usage:
    python scripts/build_cloud.py                  # everything, ~10 min
    python scripts/build_cloud.py --limit 5000     # 5000 per source, smoke run
    python scripts/build_cloud.py --meta-only      # rewrite metadata only
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cloud  # noqa: E402
import data  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "data", "cloud")
OUT_FEATURES = os.path.join(OUT_DIR, "features.npz")
COMPONENT = os.path.join(ROOT, "src", "components", "nuclei_cloud")
ATLAS_DIR = os.path.join(COMPONENT, "atlas")
COLS_DIR = os.path.join(COMPONENT, "cols")

THUMB = 32           # thumbnail edge in pixels
THUMB_CROP = 44      # centre crop fed to the thumbnail, out of 64
PER_ROW = 8          # thumbnails per atlas row -> 64 per 256x256 sheet
PER_ATLAS = PER_ROW * PER_ROW
SHARD = 1000         # atlas sheets per subdirectory
BATCH = 2048

# Sources, in global index order. Which ones carry labels is data.py's call
# (data.LABEL_COLUMN); the other eight contribute a single pseudo-class named
# after the dataset, so the legend has the same shape everywhere. `hue` is the
# dataset's colour family, `grey` puts RESTORE outside the wheel since it is a
# different acquisition rather than another cell family.
SOURCES: List[Dict] = [
    {"name": "CODEX", "hue": 0.58,
     "note": "CRC, CODEX, Hoechst — ~0,376 µm/px"},
    {"name": "HPA", "hue": 0.33,
     "note": "Human Protein Atlas, lignées cellulaires"},
    {"name": "BBBC051", "hue": 0.09,
     "note": "rein humain, crops natifs 32² — ~0,5 µm/px"},
    {"name": "TissueNet", "hue": 0.78, "note": "DAPI, multi-tissus"},
    {"name": "HelaCytoNuc", "hue": 0.03, "note": "HeLa, DAPI"},
    {"name": "DSB2018", "hue": 0.13, "note": "Data Science Bowl 2018"},
    {"name": "NuInSeg", "hue": 0.47, "note": "H&E, multi-organes"},
    {"name": "S-BSST265", "hue": 0.88, "note": "DAPI"},
    {"name": "NucleusSegData", "hue": 0.68, "note": "Huh7 / HepG2"},
    {"name": "AitslabBioimaging1", "hue": 0.21, "note": "U2OS, Hoechst"},
    {"name": "RESTORE", "hue": 0.0, "grey": True,
     "note": "confocal 3D, canal DAPI — projection Z, ~0,15 µm/px"},
]

FEATURE_NAMES = cloud.FEATURE_NAMES

_YY, _XX = np.meshgrid(np.arange(64, dtype=np.float32),
                       np.arange(64, dtype=np.float32), indexing="ij")
EPS = np.float32(1e-6)


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------

def describe(batch: np.ndarray) -> np.ndarray:
    """Morphological descriptors for a batch of 2D nuclei.

    `batch` is (b, 64, 64) float32 in [0, 1]; returns (b, len(FEATURE_NAMES)).

    The foreground mask uses a per-image relative threshold (35% of the 99th
    percentile, floored at 0.05) rather than a global one: the eleven sources
    have very different dynamic ranges -- median peak intensity runs from 0.33
    to 1.00 -- and an absolute cut would end up measuring exposure instead of
    shape.
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

    # A crop can be empty or all but empty. Its shape descriptors are not
    # small, they are meaningless -- a two-pixel mask has an axis ratio in the
    # thousands -- so they are pinned to neutral values instead of being
    # allowed to set the scale of every axis in the cloud.
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


# --------------------------------------------------------------------------
# Thumbnails
# --------------------------------------------------------------------------

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
      99.5% of nuclei (measured across the sources) and buys 1.45x of scale.
    - a per-thumbnail percentile stretch (p0.5 / p99.8) rather than min-max,
      so one hot pixel cannot set the ceiling and black out the nucleus.
    - gamma 0.65. CODEX and TissueNet nuclei are dim and thin; linear tone
      makes them a dark smudge, which is faithful and useless.
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
    """Packs thumbnails into fixed-size PNG sheets, in global index order.

    Sheets are deliberately small. Access is random, not sequential: the
    points a zoomed view needs are neighbours in the projection, which says
    nothing about where they sit in the global index, so a zoomed region
    touches roughly as many sheets as it has points. With 1024 thumbnails per
    sheet, showing 322 nuclei pulled 144 files of 240 kB.
    """

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.sheet = np.zeros((PER_ROW * THUMB, PER_ROW * THUMB), dtype=np.uint8)
        self.n = 0

    def add(self, thumbs: np.ndarray) -> None:
        for t in thumbs:
            slot = self.n % PER_ATLAS
            r, c = divmod(slot, PER_ROW)
            self.sheet[r * THUMB:(r + 1) * THUMB, c * THUMB:(c + 1) * THUMB] = t
            self.n += 1
            if self.n % PER_ATLAS == 0:
                self._flush(self.n // PER_ATLAS - 1)

    def _flush(self, k: int) -> None:
        shard = os.path.join(self.out_dir, f"{k // SHARD:03d}")
        os.makedirs(shard, exist_ok=True)
        Image.fromarray(self.sheet).save(
            os.path.join(shard, f"atlas_{k:06d}.png"), optimize=False)
        self.sheet[:] = 0

    def close(self) -> None:
        if self.n % PER_ATLAS:
            self._flush(self.n // PER_ATLAS)


# --------------------------------------------------------------------------
# Labels and colours
# --------------------------------------------------------------------------

def read_classes(name: str, n: int) -> Optional[List[str]]:
    """Per-crop class string, aligned on the crops.npy row order, or None.

    Delegates to data.load_classes so that the alignment rules live in one
    place: CODEX ships a `crop_index` column and is keyed by it, while HPA and
    BBBC051 have none and their rows are in crop order.
    """
    classes = data.load_classes(name)
    if classes is None:
        return None
    return [classes.get(i, "non étiqueté") for i in range(n)]


def hexa(h: float, s: float, l: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def class_colors(hue: float, k: int, grey: bool = False) -> List[str]:
    """One colour per class, all inside the dataset's own hue family.

    Colouring by class has to stay readable next to colouring by dataset: a
    point keeps roughly the same look under both, so the dataset groups stay
    recognisable even when the legend is expanded. Within a family, classes
    are spread over lightness and a narrow hue band rather than scattered
    across the wheel.
    """
    if grey:
        return [hexa(0.0, 0.0, 0.92 - 0.30 * (i / max(k - 1, 1))) for i in range(k)]
    if k == 1:
        return [hexa(hue, 0.62, 0.62)]
    return [hexa(hue + 0.055 * (i / (k - 1) - 0.5),
                 0.72 - 0.30 * (i % 3) / 2.0,
                 0.42 + 0.34 * (i / (k - 1)))
            for i in range(k)]


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def load_source(src: Dict, limit: Optional[int]):
    """Yield batches of (b, 64, 64) float32 in [0, 1], plus the source's size."""
    name = src["name"]
    if name == "RESTORE":
        arr = data.load_restore_nuclei()
        n = len(arr) if limit is None else min(limit, len(arr))

        def gen():
            for s in range(0, n, BATCH):
                e = min(s + BATCH, n)
                vol = np.asarray(arr[s:e, ..., 0], dtype=np.float32) / 255.0
                yield vol.max(axis=1)   # Z projection: what a 2D acquisition sees
        return gen, n

    arr = data.load_crops(name)
    n = len(arr) if limit is None else min(limit, len(arr))
    side = arr.shape[1]

    def gen():
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            b = np.asarray(arr[s:e], dtype=np.float32) / 255.0
            if side < 64:
                # BBBC051 ships native 32x32 crops and was never rescaled.
                # Padding keeps its pixel size; resizing would double the
                # apparent diameter of every nucleus and invent a difference.
                o = (64 - side) // 2
                out = np.zeros((b.shape[0], 64, 64), dtype=np.float32)
                out[:, o:o + side, o:o + side] = b
                b = out
            yield b
    return gen, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap every source at N nuclei (smoke test).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Restrict to these source names.")
    ap.add_argument("--meta-only", action="store_true",
                    help="Rewrite metadata and columns from an existing "
                         "features.npz, without recomputing descriptors.")
    args = ap.parse_args()

    if args.meta_only:
        write_component_assets()
        return

    sources = [s for s in SOURCES
               if args.only is None or s["name"] in args.only]
    os.makedirs(OUT_DIR, exist_ok=True)
    for d in (ATLAS_DIR, COLS_DIR):
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    os.remove(os.path.join(root, f))
    atlas = AtlasWriter(ATLAS_DIR)

    feats: List[np.ndarray] = []
    class_ids: List[np.ndarray] = []
    class_names: List[str] = []
    manifest: List[Dict] = []
    start = 0
    t0 = time.time()

    for src in sources:
        gen, n = load_source(src, args.limit)
        if n == 0:
            continue
        labels = read_classes(src["name"], n)
        vocab = sorted(set(labels)) if labels else [src["name"]]
        base = len(class_names)
        class_names.extend(vocab)
        lookup = {name: base + i for i, name in enumerate(vocab)}
        ids = (np.array([lookup[x] for x in labels], dtype=np.int16)
               if labels else np.full(n, base, dtype=np.int16))
        class_ids.append(ids)

        done = 0
        for batch in gen():
            feats.append(describe(batch))
            atlas.add(to_thumbs(batch))
            done += len(batch)
        print(f"{src['name']:<20s} {done:>9d}   ({time.time() - t0:.0f}s)",
              flush=True)

        manifest.append({
            "name": src["name"], "start": start, "count": n,
            "labeled": src["name"] in data.LABEL_COLUMN,
            "hue": src["hue"], "grey": bool(src.get("grey")),
            "note": src.get("note", ""),
            "class_ids": [base + i for i in range(len(vocab))],
        })
        start += n

    atlas.close()

    values = np.concatenate(feats, axis=0)
    np.savez(OUT_FEATURES,
             values=values,
             names=np.array(FEATURE_NAMES),
             class_idx=np.concatenate(class_ids),
             class_names=np.array(class_names),
             manifest=np.array(json.dumps(manifest)))

    write_component_assets()
    print(f"\n{values.shape[0]} noyaux · {len(sources)} sources · "
          f"{values.shape[1]} descripteurs · {time.time() - t0:.0f}s")


def write_component_assets() -> None:
    """Emit everything the JS component fetches over HTTP.

    At 2.6 M points, pushing positions through component args would be 10 MB
    on the websocket every time an axis changes. Each descriptor is written
    once as a normalised uint16 column instead; the component fetches the two
    it needs and the browser caches them, so Python sends only the names.
    """
    d = np.load(OUT_FEATURES, allow_pickle=False)
    values, class_idx = d["values"], d["class_idx"]
    names = [str(x) for x in d["names"]]
    class_names = [str(x) for x in d["class_names"]]
    manifest = json.loads(str(d["manifest"]))
    n = len(values)

    os.makedirs(COLS_DIR, exist_ok=True)
    class_idx.astype(np.uint8).tofile(os.path.join(COMPONENT, "classes.bin"))

    jitter = {}
    columns = list(zip(names, values.T))
    scores, ratio, loadings = cloud.pca_2d(values)
    columns += [("pca1", scores[:, 0]), ("pca2", scores[:, 1])]

    for key, col in columns:
        scaled = cloud.unit_scale(col)
        (np.clip(scaled, 0.0, 1.0) * 65535.0).astype(np.uint16).tofile(
            os.path.join(COLS_DIR, f"{key}.bin"))
        jitter[key] = cloud.jitter_step(scaled)

    counts = np.bincount(class_idx, minlength=len(class_names)).tolist()
    for entry in manifest:
        ids = entry.pop("class_ids")
        colors = class_colors(entry["hue"], len(ids), entry["grey"])
        entry["color"] = class_colors(entry["hue"], 1, entry["grey"])[0]
        entry["classes"] = [
            {"id": cid, "name": class_names[cid], "count": counts[cid],
             "color": colors[i]}
            for i, cid in enumerate(ids)]
        entry.pop("hue"); entry.pop("grey")

    meta = {
        "build": int(time.time()),
        "count": n,
        "thumb": THUMB, "per_row": PER_ROW, "per_atlas": PER_ATLAS,
        "shard": SHARD,
        "features": names,
        "labels": {k: cloud.label(k) for k in names},
        "axes": {
            "pca1": f"CP1 — {ratio[0]:.0%} de variance · "
                    f"{cloud.dominant(loadings[0], names)}",
            "pca2": f"CP2 — {ratio[1]:.0%} de variance · "
                    f"{cloud.dominant(loadings[1], names)}",
        },
        "jitter": jitter,
        "datasets": manifest,
    }
    with open(os.path.join(COMPONENT, "meta.json"), "w") as f:
        json.dump(meta, f)

    sheets = sum(len(fs) for _, _, fs in os.walk(ATLAS_DIR))
    mb = sum(os.path.getsize(os.path.join(r, f))
             for r, _, fs in os.walk(ATLAS_DIR) for f in fs) / 1e6
    print(f"{OUT_FEATURES} ({os.path.getsize(OUT_FEATURES) / 1e6:.0f} Mo)")
    print(f"{COLS_DIR} · {len(columns)} colonnes "
          f"({len(columns) * n * 2 / 1e6:.0f} Mo)")
    print(f"{ATLAS_DIR} · {sheets} atlas ({mb:.0f} Mo)")


if __name__ == "__main__":
    main()
