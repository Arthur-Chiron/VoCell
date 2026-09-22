"""Build the VoCell wordmark out of real nuclei — one nucleus per letter.

Every one of the 2 633 390 nuclei of the eleven sources is compared, up to a
rotation, against the six glyphs of "VoCell". The winner of each letter is
drawn with its own pixels, at the angle that made it win.

The comparison is a silhouette IoU, not an image correlation: what makes a
nucleus look like a V is its outline, and the crops' non-zero support already
*is* the segmentation mask (see data.THRESHOLD_SUPPORT for the one source
where it is a threshold instead).

Both shapes are put in the same canonical frame before being compared —
centred on their centroid, scaled so that their RMS radius is R0. Scaling on
the RMS radius rather than on the bounding box is what makes the frame
rotation-covariant: a bounding box grows by up to sqrt(2) when a shape turns,
the RMS radius does not move at all. It also handles the aspect ratio for
free, since the scaling stays isotropic — which is the whole point for an `l`.

The template turns, the nucleus does not: rotating the 72 orientations of six
glyphs costs nothing, rotating 2.6 M nuclei costs everything. The scan is then
a single matrix product per batch, `nuclei @ templates.T`, which BLAS runs at
a speed no per-nucleus loop could approach.

Two passes:

  --search   the exhaustive scan, ~5 min. Keeps the best score of every
             nucleus for every letter, then re-examines the top few hundred
             at four times the resolution and rejects the ones whose mask is
             not a single nucleus (fragmented support, or cut by the edge of
             the crop). Writes data/logo/candidates.json.
  --render   reads that file and draws the wordmark. Seconds, no dataset
             needed beyond the six chosen crops.

Usage:
    python scripts/build_logo.py --search            # full scan
    python scripts/build_logo.py --search --limit 20000
    python scripts/build_logo.py --render
"""

from __future__ import annotations

import argparse
import base64
import io
import zlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_cloud  # noqa: E402
import data  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "data", "logo")
CANDIDATES = os.path.join(OUT_DIR, "candidates.json")
ASSETS = os.path.join(ROOT, "assets")
COMPONENT = os.path.join(ROOT, "src", "components", "logo")

WORD = "VoCell"
GLYPHS = "VoCel"              # `l` is searched once and used twice
FONT_PATH = "/System/Library/Fonts/Avenir Next.ttc"
FONT_INDEX = 0                # Bold: thick, geometric strokes, closer to a cell

S = 32                        # canonical frame edge, in pixels
R0 = 7.0                      # RMS radius every silhouette is scaled to
N_ROT = 72                    # orientations tried, 5 degrees apart
SUB = 2                       # subsamples per axis when resampling a mask
BATCH = 8192
MIN_AREA = 120.0              # a mask too small to carry a letter's shape

S_FINE = 64                   # second pass: same frame, four times the area
R0_FINE = 14.0
N_ROT_FINE = 360
TOP_K = 600                   # candidates per letter re-examined in pass two
MIN_LARGEST_CC = 0.90         # of the mask area, else it is not one nucleus
COUNTER_PENALTY = 0.5         # weight of "the letter's counters stay empty"

EPS = np.float32(1e-6)


# --------------------------------------------------------------------------
# Canonical frame
# --------------------------------------------------------------------------

def canonicalise(masks: np.ndarray, size: int = S, r0: float = R0,
                 frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(b, H, W) masks in [0, 1] -> (b, size, size) centred, RMS-scaled.

    `frame` imposes someone else's centre and radius instead of measuring the
    mask's own — which is how a glyph's counters are put in the glyph's frame
    rather than in a frame of their own.

    Returns the canonical frames, the mask areas and their RMS radii. The
    resampling is nearest-neighbour on a SUB x SUB grid per output pixel,
    which both antialiases the result and keeps the whole thing a handful of
    vectorised gathers — a per-nucleus affine warp would be 50x slower for a
    silhouette that is about to be blurred into 32 px anyway.
    """
    b, H, W = masks.shape
    flat = masks.reshape(b, -1)
    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    basis = np.stack([np.ones(H * W, np.float32),
                      np.repeat(ys, W), np.tile(xs, H),
                      np.repeat(ys * ys, W), np.tile(xs * xs, H)], axis=1)
    mom = flat @ basis
    area = np.maximum(mom[:, 0], EPS)
    cy, cx = mom[:, 1] / area, mom[:, 2] / area
    var = (mom[:, 3] / area - cy * cy) + (mom[:, 4] / area - cx * cx)
    rms = np.sqrt(np.maximum(var, EPS))
    if frame is not None:
        cy, cx, rms = frame

    # A zero border, so that clipping the sample coordinates reads empty space
    # instead of smearing the edge pixels outwards. Without it a glyph as thin
    # as an `l`, whose frame is far wider than the glyph itself, comes back as
    # a full canvas.
    padded = np.pad(masks, ((0, 0), (1, 1), (1, 1)))
    scale = (rms / np.float32(r0)).astype(np.float32)
    g = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    offs = (np.arange(SUB, dtype=np.float32) + 0.5) / SUB - 0.5
    rows = np.arange(b)[:, None, None]
    out = np.zeros((b, size, size), np.float32)
    for dy in offs:
        yy = np.rint(cy[:, None] + (g + dy)[None, :] * scale[:, None]) + 1.0
        yi = np.clip(yy, 0, H + 1).astype(np.int32)[:, :, None]
        for dx in offs:
            xx = np.rint(cx[:, None] + (g + dx)[None, :] * scale[:, None]) + 1.0
            xi = np.clip(xx, 0, W + 1).astype(np.int32)[:, None, :]
            out += padded[rows, yi, xi]
    out /= SUB * SUB
    return out, mom[:, 0], rms


# --------------------------------------------------------------------------
# Letter templates
# --------------------------------------------------------------------------

def counter_mask(glyph: np.ndarray) -> np.ndarray:
    """What a letter needs *empty* to still be that letter.

    The convex hull of the glyph, minus the glyph: the hole of an `o`, the
    notch of a `V`, the opening of a `C`. Plain IoU under-weights exactly
    these — a filled wedge covers every pixel of a V and pays only for the
    notch, so it outranks any nucleus actually shaped like a V. Scoring the
    counters separately is what puts the eye and the number back in
    agreement.
    """
    ys, xs = np.nonzero(glyph > 0.5)
    hull = ConvexHull(np.stack([xs, ys], 1))
    poly = [tuple(map(float, hull.points[v])) for v in hull.vertices]
    img = Image.new("L", (glyph.shape[1], glyph.shape[0]), 0)
    ImageDraw.Draw(img).polygon(poly, fill=255)
    filled = (np.asarray(img) > 127).astype(np.float32)
    return np.maximum(filled - (glyph > 0.5), 0.0)


def glyph_mask(ch: str, px: int = 320) -> np.ndarray:
    """One character, alone, as a tight binary float32 image."""
    font = ImageFont.truetype(FONT_PATH, px, index=FONT_INDEX)
    img = Image.new("L", (px * 3, px * 3), 0)
    ImageDraw.Draw(img).text((px, px), ch, fill=255, font=font)
    arr = np.asarray(img)
    ys, xs = np.nonzero(arr > 127)
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return (arr > 127).astype(np.float32)


class Templates:
    """Every glyph at every orientation, both chiralities, plus its counters.

    A nucleus has no handedness — nothing distinguishes a crop from its
    mirror image — so the mirrored templates are fair game, and only the
    letter has to come out the right way round.

    The counters travel in the glyph's own frame, not in one measured on
    them: they are the same shape seen from the same place, so that a
    nucleus' overlap with both can be read off the one canonical frame.
    """

    def __init__(self, glyphs: str, size: int, r0: float, n_rot: int):
        rows: List[np.ndarray] = []
        counters: List[np.ndarray] = []
        which: List[int] = []
        pose: List[Tuple[float, int]] = []
        for gi, ch in enumerate(glyphs):
            g = glyph_mask(ch)
            pair = np.stack([g, counter_mask(g)])
            base = [Image.fromarray((a * 255).astype(np.uint8)) for a in pair]
            for k in range(n_rot):
                angle = 360.0 * k / n_rot
                rot = [np.asarray(b.rotate(angle, resample=Image.BILINEAR,
                                           expand=True), np.float32) / 255.0
                       for b in base]
                for mirror in (0, 1):
                    m = [a[:, ::-1] if mirror else a for a in rot]
                    canon, _, _ = canonicalise(m[0][None], size, r0)
                    cy, cx, rms = _frame_of(m[0])
                    ctr, _, _ = canonicalise(m[1][None], size, r0,
                                             (cy, cx, rms))
                    rows.append(canon[0].ravel())
                    counters.append(ctr[0].ravel())
                    which.append(gi)
                    pose.append((angle, mirror))
        self.glyph = np.ascontiguousarray(np.stack(rows))
        self.counter = np.ascontiguousarray(np.stack(counters))
        self.glyph_sum = self.glyph.sum(1)
        self.counter_sum = np.maximum(self.counter.sum(1), EPS)
        self.which = np.asarray(which, np.int32)
        self.pose = np.asarray(pose, np.float32)
        self.cols = [np.nonzero(self.which == g)[0] for g in range(len(glyphs))]

    def __len__(self) -> int:
        return len(self.glyph)

    def score(self, canon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(b, d) canonical frames -> (b, n) score and (b, n) plain IoU.

        Two matrix products and no branching: BLAS does in seconds what a
        per-nucleus rotation loop could not do in a day.
        """
        inter = canon @ self.glyph.T
        union = canon.sum(1)[:, None] + self.glyph_sum[None, :] - inter
        iou = inter / np.maximum(union, EPS)
        spill = (canon @ self.counter.T) / self.counter_sum[None, :]
        return iou - COUNTER_PENALTY * spill, iou


def _frame_of(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The centre and RMS radius canonicalise() measures, to impose elsewhere."""
    h, w = mask.shape
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    a = max(float(mask.sum()), 1e-6)
    cy, cx = float((mask * ys).sum() / a), float((mask * xs).sum() / a)
    var = float((mask * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum() / a)
    return (np.float32([cy]), np.float32([cx]), np.float32([max(var, 1e-6) ** 0.5]))


# --------------------------------------------------------------------------
# Pass one: every nucleus, every letter
# --------------------------------------------------------------------------

def scan(limit: Optional[int]) -> Dict:
    tmpl = Templates(GLYPHS, S, R0, N_ROT)
    print(f"{len(tmpl)} templates, {len(GLYPHS)} glyphs x {N_ROT} angles x 2")

    best: List[np.ndarray] = []      # per source: (n, n_glyphs) best IoU
    best_pose: List[np.ndarray] = []
    sizes: List[int] = []
    t0 = time.time()
    for src in build_cloud.SOURCES:
        gen, n = build_cloud.load_source(src, limit)
        sizes.append(n)
        sc = np.zeros((n, len(GLYPHS)), np.float32)
        po = np.zeros((n, len(GLYPHS)), np.int32)
        at = 0
        for batch in gen():
            masks = (batch > 0).astype(np.float32)
            canon, area, rms = canonicalise(masks)
            flat = canon.reshape(len(canon), -1)
            score, _ = tmpl.score(flat)
            bad = (area < MIN_AREA) | (rms < 2.0)
            for g, cols in enumerate(tmpl.cols):
                sub = score[:, cols]
                k = sub.argmax(1)
                sc[at:at + len(canon), g] = np.where(
                    bad, -1.0, sub[np.arange(len(canon)), k])
                po[at:at + len(canon), g] = cols[k]
            at += len(canon)
        best.append(sc)
        best_pose.append(po)
        print(f"  {src['name']:<20} {n:>9,}  {time.time() - t0:6.1f}s")

    return {"scores": best, "pose": best_pose, "sizes": sizes}


# --------------------------------------------------------------------------
# Pass two: the shortlist, at four times the resolution and with the checks
# --------------------------------------------------------------------------

def source_crop(name: str, idx: int) -> np.ndarray:
    """One nucleus as a 64x64 float image, whatever source it comes from."""
    if name == "RESTORE":
        vol = np.asarray(data.load_restore_nuclei()[idx, ..., 0], np.float32)
        return vol.max(axis=0) / 255.0
    return data.crop_2d(name, idx)


def inspect(name: str, idx: int) -> Optional[Dict]:
    """Reject anything whose mask is not one whole nucleus.

    Two ways a high IoU can be an artefact rather than a shape: a support
    broken into pieces, which HPA's intensity threshold produces by the
    hundred thousand and which can imitate any letter by accident, and a
    nucleus cut by the edge of its crop, whose outline is the crop's, not the
    cell's.
    """
    crop = source_crop(name, idx)
    mask = crop > 0
    area = float(mask.sum())
    if area < MIN_AREA:
        return None
    lab, k = ndi.label(mask)
    if k == 0:
        return None
    counts = np.bincount(lab.ravel())[1:]
    if counts.max() < MIN_LARGEST_CC * area:
        return None
    side = 32 if name == "BBBC051" else 64
    o = (64 - side) // 2
    inner = mask[o:o + side, o:o + side]
    if inner[0].any() or inner[-1].any() or inner[:, 0].any() or inner[:, -1].any():
        return None
    return {"crop": crop, "mask": mask.astype(np.float32), "area": area,
            "components": int(k)}


def refine(result: Dict) -> Dict[str, List[Dict]]:
    tmpl = Templates(GLYPHS, S_FINE, R0_FINE, N_ROT_FINE)
    names = [s["name"] for s in build_cloud.SOURCES]
    scores = np.concatenate(result["scores"])          # (N, n_glyphs)
    bounds = np.cumsum([0] + result["sizes"])

    out: Dict[str, List[Dict]] = {}
    for g, ch in enumerate(GLYPHS):
        order = np.argsort(-scores[:, g])[:TOP_K]
        kept: List[Dict] = []
        for gid in order:
            si = int(np.searchsorted(bounds, gid, side="right") - 1)
            idx = int(gid - bounds[si])
            info = inspect(names[si], idx)
            if info is None:
                continue
            canon, _, _ = canonicalise(info["mask"][None], S_FINE, R0_FINE)
            sc, iou = tmpl.score(canon.reshape(1, -1))
            cols = tmpl.cols[g]
            k = int(cols[sc[0][cols].argmax()])
            kept.append({"letter": ch, "dataset": names[si], "index": idx,
                         "score": float(sc[0][k]), "iou": float(iou[0][k]),
                         "coarse": float(scores[gid, g]),
                         "angle": float(tmpl.pose[k][0]),
                         "mirror": int(tmpl.pose[k][1]),
                         "area": info["area"], "components": info["components"]})
        kept.sort(key=lambda d: -d["score"])
        out[ch] = kept[:24]
        top = kept[0]
        print(f"  {ch}  score {top['score']:.3f} (IoU {top['iou']:.3f})  "
              f"{top['dataset']}#{top['index']}  {top['angle']:.0f}deg"
              f"{' mirrored' if top['mirror'] else ''}"
              f"   ({len(kept)}/{TOP_K} passed the checks)")
    return out


def do_search(limit: Optional[int]) -> None:
    t0 = time.time()
    result = scan(limit)
    print(f"scan done in {time.time() - t0:.0f}s, refining the shortlists")
    shortlists = refine(result)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CANDIDATES, "w", encoding="utf-8") as f:
        json.dump({"word": WORD, "glyphs": GLYPHS, "font": [FONT_PATH, FONT_INDEX],
                   "limit": limit, "shortlists": shortlists}, f, indent=1)
    print(f"wrote {CANDIDATES}")


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

SUPER = 2                     # supersampling factor of the drawing canvas
FONT_SIZE = 600               # glyph size on the supersampled canvas
EDGE = 3.0                    # alpha contrast: how crisp the silhouette gets
TRACKING = 0.05               # extra letter spacing, in em
DARK_BG = (14, 17, 23)        # the app's own background
PDF_WIDTH_PT = 480.0          # the wordmark's own width on the page, ~17 cm
PDF_MARGIN = 4                # voxels of clear space on every side
PDF_BLEED = 0.015             # of a voxel, to kill hairlines between fills
GRID_H = 40                   # voxels over the wordmark's ink height
VOXEL_ON = 0.5                # a voxel is lit or it is not, as in the 3D view


def stretched(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The nucleus' own greys, stretched the way the cloud's thumbnails are.

    p1/p99.5 rather than min/max so one hot pixel cannot set the ceiling, and
    gamma 0.65 because a CODEX or TissueNet nucleus is dim and thin and linear
    tone renders it as a dark smudge -- faithful and unreadable.
    """
    inside = crop[mask > 0]
    if inside.size == 0:
        return crop
    lo, hi = np.percentile(inside, [1.0, 99.5])
    v = np.clip((crop - lo) / max(hi - lo, 1e-6), 0.0, 1.0) ** 0.65
    return v.astype(np.float32)


def glyph_layout(word: str, size: int) -> Tuple[Tuple[int, int], List[Dict]]:
    """Where each letter of the word sits, and the box it has to fill.

    Each glyph is drawn alone at its place in the word, so everything
    measured here is already in the coordinates the nuclei get pasted into.

    The match is made on the RMS radius, but the placement is made on the
    bounding box, and the difference is not an inconsistency: the first is
    the only scale that survives a rotation, which is what a search over
    orientations needs, while the second is the one that puts six letters on
    a baseline at a common height, which is what a wordmark needs. The pose —
    angle and chirality — stays exactly the one that won.
    """
    font = ImageFont.truetype(FONT_PATH, size, index=FONT_INDEX)
    pad = size
    track = TRACKING * size
    width = int(font.getlength(word) + track * (len(word) - 1)) + 2 * pad
    height = 3 * size
    baseline = 2 * size
    out: List[Dict] = []
    for i, ch in enumerate(word):
        img = Image.new("L", (width, height), 0)
        x = pad + font.getlength(word[:i]) + track * i
        ImageDraw.Draw(img).text((x, baseline), ch, fill=255, font=font,
                                 anchor="ls")
        m = (np.asarray(img) > 127).astype(np.float32)
        ys, xs = np.nonzero(m)
        cy, cx = ys.mean(), xs.mean()
        rms = float(np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).mean()))
        out.append({"char": ch, "cy": float(cy), "cx": float(cx), "rms": rms,
                    "area": float(m.sum()), "mask": m,
                    "h": float(ys.max() - ys.min() + 1),
                    "w": float(xs.max() - xs.min() + 1),
                    "by": float((ys.max() + ys.min()) / 2),
                    "bx": float((xs.max() + xs.min()) / 2)})
    return (width, height), out


def posed_nucleus(pick: Dict, target_rms: float, box: Optional[Tuple[float, float]] = None
                  ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """The chosen crop, put back in the pose that won it the letter.

    The template was mirrored *after* being turned, so undoing it means
    mirroring first and turning back: a nucleus matched at `angle` becomes its
    letter under `rotate(-angle) . mirror`. Scale and centre are then read off
    the rotated mask rather than carried through the transform, which keeps
    the placement exact whatever the resampling did to the edges.
    """
    info = inspect(pick["dataset"], pick["index"])
    crop, mask = info["crop"], info["mask"]
    value = stretched(crop, mask)
    if pick["mirror"]:
        value, mask = value[:, ::-1], mask[:, ::-1]

    def turn(a: np.ndarray, resample: int) -> np.ndarray:
        im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
        r = im.rotate(-pick["angle"], resample=resample, expand=True)
        return np.asarray(r, np.float32) / 255.0

    value = turn(value, Image.BICUBIC)
    mask = turn(mask, Image.BILINEAR)
    ys, xs = np.nonzero(mask > 0.5)
    cy, cx = ys.mean(), xs.mean()
    rms = float(np.sqrt(((ys - cy) ** 2 + (xs - cx) ** 2).mean()))
    if box is None:
        k, ay, ax = target_rms / rms, cy, cx
    else:
        # Typographic fit: the geometric mean of the two box ratios, so that a
        # nucleus a little wider than its letter is not squashed, and the
        # anchor is the box centre — what sets a baseline and a cap height.
        nh = float(ys.max() - ys.min() + 1)
        nw = float(xs.max() - xs.min() + 1)
        k = float(np.sqrt((box[0] / nh) * (box[1] / nw)))
        ay, ax = (ys.max() + ys.min()) / 2.0, (xs.max() + xs.min()) / 2.0

    h, w = mask.shape
    dst = (max(int(round(w * k)), 1), max(int(round(h * k)), 1))
    grow = Image.LANCZOS if k > 1 else Image.BOX
    value = np.asarray(Image.fromarray((np.clip(value, 0, 1) * 255).astype(
        np.uint8)).resize(dst, grow), np.float32) / 255.0
    mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(
        dst, Image.BILINEAR), np.float32) / 255.0
    # A silhouette blown up 8x is a soft ramp; the letterform needs an edge.
    alpha = np.clip((mask - 0.5) * EDGE + 0.5, 0.0, 1.0)
    return value, alpha, (ay * k, ax * k)


def letter_planes(picks: List[Dict], word: str
                  ) -> Tuple[Tuple[int, int], List[Dict]]:
    """Every nucleus posed and placed on its letter, still one plane each.

    Kept apart rather than flattened because three things need them
    separately: the wordmark, the per-source tinting, and the voxel sprites
    the app's own header hit-tests letter by letter.
    """
    (W, H), layout = glyph_layout(word, FONT_SIZE)
    out: List[Dict] = []
    for pick, g in zip(picks, layout):
        v, a, (cy, cx) = posed_nucleus(pick, g["rms"], (g["h"], g["w"]))
        out.append({"pick": pick, "value": v, "alpha": a,
                    "y0": int(round(g["by"] - cy)),
                    "x0": int(round(g["bx"] - cx))})
    return (W, H), out


def _paste(dst: np.ndarray, acc: np.ndarray, plane: Dict,
           rgb: Optional[np.ndarray] = None) -> None:
    """Paint one letter's plane into the wordmark, nearest layer wins."""
    H, W = acc.shape
    y0, x0 = plane["y0"], plane["x0"]
    a = plane["alpha"]
    ys = slice(max(y0, 0), min(y0 + a.shape[0], H))
    xs = slice(max(x0, 0), min(x0 + a.shape[1], W))
    sy, sx = slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0)
    aa = a[sy, sx]
    src = plane["value"][sy, sx] if rgb is None else rgb[sy, sx]
    keep = aa > acc[ys, xs]
    dst[ys, xs] = np.where(keep if rgb is None else keep[..., None],
                           src, dst[ys, xs])
    acc[ys, xs] = np.maximum(acc[ys, xs], aa)


def compose(picks: List[Dict], word: str) -> Tuple[np.ndarray, np.ndarray]:
    """Drop every nucleus on its letter. Returns value and alpha planes."""
    (W, H), planes = letter_planes(picks, word)
    value = np.zeros((H, W), np.float32)
    alpha = np.zeros((H, W), np.float32)
    for plane in planes:
        _paste(value, alpha, plane)
    return value, alpha


def compose_tinted(picks: List[Dict], word: str
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """The same wordmark, each letter in the colour of its source."""
    (W, H), planes = letter_planes(picks, word)
    tint = np.zeros((H, W, 3), np.float32)
    acc = np.zeros((H, W), np.float32)
    for plane in planes:
        col = (source_colour(plane["pick"]["dataset"])[None, None]
               * (0.45 + 0.55 * plane["value"])[..., None])
        _paste(tint, acc, plane, col)
    return tint, acc


def tighten(planes: List[np.ndarray], alpha: np.ndarray, margin: float = 0.10
            ) -> Tuple[List[np.ndarray], np.ndarray]:
    ys, xs = np.nonzero(alpha > 0.02)
    m = int(margin * (ys.max() - ys.min() + 1))
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + 1 + m, alpha.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + 1 + m, alpha.shape[1])
    return [p[y0:y1, x0:x1] for p in planes], alpha[y0:y1, x0:x1]


def downsample(rgba: np.ndarray) -> Image.Image:
    img = Image.fromarray(rgba, "RGBA")
    return img.resize((img.width // SUPER, img.height // SUPER), Image.LANCZOS)


def save_rgba(path: str, rgb: np.ndarray, alpha: np.ndarray,
              background: Optional[Tuple[int, int, int]] = None) -> None:
    if background is not None:
        bg = np.asarray(background, np.float32) / 255.0
        rgb = rgb * alpha[..., None] + bg * (1.0 - alpha[..., None])
        alpha = np.ones_like(alpha)
    out = np.concatenate([np.clip(rgb, 0, 1) * 255.0,
                          np.clip(alpha, 0, 1)[..., None] * 255.0], axis=2)
    downsample(out.astype(np.uint8)).save(path)
    print(f"  {os.path.relpath(path, ROOT)}")


def source_colour(name: str) -> np.ndarray:
    src = next(s for s in build_cloud.SOURCES if s["name"] == name)
    h = build_cloud.hexa(src["hue"], 0.0 if src.get("grey") else 0.70, 0.66)
    return np.asarray([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255.0


def provenance_sheet(picks: List[Dict], total: int, path: str) -> None:
    """One row per letter: the letter, the nucleus that won it, where it is from.

    The outline of the glyph is drawn over the nucleus rather than beside it.
    It is the claim the score makes, left where it can be checked.
    """
    cell, gap, pad = 190, 24, 34
    title = ImageFont.truetype(FONT_PATH, 40, index=FONT_INDEX)
    head = ImageFont.truetype(FONT_PATH, 30, index=FONT_INDEX)
    body = ImageFont.truetype(FONT_PATH, 24, index=7)
    top = 118
    W = pad * 2 + cell * 2 + gap + 560
    H = top + len(picks) * (cell + gap) + pad + 46
    img = Image.new("RGB", (W, H), DARK_BG)
    draw = ImageDraw.Draw(img)
    draw.text((pad, pad), "VoCell — un noyau par lettre", font=title,
              fill=(238, 241, 248))
    count = f"{total:,}".replace(",", " ")
    draw.text((pad, pad + 52),
              f"le plus ressemblant, à rotation près, des {count} noyaux "
              f"des onze sources", font=body, fill=(138, 146, 162))

    r0 = cell / S * R0 * 0.80          # a little margin inside the tile
    for r, pick in enumerate(picks):
        y = top + r * (cell + gap)
        glyph = glyph_mask(pick["letter"])
        canon, _, _ = canonicalise(glyph[None], cell, r0)
        g = np.clip(canon[0], 0, 1)
        bg = np.asarray(DARK_BG, np.float32) / 255.0

        tile = bg[None, None] + (np.asarray([0.42, 0.45, 0.52], np.float32)
                                 - bg)[None, None] * g[..., None]
        img.paste(Image.fromarray((tile * 255).astype(np.uint8)), (pad, y))

        v, a, (ay, ax) = posed_nucleus(pick, r0)
        rgb = source_colour(pick["dataset"])[None, None] * (0.38 + 0.62 * v)[..., None]
        tile = np.tile(bg, (cell, cell, 1))
        ty, tx = int(cell / 2 - ay), int(cell / 2 - ax)
        ys = slice(max(ty, 0), min(ty + a.shape[0], cell))
        xs = slice(max(tx, 0), min(tx + a.shape[1], cell))
        sy, sx = slice(ys.start - ty, ys.stop - ty), slice(xs.start - tx, xs.stop - tx)
        aa = a[sy, sx][..., None]
        tile[ys, xs] = rgb[sy, sx] * aa + tile[ys, xs] * (1 - aa)
        # The glyph as a contour: an outline says where the letter is without
        # hiding the nucleus that is supposed to fill it.
        solid = g > 0.5
        edge = (solid ^ ndi.binary_erosion(solid, iterations=2))[..., None]
        tile = np.where(edge, 0.55 * tile + 0.45, tile)
        img.paste(Image.fromarray((np.clip(tile, 0, 1) * 255).astype(np.uint8)),
                  (pad + cell + gap, y))

        tx = pad + 2 * cell + 2 * gap
        colour = tuple(int(c * 255) for c in source_colour(pick["dataset"]))
        draw.text((tx, y + 34), f"« {pick['letter']} »", font=head,
                  fill=(238, 241, 248))
        draw.text((tx, y + 80), f"{pick['dataset']} #{pick['index']}",
                  font=body, fill=colour)
        draw.text((tx, y + 112),
                  f"score {pick['score']:.3f}   IoU {pick['iou']:.3f}",
                  font=body, fill=(150, 156, 170))
        draw.text((tx, y + 142),
                  f"rotation {pick['angle']:.0f}°"
                  + ("   miroir" if pick["mirror"] else ""),
                  font=body, fill=(150, 156, 170))
    draw.text((pad, H - 56),
              "score = IoU de la silhouette − ½ du remplissage des "
              "contrepoinçons de la lettre", font=body, fill=(110, 117, 132))
    img.save(path)
    print(f"  {os.path.relpath(path, ROOT)}")


def _png64(arr: np.ndarray, mode: str) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def voxel_letters(planes: List[Dict]) -> Tuple[Dict, List[Dict]]:
    """Turn the posed nuclei into one grid of lit-or-not voxels.

    The app renders a nucleus as cubes above a threshold; the wordmark says
    the same thing in two dimensions. Every letter is binned on the *same*
    grid, aligned on the whole wordmark rather than on each letter, so that
    one voxel is one size everywhere — six letters each quantised in their
    own box would come out with six different pixel sizes.

    A voxel's *presence* is a threshold, its *value* is the alpha-weighted
    mean of what falls in it: averaging the alpha instead would fringe every
    edge with half-lit cells and lose exactly the blockiness wanted. Colour
    is left to the caller, since the same grid is inked four different ways.
    """
    boxes = []
    for p in planes:
        ys, xs = np.nonzero(p["alpha"] > 0.02)
        boxes.append((p["y0"] + ys.min(), p["y0"] + ys.max() + 1,
                      p["x0"] + xs.min(), p["x0"] + xs.max() + 1))
    uy0, uy1 = min(b[0] for b in boxes), max(b[1] for b in boxes)
    ux0, ux1 = min(b[2] for b in boxes), max(b[3] for b in boxes)

    voxel = max(int(round((uy1 - uy0) / GRID_H)), 1)
    gh = int(np.ceil((uy1 - uy0) / voxel))
    gw = int(np.ceil((ux1 - ux0) / voxel))
    uh, uw = gh * voxel, gw * voxel

    letters: List[Dict] = []
    for p in planes:
        fa = np.zeros((uh, uw), np.float32)
        fv = np.zeros((uh, uw), np.float32)
        y0, x0 = p["y0"] - uy0, p["x0"] - ux0
        a, v = p["alpha"], p["value"]
        ys = slice(max(y0, 0), min(y0 + a.shape[0], uh))
        xs = slice(max(x0, 0), min(x0 + a.shape[1], uw))
        sy, sx = slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0)
        fa[ys, xs] = a[sy, sx]
        fv[ys, xs] = v[sy, sx]

        cells = (gh, voxel, gw, voxel)
        va = fa.reshape(cells).mean(axis=(1, 3))
        vv = (fv * fa).reshape(cells).mean(axis=(1, 3)) / np.maximum(va, EPS)
        on = va >= VOXEL_ON
        if not on.any():
            continue
        gy, gx = np.nonzero(on)
        y, x = int(gy.min()), int(gx.min())
        h, w = int(gy.max()) - y + 1, int(gx.max()) - x + 1
        letters.append({"pick": p["pick"], "x": x, "y": y, "w": w, "h": h,
                        "on": on[y:y + h, x:x + w],
                        "value": np.clip(vv[y:y + h, x:x + w], 0.0, 1.0)})
    return {"w": gw, "h": gh, "voxel": voxel}, letters


# Four ways to ink the same grid. Each takes the voxel values of one letter
# and its pick, and returns an (h, w, 3) array in [0, 1]. They are the ramps
# the smooth PNGs already use, so a voxel logo and its continuous twin carry
# the same tone.

def ink_mono(value: np.ndarray, pick: Dict) -> np.ndarray:
    return np.ones(value.shape + (3,), np.float32) * (0.55 + 0.45 * value)[..., None]


def ink_dark_on_white(value: np.ndarray, pick: Dict) -> np.ndarray:
    ink = np.asarray([0.05, 0.07, 0.11], np.float32)
    faint = np.asarray([0.30, 0.34, 0.43], np.float32)
    return faint[None, None] + (ink - faint)[None, None] * (value ** 0.7)[..., None]


def ink_source(value: np.ndarray, pick: Dict) -> np.ndarray:
    return (source_colour(pick["dataset"])[None, None]
            * (0.45 + 0.55 * value)[..., None])


def ink_source_on_white(value: np.ndarray, pick: Dict) -> np.ndarray:
    """The dataset colours, re-ramped for paper.

    On a dark page a dense voxel is the brightest one; on a white page that
    reads as the faintest. The ramp is inverted rather than the hue changed,
    so a letter keeps the colour that identifies its source.
    """
    return (source_colour(pick["dataset"])[None, None]
            * (0.75 - 0.40 * (value ** 0.7))[..., None])


def _pdf(width: float, height: float, content: bytes) -> bytes:
    """A one-page PDF around an already-built content stream.

    Hand-rolled rather than pulled from a library: the whole document is a
    catalogue, a page and a list of filled rectangles, and adding reportlab
    to requirements.txt to draw four thousand squares would be the larger
    change.
    """
    stream = zlib.compress(content, 9)
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        ("<</Type/Page/Parent 2 0 R/MediaBox[0 0 %.2f %.2f]"
         "/Contents 4 0 R/Resources<<>>>>" % (width, height)).encode(),
        b"<</Length %d/Filter/FlateDecode>>\nstream\n" % len(stream)
        + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, xref))
    return bytes(out)


def _rects(cells: List[Dict], ink, scale: float, height: float) -> bytes:
    """Every lit voxel as a filled rectangle, runs of one colour merged.

    PDF's origin is bottom-left and the grid's is top-left, hence the flip.

    Each letter is painted twice. First its whole silhouette, as one path in
    the letter's mean colour: subpaths of a single fill share one coverage
    computation, so that layer has no internal seams at all. Then the voxels
    on top, one rectangle each.

    The second layer needs the first. Two abutting antialiased fills do not
    compose to full coverage — each contributes a partial alpha to the shared
    edge pixel and `(1-a)(1-b)` of what is underneath survives, about a sixth
    at equal halves. Overlapping the rectangles reduces it but cannot close
    it, since the overlap that would be needed is half a device pixel and a
    vector file does not know what a device pixel is. So what survives is
    made to be the letter itself rather than the page: the seam becomes a
    shade, not a grid.
    """
    ops = bytearray()
    bleed = PDF_BLEED * scale
    for c in cells:
        rgb = np.clip(ink(c["value"], c["pick"]), 0.0, 1.0)
        on = c["on"]
        h, w = on.shape

        def place(row: int, col: int, run: int, grow: float) -> bytes:
            x = (PDF_MARGIN + c["x"] + col) * scale - grow
            y = height - (PDF_MARGIN + c["y"] + row + 1) * scale - grow
            return b"%.2f %.2f %.2f %.2f re\n" % (
                x, y, (run - col) * scale + 2 * grow, scale + 2 * grow)

        mean = rgb[on].mean(axis=0)
        ops += b"%.3f %.3f %.3f rg\n" % tuple(mean)
        for row in range(h):
            col = 0
            while col < w:
                if not on[row, col]:
                    col += 1
                    continue
                run = col + 1
                while run < w and on[row, run]:
                    run += 1
                ops += place(row, col, run, bleed)
                col = run
        ops += b"f\n"

        q = np.round(rgb * 255.0).astype(np.int16)      # merge on equal shades
        for row in range(h):
            col = 0
            while col < w:
                if not on[row, col]:
                    col += 1
                    continue
                run = col + 1
                while run < w and on[row, run] and (q[row, run] == q[row, col]).all():
                    run += 1
                r, g, b = rgb[row, col]
                ops += b"%.3f %.3f %.3f rg\n" % (r, g, b)
                ops += place(row, col, run, bleed) + b"f\n"
                col = run
    return bytes(ops)


def save_voxel_pdf(path: str, grid: Dict, cells: List[Dict], ink,
                   background: Optional[Tuple[int, int, int]] = None) -> None:
    scale = PDF_WIDTH_PT / grid["w"]
    width = (grid["w"] + 2 * PDF_MARGIN) * scale
    height = (grid["h"] + 2 * PDF_MARGIN) * scale
    ops = bytearray()
    if background is not None:
        r, g, b = (v / 255.0 for v in background)
        ops += b"%.3f %.3f %.3f rg 0 0 %.2f %.2f re f\n" % (r, g, b, width, height)
    ops += _rects(cells, ink, scale, height)
    with open(path, "wb") as f:
        f.write(_pdf(width, height, bytes(ops)))
    print(f"  {os.path.relpath(path, ROOT)}  ({os.path.getsize(path) / 1024:.0f} ko)")


def write_voxel_pdfs(picks: List[Dict]) -> None:
    """The wordmark as vectors: one filled rectangle per voxel.

    Not an upscaled bitmap. A logo ends up on a poster and in a slide, and a
    PDF of rectangles stays square-edged at any size, which is the whole
    point of a pixelated mark — a raster would either blur or leave the
    renderer to invent its own idea of a pixel.

    Six files rather than four, because PDF has no transparent background:
    a page with nothing painted under the letters *is* white in every
    viewer. So the two files meant to be dropped into a layout say so in
    their name (`_white`, `_ink`) and the four with a background of their
    own can be opened and read as they are. Opening `_white` alone shows an
    empty page; that is correct, it is white ink.
    """
    _, planes = letter_planes(picks, WORD)
    grid, cells = voxel_letters(planes)
    out = os.path.join(ASSETS, "logo_vocell_voxel")
    WHITE = (255, 255, 255)
    save_voxel_pdf(out + "_dark.pdf", grid, cells, ink_mono, DARK_BG)
    save_voxel_pdf(out + "_light.pdf", grid, cells, ink_dark_on_white, WHITE)
    save_voxel_pdf(out + "_sources.pdf", grid, cells, ink_source, DARK_BG)
    save_voxel_pdf(out + "_sources_light.pdf", grid, cells,
                   ink_source_on_white, WHITE)
    save_voxel_pdf(out + "_white.pdf", grid, cells, ink_mono)
    save_voxel_pdf(out + "_ink.pdf", grid, cells, ink_dark_on_white)


def write_component_assets(picks: List[Dict]) -> None:
    """What the app's header fetches: one JSON, sprites and all.

    The sprites ride inside the file as data URIs rather than sitting beside
    it. Streamlit serves a component's files as `Cache-Control: public`, so
    anything fetched separately needs a cache-busting token to ever be seen
    again after a rebuild; at 30 kB the whole set fits in the one document
    that is already read `no-store`.
    """
    _, planes = letter_planes(picks, WORD)
    grid, cells = voxel_letters(planes)
    letters = []
    for c in cells:
        pick = c["pick"]
        rgb = np.clip(ink_source(c["value"], pick), 0, 1)
        rgba = np.concatenate([rgb * 255.0, c["on"][..., None] * 255.0],
                              axis=2).astype(np.uint8)
        # No thumbnail and no mask here: hovering a letter fills the cloud's
        # own preview panel, out of the cloud's own atlases. Shipping a
        # second copy of the same two images would be a second thing to keep
        # in step with `build_cloud.to_thumbs`.
        letters.append(dict(pick, x=c["x"], y=c["y"], w=c["w"], h=c["h"],
                            colour="#%02x%02x%02x" % tuple(
                                int(v * 255) for v in source_colour(pick["dataset"])),
                            sprite=_png64(rgba, "RGBA")))
    os.makedirs(COMPONENT, exist_ok=True)
    path = os.path.join(COMPONENT, "letters.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"build": int(time.time()), "word": WORD, "grid": grid,
                   "letters": letters}, f, ensure_ascii=False)
    size = os.path.getsize(path) / 1024
    print(f"  {os.path.relpath(path, ROOT)}  "
          f"({grid['w']}x{grid['h']} voxels, {size:.0f} ko)")


def choose(shortlists: Dict[str, List[Dict]], ranks: Dict[str, int]) -> List[Dict]:
    """One nucleus per letter of the word, best IoU first.

    The two `l` are two different nuclei: the same crop twice would be a copy
    and paste, and the second-best `l` is a nucleus in its own right.
    """
    used: Dict[str, int] = {}
    picks: List[Dict] = []
    for ch in WORD:
        seen = used.get(ch, 0)
        rank = ranks.get(ch, 0) + seen
        picks.append(dict(shortlists[ch][rank]))
        used[ch] = seen + 1
    return picks


def do_render(ranks: Dict[str, int]) -> None:
    with open(CANDIDATES, encoding="utf-8") as f:
        spec = json.load(f)
    picks = choose(spec["shortlists"], ranks)
    os.makedirs(ASSETS, exist_ok=True)

    value, alpha = compose(picks, WORD)
    (value,), alpha = tighten([value], alpha)

    white = np.full(value.shape + (3,), 1.0, np.float32) * (0.55 + 0.45 * value)[..., None]
    save_rgba(os.path.join(ASSETS, "logo_vocell.png"), white, alpha)
    save_rgba(os.path.join(ASSETS, "logo_vocell_dark.png"), white, alpha, DARK_BG)

    # On white, the dense part of a nucleus is the darkest ink; scaling one
    # near-black colour would flatten the texture, so the two ends of the
    # range are two colours and the greys ride between them.
    ink = np.asarray([0.05, 0.07, 0.11], np.float32)
    faint = np.asarray([0.30, 0.34, 0.43], np.float32)
    dark = faint[None, None] + (ink - faint)[None, None] * (value ** 0.7)[..., None]
    save_rgba(os.path.join(ASSETS, "logo_vocell_light.png"), dark, alpha,
              (255, 255, 255))

    # Same wordmark, each letter in its source's colour: the logo doubles as a
    # legend for the eleven datasets it was drawn from.
    tint, acc = compose_tinted(picks, WORD)
    (tint,), acc = tighten([tint], acc)
    save_rgba(os.path.join(ASSETS, "logo_vocell_sources.png"), tint, acc, DARK_BG)

    write_component_assets(picks)
    write_voxel_pdfs(picks)

    total = sum(build_cloud.load_source(src, None)[1]
                for src in build_cloud.SOURCES)
    provenance_sheet(picks, total,
                     os.path.join(ASSETS, "logo_vocell_provenance.png"))
    with open(os.path.join(ASSETS, "logo_vocell.json"), "w", encoding="utf-8") as f:
        json.dump({"word": WORD, "font": f"{FONT_PATH}#{FONT_INDEX}",
                   "letters": picks}, f, indent=1, ensure_ascii=False)
    print(f"  {os.path.relpath(os.path.join(ASSETS, 'logo_vocell.json'), ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the nuclei per source, for a smoke run")
    ap.add_argument("--rank", action="append", default=[], metavar="L=N",
                    help="use the Nth best nucleus for letter L instead of the"
                         " best, e.g. --rank e=2")
    args = ap.parse_args()
    if args.search:
        do_search(args.limit)
    if args.render:
        ranks = dict((k, int(v)) for k, v in
                     (r.split("=", 1) for r in args.rank))
        do_render(ranks)
    if not (args.search or args.render):
        ap.error("nothing to do: pass --search and/or --render")


if __name__ == "__main__":
    main()
