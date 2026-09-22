"""Is cosine similarity a meaningful way to separate nuclei, and where?

Cosine is a metric, not a method: it is computable in any vector space. What is
not automatic is that the angle *means* anything. This script measures that, in
the two spaces VoCell has, on identical samples and one seed:

    descripteurs   the 10 morphological columns of data/cloud/features.npz
    h (512)        SimCLR backbone output   (scripts/extract_embeddings.py)
    z (128)        SimCLR projector output, where InfoNCE is literally defined

Five measurements:

  A. dynamic range   how much angle is there to work with at all
  B. separability    k-NN label agreement, cosine vs euclidean, per scaling
  C. arbitrariness   does a change of UNIT on one descriptor reshuffle the
                     neighbours (descriptor space only -- a latent has no units)
  D. direction/norm  cosine discards the norm; does the norm carry the signal
  E. batch effect    does either space separate the 11 acquisition sources

Run scripts/extract_embeddings.py first; without it only A and C run.

    python scripts/latent_probe.py
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLOUD = os.path.join(ROOT, "data", "cloud")

K = 10                 # neighbours
PER_CLASS = 350        # balanced subsample
UNLABELED = "non étiqueté"

RNG = np.random.default_rng(0)

d = np.load(os.path.join(CLOUD, "features.npz"))
VALUES, CLASS_IDX = d["values"], d["class_idx"]
NAMES = [str(x) for x in d["names"]]
CLASS_NAMES = [str(x) for x in d["class_names"]]
MANIFEST = json.loads(str(d["manifest"]))
LABELED = np.array([n != UNLABELED for n in CLASS_NAMES])

_h_path = os.path.join(CLOUD, "emb_h.npy")
_z_path = os.path.join(CLOUD, "emb_z.npy")
HAS_EMB = os.path.exists(_h_path) and os.path.exists(_z_path)
EMB_H = np.load(_h_path, mmap_mode="r") if HAS_EMB else None
EMB_Z = np.load(_z_path, mmap_mode="r") if HAS_EMB else None

# Percentiles and the median/IQR are taken on a stride of the full table, not
# on the sample: the scaling has to be the one the cloud actually uses.
REF = VALUES[::97]
_MED = np.median(REF, axis=0)
_Q1, _Q3 = np.percentile(REF, [25.0, 75.0], axis=0)
_IQR = np.maximum((_Q3 - _Q1) / 1.349, 1e-6)


def robust_z(x, med=None, iqr=None):
    med = _MED if med is None else med
    iqr = _IQR if iqr is None else iqr
    return np.clip((x - med) / iqr, -4.0, 4.0).astype(np.float64)


def unit_scale(x, ref):
    lo, hi = np.percentile(ref, [0.5, 99.5], axis=0)
    return np.clip((x - lo) / np.maximum(hi - lo, 1e-9), 0.0, 1.0).astype(np.float64)


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def neighbours(x, k=K, metric="euclidean"):
    """Indices of the k nearest neighbours of every row, excluding itself."""
    x = np.asarray(x, dtype=np.float64)
    if metric == "cosine":
        s = l2(x) @ l2(x).T
        np.fill_diagonal(s, -np.inf)
        return np.argpartition(-s, k, axis=1)[:, :k]
    sq = (x * x).sum(1)
    dist = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.fill_diagonal(dist, np.inf)
    return np.argpartition(dist, k, axis=1)[:, :k]


def agreement(x, y, metric="euclidean"):
    return float((y[neighbours(x, metric=metric)] == y[:, None]).mean())


def spaces(idx):
    """The three spaces for one set of STRICTLY INCREASING global indices.

    Sorted is not cosmetic: the arrays are memory-mapped, and the labels are
    taken from the same sorted indices. Indexing them in different orders
    silently decorrelates features from labels and every score falls to chance.
    """
    assert np.all(np.diff(idx) > 0), "spaces() needs strictly increasing indices"
    desc = robust_z(VALUES[idx])
    if not HAS_EMB:
        return desc, None, None
    return desc, np.asarray(EMB_H[idx], np.float64), np.asarray(EMB_Z[idx], np.float64)


def balanced(src):
    """Sorted global indices, PER_CLASS per labelled class of one source."""
    lo = src["start"]
    cls = CLASS_IDX[lo:lo + src["count"]]
    pool = np.flatnonzero(LABELED[cls])
    picks = []
    for u in np.unique(cls[pool]):
        w = pool[cls[pool] == u]
        if len(w) >= 40:
            picks.append(RNG.choice(w, min(PER_CLASS, len(w)), replace=False))
    local = np.sort(np.concatenate(picks))
    return lo + local, cls[local]


def chance_level(y):
    _, cnt = np.unique(y, return_counts=True)
    return float(((cnt / cnt.sum()) ** 2).sum())


def line(tag, score, chance):
    print(f"        {tag:<30s} {score:.3f}   (x{score / chance:.2f})")


# --------------------------------------------------------------------- A
print(f"{len(VALUES):,} noyaux · descripteurs {VALUES.shape[1]}"
      + (f" · h {EMB_H.shape[1]} · z {EMB_Z.shape[1]}" if HAS_EMB else
         " · embeddings absents (scripts/extract_embeddings.py)"))

print("\nA. Similarité cosinus entre paires (4 000 noyaux, toutes sources)")
idx = np.sort(RNG.choice(len(VALUES), 4000, replace=False))
desc, h, z = spaces(idx)
candidates = [
    ("descripteurs bruts", VALUES[idx].astype(np.float64)),
    ("descripteurs unit-scale", unit_scale(VALUES[idx], REF)),
    ("descripteurs robust-z", desc),
]
if HAS_EMB:
    candidates += [("h (512), brut", h), ("h (512), centre", h - h.mean(0)),
                   ("z (128), brut", z), ("z (128), centre", z - z.mean(0))]
for tag, x in candidates:
    s = (l2(np.asarray(x, np.float64)) @ l2(np.asarray(x, np.float64)).T)
    s = s[np.triu_indices(len(s), 1)]
    p1, med, p99 = np.percentile(s, [1, 50, 99])
    print(f"   {tag:<26s} p1={p1:+.3f} med={med:+.3f} p99={p99:+.3f}"
          f"   (étendue {p99 - p1:.3f})")

# --------------------------------------------------------------------- B
print(f"\nB. Accord d'étiquette des {K} plus proches voisins")
for src in MANIFEST:
    if not src["labeled"]:
        continue
    idx, y = balanced(src)
    desc, h, z = spaces(idx)
    chance = chance_level(y)
    print(f"   {src['name']:<9s} {len(np.unique(y)):2d} classes, {len(y):5d} "
          f"noyaux, hasard={chance:.3f}")
    line("descripteurs eucl./robust-z", agreement(desc, y), chance)
    line("descripteurs cosinus/brut",
         agreement(VALUES[idx].astype(np.float64), y, "cosine"), chance)
    if HAS_EMB:
        line("h  cosinus", agreement(h, y, "cosine"), chance)
        line("h  euclidien", agreement(h, y), chance)
        line("z  cosinus  <- la loss", agreement(z, y, "cosine"), chance)
        line("z  euclidien", agreement(z, y), chance)

# --------------------------------------------------------------------- C
print(f"\nC. On multiplie par 10 l'unité d'UN descripteur. Combien des {K} "
      "voisins survivent ?")
print("   (sans objet pour h et z : leurs axes sortent tous du même ReLU)")
idx = np.sort(RNG.choice(len(VALUES), 3000, replace=False))
sub = VALUES[idx].astype(np.float64)
base_cos = neighbours(sub, metric="cosine")
base_euc = neighbours(robust_z(sub), metric="euclidean")


def overlap(a, b):
    return float(np.mean([len(set(u) & set(v)) for u, v in zip(a, b)])) / K


for j, name in enumerate(NAMES):
    scaled = sub.copy()
    scaled[:, j] *= 10.0
    med, iqr = _MED.copy(), _IQR.copy()
    med[j] *= 10.0
    iqr[j] *= 10.0
    o_cos = overlap(base_cos, neighbours(scaled, metric="cosine"))
    o_euc = overlap(base_euc, neighbours(robust_z(scaled, med, iqr)))
    print(f"   {name:<16s} cosinus/brut {o_cos * 100:5.1f} %"
          f"   euclidien/robust-z {o_euc * 100:5.1f} %")

if not HAS_EMB:
    raise SystemExit("\nD et E sautées : lancer scripts/extract_embeddings.py.")

# --------------------------------------------------------------------- D
print("\nD. L'information est-elle dans la direction ou dans la norme ?")
src = next(s for s in MANIFEST if s["name"] == "HPA")
idx, y = balanced(src)
desc, h, z = spaces(idx)
chance = chance_level(y)
area = VALUES[idx, NAMES.index("area")].astype(np.float64)
print(f"   échantillon HPA équilibré, hasard={chance:.3f}")
for tag, x in (("h", h), ("z", z)):
    norm = np.linalg.norm(x, axis=1)
    a_dir = agreement(l2(x), y)
    a_nrm = agreement(norm[:, None], y)
    r = float(np.corrcoef(norm, area)[0, 1])
    print(f"   {tag}: direction seule {a_dir:.3f} (x{a_dir / chance:.2f})"
          f"  |  norme seule {a_nrm:.3f} (x{a_nrm / chance:.2f})"
          f"  |  corrélation ||{tag}|| vs aire = {r:+.2f}")

# --------------------------------------------------------------------- E
print(f"\nE. Accord de SOURCE des {K} plus proches voisins (400 par source)")
idx = np.sort(np.concatenate([
    RNG.choice(np.arange(s["start"], s["start"] + s["count"]), 400, replace=False)
    for s in MANIFEST]))
source_of = np.zeros(len(VALUES), np.int16)
for i, s in enumerate(MANIFEST):
    source_of[s["start"]:s["start"] + s["count"]] = i
y = source_of[idx]
desc, h, z = spaces(idx)
chance = chance_level(y)
print(f"   hasard={chance:.3f}")
line("descripteurs eucl./robust-z", agreement(desc, y), chance)
line("h  cosinus", agreement(h, y, "cosine"), chance)
line("z  cosinus", agreement(z, y, "cosine"), chance)
