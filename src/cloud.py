"""Layout maths for the nucleus point cloud. Pure NumPy, no Streamlit.

Consumed by scripts/build_cloud.py, which turns the descriptor table into the
normalised columns the JS component fetches. The app itself does no maths for
the cloud any more: at 2.6 M nuclei, computing a projection per rerun and
pushing it over the websocket costs 10 MB a keystroke, so every descriptor is
normalised once at build time and served as a uint16 column instead.

Descriptors are in raw pixels, never microns. The eleven sources were never
rescaled to a common pixel size -- median nucleus diameter runs from 13 px to
42 px across them -- and that is left in on purpose (see CLAUDE.md): any axis
involving a size separates the datasets, which is the honest picture of what a
model trained on one and shown another would face.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Column order of the descriptor table. build_cloud.py computes them in this
# order; changing it invalidates an existing features.npz.
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

# Display labels. The keys are the column names; the values are what the UI
# shows, so they are French like the rest of it.
FEATURE_LABELS: Dict[str, str] = {
    "area": "Aire du masque (px²)",
    "elongation": "Élongation (rapport d'axes)",
    "mean_intensity": "Intensité moyenne",
    "total_intensity": "Intensité intégrée",
    "contrast": "Contraste (p98 − p2)",
    "sharpness": "Netteté du contour",
    "concentration": "Concentration centrale",
    "fill": "Remplissage de la boîte",
    "cv_intensity": "Hétérogénéité (CV)",
    "off_center": "Décentrage (px)",
}

LAYOUT_MODES = ["Deux descripteurs", "ACP"]

# Offered on top of LAYOUT_MODES only when scripts/extract_embeddings.py has
# run: the latent columns need a torch checkpoint the app itself never loads.
LATENT_MODE = "ACP latente"

_CLIP_LO, _CLIP_HI = 0.5, 99.5

# A descriptor counted in pixels of a 64x64 crop cannot take more than 4096
# distinct values, however many nuclei there are. That absolute ceiling is
# what makes a column discrete -- a ratio to the sample size is not, since the
# same `area` column reads continuous at 30 k nuclei and discrete at 2.6 M.
DISCRETE_LEVELS = 4096


def label(key: str) -> str:
    return FEATURE_LABELS.get(key, key)


def unit_scale(col: np.ndarray) -> np.ndarray:
    """Map one descriptor onto [0, 1], clipped at the 0.5/99.5 percentiles.

    Plain min-max would let a single outlier crush the whole cloud into a
    corner: `area` has crops an order of magnitude above the bulk. Clipping
    costs the extremes their exact position, which a scatter plot of millions
    of points could not have shown anyway. The percentiles are global across
    all sources, since the axis is shared.
    """
    lo, hi = np.percentile(col, [_CLIP_LO, _CLIP_HI])
    if hi <= lo:
        return np.full_like(col, 0.5, dtype=np.float32)
    return np.clip((col - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def robust_z(values: np.ndarray) -> np.ndarray:
    """Column-wise robust standardisation, for PCA.

    Median and IQR rather than mean and standard deviation, then clipped at
    ±4: otherwise `area`, whose numbers are in the hundreds, would simply be
    PC1 and the other nine descriptors would not get a say.
    """
    med = np.median(values, axis=0)
    q1, q3 = np.percentile(values, [25.0, 75.0], axis=0)
    scale = np.maximum((q3 - q1) / 1.349, 1e-6)
    return np.clip((values - med) / scale, -4.0, 4.0).astype(np.float32)


def pca_2d(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First two principal components.

    Returns (scores (N, 2), explained variance ratio (2,), loadings (2, K)).
    The covariance is only K×K, so this is an eigendecomposition of a 10×10
    matrix regardless of how many nuclei there are.
    """
    z = robust_z(values)
    z = z - z.mean(axis=0)
    cov = (z.T @ z) / max(len(z) - 1, 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]

    # An eigenvector's sign is arbitrary. Pin it so the descriptor that
    # dominates a component always reads positive, otherwise the axis labels
    # flip meaning between two builds of the same data.
    top = np.abs(eigvec[:, :2]).argmax(axis=0)
    flip = np.sign(eigvec[top, [0, 1]])
    flip[flip == 0] = 1.0
    eigvec[:, :2] *= flip

    scores = z @ eigvec[:, :2]
    ratio = eigval[:2] / max(float(eigval.sum()), 1e-12)
    return scores.astype(np.float32), ratio.astype(np.float32), eigvec[:, :2].T


def jitter_step(scaled: np.ndarray, sample: int = 200_000) -> float:
    """How far a point may be nudged along a discrete axis, or 0.

    `area` is a pixel count and `elongation` is quantised by the mask, so
    hundreds of thousands of nuclei land on identical coordinates and the
    cloud turns into stripes of invisible depth. The component offsets each
    point by less than half the spacing between two adjacent levels, so no
    point crosses into a neighbour's column -- it is cosmetic, and the UI
    says so. Continuous columns get 0 and are left alone.

    The estimate runs on a sample: `np.unique` over millions of values is a
    full sort, and the spacing of a quantised axis is visible long before
    then.
    """
    if len(scaled) > sample:
        step_idx = len(scaled) // sample
        scaled = scaled[::step_idx]
    uniq = np.unique(scaled)
    if uniq.size < 2 or uniq.size > DISCRETE_LEVELS:
        return 0.0
    return float(np.median(np.diff(uniq)))


def dominant(loading: np.ndarray, names: List[str], k: int = 2) -> str:
    """The descriptors a principal component is mostly made of."""
    top = np.argsort(np.abs(loading))[::-1][:k]
    parts = []
    for i in top:
        sign = "+" if loading[i] >= 0 else "−"
        parts.append(f"{sign}{label(names[i]).split(' (')[0].lower()}")
    return ", ".join(parts)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    """Every row scaled to length 1, zero rows left at zero."""
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def pca_latent_2d(emb, chunk: int = 50_000) -> Tuple[np.ndarray, np.ndarray]:
    """First two principal components of an embedding, on its cosine geometry.

    Each row is scaled to unit length before centring, so this projects the
    DIRECTION of an embedding and not its length. That is measured, not a
    matter of taste (scripts/latent_probe.py): on the SimCLR checkpoint of
    cellf-supervised, the direction of a backbone vector agrees with the cell
    line of its nucleus 2.94x above chance while its norm alone agrees 1.28x,
    and that norm correlates -0.33 with the nucleus area. The length is mostly
    size, which is the nuisance already separating the eleven sources. Cosine
    discards it, and so does this.

    Streamed in two passes: 2.6 M x 512 in float32 is 5.4 GB, while the
    covariance it feeds is 512x512. The array is only ever read a chunk at a
    time, so a memory map never has to be materialised.

    Returns (scores (N, 2) float32, explained variance ratio (2,)).
    """
    n, k = emb.shape
    total = np.zeros(k, dtype=np.float64)
    gram = np.zeros((k, k), dtype=np.float64)

    for lo in range(0, n, chunk):
        u = _unit_rows(np.asarray(emb[lo:lo + chunk], dtype=np.float32))
        u64 = u.astype(np.float64)
        total += u64.sum(axis=0)
        gram += u64.T @ u64

    mean = total / n
    cov = gram / n - np.outer(mean, mean)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]

    # An eigenvector's sign is arbitrary, and a latent axis has no descriptor
    # to pin it to. Pin it on the largest loading instead: any rule will do as
    # long as two builds of the same data agree on it.
    top = np.abs(eigvec[:, :2]).argmax(axis=0)
    flip = np.sign(eigvec[top, [0, 1]])
    flip[flip == 0] = 1.0
    axes = eigvec[:, :2] * flip

    scores = np.empty((n, 2), dtype=np.float32)
    for lo in range(0, n, chunk):
        u = _unit_rows(np.asarray(emb[lo:lo + chunk], dtype=np.float32))
        scores[lo:lo + len(u)] = (u - mean) @ axes

    positive = np.maximum(eigval, 0.0)
    ratio = positive[:2] / max(float(positive.sum()), 1e-12)
    return scores, ratio.astype(np.float32)


def closest_descriptor(score: np.ndarray, values: np.ndarray,
                       names: List[str], stride: int = 97) -> Tuple[str, float]:
    """The descriptor a latent component tracks most closely, and its r.

    A latent axis carries no unit and no name, so `dominant` has nothing to
    read. Naming it by the morphological descriptor it correlates with is the
    only honest handle the UI can give: it says what the axis happens to line
    up with, not what it is made of.

    The stride is not an approximation worth worrying about -- a correlation
    over 27 000 nuclei is already settled well past the two decimals shown.
    """
    s = score[::stride].astype(np.float64)
    s = s - s.mean()
    ss = float((s * s).sum())
    best, best_r = names[0], 0.0
    for i, name in enumerate(names):
        v = values[::stride, i].astype(np.float64)
        v = v - v.mean()
        denom = np.sqrt(ss * float((v * v).sum()))
        r = float((s * v).sum() / denom) if denom > 0 else 0.0
        if abs(r) > abs(best_r):
            best, best_r = name, r
    return best, best_r
