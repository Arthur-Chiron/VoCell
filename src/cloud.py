"""Layout maths for the nucleus point cloud. Pure NumPy, no Streamlit.

Takes the descriptor table built by scripts/build_cloud.py and turns it into
2D positions in the unit square, which is the only thing the JS component
knows how to draw. Two layouts:

  "Deux descripteurs"  one descriptor per axis, read directly.
  "ACP"                the first two principal components of all of them.

Descriptors are in raw pixels, never microns: CODEX is ~0.377 um/px and
RESTORE ~0.15 um/px, and that gap is left in on purpose (see CLAUDE.md), so
any axis involving a size will separate the two datasets. That is the honest
picture of what a model trained on one and shown the other would face.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Display labels. The keys are the column names written by build_cloud.py;
# the values are what the UI shows, so they are French like the rest of it.
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

_CLIP_LO, _CLIP_HI = 0.5, 99.5


def label(key: str) -> str:
    return FEATURE_LABELS.get(key, key)


def unit_scale(col: np.ndarray) -> np.ndarray:
    """Map one descriptor onto [0, 1], clipped at the 0.5/99.5 percentiles.

    Plain min-max would let a single outlier crush the whole cloud into a
    corner: `area` has a few crops an order of magnitude above the bulk.
    Clipping costs the extremes their exact position, which a scatter plot of
    172k points could not have shown anyway.
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


def dither(coords: np.ndarray, seed: int = 0) -> np.ndarray:
    """Spread points that a discrete descriptor stacks exactly on top of.

    `area` is a pixel count and `elongation` is quantised by the mask, so tens
    of thousands of nuclei land on identical coordinates and the cloud turns
    into stripes of invisible depth. The offset is under half the spacing
    between adjacent levels, so no point crosses into a neighbour's column --
    it is cosmetic, and the UI says so.
    """
    out = coords.copy()
    rng = np.random.default_rng(seed)
    for axis in (0, 1):
        col = out[:, axis]
        uniq = np.unique(col)
        if uniq.size < 2 or uniq.size > len(col) / 20:
            continue                      # continuous enough already
        step = float(np.median(np.diff(uniq)))
        out[:, axis] = col + rng.uniform(-0.4 * step, 0.4 * step, size=len(col))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def project(values: np.ndarray, names: List[str], mode: str,
            x_key: str = "area", y_key: str = "mean_intensity",
            jitter: bool = True) -> Tuple[np.ndarray, Dict[str, str]]:
    """Descriptor table -> (N, 2) positions in [0, 1] plus the axis labels."""
    if mode == "ACP":
        scores, ratio, loadings = pca_2d(values)
        coords = np.stack([unit_scale(scores[:, 0]), unit_scale(scores[:, 1])], axis=1)
        axes = {
            "x": f"CP1 — {ratio[0]:.0%} de variance · {_dominant(loadings[0], names)}",
            "y": f"CP2 — {ratio[1]:.0%} de variance · {_dominant(loadings[1], names)}",
        }
    else:
        ix, iy = names.index(x_key), names.index(y_key)
        coords = np.stack([unit_scale(values[:, ix]), unit_scale(values[:, iy])], axis=1)
        axes = {"x": label(x_key), "y": label(y_key)}

    if jitter:
        coords = dither(coords)
    return coords, axes


def _dominant(loading: np.ndarray, names: List[str], k: int = 2) -> str:
    """The descriptors a principal component is mostly made of."""
    top = np.argsort(np.abs(loading))[::-1][:k]
    parts = []
    for i in top:
        sign = "+" if loading[i] >= 0 else "−"
        parts.append(f"{sign}{label(names[i]).split(' (')[0].lower()}")
    return ", ".join(parts)


def quantize(coords: np.ndarray) -> bytes:
    """Pack positions as interleaved uint16 for the component.

    172 358 points as float32 pairs is 1.4 MB on the websocket every time the
    projection changes; as uint16 it is 690 kB, and 1/65535 of the unit square
    is far below anything the canvas can resolve, even zoomed all the way in.
    """
    q = np.clip(coords, 0.0, 1.0) * 65535.0
    return q.astype(np.uint16).tobytes()
