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

# The layout picker is two-level: a SPACE (two raw descriptors, the whole
# descriptor table, or one model's latent space) and, for every space but the
# first, a PROJECTION of it to 2D. Which (space, projection) pairs exist is
# the build's call, written to meta.json["layouts"]; the app only reads it.
PAIR_MODE = "Deux descripteurs"
DESCRIPTOR_SPACE = "desc"
PROJECTIONS: Dict[str, str] = {"pca": "ACP", "umap": "UMAP", "tsne": "t-SNE"}

# SimCLR checkpoints of the sibling cellf-supervised repo, keyed by what
# follows CKPT_PREFIX in their file name. scripts/extract_embeddings.py runs
# every checkpoint it finds there, listed or not; an unlisted one simply
# shows up under its bare key. Written by hand from cellf-supervised's
# history -- a file name is all a checkpoint says about itself.
CKPT_PREFIX = "simclr_resnet18_"
MODEL_LABELS: Dict[str, str] = {
    "outHPA_augcell": "SimCLR · aug. cellaug",
    "outHPA_augcifar": "SimCLR · aug. CIFAR",
    # Commit 413b22b of cellf-supervised: "wrongly include test data in ssl
    # training set". Its HPA-test nuclei were seen during pretraining.
    "cifar_tr": "SimCLR · aug. CIFAR, HPA test vu",
}


def model_key(filename: str) -> str:
    """'…/simclr_resnet18_outHPA_augcell.pt' -> 'outHPA_augcell'."""
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem[len(CKPT_PREFIX):] if stem.startswith(CKPT_PREFIX) else stem


def model_order(keys: List[str]) -> List[str]:
    """Listed models first, in MODEL_LABELS order, then the others by name."""
    known = [k for k in MODEL_LABELS if k in keys]
    return known + sorted(k for k in keys if k not in MODEL_LABELS)


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


def latent_pca(emb, k: int = 50,
               chunk: int = 50_000) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k principal components of an embedding, on its cosine geometry.

    Each row is scaled to unit length before centring, so this projects the
    DIRECTION of an embedding and not its length. That is measured, not a
    matter of taste (scripts/latent_probe.py): on the SimCLR checkpoint of
    cellf-supervised, the direction of a backbone vector agrees with the cell
    line of its nucleus 2.94x above chance while its norm alone agrees 1.28x,
    and that norm correlates -0.33 with the nucleus area. The length is mostly
    size, which is the nuisance already separating the eleven sources. Cosine
    discards it, and so does this.

    The first two columns are the cloud's ACP layout; all k feed UMAP and
    t-SNE, which on unit rows makes their euclidean distance a cosine one.

    Streamed in two passes: 2.6 M x 512 in float32 is 5.4 GB, while the
    covariance it feeds is 512x512. The array is only ever read a chunk at a
    time, so a memory map never has to be materialised.

    Returns (scores (N, k) float32, explained variance ratio (k,)).
    """
    n, dim = emb.shape
    k = min(k, dim)
    total = np.zeros(dim, dtype=np.float64)
    gram = np.zeros((dim, dim), dtype=np.float64)

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
    top = np.abs(eigvec[:, :k]).argmax(axis=0)
    flip = np.sign(eigvec[top, np.arange(k)])
    flip[flip == 0] = 1.0
    axes = (eigvec[:, :k] * flip).astype(np.float32)
    mean32 = mean.astype(np.float32)

    scores = np.empty((n, k), dtype=np.float32)
    for lo in range(0, n, chunk):
        u = _unit_rows(np.asarray(emb[lo:lo + chunk], dtype=np.float32))
        scores[lo:lo + len(u)] = (u - mean32) @ axes

    positive = np.maximum(eigval, 0.0)
    ratio = positive[:k] / max(float(positive.sum()), 1e-12)
    return scores, ratio.astype(np.float32)


# --------------------------------------------------------------------------
# Non-linear projections
# --------------------------------------------------------------------------

# UMAP and t-SNE are fitted on a sample and the rest of the nuclei are placed
# by their neighbours in it. At 2.6 M points a full fit is hours per space and
# per method, for a picture the eye cannot tell from this one: a scatter at
# screen resolution has fewer pixels than the sample has points.
FIT_SAMPLE = 200_000
FIT_FLOOR = 5_000        # per source, so AitslabBioimaging1 (1 735) is all in
PLACE_K = 10


def fit_sample(starts: List[int], counts: List[int], n_fit: int = FIT_SAMPLE,
               floor: int = FIT_FLOOR, seed: int = 0) -> np.ndarray:
    """Sorted global indices of the nuclei the projection is fitted on.

    Uniform across the cloud, with a floor per source. Uniform alone would
    give AitslabBioimaging1 some 130 points out of 200 000 -- too few for its
    own neighbourhood to exist in the fit, so every one of its nuclei would be
    interpolated into somebody else's. The floor over-represents the small
    sources a little; t-SNE and UMAP do not preserve density anyway.
    """
    rng = np.random.default_rng(seed)
    total = max(sum(counts), 1)
    picked = []
    for start, count in zip(starts, counts):
        quota = min(count, max(int(round(n_fit * count / total)), floor))
        picked.append(start + rng.choice(count, size=quota, replace=False))
    return np.sort(np.concatenate(picked))


def _fit_2d(x: np.ndarray, method: str, seed: int = 0) -> np.ndarray:
    if method == "umap":
        import umap  # build-only dependency, see requirements.txt
        # No random_state: it would serialise the optimisation onto one core.
        # The build caches its projections, so a rerun does not reshuffle.
        return umap.UMAP(n_neighbors=30, min_dist=0.1, n_jobs=-1,
                         verbose=False).fit_transform(x).astype(np.float32)
    if method == "tsne":
        from openTSNE import TSNE  # build-only dependency
        return np.asarray(TSNE(perplexity=30, initialization="pca",
                               n_jobs=-1, random_state=seed,
                               verbose=False).fit(x), dtype=np.float32)
    raise ValueError(method)


def nonlinear_2d(x: np.ndarray, fit_idx: np.ndarray, methods: List[str],
                 k: int = PLACE_K, chunk: int = 100_000) -> Dict[str, np.ndarray]:
    """UMAP / t-SNE layouts of every row of `x`, fitted on `x[fit_idx]`.

    The fitted rows keep their own position. Every other row is placed at the
    coordinate-wise MEDIAN of its k nearest fitted rows, in the input space.
    The median and not the mean: a nucleus whose neighbours straddle two
    clusters would otherwise land in the empty space between them, drawing a
    bridge that is in neither the data nor the projection. It is openTSNE's
    own `initialization="median"` for new points, without the optimisation
    that follows -- at 2.4 M points that step is the whole cost.

    The neighbour search is done once and shared by every method.
    Returns {method: (N, 2) float32}.
    """
    from pynndescent import NNDescent  # comes with umap-learn

    fit_x = np.ascontiguousarray(x[fit_idx], dtype=np.float32)
    fitted = {m: _fit_2d(fit_x, m) for m in methods}

    index = NNDescent(fit_x, n_neighbors=30, n_jobs=-1, random_state=0)
    index.prepare()
    rest = np.ones(len(x), dtype=bool)
    rest[fit_idx] = False
    rest_idx = np.flatnonzero(rest)

    out = {m: np.empty((len(x), 2), dtype=np.float32) for m in methods}
    for m in methods:
        out[m][fit_idx] = fitted[m]
    for lo in range(0, len(rest_idx), chunk):
        ids = rest_idx[lo:lo + chunk]
        nn, _ = index.query(np.asarray(x[ids], dtype=np.float32), k=k)
        for m in methods:
            out[m][ids] = np.median(fitted[m][nn], axis=1)
    return out


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


# --------------------------------------------------------------------------
# Class palette: close colours for nuclei that look alike
# --------------------------------------------------------------------------
#
# Which classes look alike is written down here from what is known of their
# nuclei -- NOT measured on the data. Deriving it from the descriptors or the
# SimCLR space would make the cloud agree with itself by construction: colour
# the classes by how close they fall, and of course the colours then look
# well sorted. This way the palette is a prior, and the cloud is free to
# confirm it or not.
#
# Each dataset is a list of families, in an order where neighbouring
# families are the most alike; within a family, the order is the same idea
# at a finer grain. The two ends of the list are the two most unlike.

CLASS_KINSHIP: Dict[str, List[List[str]]] = {
    # Colorectal cancer. From small, round, dense lymphocytes to the large,
    # pleomorphic, vesicular nuclei of the tumour.
    "CODEX": [
        ["B cells", "T cells", "Tregs", "NK cells"],   # small, round, dense
        ["Granulocytes"],                   # dense too, but lobed
        ["Monocytes", "Dendritic cells", "Macrophages"],   # bean, irregular,
                                            # then larger and paler
        ["Vasculature", "Smooth muscle cells", "Nerves"],  # elongated stroma
        ["Adipocyte"],                      # flat crescent at the rim
        ["Neoplastic cells"],               # large, pleomorphic, nucleolated
    ],
    # Cell lines in culture. From small, round nuclei to large, flat,
    # pleomorphic ones; the carcinomas in the middle, glandular then squamous.
    "HPA": [
        ["RH-30", "SH-SY5Y", "HAP1", "HEK 293"],   # small, round
        ["SK-MEL-30"],                             # melanoma, neural crest
        ["A549", "PC-3", "EFO-21", "RPTEC TERT1"],   # glandular epithelia
        ["HeLa", "SiHa", "A-431", "HaCaT", "hTCEpi"],   # squamous epithelia
        ["HUVEC TERT2"],                           # endothelial, oval
        ["U-251 MG", "U-2 OS"],                    # large, flat, pleomorphic
    ],
    # Human kidney. From the small dense immune nuclei, through the vessels
    # and the glomerulus, then down the nephron: proximal tubule, loop,
    # distal tubule, collecting duct.
    "BBBC051": [
        ["CD45"],                           # immune
        ["CD31_inter", "CD31_glom"],        # endothelium, flat
        ["Nestin"],                         # podocytes
        ["S1S2", "S2S3"],                   # proximal tubule, large round
        ["TAL", "DCT"],                     # loop and distal tubule
        ["CNT", "CD_CNT", "CD"],            # connecting tubule, collecting duct
    ],
}

# Classes that name no cell type: a mix, or no label at all. They stay grey
# so they never pass for one more family.
NEUTRAL_CLASSES = {"non étiqueté": "#8a8f98", "Others": "#b8bcc4"}

PALETTE_ARC = 280.0      # degrees of hue wheel used; the ends must not meet
PALETTE_START = 250.0    # hue of the first class (blue)
FAMILY_GAP = 3.0         # step between families, in steps within one


def _oklch_hex(l: float, c: float, h_deg: float) -> str:
    """OKLCH to sRGB hex, chroma reduced until the colour is in gamut."""
    h = np.radians(h_deg)
    for _ in range(40):
        a, b = c * np.cos(h), c * np.sin(h)
        l_ = (l + 0.3963377774 * a + 0.2158037573 * b) ** 3
        m_ = (l - 0.1055613458 * a - 0.0638541728 * b) ** 3
        s_ = (l - 0.0894841775 * a - 1.2914855480 * b) ** 3
        lin = np.array([
            4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
            -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
            -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076720010 * s_])
        if lin.min() >= -1e-4 and lin.max() <= 1 + 1e-4:
            break
        c *= 0.95
    lin = np.clip(lin, 0.0, 1.0)
    rgb = np.where(lin <= 0.0031308, 12.92 * lin,
                   1.055 * lin ** (1 / 2.4) - 0.055)
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in rgb)


def kinship_palette(dataset: str, classes: List[str]) -> Dict[str, str]:
    """One distinct colour per class, closer for classes that look alike.

    The families of CLASS_KINSHIP are laid in order on an arc of the OKLCH
    hue wheel -- perceptually even, so equal steps of hue read as equal steps
    of colour. Neighbours inside a family are one step apart, neighbouring
    families FAMILY_GAP steps. The arc stops short of a full turn so the two
    ends, the two most unlike, do not come back together.

    Neighbours then alternate between two lightnesses. It works a little
    against the "close means alike" reading, but 17 HPA lines on one arc
    leave some neighbours ~12 degrees apart, which nobody sees on a 2 px
    point.

    A class the table does not know becomes a family of its own at the end
    of the arc rather than being dropped, so a relabelled dataset still
    builds; the table should then be updated.

    Returns {class: hex}, in palette order, the neutral classes last.
    """
    known = {c for fam in CLASS_KINSHIP.get(dataset, []) for c in fam}
    present = set(classes)
    families = [[c for c in fam if c in present]
                for fam in CLASS_KINSHIP.get(dataset, [])]
    families = [f for f in families if f]
    families += [[c] for c in classes
                 if c not in known and c not in NEUTRAL_CLASSES]

    ranked, steps = [], []
    for f, fam in enumerate(families):
        for j, c in enumerate(fam):
            if ranked:
                steps.append(1.0 if j else FAMILY_GAP)
            ranked.append(c)

    out: Dict[str, str] = {}
    if ranked:
        span = max(sum(steps), 1e-12)
        pos = np.concatenate([[0.0], np.cumsum(steps)]) / span
        for rank, c in enumerate(ranked):
            out[c] = _oklch_hex(0.70 if rank % 2 else 0.82, 0.15,
                                (PALETTE_START + PALETTE_ARC * pos[rank]) % 360.0)
    out.update({c: NEUTRAL_CLASSES[c] for c in classes if c in NEUTRAL_CLASSES})
    return out
