import csv
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

# --- Sources -----------------------------------------------------------------
# The ten harmonised crop sets of the neighbouring `cellf-supervised` repo,
# reached through the data/crops symlink, plus VoCell's own RESTORE volumes.
# Order matches scripts/build_cloud.py, which is what the cloud's global index
# space is built from.

CROPS_ROOT = 'data/crops'
RESTORE_PATH = 'data/RESTORE/nuclei.npy'

# Datasets whose crops are natively 2D. Every one of them can be pushed
# through the synthetic-depth reconstruction, so the explorer accepts them
# all; only RESTORE brings a real volume.
CROP_DATASETS: List[str] = [
    'CODEX', 'HPA', 'BBBC051', 'TissueNet', 'HelaCytoNuc', 'DSB2018',
    'NuInSeg', 'S-BSST265', 'NucleusSegData', 'AitslabBioimaging1',
]
DATASETS: List[str] = CROP_DATASETS + ['RESTORE']

# Column holding the cell-type label, for the sources that have one.
LABEL_COLUMN: Dict[str, str] = {
    'CODEX': 'classes', 'HPA': 'classes', 'BBBC051': 'classes',
}
# CODEX ships an explicit position column; HPA and BBBC051 rows are in crop
# order (verified: median nucleus area varies 1.70x across HPA cell lines with
# the real labels, 1.08x once shuffled).
INDEX_COLUMN: Dict[str, str] = {'CODEX': 'crop_index'}

# Sources whose zero background is an intensity threshold rather than a
# segmentation mask. `generate_crops` upstream multiplies each crop by
# `mask_crop == label`, so for everyone else the non-zero support *is* the
# nucleus mask -- measured median of 1 connected component. HPA never went
# through a mask: median 8 components, 40% of crops touching the border. The
# UI says which is which rather than passing the second off as the first.
THRESHOLD_SUPPORT = {'HPA'}

# --- Class Mapping (Biomarker groups) ---
CLASS_MAPPING = {
    'adipocytes':             'Adipocyte',
    'B cells':                'B cells',
    'plasma cells':           'B cells',
    'CD3+ T cells':           'T cells',
    'CD4+ T cells':           'T cells',
    'CD4+ T cells CD45RO+':   'T cells',
    'CD4+ T cells GATA3+':    'T cells',
    'CD8+ T cells':           'T cells',
    'granulocytes':           'Granulocytes',
    'CD11b+CD68+ macrophages':'Macrophages',
    'CD163+ macrophages':     'Macrophages',
    'CD68+ macrophages':      'Macrophages',
    'CD68+ macrophages GzmB+':'Macrophages',
    'CD68+CD163+ macrophages':'Macrophages',
    'NK cells':               'NK cells',
    'nerves':                 'Nerves',
    'CD11b+ monocytes':       'Monocytes',
    'smooth muscle':          'Smooth muscle cells',
    'Tregs':                  'Tregs',
    'tumor cells':            'Neoplastic cells',
    'lymphatics':             'Vasculature',
    'vasculature':            'Vasculature',
    'CD11c+ DCs':             'Dendritic cells',
    'stroma':                 'Others',
}
CLASS_REMAP: Dict[str, Dict[str, str]] = {'CODEX': CLASS_MAPPING}


# --- Loading -----------------------------------------------------------------

@st.cache_resource
def load_crops(dataset: str = 'CODEX') -> np.ndarray:
    """Memory-map a dataset's 2D crops. Shape (N, S, S), uint8.

    Memory-mapped and cached as a resource rather than loaded and cached as
    data: TissueNet alone is 5.5 GB, and @st.cache_data would pickle every
    byte of it to serialise the cache entry.
    """
    return np.load(os.path.join(CROPS_ROOT, dataset, 'crops.npy'), mmap_mode='r')


@st.cache_resource
def load_restore_nuclei(filepath: str = RESTORE_PATH) -> np.ndarray:
    """Memory-map the RESTORE dataset (3D nuclei). Shape (N, 64, 64, 64, 1)."""
    return np.load(filepath, mmap_mode='r')


def dataset_size(dataset: str) -> int:
    return len(load_restore_nuclei() if dataset == 'RESTORE'
               else load_crops(dataset))


@st.cache_data
def load_classes(dataset: str) -> Optional[Dict[int, str]]:
    """Cell-type label per crop index, or None for the unlabelled sources."""
    col = LABEL_COLUMN.get(dataset)
    if not col:
        return None
    remap = CLASS_REMAP.get(dataset, {})
    index_col = INDEX_COLUMN.get(dataset)
    path = os.path.join(CROPS_ROOT, dataset, 'crop_metadata.csv')
    out: Dict[int, str] = {}
    with open(path, mode='r', encoding='utf-8', newline='') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if index_col:
                try:
                    i = int(row[index_col])
                except (ValueError, KeyError, TypeError):
                    continue
            raw = (row.get(col) or '').strip()
            out[i] = remap.get(raw, raw) if raw else 'non étiqueté'
    return out


def crop_2d(dataset: str, idx: int) -> np.ndarray:
    """One nucleus as a 64x64 float32 image in [0, 1].

    BBBC051 ships native 32x32 crops and was never rescaled to the others'
    pixel size, so it is zero-padded rather than resized: padding keeps its
    scale, resizing would double the apparent diameter of every nucleus and
    invent a difference that is not in the data.
    """
    img = np.asarray(load_crops(dataset)[idx], dtype=np.float32) / 255.0
    if img.shape[0] < 64:
        o = (64 - img.shape[0]) // 2
        padded = np.zeros((64, 64), dtype=np.float32)
        padded[o:o + img.shape[0], o:o + img.shape[1]] = img
        return padded
    return img


# --- Volumes -----------------------------------------------------------------

def support_2d(dataset: str, idx: int) -> np.ndarray:
    """The nucleus mask of a crop, as a 64x64 float32 image of 0 and 1.

    Not computed: the crops are already masked upstream, so the mask is
    exactly the non-zero support. See THRESHOLD_SUPPORT for the one source
    where that support is a threshold instead.
    """
    return (crop_2d(dataset, idx) > 0).astype(np.float32)


def support_label(dataset: str) -> str:
    return "Support (seuil)" if dataset in THRESHOLD_SUPPORT else "Masque"


def get_crop_volume(dataset: str, idx: int, interpolation_method: str = "Gaussien",
                    params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Generate a synthetic 3D volume from a 2D crop.
    Simulates depth via Gaussian or Linear attenuation.
    """
    crop = crop_2d(dataset, idx)
    depth = 64
    volume_3d = np.zeros((depth, crop.shape[0], crop.shape[1]), dtype=np.float32)
    center_z = depth // 2

    if params is None:
        params = {}

    # Slice-by-slice weight assignment
    for z in range(depth):
        weight = 0.0
        if interpolation_method == "Gaussien":
            sigma = params.get('sigma', 2.5)
            weight = np.exp(-((z - center_z)**2) / (2 * sigma**2))
        elif interpolation_method == "Linéaire":
            thickness = params.get('thickness', 16)
            thickness_d2 = thickness / 2.0
            dist = abs(z - center_z)
            if dist < thickness_d2:
                weight = 1.0 - (dist / thickness_d2)
        elif interpolation_method == "Aucune":
            if z == center_z:
                weight = 1.0
        volume_3d[z] = crop * weight

    return volume_3d


def get_restore_volume(idx: int, interpolation: str = "Aucune") -> np.ndarray:
    """
    Returns the real 3D volume from RESTORE dataset.
    Optionally interpolates between sparse confocal slices.
    """
    nuclei = load_restore_nuclei()
    vol = np.asarray(nuclei[idx, ..., 0], dtype=np.float32) / 255.0

    if interpolation == "Linéaire":
        # Indices of slices with signal
        z_indices = [z for z in range(64) if np.max(vol[z]) > 0.01]

        # Linear interpolation between detected slices
        if len(z_indices) > 1:
            for i in range(len(z_indices) - 1):
                z1, z2 = z_indices[i], z_indices[i+1]
                dist = z2 - z1
                if dist > 1:
                    for z in range(z1 + 1, z2):
                        alpha = (z - z1) / dist
                        vol[z] = (1.0 - alpha) * vol[z1] + alpha * vol[z2]

    return vol


def get_ai_reconstructed_volume(dataset: str, idx: int) -> Optional[np.ndarray]:
    """
    Checks for a SAM3D AI reconstruction in Streamlit session state.
    """
    return st.session_state.get(ai_cache_key(dataset, idx))


def ai_cache_key(dataset: str, idx: int) -> str:
    return f"ai_vol_{dataset}_{idx}"


def get_source_2d(dataset: str, idx: int) -> np.ndarray:
    """The 2D image the ObliqueSection augmentation is applied to.

    The crop sets are natively 2D. RESTORE is a real stack, so we project it
    along Z: that projection is what a 2D acquisition of the same nucleus
    would give, and therefore the fair input for a 2D augmentation.
    """
    if dataset != "RESTORE":
        return crop_2d(dataset, idx)
    nuclei = load_restore_nuclei()
    return np.asarray(nuclei[idx, ..., 0], dtype=np.float32).max(axis=0) / 255.0


# --- Point cloud (scripts/build_cloud.py) ---

CLOUD_META = 'src/components/nuclei_cloud/meta.json'
LOGO_LETTERS = 'src/components/logo/letters.json'


def cloud_available(filepath: str = CLOUD_META) -> bool:
    """The cloud view needs assets a fresh clone does not have."""
    return os.path.exists(filepath)


def logo_available(filepath: str = LOGO_LETTERS) -> bool:
    """The header wordmark. Unlike the cloud's, its assets are committed:
    six voxel sprites weigh 15 kB, so the logo works on a fresh clone with no
    data at all. scripts/build_logo.py regenerates them."""
    return os.path.exists(filepath)


@st.cache_data
def _read_cloud_meta(filepath: str, mtime: float) -> Dict[str, Any]:
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)


def load_cloud_meta(filepath: str = CLOUD_META) -> Dict[str, Any]:
    """Descriptor names, labels and dataset manifest for the cloud controls.

    Only the small header: everything per-nucleus is fetched by the component
    itself over HTTP, and never travels through Python.

    Cached on the file's mtime and not on its path alone. build_cloud.py
    rewrites meta.json in place, so a cache keyed on the path would serve the
    previous build until the server is restarted -- a new layout mode or a new
    source would simply not appear. The component has the same problem with
    Streamlit's `Cache-Control: public` and solves it the same way, with a
    no-store fetch and a build id.
    """
    return _read_cloud_meta(filepath, os.path.getmtime(filepath))
