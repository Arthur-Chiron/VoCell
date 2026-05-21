import numpy as np
import streamlit as st
import csv
from typing import Dict, Optional, Any

# --- Class Mapping (Biomarker groups) ---
CLASS_MAPPING = {
    'adipocytes':             'Adipocyte',
    'B cells':                'B cells',
    'plasma cells':           'B cells',
    'CD3+ T cells':           'T Cells',
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

@st.cache_data
def load_all_crops(filepath: str = 'data/CODEX/crops.npy') -> np.ndarray:
    """Load the NumPy file containing 2D CODEX crops."""
    return np.load(filepath)

@st.cache_data
def load_metadata(filepath: str = 'data/CODEX/crop_metadata.csv') -> Dict[int, str]:
    """Load cell descriptions from CSV and map to consolidated classes."""
    metadata = {}
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row['crop_index'])
                raw_class = row['classes']
                mapped_class = CLASS_MAPPING.get(raw_class, raw_class)
                metadata[idx] = mapped_class
            except (ValueError, KeyError):
                continue
    return metadata

@st.cache_data
def load_restore_nuclei(filepath: str = 'data/RESTORE/nuclei.npy') -> np.ndarray:
    """Load RESTORE dataset (3D nuclei). Shape: (N, 64, 64, 64, 1)."""
    return np.load(filepath)

def get_codex_volume(idx: int, interpolation_method: str = "Gaussien", params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Generate a synthetic 3D volume from a 2D CODEX crop.
    Simulates depth via Gaussian or Linear attenuation.
    """
    crops = load_all_crops()
    crop_2d = crops[idx].astype(np.float32) / 255.0
    depth = 64
    volume_3d = np.zeros((depth, crop_2d.shape[0], crop_2d.shape[1]), dtype=np.float32)
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
        volume_3d[z] = crop_2d * weight

    return volume_3d

def get_restore_volume(idx: int, interpolation: str = "Aucune") -> np.ndarray:
    """
    Returns the real 3D volume from RESTORE dataset.
    Optionally interpolates between sparse confocal slices.
    """
    nuclei = load_restore_nuclei()
    vol_uint8 = nuclei[idx, ..., 0]  # Extract single channel
    vol = vol_uint8.astype(np.float32) / 255.0
    
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

def get_ai_reconstructed_volume(idx: int) -> Optional[np.ndarray]:
    """
    Checks for a SAM3D AI reconstruction in Streamlit session state.
    """
    cache_key = f"ai_vol_{idx}"
    return st.session_state.get(cache_key)
