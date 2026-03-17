import numpy as np
import streamlit as st
import csv

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
def load_all_crops(filepath='data/CODEX/crops.npy'):
    """Charge le fichier NumPy contenant les crops 2D."""
    return np.load(filepath)

@st.cache_data
def load_metadata(filepath='data/CODEX/crop_metadata.csv'):
    """Charge les descriptions des noyaux depuis le CSV et map les classes."""
    metadata = {}
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row['crop_index'])
                raw_class = row['classes']
                mapped_class = CLASS_MAPPING.get(raw_class, raw_class) # Fallback if not found
                metadata[idx] = {
                    'raw_class': raw_class,
                    'mapped_class': mapped_class
                }
            except (ValueError, KeyError):
                pass
    return metadata

def get_nucleus_volume(idx, crops, depth=64, sigma=2.5):
    """
    Génère un volume 3D à partir d'un crop 2D en appliquant un profil 
    d'intensité Gaussien sur l'axe Z pour simuler l'épaisseur du noyau.
    """
    crop_2d = crops[idx].astype(np.float32) / 255.0
    volume_3d = np.zeros((depth, crop_2d.shape[0], crop_2d.shape[1]), dtype=np.float32)
    center_z = depth // 2
    
    for z in range(depth):
        weight = np.exp(-((z - center_z)**2) / (2 * sigma**2))
        volume_3d[z] = crop_2d * weight
        
    return volume_3d
