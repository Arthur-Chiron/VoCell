import numpy as np
import streamlit as st

@st.cache_data
def load_all_crops(filepath='data/CODEX/crops.npy'):
    """Charge le fichier NumPy contenant les crops 2D."""
    return np.load(filepath)

def get_nucleus_volume(idx, crops, depth=64, sigma=2.5):
    """
    Génère un volume 3D à partir d'un crop 2D en appliquant un profil 
    d'intensité Gaussien sur l'axe Z pour simuler l'épaisseur du noyau.
    """
    # Normalisation entre 0 et 1 (les données étant entre 0 et 255)
    crop_2d = crops[idx].astype(np.float32) / 255.0
    
    # Création d'un volume 3D vide
    volume_3d = np.zeros((depth, crop_2d.shape[0], crop_2d.shape[1]), dtype=np.float32)
    center_z = depth // 2
    
    for z in range(depth):
        # On calcule un poids de 1.0 au centre qui diminue doucement vers 0 sur les bords
        weight = np.exp(-((z - center_z)**2) / (2 * sigma**2))
        
        # Le noyau garde sa forme 2D brute à chaque coupe, 
        # mais son intensité s'éteint au fur et à mesure qu'on s'éloigne du centre
        volume_3d[z] = crop_2d * weight
        
    return volume_3d
