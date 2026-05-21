import streamlit as st
import numpy as np
import data as data
import geometry as geom
import visualization as vis
import sam3d_engine as sam3d
from typing import Tuple, Optional

def render_sidebar_dataset() -> str:
    """Renders the dataset selection radio button in the sidebar."""
    with st.sidebar:
        st.markdown("## Dataset")
        dataset = st.radio(
            "Choix du dataset",
            ["CODEX", "RESTORE"],
            horizontal=True,
            label_visibility="collapsed"
        )
        st.markdown("---")
    return dataset

def render_nucleus_selector(dataset: str) -> int:
    """Renders the nucleus selection controls (Dice, Number Input, Load button)."""
    state_key_idx = f"nucleus_idx_{dataset}"
    state_key_inp = f"idx_input_{dataset}"

    # Initialize session state if needed
    if state_key_idx not in st.session_state:
        if dataset == "CODEX":
            n_nuclei = len(data.load_all_crops())
        else:
            n_nuclei = len(data.load_restore_nuclei())
        st.session_state[state_key_idx] = np.random.randint(0, n_nuclei)

    if state_key_inp not in st.session_state:
        st.session_state[state_key_inp] = st.session_state[state_key_idx]

    with st.sidebar.container():
        st.markdown("### Choix du noyau")
        if dataset == "CODEX":
            n_nuclei = len(data.load_all_crops())
        else:
            n_nuclei = len(data.load_restore_nuclei())
        max_idx = n_nuclei - 1

        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("🎲", help="Générer un index aléatoire", key=f"dice_{dataset}"):
                st.session_state[state_key_inp] = np.random.randint(0, n_nuclei)
                st.session_state[state_key_idx] = st.session_state[state_key_inp]
                st.rerun()

        with col2:
            st.number_input('Index', min_value=0, max_value=max_idx, key=state_key_inp, step=1, label_visibility="collapsed")
            
        with col3:
            if st.button('➤', help="Charger le noyau", key=f"load_{dataset}"):
                st.session_state[state_key_idx] = st.session_state[state_key_inp]
                st.rerun()

    return st.session_state[state_key_idx]

def render_reconstruction_settings(dataset: str, current_idx: int) -> np.ndarray:
    """Renders 3D reconstruction profile settings and handles logic for generating volumes."""
    if dataset == "CODEX":
        volume_ai = data.get_ai_reconstructed_volume(current_idx)
        
        with st.sidebar.container():
            st.markdown("### Reconstruction 3D")
            recon_mode = st.selectbox(
                "Profil de reconstruction 3D",
                ["Aucune", "Gaussien", "Linéaire", "SAM3D (IA)"],
                index=3 if volume_ai is not None else 0,
                help="Méthode de génération du volume 3D."
            )

            if recon_mode == "SAM3D (IA)":
                if volume_ai is not None:
                    st.success("✨ Sculpture IA chargée")
                    if st.button("🔄 Refaire la sculpture", width="stretch"):
                        del st.session_state[f"ai_vol_{current_idx}"]
                        st.rerun()
                    return volume_ai
                else:
                    st.info("Utilisez l'IA pour sculpter un volume réaliste.")
                    if st.button("✨ Lancer la sculpture SAM3D", help="Génère un volume 3D réaliste via IA", width="stretch"):
                        with st.spinner("L'IA sculpte le noyau..."):
                            try:
                                engine = sam3d.get_sam3d_engine()
                                crops = data.load_all_crops()
                                crop_2d = crops[current_idx]
                                volume_res = engine.generate_voxels(crop_2d)
                                st.session_state[f"ai_vol_{current_idx}"] = volume_res
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erreur IA : {e}")
                    return np.zeros((64, 64, 64), dtype=np.float32)
            else:
                params = {}
                if recon_mode == "Gaussien":
                    params['sigma'] = st.slider("Sigma (Écart-type)", 0.5, 10.0, 2.5, 0.1)
                elif recon_mode == "Linéaire":
                    params['thickness'] = st.slider("Épaisseur du Noyau", 1, 32, 16, 1)
                return data.get_codex_volume(current_idx, interpolation_method=recon_mode, params=params)
    else:
        # RESTORE Dataset
        st.sidebar.markdown("### Reconstruction 3D")
        interpolation = st.sidebar.selectbox(
            "Mode de reconstruction Z",
            ["Aucune", "Linéaire"],
            index=0,
            help="Choix du remplissage de l'espace vide entre les coupes réelles."
        )
        st.sidebar.info("Grille spatiale 64x64x64 finale. Canal DAPI.")
        return data.get_restore_volume(current_idx, interpolation=interpolation)

def render_slicing_controls() -> Tuple[float, float, float, str, bool, float, float]:
    """Renders the slicing sliders and returns the selected values."""
    with st.sidebar.container():
        st.markdown("### Coupe")
        azimuth = st.slider("Azimut (Longitude)", -180, 180, 0)
        elevation = st.slider("Élévation (Latitude)", -90, 90, 90)
        slice_offset = st.slider("Position de la coupe", -40, 40, 0)
        visibility_mode = st.radio(
            "Visibilité 3D relative à la coupe",
            ["Tout afficher", "Masquer au-dessus", "Masquer au-dessous"],
            index=0
        )
        show_cut_plane = st.toggle("Afficher le carré de coupe", value=True)

        st.markdown("### Rendu Volumétrique")
        threshold_3d = st.slider("Seuil d'intensité", 0.0, 0.9, 0.05, 0.01)
        opacity_3d = st.slider("Opacité globale", 0.1, 1.0, 0.5, 0.05)
        
    return azimuth, elevation, slice_offset, visibility_mode, show_cut_plane, threshold_3d, opacity_3d

def render_2d_info(dataset: str, current_idx: int):
    """Renders the top panel with 2D information and native slices."""
    if dataset == "CODEX":
        crops = data.load_all_crops()
        st.markdown("### Noyau Original (CODEX)")
        info_col1, info_col2 = st.columns([1, 6])
        with info_col1:
            orig_crop = crops[current_idx].astype(np.float32) / 255.0
            # Simple 4x zoom for display
            orig_pixelated = np.repeat(np.repeat(orig_crop, 4, axis=0), 4, axis=1)
            st.image(orig_pixelated, width=None, use_container_width=True, clamp=True)
        with info_col2:
            metadata = data.load_metadata()
            cell_type = metadata.get(current_idx, "Inconnu")
            st.markdown(f"**Index :** `#{current_idx}` | **Classe :** `{cell_type}`")
    else:
        st.markdown("### Coupes natives (RESTORE — DAPI)")
        nuclei = data.load_restore_nuclei()
        raw_vol = nuclei[current_idx, ..., 0].astype(np.float32) / 255.0
        slices = [raw_vol[z] for z in range(64) if np.max(raw_vol[z]) > 0.01]
        
        if slices:
            concat_img = np.concatenate(slices, axis=1)
            concat_pixelated = np.repeat(np.repeat(concat_img, 2, axis=0), 2, axis=1)
            st.image(concat_pixelated, width=None, use_container_width=True, clamp=True)
        st.markdown(f"**Index :** `#{current_idx}` | **Dataset :** RESTORE | **Coupes :** `{len(slices)}`")
