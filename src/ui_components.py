import os

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cloud
import data as data
import geometry as geom
import visualization as vis
import sam3d_engine as sam3d
import augment
from typing import Tuple, Optional, Dict


def thousands(n: int) -> str:
    """French thousands separator: a narrow no-break space, not a comma."""
    return f"{n:,}".replace(",", "\u202f")


def render_sidebar_dataset() -> str:
    """Renders the source selector in the sidebar."""
    with st.sidebar:
        if data.cloud_available():
            if st.button("← Nuage de noyaux", width="stretch",
                         help="Revenir à la vue d'ensemble des onze sources"):
                st.session_state["view"] = "cloud"
                st.rerun()
        st.markdown("## Dataset")
        dataset = st.selectbox(
            "Choix du dataset",
            data.DATASETS,
            label_visibility="collapsed",
            key="dataset_choice",
        )
        st.markdown("---")
    return dataset

def render_nucleus_selector(dataset: str) -> int:
    """Renders the nucleus selection controls (Dice, Number Input, Load button)."""
    state_key_idx = f"nucleus_idx_{dataset}"
    state_key_inp = f"idx_input_{dataset}"
    n_nuclei = data.dataset_size(dataset)

    # Initialize session state if needed
    if state_key_idx not in st.session_state:
        st.session_state[state_key_idx] = np.random.randint(0, n_nuclei)

    if state_key_inp not in st.session_state:
        st.session_state[state_key_inp] = st.session_state[state_key_idx]

    with st.sidebar.container():
        st.markdown("### Choix du noyau")
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
        st.caption(f"{thousands(n_nuclei)} noyaux")

    return min(st.session_state[state_key_idx], n_nuclei - 1)

def render_reconstruction_settings(dataset: str, current_idx: int) -> np.ndarray:
    """Renders 3D reconstruction profile settings and handles logic for generating volumes.

    Every source except RESTORE is natively 2D, so they all go through the
    same synthetic-depth reconstruction — there is nothing CODEX-specific
    about extruding a crop along Z. RESTORE is the one that brings a real
    stack, and keeps its own control.
    """
    if dataset == "RESTORE":
        st.sidebar.markdown("### Reconstruction 3D")
        interpolation = st.sidebar.selectbox(
            "Mode de reconstruction Z",
            ["Aucune", "Linéaire"],
            index=0,
            help="Choix du remplissage de l'espace vide entre les coupes réelles."
        )
        st.sidebar.info("Grille spatiale 64x64x64 finale. Canal DAPI.")
        return data.get_restore_volume(current_idx, interpolation=interpolation)

    volume_ai = data.get_ai_reconstructed_volume(dataset, current_idx)

    with st.sidebar.container():
        st.markdown("### Reconstruction 3D")
        recon_mode = st.selectbox(
            "Profil de reconstruction 3D",
            ["Aucune", "Gaussien", "Linéaire", "SAM3D (IA)"],
            index=3 if volume_ai is not None else 0,
            help="Méthode de génération du volume 3D."
        )

        if recon_mode == "SAM3D (IA)":
            cache_key = data.ai_cache_key(dataset, current_idx)
            if volume_ai is not None:
                st.success("✨ Sculpture IA chargée")
                if st.button("🔄 Refaire la sculpture", width="stretch"):
                    del st.session_state[cache_key]
                    st.rerun()
                return volume_ai
            st.info("Utilisez l'IA pour sculpter un volume réaliste.")
            if st.button("✨ Lancer la sculpture SAM3D", help="Génère un volume 3D réaliste via IA", width="stretch"):
                with st.spinner("L'IA sculpte le noyau..."):
                    try:
                        engine = sam3d.get_sam3d_engine()
                        crop_2d = (data.crop_2d(dataset, current_idx) * 255).astype(np.uint8)
                        st.session_state[cache_key] = engine.generate_voxels(crop_2d)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur IA : {e}")
            return np.zeros((64, 64, 64), dtype=np.float32)

        params = {}
        if recon_mode == "Gaussien":
            params['sigma'] = st.slider("Sigma (Écart-type)", 0.5, 10.0, 2.5, 0.1)
        elif recon_mode == "Linéaire":
            params['thickness'] = st.slider("Épaisseur du Noyau", 1, 32, 16, 1)
        return data.get_crop_volume(dataset, current_idx,
                                    interpolation_method=recon_mode, params=params)

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

def render_oblique_controls(volume: np.ndarray) -> Optional[Dict]:
    """Renders the ObliqueSection comparison panel; returns None when disabled.

    ObliqueSection lives in the shared `cellaug` package — see CLAUDE.md. It
    approximates in 2D what the rest of the app does geometrically in 3D, so
    the point of this panel is to put the two side by side on the same plane.
    """
    with st.sidebar.container():
        st.markdown("### Augmentation 2D — ObliqueSection")
        if not st.toggle("Comparer à l'augmentation", value=False,
                         help="Coupe oblique simulée en 2D (paquet partagé cellaug), "
                              "sans passer par le volume."):
            return None

        mode = st.radio(
            "Plan de l'augmentation",
            ["Suivre le plan de coupe", "Tirage aléatoire"],
            index=0,
            help="Suivre : le même plan que la coupe géométrique, pour comparer. "
                 "Tirage : un plan au hasard, comme à l'entraînement SimCLR.",
        )

        if "oblique_aspect" not in st.session_state:
            st.session_state["oblique_aspect"] = 0.4

        aspect = st.slider(
            "Aplatissement (demi-hauteur / rayon)", 0.1, 1.5,
            st.session_state["oblique_aspect"], 0.05,
            help="0.4 : noyau aplati en culture. 0.7 : noyau plus sphérique.",
        )
        if st.button("📐 Mesurer sur le volume", width="stretch",
                     help="Calibre l'approximation 2D sur le volume que la coupe "
                          "géométrique tranche réellement."):
            # borné sur la plage du curseur, sinon Streamlit refuse la valeur
            st.session_state["oblique_aspect"] = float(
                np.clip(augment.aspect_from_volume(volume), 0.1, 1.5))
            st.rerun()

        physical = {"aspect": aspect}
        with st.expander("Paramètres physiques"):
            physical["defocus_rate"] = st.slider("Défocalisation", 0.0, 1.0, 0.35, 0.05)
            physical["haze"] = st.slider("Voile diffus", 0.0, 1.0, 0.30, 0.05)
            physical["shape"] = "ellipse" if st.toggle(
                "Enveloppe elliptique", value=True,
                help="Suit le contour réel du noyau et préserve son élongation. "
                     "Désactivé : enveloppe circulaire.") else "circle"
            physical["preserve_support"] = st.toggle(
                "Préserver le fond nul", value=True,
                help="Les pixels nuls en entrée le restent : sinon le halo trahit "
                     "la vue augmentée.")
            photons = st.slider("Bruit de photons", 0, 500, 0, 10,
                                help="0 : désactivé.")
            physical["photons"] = photons or None

        settings = {"mode": mode, "physical": physical}

        if mode == "Tirage aléatoire":
            settings["max_tilt_deg"] = st.slider("Inclinaison max (°)", 0.0, 30.0, 6.0, 0.5)
            settings["max_offset"] = st.slider("Décalage max (fraction)", 0.0, 1.0, 0.35, 0.05)
            if "oblique_seed" not in st.session_state:
                st.session_state["oblique_seed"] = 0
            if st.button("🎲 Retirer un plan", width="stretch"):
                st.session_state["oblique_seed"] += 1
                st.rerun()
            settings["seed"] = st.session_state["oblique_seed"]

    return settings


def render_2d_info(dataset: str, current_idx: int):
    """Renders the top panel with 2D information and native slices."""
    if dataset != "RESTORE":
        st.markdown(f"### Noyau Original ({dataset})")
        info_col1, info_col2 = st.columns([1, 6])
        with info_col1:
            orig_crop = data.crop_2d(dataset, current_idx)
            # Simple 4x zoom for display
            orig_pixelated = np.repeat(np.repeat(orig_crop, 4, axis=0), 4, axis=1)
            st.image(orig_pixelated, width=None, use_container_width=True, clamp=True)
        with info_col2:
            line = f"**Index :** `#{current_idx}` | **Dataset :** {dataset}"
            classes = data.load_classes(dataset)
            if classes is not None:
                line += f" | **Classe :** `{classes.get(current_idx, 'Inconnu')}`"
            st.markdown(line)
            native = data.load_crops(dataset).shape[1]
            if native < 64:
                st.caption(
                    f"Crops natifs {native}×{native}, complétés par du noir "
                    "jusqu'à 64² : ce dataset n'a pas été rééchantillonné à la "
                    "taille de pixel des autres, et le recadrer préserve son "
                    "échelle là où l'agrandir inventerait une différence.")
    else:
        st.markdown("### Coupes natives (RESTORE — DAPI)")
        nuclei = data.load_restore_nuclei()
        raw_vol = np.asarray(nuclei[current_idx, ..., 0], dtype=np.float32) / 255.0
        slices = [raw_vol[z] for z in range(64) if np.max(raw_vol[z]) > 0.01]

        if slices:
            concat_img = np.concatenate(slices, axis=1)
            concat_pixelated = np.repeat(np.repeat(concat_img, 2, axis=0), 2, axis=1)
            st.image(concat_pixelated, width=None, use_container_width=True, clamp=True)
        st.markdown(f"**Index :** `#{current_idx}` | **Dataset :** RESTORE | **Coupes :** `{len(slices)}`")


# --- Point cloud view -------------------------------------------------------
# The cloud is a hand-rolled Streamlit component rather than a Plotly chart
# for one reason: Streamlit exposes click and lasso events on a chart, but not
# hover, and a hover that costs a server round-trip is not a hover. Here the
# 172 358 points are drawn and picked entirely in the browser, and only a
# click travels back to Python.

_CLOUD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "components", "nuclei_cloud")
_nuclei_cloud = components.declare_component("nuclei_cloud", path=_CLOUD_DIR)


@st.cache_data(show_spinner=False)
def _cloud_projection(mode: str, x_key: str, y_key: str, jitter: bool):
    """Positions for the cloud, already packed for the component."""
    feats = data.load_cloud_features()
    coords, axes = cloud.project(feats["values"], feats["names"],
                                 mode, x_key, y_key, jitter)
    return cloud.quantize(coords), axes


def render_cloud_view() -> None:
    """Full-page nucleus cloud. Selecting a point switches to the explorer."""
    st.markdown("## Nuage de noyaux — CODEX & RESTORE")
    st.caption(
        "Les 172 358 noyaux des deux datasets, placés par leurs descripteurs "
        "morphologiques. Survoler affiche le noyau, cliquer l'ouvre dans "
        "l'explorateur 3D. Molette pour zoomer, glisser pour déplacer : sous "
        "un millier de points visibles, les vignettes remplacent les points."
    )

    feats = data.load_cloud_features()
    names = feats["names"]
    labels = {k: cloud.label(k) for k in names}

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1.4])
    with c1:
        mode = st.selectbox("Disposition", cloud.LAYOUT_MODES, index=0)
    disabled = mode == "ACP"
    with c2:
        x_key = st.selectbox("Axe x", names, index=names.index("area"),
                             format_func=lambda k: labels[k], disabled=disabled)
    with c3:
        y_key = st.selectbox("Axe y", names,
                             index=names.index("mean_intensity"),
                             format_func=lambda k: labels[k], disabled=disabled)
    with c4:
        jitter = st.toggle("Dispersion", value=True,
                           help="Écarte les noyaux empilés sur une valeur "
                                "identique (l'aire est un compte de pixels). "
                                "Purement cosmétique : le décalage reste "
                                "inférieur au pas entre deux valeurs.")

    xy, axes = _cloud_projection(mode, x_key, y_key, jitter)

    selection = _nuclei_cloud(xy=xy, labels=axes, height=640,
                              key="cloud_selection", default=None)

    # A click is delivered again on every rerun, so it is the nonce, not the
    # payload, that says "this is new".
    if selection and selection.get("nonce") != st.session_state.get("cloud_nonce"):
        st.session_state["cloud_nonce"] = selection["nonce"]
        ds, idx = selection["dataset"], int(selection["index"])
        st.session_state["dataset_choice"] = ds
        st.session_state[f"nucleus_idx_{ds}"] = idx
        st.session_state[f"idx_input_{ds}"] = idx
        st.session_state["view"] = "explorer"
        st.rerun()

    st.caption(
        "Descripteurs calculés en pixels, sans harmoniser les échelles "
        "(CODEX ~0,377 µm/px, RESTORE ~0,15 µm/px) : l'écart entre les deux "
        "amas est le fossé de domaine entre les deux acquisitions, pas un "
        "artefact. RESTORE est réduit à sa projection maximale selon Z, comme "
        "pour l'augmentation 2D."
    )
