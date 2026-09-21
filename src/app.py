import streamlit as st
import numpy as np

import ui_components as ui
import geometry as geom
import visualization as vis
import augment
import data

# Page configuration for a wide, premium layout
st.set_page_config(layout="wide", page_title="VoCell — 3D Nucleus Explorer")

# --- VIEW ROUTING ---
# Two views share this script: the cloud of every nucleus in both datasets,
# and the explorer below. The cloud is the entry point when its assets exist
# (scripts/build_cloud.py); clicking a point writes the dataset and index the
# explorer reads and switches over.
if "view" not in st.session_state:
    st.session_state["view"] = "cloud" if data.cloud_available() else "explorer"

if st.session_state["view"] == "cloud":
    ui.render_cloud_view()
    st.stop()

# --- SIDEBAR: Dataset & Navigation ---
dataset = ui.render_sidebar_dataset()
current_idx = ui.render_nucleus_selector(dataset)
st.sidebar.markdown("---")

# --- SIDEBAR: 3D Reconstruction Settings ---
volume = ui.render_reconstruction_settings(dataset, current_idx)
st.sidebar.markdown("---")

# --- SIDEBAR: Slicing and Visualization Controls ---
azimuth, elevation, slice_offset, visibility_mode, show_cut_plane, threshold_3d, opacity_3d = ui.render_slicing_controls()
st.sidebar.markdown("---")

# --- SIDEBAR: 2D augmentation (shared `cellaug` package) ---
oblique = ui.render_oblique_controls(volume)
st.sidebar.markdown("---")

# --- CORE GEOMETRY: Plane and Clipping ---
# 1. Calculate the 3D plane vectors
cx, cy, cz = volume.shape[2] / 2.0, volume.shape[1] / 2.0, volume.shape[0] / 2.0
normal_vec, P0, u, v = geom.get_plane_vectors(azimuth, elevation, slice_offset, cx, cy, cz)

# 2. Apply clipping mask for 3D view
volume_to_render = geom.apply_clipping(volume, normal_vec[0], normal_vec[1], normal_vec[2], slice_offset, cx, cy, cz, visibility_mode)

# --- PANEL: 2D Information & Slice ---
ui.render_2d_info(dataset, current_idx)
st.markdown("---")

# --- MAIN VIEW: 3D RENDER, 2D SLICE, and optionally the 2D augmentation ---
columns = st.columns(3 if oblique else 2)
col_3d, col_2d = columns[0], columns[1]

with col_3d:
    st.subheader("Rendu Volumétrique 3D")
    
    # Generate mesh data from voxels above threshold
    mesh_data = geom.get_voxel_mesh_data(volume_to_render, threshold=threshold_3d)

    # Optional wireframe for the cutting plane
    plane_wireframe = geom.get_plane_wireframe(P0, u, v, plane_size=40) if show_cut_plane else None

    # Final Plotly figure
    fig_3d = vis.create_3d_figure(dataset, current_idx, mesh_data, opacity_3d, show_cut_plane, plane_wireframe)
    st.plotly_chart(fig_3d, use_container_width=True)

with col_2d:
    st.subheader("Coupe Transversale 2D")

    # Extract dynamic 2D slice image from the volume
    # slice_size=90 allows to see a bit beyond the 64x64 core if needed (with padding)
    img_pixelated = geom.extract_2d_slice(volume, P0, u, v, slice_size=90)
    st.image(img_pixelated, use_container_width=True, clamp=True)
    st.caption("Échantillonnage du volume 3D le long du plan.")

# --- THIRD PANEL: the same plane, approximated in 2D by ObliqueSection ---
# The geometric slice above needs a volume; this one never builds one. Putting
# them side by side is the whole point: it shows what the SimCLR augmentation
# of cellf-supervised does and does not reproduce.
if oblique:
    with columns[2]:
        st.subheader("Augmentation ObliqueSection")

        source_2d = data.get_source_2d(dataset, current_idx)

        if oblique["mode"] == "Tirage aléatoire":
            aug_img, aug_params = augment.apply_random(
                source_2d, oblique["seed"], oblique["physical"],
                oblique["max_tilt_deg"], oblique["max_offset"],
            )
        else:
            plane = augment.plane_to_oblique(
                azimuth, elevation, slice_offset, source_2d,
                oblique["physical"]["aspect"],
            )
            if plane["clamped"]:
                st.warning(
                    f"Plan trop redressé (élévation < {augment.MIN_ELEVATION_DEG:.0f}°) : "
                    "ObliqueSection modèle une coupe d'un noyau aplati et ne "
                    "représente pas ce cas. Inclinaison bridée."
                )
            aug_img, aug_params = augment.apply_forced(
                source_2d, plane, oblique["physical"],
            )

        st.image(np.repeat(np.repeat(aug_img, 8, axis=0), 8, axis=1),
                 use_container_width=True, clamp=True)

        signal = aug_img.sum() / max(float(source_2d.sum()), 1e-9)
        st.caption(
            f"Inclinaison `{aug_params['tilt_deg']:.1f}°` · "
            f"azimut `{np.degrees(aug_params['phi']) % 360:.0f}°` · "
            f"décalage `{aug_params['offset_frac']:+.2f}` demi-hauteur · "
            f"rayon estimé `{aug_params['radius_px']:.1f} px` · "
            f"signal conservé `{signal:.0%}`"
        )
        if dataset == "RESTORE":
            st.caption("Entrée : projection maximale du volume selon Z.")
