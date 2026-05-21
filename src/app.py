import streamlit as st
import numpy as np

import ui_components as ui
import geometry as geom
import visualization as vis

# Page configuration for a wide, premium layout
st.set_page_config(layout="wide", page_title="VoCell — 3D Nucleus Explorer")

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

# --- CORE GEOMETRY: Plane and Clipping ---
# 1. Calculate the 3D plane vectors
cx, cy, cz = volume.shape[2] / 2.0, volume.shape[1] / 2.0, volume.shape[0] / 2.0
normal_vec, P0, u, v = geom.get_plane_vectors(azimuth, elevation, slice_offset, cx, cy, cz)

# 2. Apply clipping mask for 3D view
volume_to_render = geom.apply_clipping(volume, normal_vec[0], normal_vec[1], normal_vec[2], slice_offset, cx, cy, cz, visibility_mode)

# --- PANEL: 2D Information & Slice ---
ui.render_2d_info(dataset, current_idx)
st.markdown("---")

# --- MAIN VIEW: 3D RENDER & 2D SLICE ---
col_3d, col_2d = st.columns(2)

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
