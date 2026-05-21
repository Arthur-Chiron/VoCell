import plotly.graph_objects as go
from typing import Tuple, Optional, Any

def create_3d_figure(
    dataset: str, 
    current_idx: int, 
    mesh_data: Tuple, 
    opacity_3d: float, 
    show_cut_plane: bool, 
    plane_wireframe: Optional[Tuple[list, list, list]] = None
) -> go.Figure:
    """
    Renders the 3D Plotly figure representing the nucleus voxel mesh and 
    the optional red cut plane indicator.
    
    Args:
        dataset: "CODEX" or "RESTORE".
        current_idx: Index of the nucleus.
        mesh_data: Tuple containing (vx, vy, vz, vi, vj, vk, vc, v_min, v_max).
        opacity_3d: Global opacity (0.1 to 1.0).
        show_cut_plane: Whether to draw the red square.
        plane_wireframe: Coordinates of the plane corners.
    """
    vx, vy, vz, vi, vj, vk, vc, v_min, v_max = mesh_data

    # Main voxel mesh (Mesh3d is efficient for voxel blocks)
    fig_3d = go.Figure(data=go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=vi, j=vj, k=vk,
        intensity=vc,
        cmin=v_min, cmax=v_max,
        colorscale='Viridis',
        opacity=opacity_3d,
        flatshading=True,
        name='Nucleus Volume'
    ))

    # Axis physical scale in micrometers
    # CODEX: ~0.377 um/px | RESTORE: ~0.15 um/px
    scale_um = 0.377 if dataset == "CODEX" else 0.15
    tick_vals = [0, 16, 32, 48, 64]
    tick_text = [f"{v * scale_um:.1f}" for v in tick_vals]

    fig_3d.update_layout(
        uirevision=f"{dataset}_{current_idx}",
        scene=dict(
            xaxis=dict(title='X (µm)', range=[0, 64], tickvals=tick_vals, ticktext=tick_text),
            yaxis=dict(title='Y (µm)', range=[0, 64], tickvals=tick_vals, ticktext=tick_text),
            zaxis=dict(title='Z (µm)', range=[0, 64], tickvals=tick_vals, ticktext=tick_text),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )

    # Visual cut plane (Red square wireframe)
    if show_cut_plane and plane_wireframe is not None:
        px, py, pz = plane_wireframe
        fig_3d.add_trace(go.Scatter3d(
            x=px, y=py, z=pz,
            mode='lines',
            line=dict(color='red', width=5),
            name='Plan de coupe',
            hoverinfo='skip'
        ))

    return fig_3d
