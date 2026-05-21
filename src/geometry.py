import numpy as np
import scipy.ndimage as ndi
from typing import Tuple

def get_plane_vectors(azimuth: float, elevation: float, slice_offset: float, cx: float, cy: float, cz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the normal vector, origin point (P0), and plane basis vectors (u, v) 
    based on spherical coordinates (azimuth, elevation) and an offset.
    
    Args:
        azimuth: Longitude in degrees.
        elevation: Latitude in degrees.
        slice_offset: Distance from center.
        cx, cy, cz: Volume center coordinates.
    """
    theta = np.radians(azimuth)
    phi = np.radians(elevation)

    nx = np.cos(phi) * np.cos(theta)
    ny = np.cos(phi) * np.sin(theta)
    nz = np.sin(phi)
    normal_vec = np.array([nx, ny, nz])

    # Plane origin point (P0)
    P0 = np.array([cx, cy, cz]) + slice_offset * normal_vec

    # Arbitrary 'Up' vector to orient the 2D slice projection.
    # We switch if the 'Up' vector is too close to the normal.
    up = np.array([0, 0, 1.0])
    if abs(np.dot(up, normal_vec)) > 0.99:
        up = np.array([0, 1.0, 0])
        
    u = np.cross(up, normal_vec)
    u /= (np.linalg.norm(u) + 1e-8)
    v = np.cross(normal_vec, u)
    
    return normal_vec, P0, u, v

def apply_clipping(volume: np.ndarray, nx: float, ny: float, nz: float, slice_offset: float, cx: float, cy: float, cz: float, mode: str) -> np.ndarray:
    """
    Mask a part of the volume (above or below) relative to the cutting plane.
    
    Optimization: Uses broadcasting instead of np.mgrid for speed.
    """
    if mode == "Tout afficher":
        return volume

    volume_to_render = volume.copy()
    sz, sy, sx = volume.shape
    
    # Construct 1D coordinate arrays for broadcasting
    z = np.arange(sz)
    y = np.arange(sy)
    x = np.arange(sx)
    
    # Calculate signed distance from the plane: (x - cx)*nx + (y - cy)*ny + (z - cz)*nz - offset
    # Broadcasting happens automatically: [sz, 1, 1], [1, sy, 1], [1, 1, sx]
    dist = (x[None, None, :] - cx) * nx + \
           (y[None, :, None] - cy) * ny + \
           (z[:, None, None] - cz) * nz - slice_offset
    
    # Apply masking based on the selected mode
    if mode == "Masquer au-dessus":
        volume_to_render[dist > 0] = 0
    elif mode == "Masquer au-dessous":
        volume_to_render[dist < 0] = 0
        
    return volume_to_render

def get_voxel_mesh_data(volume_to_render: np.ndarray, threshold: float = 0.05) -> Tuple:
    """
    Generate mesh data (vertices, faces, intensities) for optimized 
    voxel block rendering in Plotly's Mesh3d.
    Only processes voxels above the specified intensity threshold.
    """
    z_idx, y_idx, x_idx = np.where(volume_to_render > threshold)
    values = volume_to_render[volume_to_render > threshold]
    
    if len(x_idx) == 0:
        return [], [], [], [], [], [], [], 0, 1
        
    # Voxel corner offsets
    dx = np.array([-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5])
    dy = np.array([-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5])
    dz = np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5])
    
    # Triangle face indices for a cube
    faces_i = np.array([7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2])
    faces_j = np.array([3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3])
    faces_k = np.array([0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6])
    
    n_voxels = len(x_idx)
    # Flatten all vertices for all active voxels
    vx = (x_idx[:, None] + dx).flatten()
    vy = (y_idx[:, None] + dy).flatten()
    vz = (z_idx[:, None] + dz).flatten()
    
    vc = np.repeat(values, 8)
    
    # Global face indexing
    offsets = np.arange(n_voxels)[:, None] * 8
    vi = (faces_i + offsets).flatten()
    vj = (faces_j + offsets).flatten()
    vk = (faces_k + offsets).flatten()
    
    v_min, v_max = float(values.min()), float(values.max())
    
    return vx, vy, vz, vi, vj, vk, vc, v_min, v_max

def extract_2d_slice(volume: np.ndarray, P0: np.ndarray, u: np.ndarray, v: np.ndarray, slice_size: int = 90) -> np.ndarray:
    """
    Extract a 2D transversal slice from the 3D volume along the specified plane.
    Uses linear interpolation for the sampling.
    """
    # Create coordinate grid for the 2D plane
    grid_x, grid_y = np.meshgrid(
        np.arange(slice_size) - slice_size // 2,
        np.arange(slice_size) - slice_size // 2
    )
    
    # Map 2D plane grid to 3D volume coordinates
    X_sample = P0[0] + grid_x * u[0] + grid_y * v[0]
    Y_sample = P0[1] + grid_x * u[1] + grid_y * v[1]
    Z_sample = P0[2] + grid_x * u[2] + grid_y * v[2]
    
    # Sample volume using map_coordinates
    coords = np.stack([Z_sample, Y_sample, X_sample])
    img_slice = ndi.map_coordinates(volume, coords, order=1, cval=0.0)
    
    # Upscale for display using block-repetition (nearest neighbor look)
    return np.repeat(np.repeat(img_slice, 8, axis=0), 8, axis=1)

def get_plane_wireframe(P0: np.ndarray, u: np.ndarray, v: np.ndarray, plane_size: float = 40) -> Tuple[list, list, list]:
    """
    Calculate corner coordinates for a square representing the cutting plane.
    """
    corners = [
        P0 - (plane_size/2)*u - (plane_size/2)*v,
        P0 + (plane_size/2)*u - (plane_size/2)*v,
        P0 + (plane_size/2)*u + (plane_size/2)*v,
        P0 - (plane_size/2)*u + (plane_size/2)*v,
        P0 - (plane_size/2)*u - (plane_size/2)*v, 
    ]
    px = [c[0] for c in corners]
    py = [c[1] for c in corners]
    pz = [c[2] for c in corners]
    return px, py, pz
