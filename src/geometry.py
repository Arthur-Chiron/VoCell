import numpy as np
import scipy.ndimage as ndi

def get_plane_vectors(azimuth, elevation, slice_offset, cx, cy, cz):
    """
    Calcule les vecteurs normaux et la base (u, v) du plan de coupe 
    dans l'espace 3D en fonction de l'azimut et de l'élévation.
    """
    theta = np.radians(azimuth)
    phi = np.radians(elevation)

    nx = np.cos(phi) * np.cos(theta)
    ny = np.cos(phi) * np.sin(theta)
    nz = np.sin(phi)
    normal_vec = np.array([nx, ny, nz])

    # Point d'origine du plan
    P0 = np.array([cx, cy, cz]) + slice_offset * normal_vec

    # Vecteur "Up" arbitraire pour orienter l'image 2D
    up = np.array([0, 0, 1.0])
    if abs(np.dot(up, normal_vec)) > 0.99:
        up = np.array([0, 1.0, 0])
        
    u = np.cross(up, normal_vec)
    u /= np.linalg.norm(u)
    v = np.cross(normal_vec, u)
    
    return normal_vec, P0, u, v

def apply_clipping(volume, nx, ny, nz, slice_offset, cx, cy, cz, mode):
    """
    Masque une partie du volume (au-dessus ou au-dessous) par rapport au plan de coupe.
    """
    volume_to_render = volume.copy()
    if mode != "Tout afficher":
        zz, yy, xx = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
        # Distance signée de chaque point par rapport au plan
        dist = (xx - cx)*nx + (yy - cy)*ny + (zz - cz)*nz - slice_offset
        if mode == "Masquer au-dessus":
            volume_to_render[dist > 0] = 0
        elif mode == "Masquer au-dessous":
            volume_to_render[dist < 0] = 0
    return volume_to_render

def get_voxel_mesh_data(volume_to_render, threshold=0.05):
    """
    Génère les données (sommets, faces, intensités) pour un rendu 
    Minecraft (Mesh3d) optimisé.
    """
    z, y, x = np.where(volume_to_render > threshold)
    values = volume_to_render[volume_to_render > threshold]
    
    if len(x) == 0:
        return [], [], [], [], [], [], [], 0, 1
        
    dx = np.array([-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5])
    dy = np.array([-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5])
    dz = np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5])
    
    faces_i = np.array([7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2])
    faces_j = np.array([3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3])
    faces_k = np.array([0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6])
    
    n_voxels = len(x)
    vx = (x[:, None] + dx).flatten()
    vy = (y[:, None] + dy).flatten()
    vz = (z[:, None] + dz).flatten()
    
    vc = np.repeat(values, 8)
    
    offsets = np.arange(n_voxels)[:, None] * 8
    vi = (faces_i + offsets).flatten()
    vj = (faces_j + offsets).flatten()
    vk = (faces_k + offsets).flatten()
    
    v_min, v_max = float(values.min()), float(values.max())
    
    return vx, vy, vz, vi, vj, vk, vc, v_min, v_max

def extract_2d_slice(volume, P0, u, v, slice_size=90):
    """
    Extrait une coupe 2D du volume 3D le long du plan spécifié.
    """
    grid_x, grid_y = np.meshgrid(np.arange(slice_size), np.arange(slice_size))
    grid_x = grid_x - slice_size // 2
    grid_y = grid_y - slice_size // 2
    
    # Projection des coordonnées 2D en 3D
    X_sample = P0[0] + grid_x * u[0] + grid_y * v[0]
    Y_sample = P0[1] + grid_x * u[1] + grid_y * v[1]
    Z_sample = P0[2] + grid_x * u[2] + grid_y * v[2]
    
    coords = np.stack([Z_sample, Y_sample, X_sample])
    img_slice = ndi.map_coordinates(volume, coords, order=1, cval=0.0)
    
    # Agrandissement des pixels sans interpolation
    return np.repeat(np.repeat(img_slice, 8, axis=0), 8, axis=1)

def get_plane_wireframe(P0, u, v, plane_size=40):
    """
    Calcule les coordonnées des coins pour dessiner un cadre (wireframe).
    """
    corners = [
        P0 - (plane_size/2)*u - (plane_size/2)*v,
        P0 + (plane_size/2)*u - (plane_size/2)*v,
        P0 + (plane_size/2)*u + (plane_size/2)*v,
        P0 - (plane_size/2)*u + (plane_size/2)*v,
        P0 - (plane_size/2)*u - (plane_size/2)*v, # fermer la boucle
    ]
    px = [c[0] for c in corners]
    py = [c[1] for c in corners]
    pz = [c[2] for c in corners]
    return px, py, pz
