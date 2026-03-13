import streamlit as st
import numpy as np
import scipy.ndimage as ndi
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🔬 Nuclei 3D Voxel Explorer")

# --- Fonctions de génération / Chargement ---
@st.cache_data
def load_all_crops():
    return np.load('data/CODEX/crops.npy')

if "nucleus_idx" not in st.session_state:
    crops = load_all_crops()
    st.session_state.nucleus_idx = np.random.randint(0, len(crops))

def get_nucleus_volume(idx):
    crops = load_all_crops()
    # Normalisation entre 0 et 1 (les données étant entre 0 et 255)
    crop_2d = crops[idx].astype(np.float32) / 255.0
    
    # Création d'un volume 3D vide 64x64x64
    depth = 64
    volume_3d = np.zeros((depth, crop_2d.shape[0], crop_2d.shape[1]), dtype=np.float32)
    center_z = depth // 2
    
    # Épaisseur du noyau sur l'axe Z (contrôle l'étalement de la courbe Gaussienne)
    sigma = 2.5
    
    for z in range(depth):
        # On calcule un poids de 1.0 au centre qui diminue doucement vers 0 sur les bords
        weight = np.exp(-((z - center_z)**2) / (2 * sigma**2))
        
        # Le noyau garde sa forme 2D brute à chaque coupe, 
        # mais son intensité s'éteint au fur et à mesure qu'on s'éloigne du centre
        volume_3d[z] = crop_2d * weight
        
    return volume_3d

# --- Sidebar : Contrôles ---
if st.sidebar.button("🎲 Nouveau noyau aléatoire"):
    crops = load_all_crops()
    st.session_state.nucleus_idx = np.random.randint(0, len(crops))

volume = get_nucleus_volume(st.session_state.nucleus_idx)

st.sidebar.header("Paramètres de Coupe")
azimuth = st.sidebar.slider("Azimut (Longitude)", -180, 180, 0)
elevation = st.sidebar.slider("Élévation (Latitude)", -90, 90, 90)
slice_offset = st.sidebar.slider("Position de la coupe", -40, 40, 0)

# Option pour masquer une des moitiés
visibility_mode = st.sidebar.radio("Visibilité 3D relative à la coupe", ["Tout afficher", "Masquer au-dessus", "Masquer au-dessous"], index=1)

# Préparation géométrique du plan de coupe
theta = np.radians(azimuth)
phi = np.radians(elevation)

nx = np.cos(phi) * np.cos(theta)
ny = np.cos(phi) * np.sin(theta)
nz = np.sin(phi)
normal_vec = np.array([nx, ny, nz])

# Centre du volume
cz, cy, cx = volume.shape[0]/2.0, volume.shape[1]/2.0, volume.shape[2]/2.0

# Calcul des vecteurs de base de notre plan 2D dans l'espace 3D
P0 = np.array([cx, cy, cz]) + slice_offset * normal_vec

# Vecteur "Up" arbitraire pour orienter l'image 2D
up = np.array([0, 0, 1.0])
if abs(np.dot(up, normal_vec)) > 0.99:
    up = np.array([0, 1.0, 0])
    
u = np.cross(up, normal_vec)
u /= np.linalg.norm(u)
v = np.cross(normal_vec, u)

# --- Mise en page : Colonnes ---
col1, col2 = st.columns(2)

with col1:
    # Affichage 3D avec Plotly
    st.subheader("Rendu Volumétrique 3D")
    
    # Masquage conditionnel du volume arbitraire avant la construction des voxels
    volume_to_render = volume.copy()
    if visibility_mode != "Tout afficher":
        # Grille de coordonnées 3D
        zz, yy, xx = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
        # Distance signée de chaque point par rapport au plan
        dist = (xx - cx)*nx + (yy - cy)*ny + (zz - cz)*nz - slice_offset
        if visibility_mode == "Masquer au-dessus":
            volume_to_render[dist > 0] = 0
        elif visibility_mode == "Masquer au-dessous":
            volume_to_render[dist < 0] = 0

    # Pour un rendu de type Voxel "Minecraft" strict et cubique, on génère un maillage (Mesh3d).
    # On filtre les pixels vides pour des questions de performances
    threshold = 0.05
    z, y, x = np.where(volume_to_render > threshold)
    values = volume_to_render[volume_to_render > threshold]
    
    # Construction vectorisée et ultra-rapide de cubes (8 sommets et 12 triangles par voxel)
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
    
    # Les couleurs sont répétées pour peindre les 8 sommets de la même façon
    vc = np.repeat(values, 8)
    
    offsets = np.arange(n_voxels)[:, None] * 8
    vi = (faces_i + offsets).flatten()
    vj = (faces_j + offsets).flatten()
    vk = (faces_k + offsets).flatten()
    
    v_min, v_max = float(values.min()) if len(values)>0 else 0, float(values.max()) if len(values)>0 else 1
    
    # On génère TOUJOURS le Mesh3d, même s'il est vide (x=[], y=[] etc...)
    # C'est vital pour que Plotly WebGL ne détruise pas la scène en changeant de type de trace.
    fig_3d = go.Figure(data=go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=vi, j=vj, k=vk,
        intensity=vc,
        cmin=v_min, cmax=v_max,
        colorscale='Viridis',
        opacity=1.0,
        flatshading=True # Indispensable pour l'aspect arrêtes saillantes et cubiques
    ))
    
    fig_3d.update_layout(
        # Lier uirevision à l'ID du noyau permet de figer la caméra quand on manipule la coupe,
        # mais de bien la réinitialiser si l'on clique sur "Nouveau noyau aléatoire" !
        uirevision=str(st.session_state.nucleus_idx), 
        scene=dict(
            xaxis=dict(title='X', range=[0, 64]),
            yaxis=dict(title='Y', range=[0, 64]),
            zaxis=dict(title='Z', range=[0, 64]),
            aspectmode='cube' # Force l'affichage cubique vu que nos axes sont tous 0-64
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    # --- Ajout du cadre / plan de coupe visuel ---
    plane_size = 40 # Taille réduite du carré pour s'adapter au noyau et ne pas sortir du cadre (0-64)
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
    
    fig_3d.add_trace(go.Scatter3d(
        x=px, y=py, z=pz,
        mode='lines',
        line=dict(color='red', width=5),
        name='Plan de coupe',
        hoverinfo='skip'
    ))
    
    st.plotly_chart(fig_3d, use_container_width=True)

with col2:
    # --- Affichage de la Coupe 2D ---
    st.subheader("Coupe Transversale Arbitraire")
    
    # Création d'une grille 2D pour recueillir les pixels de la coupe
    # 90x90 permet d'avoir assez de place pour piocher en diagonale dans le cube 64x64x64
    slice_size = 90
    grid_x, grid_y = np.meshgrid(np.arange(slice_size), np.arange(slice_size))
    grid_x = grid_x - slice_size // 2
    grid_y = grid_y - slice_size // 2
    
    # Projection des coordonnées 2D (pixels) en absolu 3D
    X_sample = P0[0] + grid_x * u[0] + grid_y * v[0]
    Y_sample = P0[1] + grid_x * u[1] + grid_y * v[1]
    Z_sample = P0[2] + grid_x * u[2] + grid_y * v[2]
    
    # Échantillonnage / Interpolation des valeurs du volume aux coordonnées calculées
    coords = np.stack([Z_sample, Y_sample, X_sample])
    img_slice = ndi.map_coordinates(volume, coords, order=1, cval=0.0)
    
    # Agrandissement des pixels sans interpolation pour un effet net / pixelisé
    img_pixelated = np.repeat(np.repeat(img_slice, 8, axis=0), 8, axis=1)
    
    st.image(img_pixelated, use_container_width=True, clamp=True)
