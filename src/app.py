import streamlit as st
import numpy as np
import scipy.ndimage as ndi
import plotly.graph_objects as go
import csv

st.set_page_config(layout="wide")

# --- Mapping des classes ---
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

# --- Fonctions de génération / Chargement ---
@st.cache_data
def load_all_crops():
    return np.load('data/CODEX/crops.npy')

@st.cache_data
def load_metadata():
    metadata = {}
    with open('data/CODEX/crop_metadata.csv', mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row['crop_index'])
                raw_class = row['classes']
                mapped_class = CLASS_MAPPING.get(raw_class, raw_class)
                metadata[idx] = mapped_class
            except (ValueError, KeyError):
                pass
    return metadata

if "nucleus_idx" not in st.session_state:
    crops = load_all_crops()
    st.session_state.nucleus_idx = np.random.randint(0, len(crops))

def get_nucleus_volume(idx, interpolation_method="Gaussien", params=None):
    crops = load_all_crops()
    # Normalisation entre 0 et 1
    crop_2d = crops[idx].astype(np.float32) / 255.0
    
    depth = 64
    volume_3d = np.zeros((depth, crop_2d.shape[0], crop_2d.shape[1]), dtype=np.float32)
    center_z = depth // 2
    
    if params is None:
        params = {}

    for z in range(depth):
        weight = 0.0
        if interpolation_method == "Gaussien":
            sigma = params.get('sigma', 2.5)
            weight = np.exp(-((z - center_z)**2) / (2 * sigma**2))
        elif interpolation_method == "Linéaire":
            thickness = params.get('thickness', 16)
            thickness_d2 = thickness / 2.0
            dist = abs(z - center_z)
            if dist < thickness_d2:
                weight = 1.0 - (dist / thickness_d2)

        volume_3d[z] = crop_2d * weight
        
    return volume_3d

# --- Sidebar : Contrôles ---
with st.sidebar.container():
    st.markdown("### Choix du noyau")

    # Custom CSS to hide the form's border and attempt to align the dice button.
    # Note: Perfect vertical alignment is tricky and this is an approximation.
    st.markdown("""
        <style>
        [data-testid="stForm"] {
            border: none;
            padding: 0px;
        }
        /* Add a small top margin to the dice button to align it visually */
        div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
            margin-top: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    if 'idx_input' not in st.session_state:
        st.session_state.idx_input = st.session_state.nucleus_idx

    col1, col2 = st.columns([1, 5])

    with col1:
        # This button just updates the input state without submitting
        if st.button("🎲", help="Générer un index aléatoire"):
            crops = load_all_crops()
            st.session_state.idx_input = np.random.randint(0, len(crops))
            
    with col2:
        # The form only contains the input and the submit button
        with st.form(key='index_selector'):
            form_cols = st.columns([4, 1])
            with form_cols[0]:
                st.number_input(
                    'Index', 
                    min_value=0, 
                    max_value=len(load_all_crops())-1, 
                    key='idx_input', 
                    step=1, 
                    label_visibility="collapsed"
                )
            with form_cols[1]:
                submitted = st.form_submit_button('➤', help="Charger le noyau")

            if submitted:
                st.session_state.nucleus_idx = st.session_state.idx_input

current_idx = st.session_state.nucleus_idx
crops = load_all_crops()

st.sidebar.markdown("---")

with st.sidebar.container():
    st.markdown("### Reconstruction 3D")
    interpolation = st.selectbox(
        "Profil de reconstruction 3D",
        ["Gaussien", "Linéaire"],
        index=0,
        help="Choix du profil d'intensité le long de l'axe Z pour reconstruire le volume 3D."
    )

    params = {}
    if interpolation == "Gaussien":
        sigma = st.slider(
            "Sigma (Écart-type)", 
            min_value=0.5, 
            max_value=10.0, 
            value=2.5, 
            step=0.1,
            help="Contrôle la dispersion du profil Gaussien. Une valeur plus élevée donne un noyau plus 'épais' et flou."
        )
        params['sigma'] = sigma
    elif interpolation == "Linéaire":
        thickness = st.slider(
            "Épaisseur du Noyau",
            min_value=1,
            max_value=32,
            value=16,
            step=1,
            help="Contrôle l'épaisseur totale (en 'voxels') du profil linéaire. Le centre est à 1.0 et les bords à 0.0."
        )
        params['thickness'] = thickness

volume = get_nucleus_volume(current_idx, interpolation_method=interpolation, params=params)
metadata = load_metadata()
cell_type = metadata.get(current_idx, "Inconnu")

st.sidebar.markdown("---")

with st.sidebar.container():
    st.markdown("### Coupe")
    azimuth = st.slider("Azimut (Longitude)", -180, 180, 0)
    elevation = st.slider("Élévation (Latitude)", -90, 90, 90)
    slice_offset = st.slider("Position de la coupe", -40, 40, 0)
    # Option pour masquer une des moitiés
    visibility_mode = st.radio("Visibilité 3D relative à la coupe", ["Tout afficher", "Masquer au-dessus", "Masquer au-dessous"], index=1)

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

# --- En-tête : Crop Original et Métadonnées ---
st.markdown("### Noyau Original")
info_col1, info_col2 = st.columns([1, 6])
with info_col1:
    orig_crop = crops[current_idx].astype(np.float32) / 255.0
    orig_pixelated = np.repeat(np.repeat(orig_crop, 4, axis=0), 4, axis=1) # Rendu plus petit (x4)
    st.image(orig_pixelated, use_container_width=True, clamp=True)
with info_col2:
    st.markdown(f"**Index dans le dataset :** `#{current_idx}`")
    st.markdown(f"**Classe cellulaire :** `{cell_type}`")

st.markdown("---")

# --- Mise en page : Colonnes 3D et Transformée ---
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
    st.subheader("Coupe Transversale 2D Associée")
    
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
