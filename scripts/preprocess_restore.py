"""
preprocess_restore.py
---------------------
Prétraitement du dataset RESTORE (2021) :
  - Lit les paires (.ims, _mask.npy) depuis data/2021/
  - Maintient la forme 3D d'origine en Z (pas de padding / redimensionnement arbitraire).
  - Normalise proportionnellement (crop carré en XY) puis retaille en 64x64.
  - Sauvegarde : data/RESTORE/nuclei.npy → array d'objets (liste de dicts {volume, aspect_z}).
"""

import os
import glob
import numpy as np
import h5py
import scipy.ndimage as ndi

DATA_ROOT    = "data/2021"
OUTPUT_DIR   = "data/RESTORE"
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "nuclei.npy")
TARGET_HW    = 64
CHANNEL_IDX  = 0
MIN_DEPTH    = 3
MIN_SPATIAL  = 8

def get_voxel_sizes(ims_path):
    """Extrait la taille physique des voxels (Z, X) depuis le HDF5 Imaris."""
    with h5py.File(ims_path, 'r') as f:
        if "DataSetInfo/Image" not in f:
            return 1.0, 1.0
        
        info = f["DataSetInfo/Image"]
        def get_val(key, default):
            v = info.attrs.get(key, [str(default).encode('ascii')])
            try:
                s = "".join([c.decode('ascii') for c in v])
                return float(s)
            except:
                return default
                
        ext_max_x = get_val('ExtMax0', 1.0)
        ext_min_x = get_val('ExtMin0', 0.0)
        size_x = get_val('X', 1.0)
        
        ext_max_z = get_val('ExtMax2', 1.0)
        ext_min_z = get_val('ExtMin2', 0.0)
        size_z = get_val('Z', 1.0)
        
        res_x = (ext_max_x - ext_min_x) / size_x if size_x > 0 else 1.0
        res_z = (ext_max_z - ext_min_z) / size_z if size_z > 0 else 1.0
        
        return res_z, res_x

def read_dapi_volume(ims_path: str) -> np.ndarray:
    with h5py.File(ims_path, "r") as f:
        key = f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {CHANNEL_IDX}/Data"
        volume = f[key][:]
    return volume

def extract_nuclei(volume: np.ndarray, mask: np.ndarray, res_z: float, res_x: float):
    slices_list = ndi.find_objects(mask)

    for label_id, slc in enumerate(slices_list, start=1):
        if slc is None:
            continue

        sd, sh, sw = slc

        depth   = sd.stop - sd.start
        height  = sh.stop - sh.start
        width   = sw.stop - sw.start
        if depth < MIN_DEPTH or height < MIN_SPATIAL or width < MIN_SPATIAL:
            continue

        # Taille originale max
        s = max(height, width)
        
        # Le crop XY doit être doublé (zoom arrière x2)
        s_new = s * 2
        
        cy = (sh.start + sh.stop) / 2.0
        cx = (sw.start + sw.stop) / 2.0
        
        y_start = int(round(cy - s_new / 2.0))
        y_stop  = y_start + s_new
        x_start = int(round(cx - s_new / 2.0))
        x_stop  = x_start + s_new
        
        # Dimensions de l'image
        max_z, max_y, max_x = volume.shape
        
        # Limites valides pour le découpage
        y_start_v = max(0, y_start)
        y_stop_v  = min(max_y, y_stop)
        x_start_v = max(0, x_start)
        x_stop_v  = min(max_x, x_stop)
        
        # Découpe
        vol_crop_raw  = volume[sd.start:sd.stop, y_start_v:y_stop_v, x_start_v:x_stop_v].astype(np.float32)
        mask_crop_raw = mask[sd.start:sd.stop, y_start_v:y_stop_v, x_start_v:x_stop_v]
        
        # Création des versions "pleines" s_new x s_new avec padding
        vol_crop  = np.zeros((depth, s_new, s_new), dtype=np.float32)
        mask_crop = np.zeros((depth, s_new, s_new), dtype=mask.dtype)
        
        off_y = y_start_v - y_start
        off_x = x_start_v - x_start
        h_v = y_stop_v - y_start_v
        w_v = x_stop_v - x_start_v
        
        vol_crop[:, off_y:off_y+h_v, off_x:off_x+w_v]  = vol_crop_raw
        mask_crop[:, off_y:off_y+h_v, off_x:off_x+w_v] = mask_crop_raw

        # Isolation du noyau (fond = 0)
        vol_crop[mask_crop != label_id] = 0.0

        # Normalisation
        pix = vol_crop[mask_crop == label_id]
        if pix.size == 0:
            continue
        lo, hi = np.percentile(pix, 1), np.percentile(pix, 99)
        if hi <= lo:
            continue
        vol_crop = np.clip((vol_crop - lo) / (hi - lo), 0.0, 1.0)
        vol_crop[mask_crop != label_id] = 0.0

        # Espace 3D de destination : TARGET_HW^3
        vol_64 = np.zeros((TARGET_HW, TARGET_HW, TARGET_HW), dtype=np.float32)
        
        # Calcul du pas Z dans le nouveau repère
        # s_new pixels originaux = 64 pixels cibles.
        # res_xy finale = (s_new * res_x) / 64
        # res_z reste inchangé car on replace les coupes entières.
        res_xy_new = (s_new * res_x) / 64.0
        step_z = res_z / res_xy_new if res_xy_new > 0 else 0
        
        # Chaque coupe D est redimensionnée en 64x64 puis insérée à l'index Z correspondant
        # Le milieu est à 31.5 (index 31/32)
        zoom_xy = TARGET_HW / s_new
        
        for i in range(depth):
            # Position visée dans l'espace 64x64x64
            z_idx = int(round(31.5 + (i - (depth - 1) / 2.0) * step_z))
            
            if 0 <= z_idx < TARGET_HW:
                slice_2d = vol_crop[i, :, :]
                # Redimensionnement xy
                slice_resized = ndi.zoom(slice_2d, zoom=(zoom_xy, zoom_xy), order=1)
                
                # Assignation (il peut y avoir des z_idx qui fusionnent ou sautent des pas, 
                # on prend le max si jamais deux originaux atterrissent sur le même z_idx)
                vol_64[z_idx, :, :] = np.maximum(vol_64[z_idx, :, :], slice_resized)
                
        # Remise à [0, 255] uint8 pour économiser de la RAM
        vol_final = np.clip(vol_64 * 255.0, 0, 255).astype(np.uint8)
        vol_final = vol_final[..., np.newaxis]  # (64, 64, 64, 1)

        yield vol_final

def find_pairs(root: str):
    pairs = []
    for ims_path in glob.glob(os.path.join(root, "**", "*.ims"), recursive=True):
        mask_path = ims_path.replace(".ims", "_mask.npy")
        if os.path.exists(mask_path):
            pairs.append((ims_path, mask_path))
    return sorted(pairs)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pairs = find_pairs(DATA_ROOT)
    all_nuclei = []

    for i, (ims_path, mask_path) in enumerate(pairs):
        ims_name = os.path.basename(ims_path)
        print(f"\n[{i+1}/{len(pairs)}] {ims_name}")
        
        try:
            res_z, res_x = get_voxel_sizes(ims_path)
            print(f"  → Voxel: Z={res_z:.3f} um, X={res_x:.3f} um")
            volume = read_dapi_volume(ims_path)
        except Exception as e:
            print(f"  ⚠️  Erreur lecture volume : {e}")
            continue

        try:
            mask = np.load(mask_path, allow_pickle=True)
            if not isinstance(mask, np.ndarray):
                print(f"  ⚠️  Masque invalide (pas un array).")
                continue
        except Exception as e:
            print(f"  ⚠️  Erreur lecture masque : {e}")
            continue

        count_before = len(all_nuclei)
        for nucleus in extract_nuclei(volume, mask, res_z, res_x):
            all_nuclei.append(nucleus)
        
        print(f"  ✓ {len(all_nuclei) - count_before} noyaux (total: {len(all_nuclei)})")

    # Stack des noyaux en (N, 64, 64, 64, 1) uint8
    if all_nuclei:
        dataset = np.stack(all_nuclei, axis=0)
        print(f"\nFinal dataset shape: {dataset.shape}, dtype: {dataset.dtype}")
        np.save(OUTPUT_FILE, dataset)
        print(f"Sauvegardé {len(dataset)} noyaux dans {OUTPUT_FILE} ✓")
    else:
        print("\n⚠️  Aucun noyau extrait !")

if __name__ == "__main__":
    main()
