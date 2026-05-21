import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
import trimesh
from scipy.spatial import KDTree
import scipy.ndimage as ndi

# Add SAM3D project paths to sys.path
sam3d_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sam-3d-objects"))
sys.path.append(sam3d_dir)
sys.path.append(os.path.join(sam3d_dir, "notebook"))

try:
    from inference import Inference
except ImportError:
    print("[Error] Could not import 'inference' from sam-3d-objects. Ensure path is correct.")
    sys.exit(1)

def medical_gaussian_filter(volume: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Slight smoothing for an organic, realistic look."""
    return ndi.gaussian_filter(volume, sigma=sigma, mode='nearest')

def main():
    parser = argparse.ArgumentParser(description="SAM3D Standalone Inference for VoCell")
    parser.add_argument("--input", type=str, required=True, help="Input .npy file (2D crop)")
    parser.add_argument("--output", type=str, required=True, help="Output .npy file (3D voxels)")
    args = parser.parse_args()

    print(f"[SAM3D-Engine] Loading input from {args.input}...")
    crop_2d = np.load(args.input)

    # Intensity Preprocessing (Normalization)
    v_min, v_max = crop_2d.min(), crop_2d.max()
    if v_max > v_min:
        p2, p98 = np.percentile(crop_2d, (2, 98))
        img_np = np.clip(crop_2d, p2, p98)
        img_np = ((img_np - p2) / (p98 - p2 + 1e-6) * 255).astype(np.uint8)
    else:
        img_np = np.zeros_like(crop_2d, dtype=np.uint8)
        
    # Generate binary mask for SAM3D
    threshold = v_min + (v_max - v_min) * 0.1
    mask_np = (crop_2d > threshold).astype(np.uint8)

    # SAM3D expects RGB input
    img_input = np.stack([img_np]*3, axis=-1)
    mask_input = mask_np 

    print("[SAM3D-Engine] Initializing SAM3D pipeline...")
    # Load inference pipeline using local config
    config_path = os.path.join(sam3d_dir, "checkpoints/hf/pipeline.yaml")
    inference = Inference(config_path, compile=False)

    print("[SAM3D-Engine] Running inference...")
    output = inference(img_input, mask_input, seed=42)
    
    # Mesh extraction and voxelization
    if "mesh" not in output or not output["mesh"]:
        print("[SAM3D-Engine] Error: No mesh returned.")
        sys.exit(1)
        
    res = output["mesh"][0]
    if not res.success:
        print("[SAM3D-Engine] Error: Mesh extraction unsuccessful.")
        sys.exit(1)

    print("[SAM3D-Engine] Post-processing mesh to solid 64x64x64 voxels...")
    vertices = res.vertices.cpu().numpy()
    faces = res.faces.cpu().numpy()
    t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    res_val = 64
    import trimesh.voxel.creation as voxel_creation
    pitch = 1.0 / res_val
    voxels = voxel_creation.voxelize(t_mesh, pitch=pitch)
    voxels = voxels.fill() # Make the nucleus solid
    
    occupancy = np.zeros((res_val, res_val, res_val), dtype=np.float32)
    m = voxels.matrix
    s = m.shape
    
    # Robust centering of the generated volume
    tx1 = max(0, (res_val - s[0]) // 2)
    ty1 = max(0, (res_val - s[1]) // 2)
    tz1 = max(0, (res_val - s[2]) // 2)
    tx2 = min(res_val, tx1 + s[0])
    ty2 = min(res_val, ty1 + s[1])
    tz2 = min(res_val, tz1 + s[2])
    
    sx1 = max(0, -( (res_val - s[0]) // 2 ) )
    sy1 = max(0, -( (res_val - s[1]) // 2 ) )
    sz1 = max(0, -( (res_val - s[2]) // 2 ) )
    sx2 = sx1 + (tx2 - tx1)
    sy2 = sy1 + (ty2 - ty1)
    sz2 = sz1 + (tz2 - tz1)
    
    occupancy[tx1:tx2, ty1:ty2, tz1:tz2] = m[sx1:sx2, sy1:sy2, sz1:sz2]

    # Transfer vertex intensities (biomarker signal) to voxels using KD-Tree
    if res.vertex_attrs is not None and len(res.vertex_attrs) > 0:
        v_points = vertices
        v_attrs = res.vertex_attrs.cpu().numpy()
        v_intensities = v_attrs[:, 0] if v_attrs.ndim > 1 else v_attrs
        
        tree = KDTree(v_points)
        vx, vy, vz = np.where(occupancy > 0)
        if len(vx) > 0:
            voxel_coords = np.stack([vz, vy, vx], axis=-1)
            voxel_coords_norm = (voxel_coords + 0.5) / 64.0 - 0.5
            dist, idx = tree.query(voxel_coords_norm)
            occupancy[vx, vy, vz] = v_intensities[idx]

    # Final touch-ups
    occupancy = occupancy.astype(np.float32)
    occupancy = medical_gaussian_filter(occupancy, sigma=0.8)
    occupancy = np.clip(occupancy, 0.0, 1.0)
    
    print(f"[SAM3D-Engine] Saving output volume to {args.output}...")
    np.save(args.output, occupancy)
    print("[SAM3D-Engine] Done.")

if __name__ == "__main__":
    main()
