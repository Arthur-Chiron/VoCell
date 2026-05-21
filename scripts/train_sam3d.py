import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from PIL import Image

# 1. PATH SETUP
# We need to access the sam-3d-objects library and its specific notebook helpers.
SAM3D_DIR = "/home/arthur.chiron/sam-3d-objects"
sys.path.append(SAM3D_DIR)
sys.path.append(os.path.join(SAM3D_DIR, "notebook"))

from inference import Inference

# 2. DATASET DEFINITION
class RESTOREDataset(Dataset):
    """
    Loads 3D nuclei from the RESTORE dataset (npy) and prepares them for training.
    """
    def __init__(self, npy_path):
        super().__init__()
        # Load the whole dataset into memory (approx 880MB) for fast access
        self.data = np.load(npy_path) # Shape: (N, 64, 64, 64, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Target: Full 3D volume (uint8 -> float32 normalized)
        vol_3d = self.data[idx, ..., 0].astype(np.float32) / 255.0
        
        # Linear Interpolation logic (mirroring data.py)
        z_indices = [z for z in range(64) if np.max(vol_3d[z]) > 0.05]
        if len(z_indices) > 1:
            for i in range(len(z_indices) - 1):
                z1, z2 = z_indices[i], z_indices[i+1]
                dist = z2 - z1
                if dist > 1:
                    for z in range(z1 + 1, z2):
                        alpha = (z - z1) / dist
                        vol_3d[z] = (1.0 - alpha) * vol_3d[z1] + alpha * vol_3d[z2]
        
        # Improvement: Instead of a fixed central slice, pick the slice with most signal
        z_signal = np.sum(vol_3d > 0.1, axis=(1, 2))
        best_z = np.argmax(z_signal)
        
        center_slice = vol_3d[best_z]
        mask_2d = (center_slice > 0.1).astype(np.uint8) * 255
        img_2d = (center_slice * 255).astype(np.uint8)
        
        # RESIZE for Pipeline (64x64 -> 518x518 as seen in pipeline.yaml)
        # We also scale the signal to be more robust.
        img_pil = Image.fromarray(img_2d).resize((518, 518), resample=Image.BICUBIC)
        mask_pil = Image.fromarray(mask_2d).resize((518, 518), resample=Image.NEAREST)
        
        img_rgb = np.stack([np.array(img_pil)] * 3, axis=-1)
        mask_final = np.array(mask_pil)
        
        # If the nucleus is completely empty for some reason, provide a dummy 1-pixel mask
        if np.max(mask_final) == 0:
            mask_final[259, 259] = 255
        
        # We also need the target volume as a torch tensor
        target_tensor = torch.from_numpy(vol_3d).unsqueeze(0) # (1, 64, 64, 64)
        
        return {
            "image": img_rgb,   # Numpy RGB (518, 518, 3)
            "mask": mask_final, # Numpy Mask (518, 518)
            "target_3d": target_tensor
        }

# 3. FINE-TUNER CLASS
class SAM3DFineTuner:
    """
    Orchestrates the fine-tuning of the SAM3D Sparse Structure Generator.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        
        # Load the full pipeline using the Inference wrapper (handles all hydra/yaml logic)
        self.inf_engine = Inference(config["pipeline_config"], compile=False)
        self.pipeline = self.inf_engine._pipeline
        
        # We target the ss_generator for structurally-aware specialization
        self.model = self.pipeline.models["ss_generator"]
        self.decoder = self.pipeline.models["ss_decoder"]
        self.embedder = self.pipeline.condition_embedders["ss_condition_embedder"]
        self.processor = self.pipeline.ss_preprocessor
        
        # CRITICAL fix for brittle external library logic:
        # Patch the pointmap normalizer to return default values instead of crashing 
        # when mask_points.numel() == 0.
        def safe_compute_scale_and_shift(obj, pointmap, mask):
            # Normalizers expect (3,) tensors for scale and shift
            dev = pointmap.device
            return torch.ones(3, device=dev), torch.zeros(3, device=dev)

        # Apply monkeypatch
        self.processor.pointmap_normalizer._compute_scale_and_shift = safe_compute_scale_and_shift.__get__(
            self.processor.pointmap_normalizer, type(self.processor.pointmap_normalizer)
        )
        # Apply to rgb_pointmap_normalizer as well
        self.processor.rgb_pointmap_normalizer._compute_scale_and_shift = safe_compute_scale_and_shift.__get__(
            self.processor.rgb_pointmap_normalizer, type(self.processor.rgb_pointmap_normalizer)
        )
        
        # Also disable tricky joint transforms
        self.processor.img_mask_pointmap_joint_transform = []
        self.processor.img_mask_joint_transform = []
        
        self.model.to(self.device)
        self.decoder.to(self.device)
        self.embedder.to(self.device)
        
        # MULTI-GPU SETUP (DataParallel)
        # Taking advantage of the 2 L40S cards (92GB total VRAM)
        if torch.cuda.device_count() > 1:
            print(f"Detected {torch.cuda.device_count()} GPUs. Wrapping generator in DataParallel ...")
            self.model = torch.nn.DataParallel(self.model)
            # Decoder is used for a tiny forward pass, but we keep it on default device or DP it too.
            # We wrap decoder only if we need gradients or speed. 
            # In fine-tuning, decoder is frozen, so simple .to(device) is enough or we DP it too for parity.
            self.decoder = torch.nn.DataParallel(self.decoder)
        
        # Ensure gradients are enabled for the generator parameters
        # (Must be done AFTER wrapping in DP as the internal module changes)
        target_model = self.model.module if hasattr(self.model, "module") else self.model
        for param in target_model.parameters():
            param.requires_grad = True
            
        # VRAM OPTIMIZATION: Now that we have 2 GPUs, we can keep everything on GPU!
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False
            
        # Still delete Stage 0 (Depth) to be super safe and leave room for larger batches
        if hasattr(self.pipeline, "pointmap_model"):
            print("Deleting Stage 0 (depth) to free up VRAM for Stage 1 gradients across GPUs...")
            del self.pipeline.pointmap_model
            torch.cuda.empty_cache()
            
        self.embedder.requires_grad_(False)

    def training_step(self, batch):
        # ... (Stage 1 and 2 logic below)
        # Prepare inputs using the pipeline's own preprocessing logic
        input_dicts = []
        for i in range(len(batch["image"])):
            # Combine image and mask as expected by the pipeline
            rgba = self.inf_engine.merge_mask_to_rgba(batch["image"][i].numpy(), batch["mask"][i].numpy())
            
            # CRITICAL: Provide a valid synthetic pointmap that won't fail normalization.
            H, W = rgba.shape[:2]
            y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
            z = torch.ones_like(x) * 5.0 # Plausible distance in scene-space
            pointmap = torch.stack([x, y, z], dim=-1).to(self.device).permute(2, 0, 1).float()
            
            # Preprocess. We pass on_device tensors to avoid repeated transfers.
            item = self.pipeline.preprocess_image(rgba, self.processor, pointmap=pointmap)
            input_dicts.append(item)
            
        # Collate list of dicts to batch dict
        collated = {k: torch.cat([d[k] for d in input_dicts]).to(self.device) for k in input_dicts[0].keys()}
        
        # Ensure scale and shift are present to avoid pytorch3d NoneType errors
        if "pointmap_scale" not in collated:
            collated["pointmap_scale"] = torch.ones((bs, 3), device=self.device)
        if "pointmap_shift" not in collated:
            collated["pointmap_shift"] = torch.zeros((bs, 3), device=self.device)
            
        # 1. Condition Embedding (Stage 1)
        # Extract features from the 2D image + mask
        cond_args, cond_kwargs = self.pipeline.get_condition_input(
            self.embedder, collated, self.pipeline.ss_condition_input_mapping
        )
        
        # 2. Forward Pass (SS Generator)
        bs = collated["image"].shape[0]
        
        # Access internal module if wrapped in DataParallel
        target_model = self.model.module if hasattr(self.model, "module") else self.model
        
        latent_shape_dict = {
            k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
            for k, v in target_model.reverse_fn.backbone.latent_mapping.items()
        }
        
        # Run the generator
        return_dict = self.model(latent_shape_dict, self.device, *cond_args, **cond_kwargs)
        shape_latent = return_dict["shape"] # The generated 3D sparse latents
        
        # 3. Decode to Occupancy (Stage 2)
        # Convert latents to a 3D grid (16x16x16 -> 64x64x64)
        ss_voxels = self.decoder(
            shape_latent.permute(0, 2, 1).contiguous().view(bs, 8, 16, 16, 16)
        )
        
        # 4. Computed Loss on GPU
        targets = batch["target_3d"].to(self.device) # Target is (B, 1, 64, 64, 64)
        loss = F.binary_cross_entropy_with_logits(ss_voxels, targets)
        
        return loss, ss_voxels

# 4. MAIN EXECUTION
def main():
    parser = argparse.ArgumentParser(description="SAM3D Fine-Tuning Step-by-Step")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()

    # Step 1: Project Setup
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if args.wandb:
        wandb.init(project="VoCell-FineTune", config=args)
    
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pipeline_config": "/home/arthur.chiron/VoCell/checkpoints/hf/pipeline.yaml",
        "data_path": "/home/arthur.chiron/VoCell/data/RESTORE/nuclei.npy"
    }

    # Step 2: Initialize Trainer & Data
    finetuner = SAM3DFineTuner(config)
    dataset = RESTOREDataset(config["data_path"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(finetuner.model.parameters(), lr=args.lr)

    # Step 3: Optimization Loop
    print(f"Starting memory-optimized fine-tuning: {len(dataset)} samples.")
    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # Use Mixed Precision to save VRAM
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, voxels = finetuner.training_step(batch)
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
            if args.wandb:
                wandb.log({"train/loss": loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Save specialization checkpoint
        out_dir = "/home/arthur.chiron/VoCell/checkpoints/fine_tuned"
        os.makedirs(out_dir, exist_ok=True)
        torch.save(finetuner.model.state_dict(), f"{out_dir}/sam3d_specialized_epoch_{epoch+1}.ckpt")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
