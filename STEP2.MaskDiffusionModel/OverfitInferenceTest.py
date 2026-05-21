import os
from ddpm import Unet3D, GaussianDiffusion

import sys
sys.path.append(os.getcwd())

import hydra
from omegaconf import DictConfig, open_dict
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np

from dataset.dataloader import get_loader

# 1. Bring in your helper function!
def prepare_conditional(volume_shape, data):
    conditional_feature_list = [
        "organ", "diameter_x_mm", "diameter_y_mm", "diameter_z_mm", 
        "mean_x_mm", "mean_y_mm", "mean_z_mm", "std_x_mm", "std_y_mm", 
        "std_z_mm", "volume_ml"
    ]
    conditional_volume = torch.zeros(volume_shape)
    sheets_per_feature = volume_shape[-1] // len(conditional_feature_list)
    
    for i, key in enumerate(conditional_feature_list):
        start_idx = sheets_per_feature * i
        if i == len(conditional_feature_list) - 1:
            end_idx = volume_shape[-1]
        else:
            end_idx = sheets_per_feature * (i + 1)
            
        feature_data = torch.asarray(data[key]).reshape(volume_shape[0], 1, 1, 1, 1)
        conditional_volume[:, :, :, :, start_idx:end_idx] = feature_data

    return conditional_volume


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def reconstruct(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    device = torch.device(f"cuda:{cfg.model.gpus}")

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    print("1. Initializing Model...")
    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels, 
            out_dim=1 
        ).to(device)
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
    ).to(device)

    print("2. Loading Checkpoint...")
    ckpt_path = os.path.join(cfg.model.results_folder, 'model_best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Could not find checkpoint at {ckpt_path}")
    
    data = torch.load(ckpt_path, map_location=device)
    diffusion.load_state_dict(data['ema']) 
    diffusion.eval()

    print("3. Loading Data & Diagnosing Labels...")
    train_dataloader, _, _ = get_loader(cfg.dataset)
    
    found_tumor = False
    for batch_idx, batch in enumerate(train_dataloader):
        raw_tumor_mask = batch["tumor_mask"].to(device)
        
        # Print the unique values present in the raw mask
        unique_vals = torch.unique(raw_tumor_mask).tolist()
        print(f"Batch {batch_idx} - Raw tumor_mask unique values: {unique_vals}")
        
        # Check if there is a '2' (your original target)
        if 2 in unique_vals:
            print("--> Found a label '2'! Applying mapping...")
            tumor_mask = raw_tumor_mask.clone()
            tumor_mask[tumor_mask == 1] = 0
            tumor_mask[tumor_mask == 2] = 1
            found_tumor = True
            
        # Or, check if it's already a binary mask (just 0 and 1)
        elif 1 in unique_vals and len(unique_vals) <= 2:
            print("--> Found label '1'. This mask is already binary! Skipping mapping...")
            tumor_mask = raw_tumor_mask.clone()
            # NO MAPPING NEEDED
            found_tumor = True
            
        if found_tumor:
            image = batch["image"].to(device)
            organ_mask = batch["organ_mask"].to(device)
            
            # Apply organ mask corrections safely
            if 2 in torch.unique(organ_mask):
                organ_mask[organ_mask == 1] = 0
                organ_mask[organ_mask == 2] = 1
                
            break
            
    if not found_tumor:
        raise ValueError("Still completely empty! Check if CropForeground is cropping out the tumor.")

    # Generate conditionals and slice
    conditional_features = prepare_conditional(tumor_mask.shape, batch).to(device)
    
    image = image[0:1]
    tumor_mask = tumor_mask[0:1]
    organ_mask = organ_mask[0:1]
    conditional_features = conditional_features[0:1]

    print("4. Preparing Conditioners...")
    with torch.no_grad():
        # Permute matching the forward pass
        image_p = image.permute(0, 1, -1, -3, -2)
        tumor_mask_p = tumor_mask.permute(0, 1, -1, -3, -2)
        organ_mask_p = organ_mask.permute(0, 1, -1, -3, -2)
        cond_feats_p = conditional_features.permute(0, 1, -1, -3, -2)

        # Encode Image
        if diffusion.vqgan is not None:
            img_cond = diffusion.vqgan.encode(image_p, quantize=False, include_embeddings=True)
            img_cond = ((img_cond - diffusion.vqgan.codebook.embeddings.min()) /
                        (diffusion.vqgan.codebook.embeddings.max() - diffusion.vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        else:
            img_cond = (image_p - image_p.min()) / (image_p.max() - image_p.min() + 1e-8) * 2.0 - 1.0

        # Downsample conditions to latent space
        latent_spatial = img_cond.shape[-3:]
        organ_cond = F.interpolate(organ_mask_p.float(), size=latent_spatial)
        feat_cond = F.interpolate(cond_feats_p.float(), size=latent_spatial)
        
        # Concatenate!
        cond = torch.cat([img_cond, organ_cond, feat_cond], dim=1)

        # Scale target to [-1, 1] and downsample
        target_mask = tumor_mask_p * 2.0 - 1.0
        target_latent = F.interpolate(target_mask, size=latent_spatial)

        print("5. Applying Forward Noise (q_sample)...")
        # Define how much noise to apply (e.g., 500 out of 1000 steps)
        T_START = cfg.model.timesteps // 2 
        t = torch.full((target_latent.shape[0],), T_START, device=device, dtype=torch.long)
        
        noisy_latent = diffusion.q_sample(x_start=target_latent, t=t)

        print(f"6. Running Reverse Diffusion from t={T_START}...")
        recon_latent = noisy_latent
        
        # Loop backwards from T_START down to 0
        from tqdm import tqdm
        for i in tqdm(reversed(range(T_START)), total=T_START, desc="Denoising"):
            t_i = torch.full((recon_latent.shape[0],), i, device=device, dtype=torch.long)
            recon_latent = diffusion.p_sample(recon_latent, t_i, cond=cond)

        print("7. Post-processing and Saving...")
        # Upsample back to permuted original size
        recon = F.interpolate(recon_latent, size=tumor_mask_p.shape[-3:], mode='trilinear')
        recon = recon.permute(0, 1, -2, -1, -3)
        tumor_mask_orig = tumor_mask_p.permute(0, 1, -2, -1, -3)

        # Threshold to binary
        recon = (recon + 1.0) / 2.0
        recon = (recon > 0.5).float()

        os.makedirs("debug_reconstructions", exist_ok=True)
        
        # --- NEW: Convert to uint8 (integers) for ITK-SNAP ---
        recon_np = recon[0, 0].cpu().numpy().astype(np.uint8)
        gt_np = tumor_mask_orig[0, 0].cpu().numpy().astype(np.uint8)
        
        # --- NEW: Check if there's actually a tumor here! ---
        print(f"DEBUG: Tumor pixels in Ground Truth: {gt_np.sum()}")
        print(f"DEBUG: Tumor pixels in Reconstruction: {recon_np.sum()}")

        nib.save(nib.Nifti1Image(recon_np, np.eye(4)), "debug_reconstructions/reconstruction.nii.gz")
        nib.save(nib.Nifti1Image(gt_np, np.eye(4)), "debug_reconstructions/ground_truth.nii.gz")

        print("Done! Check the 'debug_reconstructions' folder.")

if __name__ == '__main__':
    reconstruct()