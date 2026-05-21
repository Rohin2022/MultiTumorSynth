from dataset.dataloader import get_loader
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import torch
from omegaconf import DictConfig, open_dict
import hydra
import os
from ddpm import Unet3D, GaussianDiffusion
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())

def prepare_conditional(spatial_shape, data):
    """
    spatial_shape: strictly a 3-tuple representing (Depth, Height, Width). e.g., (32, 32, 32)
    Outputs tensor of shape: (Batch, 1, Depth, Height, Width)
    """
    conditional_feature_list = [
        "organ", "diameter_x_mm", "diameter_y_mm", "diameter_z_mm",
        "mean_x_mm", "mean_y_mm", "mean_z_mm", "std_x_mm", "std_y_mm",
        "std_z_mm", "volume_ml"
    ]

    batch_size = data[conditional_feature_list[0]].shape[0]
    
    # Restored to EXACTLY 1 channel so UNet receives the 4 total channels it expects
    num_channels = 1 

    conditional_volume = torch.zeros(
        (batch_size, num_channels, spatial_shape[0], spatial_shape[1], spatial_shape[2])
    )

    # Slice the features along the Depth dimension (spatial_shape[0])
    depth = spatial_shape[0]
    sheets_per_feature = depth // len(conditional_feature_list)

    for i, key in enumerate(conditional_feature_list):
        start_idx = sheets_per_feature * i
        end_idx = depth if i == len(conditional_feature_list) - 1 else sheets_per_feature * (i + 1)

        # feature_data shape: (B, 1, 1, 1, 1) - broadcasts across the depth slice, height, and width
        feature_data = torch.as_tensor(data[key]).view(batch_size, 1, 1, 1, 1)

        # Assign strictly to the DEPTH slices
        conditional_volume[:, :, start_idx:end_idx, :, :] = feature_data

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
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type
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
    train_data = next(iter(train_dataloader))

    batch_size = train_data["heatmap"].shape[0]

    # Strictly separate spatial shape from 5D tensor shapes
    spatial_shape = (32, 32, 32)
    tumor_mask_dims = (batch_size, 1, 32, 32, 32)

    # Move to GPU and ensure standard (B, C, D, H, W)
    heatmap = train_data["heatmap"].permute(0, 1, -1, -3, -2).cuda()
    organ_mask_p = train_data["organ_mask"].permute(0, 1, -1, -3, -2).cuda()
    
    # Conditional features already generated as (B, C, D, H, W)
    feat_cond = prepare_conditional(spatial_shape, train_data).cuda()

    # Concatenate along the channel dimension (dim=1)
    cond = torch.cat([organ_mask_p, feat_cond, heatmap], dim=1)
    print(f"Condition Tensor Shape: {cond.shape}") # Should be (B, 34, 32, 32, 32)

    T_START = 1000
    noisy_latent = torch.randn(tumor_mask_dims).cuda()

    for i in tqdm(reversed(range(T_START))):
        t_i = torch.full((batch_size,), i, device=device, dtype=torch.long)
        
        # Use diffusion.p_sample if you saved an EMA checkpoint
        noisy_latent = diffusion.p_sample(
            noisy_latent, t_i, cond=cond, cond_scale=2.0)

    # 6. Map from [-1, 1] back to [0, 1]
    recon = noisy_latent
    recon_normalized = (recon + 1.0) / 2.0
    generated_masks = (recon_normalized > 0.5).float()

    masks_np = generated_masks.cpu().numpy().astype(np.uint8)
    raw_np = recon_normalized.cpu().numpy()

    # Grab the ground truth mask from the dataloader so targets_np doesn't error out
    tumor_mask = train_data.get("tumor_mask", torch.zeros_like(generated_masks))
    targets_np = tumor_mask.cpu().numpy().astype(np.uint8)

    debug_folder = Path("debug_folder")
    debug_folder.mkdir(exist_ok=True)

    # Dummy spacing values so affine matrix doesn't crash
    space_x, space_y, space_z = 1.0, 1.0, 1.0 
    step = "inference"

    for b_idx in range(min(3, masks_np.shape[0])):
        pred_3d = masks_np[b_idx, 0, :, :, :]
        targ_3d = targets_np[b_idx, 0, :, :, :]
        raw_3d = raw_np[b_idx, 0, :, :, :]

        if targ_3d.sum() > 0 or True: # Added 'or True' just to ensure it saves for debugging
            affine = np.array([
                [space_x, 0, 0, 0],
                [0, space_y, 0, 0],
                [0, 0, space_z, 0],
                [0, 0, 0, 1]
            ])
            nib.save(
                nib.Nifti1Image(pred_3d, affine=affine),
                str(debug_folder / f"step_{step}_sample_{b_idx}_RECON.nii.gz")
            )
            nib.save(
                nib.Nifti1Image(targ_3d, affine=affine),
                str(debug_folder / f"step_{step}_sample_{b_idx}_GT.nii.gz")
            )
            nib.save(
                nib.Nifti1Image(raw_3d, affine=affine),
                str(debug_folder / f"step_{step}_sample_{b_idx}_RAW_RECON.nii.gz")
            )


if __name__ == '__main__':
    reconstruct()