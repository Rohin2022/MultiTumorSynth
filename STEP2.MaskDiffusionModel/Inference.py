import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import multiprocessing

import torch
import torch.nn.functional as F
from omegaconf import open_dict
from hydra import initialize, compose

# MONAI imports
from monai.transforms import FillHoles, Compose

# Custom local imports
from dataset.dataloader import get_loader
from ddpm import Unet3D, GaussianDiffusion
from ddpm import DDIMSampler 
from metrics import RadiomicsMetricsEvaluator

sys.path.append(os.getcwd())


def process_radiomics_worker(task_data):
    """
    Top-level worker function for multiprocessing pyradiomics evaluation.
    Instantiates the evaluator locally to ensure process safety.
    """
    evaluator = RadiomicsMetricsEvaluator()
    
    cleaned_pred_3d = task_data["cleaned_pred_3d"]
    post_spacing = task_data["post_spacing"]
    
    # --- METRICS EXTRACTION ---
    metrics = evaluator.compute(cleaned_pred_3d, post_spacing)
    
    results = []
    for key, target_real_val in task_data["target_vals"].items():
        results.append({
            "cond_scale": task_data["cond_scale"], 
            "column_task": task_data["column_task"], 
            "column": key, 
            "desired_val": target_real_val, 
            "actual_val": metrics.get(key, np.nan)
        })
        
    return results


def postprocess_tensor(raw_mask, scale_factor=3, threshold=0.5):
    """
    Handles both (X, Y, Z) and (B, X, Y, Z) formats with NO channel dimension.
    Accepts both PyTorch Tensors and NumPy arrays.
    """
    is_numpy = isinstance(raw_mask, np.ndarray)
    if is_numpy:
        tensor_mask = torch.from_numpy(raw_mask).float()
    else:
        tensor_mask = raw_mask.float()

    original_dims = tensor_mask.dim()
    if original_dims == 3:
        tensor_mask = tensor_mask.unsqueeze(0)
    elif original_dims == 4:
        pass
    else:
        raise ValueError(
            f"Expected 3D (X,Y,Z) or 4D (B,X,Y,Z) input, got {original_dims}D")

    tensor_mask = tensor_mask.unsqueeze(1)

    if scale_factor != 1:
        pass

    binary_mask = (tensor_mask <= threshold).to(torch.uint8)

    postprocess_transforms = Compose([
        FillHoles(),
        # KeepLargestConnectedComponent(num_components=num_components) 
    ])

    processed_batch = []
    for i in range(binary_mask.shape[0]):
        single_item = binary_mask[i]
        cleaned_item = postprocess_transforms(single_item)
        processed_batch.append(cleaned_item)

    final_tensor = torch.stack(processed_batch, dim=0)
    final_tensor = final_tensor.squeeze(1)

    if original_dims == 3:
        final_tensor = final_tensor.squeeze(0)

    # Return in the same format it was received
    if is_numpy:
        return final_tensor.cpu().numpy().astype(np.uint8)
    return final_tensor


def prepare_conditional_vector(data, device):
    """
    Extracts tabular features into a single tensor.
    Output shape: (Batch, 19) -> 9 organ classes + 10 numerical features
    """
    numerical_features = [
        "major_axis_mm", "minor_axis_mm", "least_axis_mm",
        "volume_ml", "sphericity", "surface_volume_ratio",
        "elongation", "flatness", "max_3d_diameter_mm",
        "num_components"
    ]

    # 1. Handle the categorical "organ" feature
    organ_idx = torch.as_tensor(
        data["organ"], dtype=torch.long, device=device).view(-1)

    # One-hot encode to shape (Batch, 9) and cast back to float32
    organ_one_hot = F.one_hot(organ_idx, num_classes=9).float()

    # 2. Handle the remaining continuous numerical features
    num_tensors = []
    for key in numerical_features:
        val = torch.as_tensor(
            data[key], dtype=torch.float32, device=device).view(-1)
        num_tensors.append(val)

    # Stack continuous features to shape (Batch, 10)
    continuous_vector = torch.stack(num_tensors, dim=1)

    # 3. Concatenate the one-hot organ with the continuous features
    # Resulting shape: (Batch, 19)
    cond_vector = torch.cat([organ_one_hot, continuous_vector], dim=1)

    return cond_vector


def generate_samples(train_data, step, diffusion, cond_scale=2.0, ddim_steps=50, use_ddpm=False, spacing=(3.0, 3.0, 3.0), dims=(48, 48, 48), norm_stats="dataset_norm_stats.json", save_raw=False):
    batch_size = train_data["heatmap"].shape[0]
    
    # Define generation shape: (Channels, D, H, W)
    shape = (1, *dims)

    heatmap = train_data["heatmap"].permute(0, 1, -1, -3, -2).cuda()
    organ_mask_p = train_data["organ_mask"].permute(0, 1, -1, -3, -2).cuda()

    tabular_cond = prepare_conditional_vector(train_data, heatmap.device)
    cond = torch.cat([organ_mask_p, heatmap], dim=1)

    noisy_latent = torch.randn((batch_size, *shape)).cuda()

    with torch.no_grad():
        if use_ddpm:
            # --- STANDARD DDPM SAMPLING ---
            T_START = diffusion.num_timesteps
            for i in tqdm(reversed(range(T_START)), desc="DDPM Sampling", total=T_START, leave=False):
                t_i = torch.full((batch_size,), i, device=heatmap.device, dtype=torch.long)
                noisy_latent = diffusion.p_sample(
                    noisy_latent, 
                    t_i, 
                    cond=cond, 
                    tabular_cond=tabular_cond, 
                    cond_scale=cond_scale
                )
            recon_latent = noisy_latent
        else:
            # --- FAST DDIM SAMPLING ---
            ddim_sampler = DDIMSampler(diffusion)
            recon_latent, _ = ddim_sampler.sample(
                S=ddim_steps,
                batch_size=batch_size,
                shape=shape,
                conditioning=cond,
                tabular_cond=tabular_cond,
                unconditional_guidance_scale=cond_scale,
                x_T=noisy_latent,
                eta=0.0  # Pure deterministic DDIM
            )

        # Reverse the permutation to restore original spatial dimensions!
        recon = recon_latent.permute(0, 1, -2, -1, -3)

        # Normalize using the correctly oriented tensor
        recon_normalized = (recon + 1.0) / 2.0

    raw_np = recon_normalized.cpu().numpy()

    tumor_mask = train_data.get("tumor_mask", torch.zeros_like(recon_normalized))
    targets_np = tumor_mask.cpu().numpy().astype(np.uint8)

    debug_folder = Path("inference_masks_v2")
    debug_folder.mkdir(exist_ok=True)

    base_spacing = spacing
    scale_factor = spacing[0]

    post_spacing = (
        base_spacing[0] / scale_factor,
        base_spacing[1] / scale_factor,
        base_spacing[2] / scale_factor
    )

    base_affine = np.array([
        [base_spacing[0], 0, 0, 0],
        [0, base_spacing[1], 0, 0],
        [0, 0, base_spacing[2], 0],
        [0, 0, 0, 1]
    ])

    new_affine = base_affine.copy()
    new_affine[:3, :3] /= scale_factor

    numerical_features = [
        "major_axis_mm", "minor_axis_mm", "least_axis_mm",
        "volume_ml", "sphericity", "surface_volume_ratio",
        "elongation", "flatness", "max_3d_diameter_mm",
        "num_components"
    ]

    output_tasks = []

    with open(norm_stats, "r") as f:
        normalized_stats = json.load(f)

        for b_idx in range(raw_np.shape[0]):
            raw_3d = raw_np[b_idx, 0, :, :, :]
            targ_3d = targets_np[b_idx, 0, :, :, :]

            # --- POST-PROCESSING ---
            cleaned_pred_3d = postprocess_tensor(
                raw_3d,
                scale_factor=scale_factor,
                threshold=0.5
            )
            
            # --- PREPARE DATA FOR PARALLEL METRICS EXTRACTION ---
            target_vals = {}
            for key in numerical_features:
                normalized_conditioner = train_data[key][b_idx].item()
                target_real_val = (normalized_conditioner * normalized_stats[key]["std"]) + normalized_stats[key]["mean"]
                target_vals[key] = target_real_val

            output_tasks.append({
                "cleaned_pred_3d": cleaned_pred_3d,
                "post_spacing": post_spacing,
                "cond_scale": cond_scale,
                "column_task": train_data["column_task"][b_idx] if "column_task" in train_data else "Unknown",
                "target_vals": target_vals
            })

            # Save the raw output
            if(save_raw):
                nib.save(
                    nib.Nifti1Image(raw_3d, affine=base_affine),
                    str(debug_folder / f"step_inference_{step}_sample_{b_idx}_cfg_{cond_scale}_RAW.nii.gz")
                )
            # Save the cleaned output
            nib.save(
                nib.Nifti1Image(cleaned_pred_3d, affine=new_affine),
                str(debug_folder / f"step_inference_{step}_sample_{b_idx}_cfg_{cond_scale}_CLEANED.nii.gz")
            )

    return output_tasks


def reconstruct(cfg, model_name="model_best.pt", output_file="metrics.csv", cond_scales=[1.0, 2.0, 4.0, 6.0], use_ddpm=False, ddim_steps=50, results_folder_name="mask_diffusion_train_pro_v7", norm_stats="dataset_norm_stats.json", save_raw=False):
    cfg.model.results_folder_postfix = results_folder_name

    torch.cuda.set_device(cfg.model.gpus)
    device = torch.device(f"cuda:{cfg.model.gpus}")

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix)

    print("1. Initializing Model...")
    model = Unet3D(
        dim=cfg.model.diffusion_img_size,
        dim_mults=cfg.model.dim_mults,
        channels=cfg.model.diffusion_num_channels,
        out_dim=1,
        num_continuous_conditioners=10, # Exactly 10 matching your metrics evaluator
        num_organs=9
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type
    ).to(device)

    print("2. Loading Checkpoint...")
    ckpt_path = os.path.join(cfg.model.results_folder, f'{model_name}')
    data = torch.load(ckpt_path, map_location=device)
    diffusion.load_state_dict(data['ema'])
    diffusion.eval()

    print("3. Generating Samples...")
    val_loader, _, _ = get_loader(cfg.dataset)
    loader_iter = iter(val_loader)
    step = 0
    all_radiomics_tasks = []

    for train_data in tqdm(loader_iter, desc="Generating Batches"):
        for scale in cond_scales:
            # Generate masks and collect task data for later calculation
            batch_tasks = generate_samples(
                train_data, 
                step+1, 
                diffusion, 
                cond_scale=scale, 
                ddim_steps=ddim_steps, 
                use_ddpm=use_ddpm, 
                spacing=(cfg.dataset.space_x, cfg.dataset.space_y, cfg.dataset.space_z), 
                dims=(cfg.dataset.roi_x,cfg.dataset.roi_y, cfg.dataset.roi_z), 
                norm_stats=norm_stats, 
                save_raw=save_raw
            )
            all_radiomics_tasks.extend(batch_tasks)
        
        step += 1

    print("4. Calculating Radiomics Metrics in Parallel...")
    all_metrics = []
    num_workers = cfg.dataset.num_workers if hasattr(cfg.dataset, "num_workers") and cfg.dataset.num_workers > 0 else 1
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        for result_batch in tqdm(pool.imap_unordered(process_radiomics_worker, all_radiomics_tasks), total=len(all_radiomics_tasks), desc="Extracting Radiomics"):
            all_metrics.extend(result_batch)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_file, index=False)
    print(f"Inference complete! Metrics saved to {output_file}")


if __name__ == '__main__':
    initialize(version_base=None, config_path="config")
    
    # --- HYDRA OVERRIDE ---
    # Compose the config by keeping the base models but swapping out the dataset config
    # Ensure "inference_dataset" exists in your config/ folder (e.g. inference_dataset.yaml)
    cfg = compose(config_name="base_cfg", overrides=["dataset=inference_dataset"])
    
    reconstruct(
        cfg, 
        output_file="metrics_v8.csv",
        cond_scales=[1.0, 4.0, 8.0],
        use_ddpm=True,     # <--- SET TO TRUE TO RUN STANDARD DDPM
        ddim_steps=200,    # Will be ignored if use_ddpm=True
        norm_stats="dataset_norm_stats.json",
        results_folder_name="mask_diffusion_train_pro_v8",
        save_raw=True
    )