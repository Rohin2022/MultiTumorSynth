from dataset.dataloader import get_loader
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import pandas as pd
import torch
from omegaconf import DictConfig, open_dict
import hydra
import os
from ddpm import Unet3D, GaussianDiffusion
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())

import json


def load_norm_stats(stats_file="dataset_norm_stats.json"):
    """
    Loads the per-feature z-score normalization stats (mean/std) written by
    dataset.dataloader.get_loader() during training. Needed to invert the
    z-scoring applied to attenuation_mean / attenuation_stdev (and other
    numerical features) before they can be compared against raw HU values.
    """
    with open(stats_file, "r") as f:
        return json.load(f)


def denormalize_feature(value, feature_name, norm_stats):
    """
    Inverts the z-score normalization applied in get_loader():
        normalized = (raw - mean) / (std + 1e-6)
    =>  raw        = normalized * (std + 1e-6) + mean
    """
    stats = norm_stats[feature_name]
    return value * (stats["std"] + 1e-6) + stats["mean"]


def denormalize_ct(ct_normalized, a_min, a_max, b_min=-1.0, b_max=1.0):
    """
    Inverts MONAI's ScaleIntensityRanged (with clip=True) used in get_loader():
        normalized = (raw - a_min) / (a_max - a_min) * (b_max - b_min) + b_min
    =>  raw        = (normalized - b_min) / (b_max - b_min) * (a_max - a_min) + a_min

    Note: clip=True is lossy at the tails (values originally outside
    [a_min, a_max] were clipped before scaling), so this recovers HU only
    up to that clipping — same caveat applies to the real training data.
    """
    return (ct_normalized - b_min) / (b_max - b_min) * (a_max - a_min) + a_min


def prepare_conditional_vector(data, device):
    """
    Mirrors Trainer.prepare_conditional_vector() exactly.
    Output shape: (B, 19)  ->  9 one-hot organ classes + 10 continuous features
    """
    numerical_features = [
        "attenuation_mean", "attenuation_stdev", "attenuation_delta",
        "attenuation_skew", "attenuation_10th", "attenuation_uniformity",
        "glcm_contrast", "glcm_autocorrelation", "glcm_idm", "num_components",
    ]
    organ_idx    = torch.as_tensor(data["organ"], dtype=torch.long, device=device).view(-1)
    organ_one_hot = F.one_hot(organ_idx, num_classes=9).float()          # (B, 9)

    num_tensors = [
        torch.as_tensor(data[k], dtype=torch.float32, device=device).view(-1)
        for k in numerical_features
    ]
    continuous_vector = torch.stack(num_tensors, dim=1)                  # (B, 10)
    return torch.cat([organ_one_hot, continuous_vector], dim=1)          # (B, 19)


def build_spatial_cond(image, mask, vqgan, device):
    """
    Mirrors GaussianDiffusion.forward() spatial conditioning path exactly:

        mask_     = (1 - mask).detach()          # background=1, tumor=0
        masked_img = img * mask_                  # zero out the tumor region
        # permute: (B,C,X,Y,Z) -> (B,C,Z,X,Y)
        masked_img = masked_img.permute(0,1,4,2,3)
        mask       = mask.permute(0,1,4,2,3)
        # VQGAN encode + normalise to [-1, 1]
        latent     = vqgan.encode(masked_img, ...)
        latent_n   = ((latent - emb_min) / emb_denom) * 2 - 1
        # mask channel
        cc         = interpolate(mask * 2 - 1, size=latent.shape[-3:])
        cond       = cat([latent_n, cc], dim=1)   # (B, C_lat+1, Z', X', Y')

    Returns:
        spatial_cond  (B, C_lat+1, Z', X', Y')  — ready to pass as `cond`
        latent_shape  tuple — shape of the noise tensor for sampling
    """
    # Inputs from the dataloader are (B, C, X, Y, Z)
    image = image.to(device)
    mask  = mask.to(device)
    mask = (mask==2).float()
    # 1. Zero out the tumor region in the CT (the part the model must synthesise)
    mask_bg     = (1 - mask).detach()
    masked_img  = (image * mask_bg).detach()

    # 2. Permute to the VQGAN / diffusion convention: (B,C,X,Y,Z) -> (B,C,Z,X,Y)
    masked_img_p = masked_img.permute(0, 1, 4, 2, 3)
    mask_p       = mask.permute(0, 1, 4, 2, 3)

    with torch.no_grad():
        emb_min   = vqgan.codebook.embeddings.min()
        emb_max   = vqgan.codebook.embeddings.max()
        emb_denom = emb_max - emb_min

        # 3. Encode masked CT and normalise to [-1, 1]
        latent   = vqgan.encode(masked_img_p, quantize=False, include_embeddings=True)
        latent_n = ((latent - emb_min) / emb_denom) * 2.0 - 1.0
        # latent_n: (B, C_lat, Z', X', Y')

        # 4. Downscale binary mask to latent spatial size, rescale to [-1, 1]
        cc = F.interpolate(
            mask_p * 2.0 - 1.0,
            size=latent_n.shape[-3:],
            mode='nearest',
        )   # (B, 1, Z', X', Y')

        # 5. Concatenate — matches training: cond = cat([masked_img, cc], dim=1)
        spatial_cond = torch.cat([latent_n, cc], dim=1)
        # (B, C_lat+1, Z', X', Y')

    return spatial_cond, latent_n.shape


def compute_tumor_attenuation(ct_3d, mask_3d, target_mean=None, target_std=None):
    """
    Computes mean/stdev attenuation (HU) inside the tumor mask region of a
    single generated sample, and optionally compares it against the
    conditioning target values used for that sample.

    IMPORTANT: ct_3d must already be in raw HU (i.e. denormalize_ct() has
    been applied). target_mean/target_std must already be denormalized
    out of z-score space (i.e. denormalize_feature() has been applied).
    This function does no unit conversion itself — it just computes stats
    on whatever scale it's handed.

    ct_3d, mask_3d: numpy arrays, shape (X, Y, Z). mask_3d is the binary
    tumor mask (1 = tumor) used as conditioning/ground-truth region.

    Returns a dict with the computed stats (and error vs target, if given).
    """
    tumor_voxels = ct_3d[mask_3d.astype(bool)]

    result = {
        "n_voxels": int(tumor_voxels.size),
        "gen_mean": None,
        "gen_std": None,
        "target_mean": target_mean,
        "target_std": target_std,
        "mean_error": None,
        "std_error": None,
    }

    if tumor_voxels.size == 0:
        # No tumor voxels in mask — nothing to compute
        return result

    gen_mean = float(tumor_voxels.mean())
    gen_std  = float(tumor_voxels.std())

    result["gen_mean"] = gen_mean
    result["gen_std"]  = gen_std

    if target_mean is not None:
        result["mean_error"] = gen_mean - float(target_mean)
    if target_std is not None:
        result["std_error"] = gen_std - float(target_std)

    return result


def decode_latent(latent, vqgan):
    """
    Inverse of the VQGAN normalisation used in training, then decode.
    Returns raw CT values in (B, 1, X, Y, Z).
    """
    emb_min   = vqgan.codebook.embeddings.min()
    emb_max   = vqgan.codebook.embeddings.max()
    emb_denom = emb_max - emb_min

    # Invert: latent_n = ((latent - emb_min) / emb_denom) * 2 - 1
    latent_denorm = ((latent + 1.0) / 2.0) * emb_denom + emb_min

    # Decode: output is in (B, C, Z, X, Y) — reverse training permutation back to (B,C,X,Y,Z)
    decoded = vqgan.decode(latent_denorm, quantize=True)
    decoded = decoded.permute(0, 1, 3, 4, 2).contiguous()   # (B, C, X, Y, Z)
    return decoded


def generate_samples(data, step, diffusion, vqgan, norm_stats, a_min, a_max, cond_scale=2.0):
    """
    Full inference pass for one batch:
      - builds spatial cond from the real masked CT + tumor mask
      - builds tabular cond from radiomics features
      - runs the reverse diffusion loop in latent space
      - decodes back to CT (still in [-1, 1] intensity space)
      - denormalizes both the generated CT and the attenuation targets back
        to raw HU, then computes generated tumor-region attenuation vs.
        conditioning target
      - saves one NIfTI per sample (raw HU)
    """
    device     = next(diffusion.parameters()).device
    batch_size = data["image"].shape[0]

    image = data["image"]   # (B, 1, X, Y, Z), normalized to [-1, 1]
    mask  = data["label"]   # (B, 1, X, Y, Z)  binary tumor mask

    # ---- conditioning ----
    spatial_cond, latent_shape = build_spatial_cond(image, mask, vqgan, device)
    tabular_cond = prepare_conditional_vector(data, device)

    # Target attenuation values used as conditioning, per sample in batch.
    # These come out of get_loader() z-scored — denormalize back to HU
    # before comparing against the generated CT.
    target_means_z = torch.as_tensor(data["attenuation_mean"], dtype=torch.float32).view(-1)
    target_stds_z  = torch.as_tensor(data["attenuation_stdev"], dtype=torch.float32).view(-1)
    target_means_hu = [
        denormalize_feature(v.item(), "attenuation_mean", norm_stats) for v in target_means_z
    ]
    target_stds_hu = [
        denormalize_feature(v.item(), "attenuation_stdev", norm_stats) for v in target_stds_z
    ]

    # ---- reverse diffusion in latent space ----
    noisy_latent = torch.randn(latent_shape, device=device)

    with torch.no_grad():
        for i in tqdm(
            reversed(range(diffusion.num_timesteps)),
            desc=f"Sampling cfg={cond_scale}",
            total=diffusion.num_timesteps,
            leave=False,
        ):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            noisy_latent = diffusion.p_sample(
                noisy_latent, t,
                cond=spatial_cond,
                tabular_cond=tabular_cond,
                cond_scale=cond_scale,
                clip_denoised=False
            )

        # ---- decode ----
        ct_synth = decode_latent(noisy_latent, vqgan)   # (B, 1, X, Y, Z)

    ct_np   = ct_synth.cpu().numpy()
    mask_np = mask.numpy()

    out_dir = Path("inference_output")
    out_dir.mkdir(exist_ok=True)

    spacing    = (1.0, 1.0, 1.0)
    affine     = np.diag([*spacing, 1.0])

    attenuation_records = []

    for b in range(batch_size):
        ct_3d_norm = ct_np[b, 0]   # (X, Y, Z), still in [-1, 1]
        mask_3d    = mask_np[b, 0]  # (X, Y, Z)

        # ---- denormalize generated CT back to raw HU ----
        ct_3d_hu = denormalize_ct(ct_3d_norm, a_min=a_min, a_max=a_max)

        # ---- attenuation in the generated tumor region (HU vs HU) ----
        atten_stats = compute_tumor_attenuation(
            ct_3d_hu, mask_3d,
            target_mean=target_means_hu[b],
            target_std=target_stds_hu[b],
        )
        atten_stats.update({"step": step, "sample": b, "cond_scale": cond_scale})
        attenuation_records.append(atten_stats)

        print(f"\n===== step {step}  sample {b+1}  cfg={cond_scale} =====")
        print(f"  CT   shape : {ct_3d_hu.shape}  range [{ct_3d_hu.min():.3f}, {ct_3d_hu.max():.3f}] HU")
        print(f"  Mask voxels: {mask_3d.sum().astype(int)}")
        if atten_stats["gen_mean"] is not None:
            print(
                f"  Tumor attenuation (HU): "
                f"gen mean={atten_stats['gen_mean']:.2f} (target={atten_stats['target_mean']:.2f}, "
                f"err={atten_stats['mean_error']:+.2f})  "
                f"gen std={atten_stats['gen_std']:.2f} (target={atten_stats['target_std']:.2f}, "
                f"err={atten_stats['std_error']:+.2f})"
            )
        else:
            print("  Tumor attenuation (HU): no tumor voxels in mask, skipped")

        stem = f"step{step:04d}_b{b}_cfg{cond_scale}"
        nib.save(
            nib.Nifti1Image(ct_3d_hu.astype(np.float32), affine),
            str(out_dir / f"{stem}_ct.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(mask_3d.astype(np.uint8), affine),
            str(out_dir / f"{stem}_mask.nii.gz"),
        )

    # ---- append attenuation stats for this batch to a running CSV log ----
    atten_df  = pd.DataFrame(attenuation_records)
    atten_csv = out_dir / "tumor_attenuation_log.csv"
    atten_df.to_csv(
        atten_csv,
        mode="a",
        header=not atten_csv.exists(),
        index=False,
    )


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def reconstruct(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    device = torch.device(f"cuda:{cfg.model.gpus}")

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder,
            cfg.dataset.name,
            cfg.model.results_folder_postfix,
        )

    # ---- Diffusion model (VQGAN bundled via vqgan_ckpt, loaded from checkpoint) ----
    print("1. Initializing diffusion model...")
    model = Unet3D(
        dim=cfg.model.diffusion_img_size,
        dim_mults=cfg.model.dim_mults,
        channels=cfg.model.diffusion_num_channels,
        out_dim=cfg.model.out_dim,
        num_organs=9,
        num_continuous_conditioners=10,
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
        vqgan_ckpt=cfg.model.vqgan_ckpt,  # VQGAN weights come from the checkpoint
    ).to(device)

    # ---- Checkpoint (EMA weights, includes VQGAN) ----
    print("2. Loading checkpoint...")
    ckpt_path = os.path.join(cfg.model.results_folder, 'model_best.pt')
    ckpt      = torch.load(ckpt_path, map_location=device)
    diffusion.load_state_dict(ckpt['ema'], strict=True)
    diffusion.eval()
    vqgan = diffusion.vqgan

    # ---- Inference loop ----
    print("4. Running inference...")
    val_loader, _, _ = get_loader(cfg.dataset)

    # Same z-score stats file written by get_loader() during training, and
    # the same HU clip window (a_min/a_max) passed to ScaleIntensityRanged —
    # both needed to denormalize generated CT / attenuation targets back to HU.
    norm_stats = load_norm_stats(f"dataset_norm_stats_{cfg.model.results_folder_postfix}.json")
    a_min = cfg.dataset.a_min
    a_max = cfg.dataset.a_max

    cond_scales = [6.0]

    for step, batch in enumerate(tqdm(val_loader, desc="Batches")):
        for scale in cond_scales:
            generate_samples(
                batch, step + 1, diffusion, vqgan,
                norm_stats=norm_stats, a_min=a_min, a_max=a_max,
                cond_scale=scale,
            )


if __name__ == '__main__':
    reconstruct()