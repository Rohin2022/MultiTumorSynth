from dataset.dataloader import get_loader
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import pandas as pd
import torch
from omegaconf import DictConfig, open_dict
import hydra
import os
from ddpm import Unet3D, GaussianDiffusion, ResnetBlock, Unet3D_CA
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
    numerical features) before they can be used as conditioning-vector
    provenance / debugging. NOTE: these are no longer used to build the
    *evaluation* target — see ground_truth_tumor_attenuation() below.
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
    [a_min, a_max] were clipped before scaling). This recovers HU only up
    to that clipping — this is exactly why the *evaluation* ground truth
    must be computed from the already-clipped/resampled `data["image"]`
    tensor rather than from the raw pyradiomics CSV (see
    ground_truth_tumor_attenuation()). The CSV values live in a space the
    network never saw and can legitimately fall outside [a_min, a_max].
    """
    return (ct_normalized - b_min) / (b_max - b_min) * (a_max - a_min) + a_min


def prepare_conditional_vector(data, device):
    """
    Mirrors Trainer.prepare_conditional_vector() exactly.
    Output shape: (B, 19)  ->  9 one-hot organ classes + 10 continuous features

    This vector is built from the z-scored CSV-derived radiomics (raw,
    unclipped HU space) — that's correct and unchanged, since the network
    was trained with this exact conditioning provenance. Only the
    *evaluation* comparison target changes (see below).
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
    mask = (mask == 2).float()
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
    single sample, and optionally compares it against target_mean/target_std.

    IMPORTANT: ct_3d must already be in raw HU (denormalize_ct() applied).
    target_mean/target_std must be in the SAME space as ct_3d — i.e. also
    HU, and specifically HU-after-clipping if ct_3d came out of the
    clipped training/generation pipeline (which it always does here).
    This function does no unit conversion itself — it just computes stats
    on whatever scale it's handed and diffs them.
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


def ground_truth_tumor_attenuation(image, mask, a_min, a_max):
    """
    Computes mean/std attenuation directly from the *transformed* ground
    truth image (post-Spacingd resample, post-ScaleIntensityRanged clip),
    inside the tumor region (label == 2).

    This is the fair comparison target for gen_mean/gen_std. The
    pyradiomics CSV values (attenuation_mean/attenuation_stdev) were
    computed on raw, native-resolution, unclipped ct.nii.gz — a space the
    network never operates in. Resampling blends HU across voxels
    (reducing local variance) and clip=True hard-clips the tails
    (compressing stdev / shifting extreme means) before the network ever
    sees the data. Comparing generated output — which is mechanically
    confined to [a_min, a_max] post-decode — against the unclipped CSV
    number manufactures systematic error that has nothing to do with
    generation quality, especially for hyper-/hypo-attenuating tumors.

    image: (B, 1, X, Y, Z) tensor, still normalized to [b_min, b_max]
    mask:  (B, 1, X, Y, Z) tensor, ternary label (organ=1, tumor=2)

    Returns a list of dicts (one per batch element): {"gt_mean", "gt_std"}
    """
    ct_hu = denormalize_ct(image.detach().cpu().numpy(), a_min=a_min, a_max=a_max)
    tumor_mask = (mask.detach().cpu().numpy() == 2)

    records = []
    for b in range(image.shape[0]):
        voxels = ct_hu[b, 0][tumor_mask[b, 0]]
        if voxels.size == 0:
            records.append({"gt_mean": None, "gt_std": None})
            continue
        records.append({
            "gt_mean": float(voxels.mean()),
            "gt_std": float(voxels.std()),
        })
    return records


def vqgan_reconstruction_floor(image, mask, vqgan, a_min, a_max, device):
    """
    Diagnostic: encodes the REAL (unmasked-in) tumor-containing CT through
    the VQGAN and decodes with no diffusion step at all, then computes
    tumor-region attenuation stats on that pure reconstruction.

    VQGAN quantization + spatial downsampling is itself lossy, so any
    gen_std shortfall you see in the diffusion output is a mix of
    (a) diffusion/conditioning error and (b) VQGAN compression smoothing.
    This function isolates (b): it's the best gen_std the pipeline could
    ever produce even with a perfect diffusion model, since the tumor
    region here is reconstructed from the *true* image, not synthesized.

    Read diffusion std_error relative to (gt_std - floor_std), not to 0.

    image: (B, 1, X, Y, Z) tensor, normalized to [b_min, b_max]
    mask:  (B, 1, X, Y, Z) tensor, ternary label (organ=1, tumor=2)

    Returns a list of dicts (one per batch element):
        {"floor_mean", "floor_std"}
    """
    image = image.to(device)

    with torch.no_grad():
        emb_min   = vqgan.codebook.embeddings.min()
        emb_max   = vqgan.codebook.embeddings.max()
        emb_denom = emb_max - emb_min

        # Same permutation convention as build_spatial_cond / training
        image_p = image.permute(0, 1, 4, 2, 3)

        latent    = vqgan.encode(image_p, quantize=False, include_embeddings=True)
        latent_n  = ((latent - emb_min) / emb_denom) * 2.0 - 1.0
        recon     = decode_latent(latent_n, vqgan)   # (B, 1, X, Y, Z), in [-1, 1]

    recon_np   = recon.cpu().numpy()
    ct_hu      = denormalize_ct(recon_np, a_min=a_min, a_max=a_max)
    tumor_mask = (mask.detach().cpu().numpy() == 2)

    records = []
    for b in range(image.shape[0]):
        voxels = ct_hu[b, 0][tumor_mask[b, 0]]
        if voxels.size == 0:
            records.append({"floor_mean": None, "floor_std": None})
            continue
        records.append({
            "floor_mean": float(voxels.mean()),
            "floor_std": float(voxels.std()),
        })
    return records


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

def get_denormalized_radiomics(data, idx, norm_stats):
    """
    Extracts the radiomics conditioning values for one sample and
    converts them back from z-score normalized space to raw values.
    """
    numerical_features = [
        "attenuation_mean", "attenuation_stdev", "attenuation_delta",
        "attenuation_skew", "attenuation_10th", "attenuation_uniformity",
        "glcm_contrast", "glcm_autocorrelation", "glcm_idm",
        "num_components",
    ]

    record = {}

    for key in numerical_features:
        # normalized value from dataloader
        value = data[key][idx]

        # convert torch tensor -> python float
        if torch.is_tensor(value):
            value = value.detach().cpu().item()

        # undo z-score normalization
        record[key] = float(
            denormalize_feature(
                value,
                key,
                norm_stats
            )
        )

    # organ is categorical; keep original value
    organ = data["organ"][idx]
    if torch.is_tensor(organ):
        organ = organ.detach().cpu().item()

    record["organ"] = int(organ)

    return record


def generate_samples(data, step, diffusion, vqgan, norm_stats, a_min, a_max, cond_scale=2.0):
    """
    Full inference pass for one batch:
      - builds spatial cond from the real masked CT + tumor mask
      - builds tabular cond from radiomics features (unchanged: z-scored,
        raw/unclipped-HU-derived, exactly as seen in training)
      - runs the reverse diffusion loop in latent space
      - decodes back to CT (still in [b_min, b_max] intensity space)
      - denormalizes the generated CT back to raw HU
      - computes ground-truth tumor attenuation directly from the
        transformed (resampled + clipped) data["image"]/data["label"] —
        NOT from the denormalized CSV target — so gen vs. target lives in
        the same clipped space the network actually operates in
      - also computes the VQGAN-only reconstruction floor for std, so
        std_error can be read relative to compression loss rather than 0
      - saves one NIfTI per sample (raw HU)
    """
    device     = next(diffusion.parameters()).device
    batch_size = data["image"].shape[0]

    image = data["image"]   # (B, 1, X, Y, Z), normalized to [b_min, b_max]
    mask  = data["label"]   # (B, 1, X, Y, Z)  ternary mask (organ=1, tumor=2)

    # ---- conditioning ----
    spatial_cond, latent_shape = build_spatial_cond(image, mask, vqgan, device)
    tabular_cond = prepare_conditional_vector(data, device)

    # ---- ground-truth comparison targets, computed in TRANSFORMED space ----
    # (post-Spacingd, post-ScaleIntensityRanged clip) — this replaces the
    # old norm_stats-denormalized CSV target, which lived in raw/unclipped
    # HU space the network never saw.
    gt_records = ground_truth_tumor_attenuation(image, mask, a_min=a_min, a_max=a_max)

    # ---- VQGAN-only reconstruction floor (diagnostic) ----
    floor_records = vqgan_reconstruction_floor(image, mask, vqgan, a_min=a_min, a_max=a_max, device=device)

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
                clip_denoised=True
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
        ct_3d_norm = ct_np[b, 0]   # (X, Y, Z), still in [b_min, b_max]
        mask_3d    = mask_np[b, 0]  # (X, Y, Z)

        # ---- denormalize generated CT back to raw HU ----
        ct_3d_hu = denormalize_ct(ct_3d_norm, a_min=a_min, a_max=a_max)

        # ---- attenuation in the generated tumor region vs. transformed-space GT ----
        gt_mean = gt_records[b]["gt_mean"]
        gt_std  = gt_records[b]["gt_std"]

        atten_stats = compute_tumor_attenuation(
            ct_3d_hu, mask_3d,
            target_mean=gt_mean,
            target_std=gt_std,
        )
        radiomics_record = get_denormalized_radiomics(
            data,
            b,
            norm_stats
        )

        atten_stats.update({
            "step": step,
            "sample": b,
            "cond_scale": cond_scale,
            "floor_mean": floor_records[b]["floor_mean"],
            "floor_std": floor_records[b]["floor_std"],
            "filename": f"step{step:04d}_b{b}_cfg{cond_scale}",

            # denormalized conditioning values
            **radiomics_record,
        })
        # std error relative to the VQGAN compression floor, when available
        if atten_stats["gen_std"] is not None and atten_stats["floor_std"] is not None and gt_std is not None:
            atten_stats["std_error_vs_floor"] = (
                (atten_stats["gen_std"] - atten_stats["floor_std"])
                - 0.0  # floor already reflects (recon_std - gt_std) baseline implicitly via floor_std vs gt_std
            )
            atten_stats["floor_std_gap"] = gt_std - atten_stats["floor_std"]
        else:
            atten_stats["std_error_vs_floor"] = None
            atten_stats["floor_std_gap"] = None

        attenuation_records.append(atten_stats)

        print(f"\n===== step {step}  sample {b+1}  cfg={cond_scale} =====")
        print(f"  CT   shape : {ct_3d_hu.shape}  range [{ct_3d_hu.min():.3f}, {ct_3d_hu.max():.3f}] HU")
        print(f"  Mask voxels: {mask_3d.sum().astype(int)}")
        if atten_stats["gen_mean"] is not None:
            print(
                f"  Tumor attenuation (HU, transformed-space GT): "
                f"gen mean={atten_stats['gen_mean']:.2f} (gt={atten_stats['target_mean']:.2f}, "
                f"err={atten_stats['mean_error']:+.2f})  "
                f"gen std={atten_stats['gen_std']:.2f} (gt={atten_stats['target_std']:.2f}, "
                f"err={atten_stats['std_error']:+.2f})"
            )
            if atten_stats["floor_std"] is not None:
                print(
                    f"  VQGAN-only recon std={atten_stats['floor_std']:.2f} "
                    f"(gt gap={atten_stats['floor_std_gap']:+.2f})  "
                    f"-> compression accounts for {atten_stats['floor_std_gap']:+.2f} HU "
                    f"of any diffusion std_error"
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


@hydra.main(config_path='config', config_name='inference', version_base=None)
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
    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.unet_dim,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels, # image (1) and tumor mask (1)
            out_dim=cfg.model.out_dim,
            num_continuous_conditioners=10,
            num_organs=9
        ).cuda()
    elif cfg.model.denoising_fn == 'Unet3D_CA':
        x_channels = cfg.model.out_dim
        cond_channels = cfg.model.diffusion_num_channels - cfg.model.out_dim

        model = Unet3D_CA(
            dim=cfg.model.unet_dim,
            dim_mults=cfg.model.dim_mults,
            channels=x_channels,
            out_dim=cfg.model.out_dim,
            num_continuous_conditioners=10,
            num_organs=9,
            cond_channels=cond_channels,
            num_res_blocks=2,
            attention_resolutions=(2, 4, 8),
            num_heads=8,
            # dim_head removed -- now computed internally as ch // num_heads
            # at every resolution level, matching source's legacy=True behavior
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")


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


    
    for name, block in diffusion.denoise_fn.named_modules():
        if isinstance(block, ResnetBlock) and block.mlp is not None:
            w = block.mlp[1].weight  # nn.Linear inside the Sequential
            time_dim = w.shape[1] - diffusion.denoise_fn.tabular_emb_dim
            w_time = w[:, :time_dim]
            w_tab = w[:, time_dim:]
            print(f"{name}: time_cols_mean={w_time.abs().mean().item():.5f}  tab_cols_mean={w_tab.abs().mean().item():.5f}  ratio={w_tab.abs().mean().item()/w_time.abs().mean().item():.4f}")


    vqgan = diffusion.vqgan

    # ---- Inference loop ----
    print("4. Running inference...")
    val_loader, _, _ = get_loader(cfg.dataset)

    # norm_stats is still loaded (kept for provenance / debugging the
    # tabular conditioning vector), but is no longer used to build the
    # evaluation ground truth — see ground_truth_tumor_attenuation().
    norm_stats = load_norm_stats(f"dataset_norm_stats_{cfg.model.results_folder_postfix}.json")

    # HU clip window used by ScaleIntensityRanged in the transform pipeline.
    # a_min=-1000, a_max=500, b_min=-1, b_max=1, clip=True
    a_min = getattr(cfg.dataset, "a_min", -1000)
    a_max = getattr(cfg.dataset, "a_max", 500)

    # Sweep cond_scale so mean/std error vs. gt can be inspected as a
    # function of guidance strength. If error shrinks monotonically as
    # scale drops, cond_scale=6.0 was pushing samples off-manifold
    # (CFG overshoot). If error stays large even at scale=1.0 (near the
    # unconditional/weakly-conditional regime), the tabular conditioning
    # signal itself isn't being learned/used strongly enough — that's a
    # training-side problem (FiLM magnitude, embedding scale, loss
    # weighting), not a sampling-time one.
    cond_scales = [1.0, 2.0, 4.0, 6.0]

    for step, batch in enumerate(tqdm(val_loader, desc="Batches")):
        for scale in cond_scales:
            generate_samples(
                batch, step + 1, diffusion, vqgan,
                norm_stats=norm_stats, a_min=a_min, a_max=a_max,
                cond_scale=scale,
            )


if __name__ == '__main__':
    reconstruct()