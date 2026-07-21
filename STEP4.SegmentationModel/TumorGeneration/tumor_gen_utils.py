# Tumor Generation - inference pipeline
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from omegaconf import OmegaConf
from monai.transforms import FillHoles, Compose

# TODO: adjust this import path to wherever DDIMSampler actually lives in
# your package layout (the reference mask inference script imports it as
# `from ddpm.ddim import DDIMSampler`).
from .diffusion_models.STEP2_ddpm import DDIMSampler as Mask_DDIMSampler
from .diffusion_models.STEP2_ddpm import (
    Unet3D as Mask_Unet3D,
    GaussianDiffusion as Mask_GaussianDiffusion,
    Tester as Mask_Tester,
)
from .diffusion_models.STEP3_ddpm import (
    Unet3D_CA as Tumor_Unet3D_CA,
    GaussianDiffusion as Tumor_GaussianDiffusion,
    Tester as Tumor_Tester,
)
from .diffusion_models.STEP3_ddpm import DDIMSampler as Tumor_DDIMSampler



from .radiomics_sampler.utils import synthesize_organ_radiomics, split_tumor_shape_features, apply_normalization as apply_radiomics_normalization


# --------------------------------------------------------------------------- #
# Feature ordering / organ index maps
#
# These must match exactly the column order used when the GMM radiomics banks
# were fit and when MASK_COLUMNS / TUMOR_COLUMNS were used to build the
# training-time conditioning vectors.
# --------------------------------------------------------------------------- #
MASK_COLUMNS = [
    'diameter_x_mm',
    'diameter_y_mm',
    'diameter_z_mm',
    'original_shape_Elongation',
    'original_shape_Flatness',
    'original_shape_LeastAxisLength',
    'original_shape_MajorAxisLength',
    'original_shape_Maximum2DDiameterColumn',
    'original_shape_Maximum2DDiameterRow',
    'original_shape_Maximum2DDiameterSlice',
    'original_shape_Maximum3DDiameter',
    'original_shape_MeshVolume',
    'original_shape_MinorAxisLength',
    'original_shape_Sphericity',
    'original_shape_SurfaceArea',
    'original_shape_SurfaceVolumeRatio',
    'original_shape_VoxelVolume',
]

TUMOR_COLUMNS = [
    'attenuation_delta',
    'original_firstorder_10Percentile', 'original_firstorder_90Percentile',
    'original_firstorder_Energy', 'original_firstorder_Entropy',
    'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis',
    'original_firstorder_Maximum', 'original_firstorder_Mean',
    'original_firstorder_MeanAbsoluteDeviation',
    'original_firstorder_Median',
    'original_firstorder_Minimum',
    'original_firstorder_Range',
    'original_firstorder_RobustMeanAbsoluteDeviation',
    'original_firstorder_RootMeanSquared',
    'original_firstorder_Skewness',
    'original_firstorder_TotalEnergy',
    'original_firstorder_Uniformity',
    'original_firstorder_Variance',
    'original_glcm_Autocorrelation',
    'original_glcm_ClusterProminence',
    'original_glcm_ClusterShade',
    'original_glcm_ClusterTendency',
    'original_glcm_Contrast',
    'original_glcm_Correlation',
    'original_glcm_DifferenceAverage',
    'original_glcm_DifferenceEntropy',
    'original_glcm_DifferenceVariance',
    'original_glcm_Id',
    'original_glcm_Idm',
    'original_glcm_Idmn',
    'original_glcm_Idn',
    'original_glcm_Imc1',
    'original_glcm_Imc2',
    'original_glcm_InverseVariance',
    'original_glcm_JointAverage',
    'original_glcm_JointEnergy',
    'original_glcm_JointEntropy',
    'original_glcm_MCC',
    'original_glcm_MaximumProbability',
    'original_glcm_SumAverage',
    'original_glcm_SumEntropy',
    'original_glcm_SumSquares',
    'original_gldm_DependenceEntropy',
    'original_gldm_DependenceNonUniformity',
    'original_gldm_DependenceNonUniformityNormalized',
    'original_gldm_DependenceVariance',
    'original_gldm_GrayLevelNonUniformity',
    'original_gldm_GrayLevelVariance',
    'original_gldm_HighGrayLevelEmphasis',
    'original_gldm_LargeDependenceEmphasis',
    'original_gldm_LargeDependenceHighGrayLevelEmphasis',
    'original_gldm_LargeDependenceLowGrayLevelEmphasis',
    'original_gldm_LowGrayLevelEmphasis',
    'original_gldm_SmallDependenceEmphasis',
    'original_gldm_SmallDependenceHighGrayLevelEmphasis',
    'original_gldm_SmallDependenceLowGrayLevelEmphasis',
    'original_glrlm_GrayLevelNonUniformity',
    'original_glrlm_GrayLevelNonUniformityNormalized',
    'original_glrlm_GrayLevelVariance',
    'original_glrlm_HighGrayLevelRunEmphasis',
    'original_glrlm_LongRunEmphasis',
    'original_glrlm_LongRunHighGrayLevelEmphasis',
    'original_glrlm_LongRunLowGrayLevelEmphasis',
    'original_glrlm_LowGrayLevelRunEmphasis',
    'original_glrlm_RunEntropy',
    'original_glrlm_RunLengthNonUniformity',
    'original_glrlm_RunLengthNonUniformityNormalized',
    'original_glrlm_RunPercentage',
    'original_glrlm_RunVariance',
    'original_glrlm_ShortRunEmphasis',
    'original_glrlm_ShortRunHighGrayLevelEmphasis',
    'original_glrlm_ShortRunLowGrayLevelEmphasis',
    'original_glszm_GrayLevelNonUniformity',
    'original_glszm_GrayLevelNonUniformityNormalized',
    'original_glszm_GrayLevelVariance',
    'original_glszm_HighGrayLevelZoneEmphasis',
    'original_glszm_LargeAreaEmphasis',
    'original_glszm_LargeAreaHighGrayLevelEmphasis',
    'original_glszm_LargeAreaLowGrayLevelEmphasis',
    'original_glszm_LowGrayLevelZoneEmphasis',
    'original_glszm_SizeZoneNonUniformity',
    'original_glszm_SizeZoneNonUniformityNormalized',
    'original_glszm_SmallAreaEmphasis',
    'original_glszm_SmallAreaHighGrayLevelEmphasis',
    'original_glszm_SmallAreaLowGrayLevelEmphasis',
    'original_glszm_ZoneEntropy',
    'original_glszm_ZonePercentage',
    'original_glszm_ZoneVariance',
    'original_ngtdm_Busyness',
    'original_ngtdm_Coarseness',
    'original_ngtdm_Complexity',
    'original_ngtdm_Contrast',
    'original_ngtdm_Strength',
]

ORGAN_TO_IDX = {
    "spleen": 0,
    "bladder": 1,
    "gallbladder": 2,
    "esophagus": 3,
    "stomach": 4,
    "duodenum": 5,
    "colon": 6,
    "prostate": 7,
    "uterus": 8,
}

# TODO: tune / replace with the actual volume thresholds (in voxels or mm^3)
# that define your small / medium / large tumor buckets.
SMALL_VOLUME_THRESHOLD = 500.0
LARGE_VOLUME_THRESHOLD = 5000.0

# Default width of the gaussian tumor-location heatmap, matching the
# GenerateTumorHeatmapd MONAI transform used at training time.
DEFAULT_HEATMAP_SIGMA = 8.0


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_config(path):
    """Loads the pipeline config (see the accompanying config.yaml) via OmegaConf."""
    return OmegaConf.load(path)






# --------------------------------------------------------------------------- #
# Radiomics sampling
#
# GMM loading / sampling lives in radiomics_sampler.utils
# (synthesize_organ_radiomics, split_tumor_shape_features) — this just wires
# that up for our MASK_COLUMNS / TUMOR_COLUMNS split and reshapes the result
# into the {"mask_radiomics":..., "tumor_radiomics":...} shape the rest of
# this module expects.
# --------------------------------------------------------------------------- #
def sample_radiomics(gmm_bank_path, organ):
    """Sample one synthetic radiomics vector for `organ` from the unified GMM
    bank at `gmm_bank_path`, then split it into mask-shape vs tumor-appearance
    feature dicts."""
    sampled = synthesize_organ_radiomics(gmm_bank_path, organ, num_samples=1)
    tumor_features, shape_features = split_tumor_shape_features(
        sampled, tumor_keys=TUMOR_COLUMNS, shape_keys=MASK_COLUMNS
    )

    mask_radiomics = {k: float(v[0]) for k, v in shape_features.items()}
    tumor_radiomics = {k: float(v[0]) for k, v in tumor_features.items()}
    return {"mask_radiomics": mask_radiomics, "tumor_radiomics": tumor_radiomics}


def get_size(radiomics):
    """Classify sampled tumor into a small/medium/large bucket from its volume feature."""
    mask_radiomics = radiomics["mask_radiomics"]
    volume = mask_radiomics.get("original_shape_VoxelVolume", list(mask_radiomics.values())[0])
    if volume < SMALL_VOLUME_THRESHOLD:
        return "small"
    elif volume < LARGE_VOLUME_THRESHOLD:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# Tabular conditioning vector construction
# --------------------------------------------------------------------------- #
def build_cond_vector(organ_idx, numerical_dict, columns, device, batch_size=1,
                       num_organs=len(ORGAN_TO_IDX)):
    """Builds the tabular conditioning vector: one-hot organ + numerical radiomics,
    matching the training-time `prepare_conditional_vector` layout."""
    organ_tensor = torch.full((batch_size,), organ_idx, dtype=torch.long, device=device)
    organ_one_hot = F.one_hot(organ_tensor, num_classes=num_organs).float()

    vals = [numerical_dict[c] for c in columns]
    continuous = torch.tensor(vals, dtype=torch.float32, device=device)
    continuous = continuous.unsqueeze(0).repeat(batch_size, 1)

    return torch.cat([organ_one_hot, continuous], dim=1)


# --------------------------------------------------------------------------- #
# Mask-model spatial conditioning + DDIM sampling
#
# Mirrors the reference mask inference script exactly: cond = [organ_mask,
# heatmap] (no CT volume channel), permuted (B,C,X,Y,Z) -> (B,C,Z,X,Y) to
# match the model's training-time axis convention, sampled with DDIM, then
# permuted back and thresholded. The training-time mask convention is
# background=1 / tumor=0, so the generated tumor region is recovered with
# `< mask_threshold`, not `>`.
# --------------------------------------------------------------------------- #
def build_mask_spatial_cond(organ_mask, heatmap, device):
    """
    organ_mask / heatmap: (B, 1, X, Y, Z) tensors (or ndarrays) in the same
    orientation as ct_volume.

    Returns the (B, 2, Z, X, Y) spatial conditioning tensor.
    """
    organ_mask_t = torch.as_tensor(organ_mask, dtype=torch.float32, device=device)
    heatmap_t = torch.as_tensor(heatmap, dtype=torch.float32, device=device)


    organ_mask_p = organ_mask_t.permute(0, 1, -1, -3, -2)
    heatmap_p = heatmap_t.permute(0, 1, -1, -3, -2)



    return torch.cat([organ_mask_p, heatmap_p], dim=1)


def _apply_fill_holes(binary_mask):
    """
    Applies MONAI's FillHoles per-sample to a (B, 1, X, Y, Z) binary mask
    tensor, matching the reference mask inference script's postprocess_tensor
    (KeepLargestConnectedComponent is available upstream but left disabled
    there too, so it's not applied here).
    """
    post = Compose([FillHoles()])
    cleaned = torch.stack(
        [post(binary_mask[i]) for i in range(binary_mask.shape[0])], dim=0
    )
    return cleaned


def generate_tumor_mask(mask_tester, organ, m_organ_mask, organ_mask, heatmap, radiomics, device,
                         ddim_steps=50, cond_scale=1.0, dim_size=32,
                         mask_threshold=0.5, apply_fill_holes=True):
    """
    Samples a tumor-shape mask via DDIM, matching the reference mask
    inference script.

    m_organ_mask / heatmap: (B, 1, X, Y, Z) tensors, pre-permute (the permute
    to the model's (Z, X, Y) convention happens inside this function).

    Returns a (B, 1, X, Y, Z) float tensor with 1 = tumor, 0 = background.
    """

    


    organ_idx = ORGAN_TO_IDX[organ]
    batch_size = m_organ_mask.shape[0]

    cond = build_mask_spatial_cond(m_organ_mask, heatmap, device)
    tabular_cond = build_cond_vector(
        organ_idx, radiomics["mask_radiomics"], MASK_COLUMNS, device, batch_size=batch_size
    )

    

    ddim_sampler = Mask_DDIMSampler(mask_tester.ema_model)
    with torch.no_grad():
        img_out, _ = ddim_sampler.sample(
            ddim_steps, batch_size, (1, dim_size, dim_size, dim_size),
            conditioning=cond, tabular_cond=tabular_cond, cond_scale=cond_scale,
        )

    # invert the (B,C,Z,X,Y) permutation used for conditioning, back to (B,C,X,Y,Z)
    recon = img_out.permute(0, 1, -2, -1, -3)
    recon_01 = (recon + 1.0) / 2.0


    import nibabel as nib
    dl = recon.squeeze().cpu().numpy()
    nib.save(nib.Nifti1Image(dl, np.eye(4)), "recon_test.nii.gz")

    rescaled_recon_01 = F.interpolate(recon_01, size=(128, 128, 128), mode="trilinear", align_corners=False)


    # training convention: background=1, tumor=0 -> tumor is where value < threshold
    binary_mask = (rescaled_recon_01 < mask_threshold).to(torch.uint8)
    if apply_fill_holes:
        binary_mask = _apply_fill_holes(binary_mask)

    output_mask = binary_mask.float().to(device)

    size = get_size(radiomics)

    

    if size not in ("large", "medium"):
        organ_mask_t = torch.as_tensor(organ_mask, device=device, dtype=torch.float32)
        # selects the portion of the tumor mask that lies INSIDE the organ
        output_mask = (output_mask * organ_mask_t) >= 1
        output_mask = output_mask.float()


    return output_mask


# --------------------------------------------------------------------------- #
# Tumor-model spatial conditioning + manual reverse-diffusion sampling
#
# Mirrors the reference tumor inference script exactly: this is an
# inpainting-style model. The tumor region is masked OUT of the real CT, the
# masked CT is encoded through the VQGAN, normalized to [-1, 1], and
# concatenated with a downsampled version of the tumor mask to form the
# spatial conditioning tensor. Sampling is a manual reverse-diffusion loop in
# VQGAN latent space (not GaussianDiffusion.sample()), followed by a manual
# decode.
# --------------------------------------------------------------------------- #
def build_tumor_spatial_cond(image, tumor_mask, vqgan, device):
    """
    image:       (B, 1, X, Y, Z) CT volume, normalized to [-1, 1]
    tumor_mask:  (B, 1, X, Y, Z) binary tumor mask (1 = tumor)

    Returns (spatial_cond, latent_shape). spatial_cond is
    (B, C_lat+1, Z', X', Y'), ready to pass as the `cond` kwarg to
    diffusion.p_sample(...). latent_shape is the shape to use for the initial
    noise tensor.
    """
    image_t = torch.as_tensor(image, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(tumor_mask, dtype=torch.float32, device=device)

    # 1. Zero out the tumor region in the CT (the part the model must synthesize)
    mask_bg = (1 - mask_t).detach()
    masked_img = (image_t * mask_bg).detach()

    # 2. Permute to the VQGAN / diffusion convention: (B,C,X,Y,Z) -> (B,C,Z,X,Y)
    masked_img_p = masked_img.permute(0, 1, 4, 2, 3)
    mask_p = mask_t.permute(0, 1, 4, 2, 3)

    with torch.no_grad():
        emb_min = vqgan.codebook.embeddings.min()
        emb_max = vqgan.codebook.embeddings.max()
        emb_denom = emb_max - emb_min

        # 3. Encode masked CT and normalize to [-1, 1]
        latent = vqgan.encode(masked_img_p, quantize=False, include_embeddings=True)
        latent_n = ((latent - emb_min) / emb_denom) * 2.0 - 1.0

        # 4. Downscale binary mask to latent spatial size, rescale to [-1, 1]
        cc = F.interpolate(mask_p * 2.0 - 1.0, size=latent_n.shape[-3:], mode='nearest')

        # 5. Concatenate — matches training: cond = cat([masked_img, cc], dim=1)
        spatial_cond = torch.cat([latent_n, cc], dim=1)

    return spatial_cond, latent_n.shape


def decode_latent(latent, vqgan):
    """
    Inverse of the VQGAN normalization used in build_tumor_spatial_cond, then
    decode. Returns raw CT values (still in [-1, 1]) in (B, 1, X, Y, Z).
    """
    emb_min = vqgan.codebook.embeddings.min()
    emb_max = vqgan.codebook.embeddings.max()
    emb_denom = emb_max - emb_min

    # Invert: latent_n = ((latent - emb_min) / emb_denom) * 2 - 1
    latent_denorm = ((latent + 1.0) / 2.0) * emb_denom + emb_min

    # Decode: output is (B, C, Z, X, Y) — reverse the permutation back to (B, C, X, Y, Z)
    decoded = vqgan.decode(latent_denorm, quantize=True)
    decoded = decoded.permute(0, 1, 3, 4, 2).contiguous()
    return decoded


def sample_tumor_appearance(tumor_tester, ct_volume, tumor_mask, radiomics, organ, device,
                             cond_scale=1.0):
    """
    Runs the manual reverse-diffusion loop in VQGAN latent space (matching the
    reference tumor inference script) and decodes the result.

    Returns a (B, 1, X, Y, Z) tensor, in [-1, 1].
    """
    organ_idx = ORGAN_TO_IDX[organ]
    diffusion = tumor_tester.ema_model
    vqgan = diffusion.vqgan

    spatial_cond, latent_shape = build_tumor_spatial_cond(ct_volume, tumor_mask, vqgan, device)
    tabular_cond = build_cond_vector(
        organ_idx, radiomics["tumor_radiomics"], TUMOR_COLUMNS, device, batch_size=1
    )

    batch_size = ct_volume.shape[0]
    noisy_latent = torch.randn(latent_shape, device=device)

    with torch.no_grad():
        for i in reversed(range(diffusion.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            noisy_latent = diffusion.p_sample(
                noisy_latent, t,
                cond=spatial_cond,
                tabular_cond=tabular_cond,
                cond_scale=cond_scale,
                clip_denoised=True,
            )

        ct_synth = decode_latent(noisy_latent, vqgan)  # (B, 1, X, Y, Z), in [-1, 1]

    return ct_synth


# --------------------------------------------------------------------------- #
# Model loading
#
# These take the already-loaded pipeline config (see config.yaml) rather than
# composing it themselves via hydra — load it once up front with
# load_config(...) and pass it in.
#
# Both the mask model and the tumor model are single, organ-agnostic models
# (organ identity is passed in via the one-hot tabular conditioning, not via
# separate per-organ checkpoints), so each has exactly one checkpoint file.
#
# Only the tumor diffusion model does latent diffusion — its GaussianDiffusion
# loads the VQGAN internally from cfg.tumor_model.vqgan_ckpt via the
# `vqgan_ckpt` constructor kwarg, so we never load/pass a VQGAN object
# ourselves (it's read off tumor_tester.ema_model.vqgan when needed). The mask
# model works directly in voxel space and has no VQGAN at all.
# --------------------------------------------------------------------------- #
def prepare_mask_model(device, cfg):
    m = cfg.mask_model

    mask_unet = Mask_Unet3D(
        dim=64,
        dim_mults=m.dim_mults,
        # target (1) + img_cond (VQ_dim) + organ (1) + feat (1)
        channels=m.diffusion_num_channels,
        out_dim=1,
        num_continuous_conditioners=len(MASK_COLUMNS),
        num_organs=9,
    ).to(device)

    mask_diffusion = Mask_GaussianDiffusion(
        mask_unet,
        image_size=m.diffusion_img_size,
        num_frames=m.diffusion_depth_size,
        channels=m.diffusion_num_channels,
        timesteps=m.timesteps,
        loss_type=m.loss_type,
    ).to(device)

    mask_tester = Mask_Tester(mask_diffusion)
    mask_ckpt_path = os.path.join(cfg.paths.mask_diffusion_ckpt_dir, m.checkpoint_name)
    mask_tester.load(mask_ckpt_path, map_location=device)
    mask_tester.ema_model.eval()

    print(f"LOADED: Mask Diffusion Model - {mask_ckpt_path}")

    return mask_tester


def prepare_tumor_model(device, cfg):
    t = cfg.tumor_model

    x_channels = t.out_dim
    cond_channels = t.diffusion_num_channels - t.out_dim

    unet = Tumor_Unet3D_CA(
        dim=t.unet_dim,
        dim_mults=t.dim_mults,
        channels=x_channels,
        out_dim=t.out_dim,
        num_continuous_conditioners=len(TUMOR_COLUMNS),
        num_organs=9,
        cond_channels=cond_channels,
        num_res_blocks=2,
        attention_resolutions=(2, 4, 8),
        num_heads=8,
        # dim_head removed -- now computed internally as ch // num_heads
        # at every resolution level, matching source's legacy=True behavior
    ).to(device)

    # vqgan_ckpt is passed straight through; Tumor_GaussianDiffusion loads and
    # owns the VQGAN internally (encode on the way in, decode on the way out)
    # — we never touch it directly here except by reading .vqgan off the
    # loaded model when sampling.
    diffusion = Tumor_GaussianDiffusion(
        unet,
        image_size=t.diffusion_img_size,
        num_frames=t.diffusion_depth_size,
        channels=t.diffusion_num_channels,
        timesteps=t.timesteps,
        loss_type=t.loss_type,
        vqgan_ckpt=t.vqgan_ckpt,  # VQGAN weights come from the checkpoint
    ).to(device)

    tumor_tester = Tumor_Tester(diffusion)
    tumor_ckpt_path = os.path.join(cfg.paths.tumor_diffusion_ckpt_dir, t.checkpoint_name)
    tumor_tester.load(tumor_ckpt_path, map_location=device)
    tumor_tester.ema_model.eval()


    print(f"LOADED: Tumor Diffusion Model - {tumor_ckpt_path}")


    return tumor_tester


# --------------------------------------------------------------------------- #
# Full synthesis + blending
# --------------------------------------------------------------------------- #
def synthesize_tumor(ct_volume, organ_mask, heatmap, m_organ_mask, organ_type, mask_tester, tumor_tester, gmm_bank_path, tumor_norm_stats, mask_norm_stats,
                      cond_scale=1.0, ddim_steps=50, mask_dim_size=32, mask_threshold=0.5,
                      apply_fill_holes=True, heatmap_sigma=DEFAULT_HEATMAP_SIGMA, hu_min=-1000.0, hu_max=500.0):
    """
    Per-sample synthesis call. All the sample-specific inputs are passed as
    arguments here; anything about how the models were built comes from the
    config used in prepare_mask_model / prepare_tumor_model.

    ct_volume:   (1, 1, X, Y, Z) healthy CT volume tensor, normalized to [-1, 1]
    organ_mask:  (1, 1, X, Y, Z) binary mask of the organ (tensor or ndarray)
    organ_type:  one of ORGAN_TO_IDX keys, e.g. 'spleen' | 'stomach' | 'colon'
    mask_tester / tumor_tester: Tester objects returned by prepare_mask_model /
        prepare_tumor_model (each a single organ-agnostic model)
    gmm_bank_path: path to the pickled, unified per-organ radiomics GMM bank
    cond_scale: classifier-free-guidance style conditioning scale forwarded to
        both diffusion models' sampling calls
    ddim_steps: number of DDIM steps for mask sampling
    mask_dim_size: spatial size (cube) of the mask diffusion model's working
        resolution
    mask_threshold: threshold applied to the decoded mask (tumor = value <
        mask_threshold, matching the training-time background=1/tumor=0
        convention)
    apply_fill_holes: whether to clean the thresholded mask with MONAI's
        FillHoles
    heatmap_sigma: width of the gaussian tumor-location heatmap
    """
    device = ct_volume.device

    # 1. Sample a target radiomics profile (shape + appearance) for this organ
    radiomics = sample_radiomics(gmm_bank_path, organ_type)


    normalized_radiomics = apply_radiomics_normalization(radiomics, tumor_norm_stats, mask_norm_stats)

    m_organ_mask_t = torch.as_tensor(m_organ_mask, dtype=torch.float32, device=device)
    if m_organ_mask_t.dim() == ct_volume.dim() - 1:
        m_organ_mask_t = m_organ_mask_t.unsqueeze(0)

    organ_mask_t = torch.as_tensor(organ_mask, dtype=torch.float32, device=device)
    if organ_mask_t.dim() == ct_volume.dim() - 1:
        organ_mask_t = organ_mask_t.unsqueeze(0)
    heatmap_t = heatmap
    if heatmap_t.dim() == ct_volume.dim() - 1:
        heatmap_t = heatmap_t.unsqueeze(0)


    print("GENERATING TUMOR MASK")
    # 3. Generate a tumor shape mask via DDIM (organ_mask + heatmap conditioning only)
    tumor_mask = generate_tumor_mask(
        mask_tester, organ_type, m_organ_mask_t, organ_mask_t, heatmap_t, normalized_radiomics, device,
        ddim_steps=ddim_steps, cond_scale=cond_scale, dim_size=mask_dim_size,
        mask_threshold=mask_threshold, apply_fill_holes=apply_fill_holes,
    )
    print(f"Tumor Mask contains {tumor_mask.sum()} voxels")


    tumor_mask = tumor_mask.float().to(device)
    if tumor_mask.dim() == ct_volume.dim() - 1:
        tumor_mask = tumor_mask.unsqueeze(0)

    
    print("GENERATING TUMOR")
    # 4. Synthesize the tumor appearance (inpainting-style latent diffusion)
    sample = sample_tumor_appearance(
        tumor_tester, ct_volume, tumor_mask, normalized_radiomics, organ_type, device, cond_scale=cond_scale,
    )

    # 5. Blend the synthesized tumor into the healthy volume using the tumor mask.
    # Same gaussian-blur blending rule for every organ (no organ-type branching).
    # tumor_mask is already a clean binary {0,1} mask (post-threshold/FillHoles),
    # so it's used directly here rather than being re-mapped from [-1,1].
    mask_01 = torch.clamp(tumor_mask, min=0.0, max=1.0)
    sigma = np.random.uniform(0, 4)
    mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy() * 1.0, sigma=[0, 0, sigma, sigma, sigma])

    volume_ = torch.clamp((ct_volume + 1.0) / 2.0, min=0.0, max=1.0)
    sample_ = torch.clamp((sample + 1.0) / 2.0, min=0.0, max=1.0)

    mask_01_blur = torch.from_numpy(mask_01_np_blur).to(device=device)
    final_volume_ = (1 - mask_01_blur) * volume_ + mask_01_blur * sample_
    final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0)

    final_volume_hu = final_volume_ * (hu_max - hu_min) + hu_min
    return final_volume_hu, tumor_mask, radiomics