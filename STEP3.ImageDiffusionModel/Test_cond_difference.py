"""
Isolated single-forward-pass diagnostic:
  - Loads the model/checkpoint exactly like reconstruct.py
  - Pulls ONE real batch to get a real spatial_cond (so the input
    distribution matches what the model was trained on)
  - Calls diffusion.denoise_fn(...) DIRECTLY (bypassing p_sample,
    guidance amplification, and VQGAN decode entirely)
  - Varies ONLY tabular_cond across two forward passes, everything
    else held fixed (same x, same t, same spatial_cond)
  - Reports whether the raw noise predictions actually diverge

Run this the same way you'd run reconstruct.py (same hydra config).
"""
from dataset.dataloader import get_loader
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict
import hydra
import os
from ddpm import Unet3D, GaussianDiffusion

import sys
sys.path.append(os.getcwd())


@hydra.main(config_path='config', config_name='inference', version_base=None)
def test_diff(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    device = torch.device(f"cuda:{cfg.model.gpus}")

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder,
            cfg.dataset.name,
            cfg.model.results_folder_postfix,
        )

    # ---- Diffusion model (same construction as reconstruct.py) ----
    print("1. Initializing diffusion model...")
    model = Unet3D(
        dim=cfg.model.unet_dim,
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
        vqgan_ckpt=cfg.model.vqgan_ckpt,
    ).to(device)

    # ---- Checkpoint (EMA weights) ----
    print("2. Loading checkpoint...")
    ckpt_path = os.path.join(cfg.model.results_folder, 'model_best.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = diffusion.load_state_dict(ckpt['ema'], strict=True)
    print("   missing keys:", missing)
    print("   unexpected keys:", unexpected)
    diffusion.eval()
    vqgan = diffusion.vqgan

    denoise_fn = diffusion.denoise_fn  # the Unet3D itself
    denoise_fn.eval()

    # ---- Pull ONE real batch to get a realistic spatial_cond ----
    print("3. Loading one real batch for spatial_cond...")
    val_loader, _, _ = get_loader(cfg.dataset)
    batch = next(iter(val_loader))

    image = batch["image"].to(device)   # (B, 1, X, Y, Z), in [b_min, b_max]
    mask = batch["label"].to(device)    # (B, 1, X, Y, Z) ternary

    mask_bin = (mask == 2).float()
    mask_bg = (1 - mask_bin).detach()
    masked_img = (image * mask_bg).detach()

    masked_img_p = masked_img.permute(0, 1, 4, 2, 3)
    mask_p = mask_bin.permute(0, 1, 4, 2, 3)

    with torch.no_grad():
        emb_min = vqgan.codebook.embeddings.min()
        emb_max = vqgan.codebook.embeddings.max()
        emb_denom = emb_max - emb_min

        latent = vqgan.encode(masked_img_p, quantize=False, include_embeddings=True)
        latent_n = ((latent - emb_min) / emb_denom) * 2.0 - 1.0

        cc = F.interpolate(
            mask_p * 2.0 - 1.0,
            size=latent_n.shape[-3:],
            mode='nearest',
        )
        spatial_cond = torch.cat([latent_n, cc], dim=1)
        latent_shape = latent_n.shape

    # only use the FIRST sample in the batch, so we have a single fixed x/cond
    spatial_cond_1 = spatial_cond[:1]
    latent_shape_1 = (1, *latent_shape[1:])

    # ---- Fixed noisy latent + fixed mid-range timestep ----
    torch.manual_seed(0)
    x = torch.randn(latent_shape_1, device=device)
    t = torch.full((1,), diffusion.num_timesteps // 2, device=device, dtype=torch.long)

    # ---- Two distinct tabular_cond vectors: different organ, different continuous ----
    tab_dim = denoise_fn.tabular_cond_dim  # should be 19 (9 organ + 10 continuous)
    print(f"4. tabular_cond_dim = {tab_dim}, tabular_emb_dim = {denoise_fn.tabular_emb_dim}")

    tab_a = torch.zeros(1, tab_dim, device=device)
    tab_a[0, 0] = 1.0          # organ class 0
    tab_a[0, 9:] = 0.0         # continuous features at 0 (z-scored mean)

    tab_b = torch.zeros(1, tab_dim, device=device)
    tab_b[0, 4] = 1.0          # organ class 4
    tab_b[0, 9:] = 4.0         # continuous features shifted +2 std

    print("tab_a:", tab_a)
    print("tab_b:", tab_b)

    with torch.no_grad():
        out_a = denoise_fn(x, t, cond=spatial_cond_1, tabular_cond=tab_a, null_cond_prob=0.)
        out_b = denoise_fn(x, t, cond=spatial_cond_1, tabular_cond=tab_b, null_cond_prob=0.)

        # sanity: same tabular_cond twice should give identical output (no dropout randomness)
        out_a2 = denoise_fn(x, t, cond=spatial_cond_1, tabular_cond=tab_a, null_cond_prob=0.)

    diff_ab = (out_a - out_b).abs().mean().item()
    diff_aa = (out_a - out_a2).abs().mean().item()  # should be ~exactly 0 (determinism check)

    std_a = out_a.std().item()
    std_b = out_b.std().item()

    print("\n===== RESULTS =====")
    print(f"out_a std: {std_a:.6f}")
    print(f"out_b std: {std_b:.6f}")
    print(f"|out_a - out_b|.mean(): {diff_ab:.6f}   (relative to std: {diff_ab / std_a:.4%})")
    print(f"|out_a - out_a2|.mean(): {diff_aa:.8f}   (determinism sanity check, should be ~0)")

    # ---- Also test: null-cond vs real cond (CFG null embedding effect) ----
    with torch.no_grad():
        out_null = denoise_fn(x, t, cond=spatial_cond_1, tabular_cond=tab_a, null_cond_prob=1.)
    diff_null = (out_a - out_null).abs().mean().item()
    print(f"|out_a - out_null|.mean(): {diff_null:.6f}   (relative to std: {diff_null / std_a:.4%})")

    print("\n===== INTERPRETATION =====")
    if diff_aa > 1e-5:
        print("WARNING: identical inputs produced different outputs — nondeterminism")
        print("(e.g. dropout still active, or model not in eval() mode somewhere).")
    if diff_ab / std_a > 0.05:
        print("Model output MEANINGFULLY diverges between tab_a and tab_b (>5% of output std).")
        print("-> Model-level conditioning IS working. Look downstream: VQGAN decode,")
        print("   guidance scale in the real sampling loop, or masking/compositing.")
    else:
        print("Model output does NOT meaningfully diverge between tab_a and tab_b.")
        print("-> Problem is inside the model despite healthy weight magnitudes.")
        print("   Check GroupNorm sensitivity to FiLM shift, or res_conv skip dominance.")


if __name__ == '__main__':
    test_diff()