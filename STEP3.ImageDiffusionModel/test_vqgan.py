"""
VQGAN round-trip sanity test — no diffusion model involved.

Goal: isolate whether noise in final outputs comes from the VQGAN
autoencoder itself (codebook health / encoder-decoder mismatch) or
from the diffusion model's sampling process.

Pulls the first batch from the real dataloader (so preprocessing exactly
matches training), encodes through the VQGAN, and decodes it straight
back — with both quantize=True and quantize=False — so you can compare:

  1. image                -> original CT
  2. recon_quantized       -> encode -> decode(quantize=True)   (snaps to codebook)
  3. recon_continuous      -> encode -> decode(quantize=False)  (no snapping)

If recon_quantized is noisy but recon_continuous is clean:
    -> the codebook / quantization step is introducing the noise
       (e.g. poor codebook utilization, codes landing in bad regions)

If BOTH are noisy:
    -> the noise is baked into the autoencoder itself (encoder/decoder
       weights, not the discrete bottleneck) — go back to STEP1 training,
       not the diffusion code.

If BOTH are clean:
    -> the VQGAN is fine. The noise you're seeing in full pipeline output
       is coming from the diffusion model's sampling, not the autoencoder.

Usage:
    python test_vqgan_roundtrip.py model.vqgan_ckpt=/path/to/vqgan.ckpt
    (any other Hydra overrides work the same way you'd call reconstruct.py)
"""

import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import nibabel as nib
import torch
import hydra
from omegaconf import DictConfig, open_dict
from pathlib import Path

from dataset.dataloader import get_loader
from vq_gan_3d.model.vqgan import VQGAN


def describe(name, t):
    t = t.detach().float()
    print(
        f"  {name:18s} shape={tuple(t.shape)}  "
        f"min={t.min().item():.4f}  max={t.max().item():.4f}  "
        f"mean={t.mean().item():.4f}  std={t.std().item():.4f}"
    )


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def main(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    device = torch.device(f"cuda:{cfg.model.gpus}")

    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder,
            cfg.dataset.name,
            cfg.model.results_folder_postfix,
        )

    # ---- Load VQGAN directly from checkpoint (mirrors GaussianDiffusion's loading) ----
    print("1. Loading VQGAN checkpoint...")
    vqgan = VQGAN.load_from_checkpoint(cfg.model.vqgan_ckpt, weights_only=False).to(device)
    vqgan.eval()

    # ---- Pull first real batch from the actual dataloader ----
    print("2. Loading first batch from dataloader...")
    val_loader, _, _ = get_loader(cfg.dataset)
    batch = next(iter(val_loader))

    image = batch["image"].to(device)   # (B, C, X, Y, Z) — same convention as reconstruct.py

    # ---- Permute to VQGAN convention: (B,C,X,Y,Z) -> (B,C,Z,X,Y) ----
    # Mirrors build_spatial_cond's permute exactly.
    image_p = image.permute(0, 1, 4, 2, 3)

    out_dir = Path("vqgan_roundtrip_test")
    out_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        # ---- Encode once (continuous, pre-quantization) ----
        print("3. Encoding...")
        latent = vqgan.encode(image_p, quantize=False, include_embeddings=True)
        describe("latent (raw)", latent)

        # For reference: where does this sit relative to the codebook itself?
        emb = vqgan.codebook.embeddings
        describe("codebook", emb)

        # ---- Decode WITH quantization (snap to nearest codebook vector) ----
        print("4. Decoding with quantize=True...")
        recon_quantized = vqgan.decode(latent, quantize=True)
        recon_quantized = recon_quantized.permute(0, 1, 3, 4, 2).contiguous()  # back to (B,C,X,Y,Z)

        # ---- Decode WITHOUT quantization (straight continuous decode) ----
        print("5. Decoding with quantize=False...")
        recon_continuous = vqgan.decode(latent, quantize=False)
        recon_continuous = recon_continuous.permute(0, 1, 3, 4, 2).contiguous()

    print("\n--- Stats ---")
    describe("original image", image)
    describe("recon_quantized", recon_quantized)
    describe("recon_continuous", recon_continuous)

    diff_q = (recon_quantized - image).abs()
    diff_c = (recon_continuous - image).abs()
    describe("abs diff (quantized)", diff_q)
    describe("abs diff (continuous)", diff_c)

    # ---- Save NIfTIs for visual inspection ----
    spacing = (1.0, 1.0, 1.0)
    affine = np.diag([*spacing, 1.0])

    batch_size = image.shape[0]
    for b in range(batch_size):
        stem = f"sample{b}"

        nib.save(
            nib.Nifti1Image(image[b, 0].cpu().numpy().astype(np.float32), affine),
            str(out_dir / f"{stem}_original.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(recon_quantized[b, 0].cpu().numpy().astype(np.float32), affine),
            str(out_dir / f"{stem}_recon_quantized.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(recon_continuous[b, 0].cpu().numpy().astype(np.float32), affine),
            str(out_dir / f"{stem}_recon_continuous.nii.gz"),
        )

    print(f"\nSaved {batch_size} sample(s) to {out_dir.resolve()}")
    print("\nCompare *_original vs *_recon_quantized vs *_recon_continuous in your NIfTI viewer.")
    print("  - Both noisy            -> autoencoder/codebook itself is the problem (STEP1)")
    print("  - Only quantized noisy  -> quantization/codebook-snap step is the problem")
    print("  - Both clean            -> VQGAN is fine, noise comes from diffusion sampling")


if __name__ == '__main__':
    main()