import json
import hydra
from dataset.dataloader import get_healthy_loader
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import pandas as pd
import torch
from omegaconf import DictConfig, open_dict
import hydra
import os
from TumorGeneration.tumor_gen_utils import *
from TumorGeneration.diffusion_models.STEP2_ddpm import Tester as MaskTester, GaussianDiffusion as Mask_GaussianDiffusion
from TumorGeneration.diffusion_models.STEP3_ddpm import Tester as TumorTester, GaussianDiffusion as Tumor_GaussianDiffusion

from pathlib import Path
import sys
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())

import nibabel as nib
import numpy as np
import torch



def load_normalization_dicts(tumor_norm_stats_path, mask_norm_stats_path):
    tumor_stats_file = tumor_norm_stats_path
    if os.path.exists(tumor_stats_file):
        with open(tumor_stats_file, "r") as f:
            tumor_normalization_stats = json.load(f)
    else:
        raise RuntimeError("No tumor normalization stats json was provided")



    mask_stats_file = mask_norm_stats_path
    if os.path.exists(mask_stats_file):
        with open(mask_stats_file, "r") as f:
            mask_normalization_stats = json.load(f)
    else:
        raise RuntimeError("No mask normalization stats json was provided")

    return tumor_normalization_stats, mask_normalization_stats


def save_synthesis_outputs(final_volume, tumor_mask, organ_mask, out_volume_path, out_mask_path, out_organ_path,
                            batch_idx=0, affine=None):
    if affine is None:
        affine = np.eye(4)

    vol_np = final_volume[batch_idx, 0].detach().cpu().numpy().astype(np.float32)
    mask_np = tumor_mask[batch_idx, 0].detach().cpu().numpy().astype(np.uint8)
    organ_np = organ_mask[batch_idx, 0].detach().cpu().numpy().astype(np.uint8)

    nib.save(nib.Nifti1Image(vol_np, affine), out_volume_path)
    nib.save(nib.Nifti1Image(mask_np, affine), out_mask_path)
    nib.save(nib.Nifti1Image(organ_np, affine), out_organ_path)


def save_radiomics_manifest(manifest, out_path):
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)


@hydra.main(config_path='config', config_name='synthesis', version_base=None)
def generate_samples(cfg: DictConfig):
    torch.cuda.set_device(cfg.inference.gpu_idxs)
    device = torch.device(f"cuda:{cfg.inference.gpu_idxs}")

    healthy_loader, _, _ = get_healthy_loader(cfg.dataset)
    gmm_bank_path = cfg.paths.radiomics_gmm_bank

    mask_tester = prepare_mask_model(device, cfg)
    tumor_tester = prepare_tumor_model(device, cfg)

    tumor_norm_stats, mask_norm_stats = load_normalization_dicts(cfg.dataset.tumor_norm_stats, cfg.dataset.mask_norm_stats)


    out_dir = Path("/scratch/rpinise1/MultiTumorSynthesis/SyntheticSamplesV1")
    out_dir.mkdir(parents=True, exist_ok=True)


    radiomics_manifest = {}
    radiomics_json_path = out_dir / "radiomics_manifest_3.json"

    for step, batch in enumerate(tqdm(healthy_loader, desc="Batches")):
        ct = batch["image"].to(device)
        organ_mask = batch["organ_mask"].to(device)
        m_organ_mask = batch["m_organ_mask"].to(device)
        heatmap = batch["heatmap"].to(device)
        organ = batch["organ"][0]
        bdmap_id = batch["bdmap_id"][0]

       
        try:
            final_volume, tumor_mask, radiomics = synthesize_tumor(
                ct, organ_mask, heatmap, m_organ_mask, organ, mask_tester, tumor_tester, gmm_bank_path, tumor_norm_stats, mask_norm_stats
            )

            (out_dir / bdmap_id / "segmentations").mkdir(parents=True, exist_ok=True)

            affine = ct.meta["affine"][0] if ct.meta["affine"].ndim == 3 else ct.meta["affine"]
            affine = affine.cpu().numpy() if torch.is_tensor(affine) else np.asarray(affine)

            save_synthesis_outputs(
                final_volume, tumor_mask, organ_mask,
                out_volume_path=str(out_dir / bdmap_id / f"ct.nii.gz"),
                out_mask_path=str(out_dir / bdmap_id / "segmentations" / f"{organ}_lesion.nii.gz"),
                out_organ_path=str(out_dir / bdmap_id / "segmentations" / f"{organ}.nii.gz"),
                affine=affine
            )


            radiomics_manifest[bdmap_id] = {"organ": organ, **radiomics}
            if (step + 1) % 50 == 0:
                save_radiomics_manifest(radiomics_manifest, radiomics_json_path)

        except Exception as e:
            print(f"[FAILED] bdmap_id={bdmap_id} organ={organ}: {e}")
            torch.cuda.empty_cache()
            continue
    
    save_radiomics_manifest(radiomics_manifest, radiomics_json_path)

if __name__ == "__main__":
    generate_samples()