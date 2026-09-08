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
import threading
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


# Guards concurrent access to `radiomics_manifest` and to the per-bdmap_id
# sample-index bookkeeping below. Only matters if you parallelize batches
# across processes/threads; harmless overhead otherwise.
_manifest_lock = threading.Lock()

# Tracks how many samples have already been written for a given bdmap_id
# *within this run*, so repeated bdmap_ids (across batches, or duplicated
# within a batch) get distinct, non-overwriting sample numbers. Seeded from
# disk on first use so re-running the script on a partially-populated
# out_dir also doesn't overwrite previous runs' outputs.
_sample_counters = {}
_sample_counters_lock = threading.Lock()


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


def resolve_radiomics_source(cfg):
    """
    Decides which radiomics source to use for this run, based on
    cfg.paths.radiomics_gmm_bank and cfg.paths.radiomics_real_csv.

    - If both are set: prefer the real CSV (gmm_bank_path is still returned,
      since synthesize_organ_radiomics uses it to validate the organ and
      read feature_names even in real-csv mode, but the GMM itself won't be
      sampled from).
    - If only one is set: use that one.
    - If neither is set: raise, since there's no valid radiomics source to
      draw from.

    Returns (gmm_bank_path, radiomics_real_csv) where radiomics_real_csv is
    None when the GMM path is the one being used.
    """
    gmm_bank_path = cfg.paths.get("radiomics_gmm_bank", None)
    radiomics_real_csv = cfg.paths.get("radiomics_real_csv", None)

    if not gmm_bank_path and not radiomics_real_csv:
        raise RuntimeError(
            "No radiomics source configured: set either "
            "cfg.paths.radiomics_gmm_bank or cfg.paths.radiomics_real_csv "
            "(or both, in which case radiomics_real_csv takes priority)."
        )

    if radiomics_real_csv:
        if gmm_bank_path:
            print(
                f"Both radiomics_gmm_bank ({gmm_bank_path}) and "
                f"radiomics_real_csv ({radiomics_real_csv}) are configured; "
                f"defaulting to radiomics_real_csv."
            )
        else:
            print(f"Using radiomics_real_csv: {radiomics_real_csv}")
        return gmm_bank_path, radiomics_real_csv

    print(f"Using radiomics_gmm_bank: {gmm_bank_path}")
    return gmm_bank_path, None


def _next_sample_num(out_dir, bdmap_id):
    """
    Returns the next free sample index for `bdmap_id` under `out_dir`,
    scanning disk the first time a given bdmap_id is seen (so this is safe
    across repeated runs, not just within one process's lifetime), then
    incrementing an in-memory counter for subsequent calls within this run.

    Directory layout produced: out_dir / f"{bdmap_id}_{sample_num}" / ...
    """
    with _sample_counters_lock:
        if bdmap_id not in _sample_counters:
            existing = 0
            if out_dir.exists():
                prefix = f"{bdmap_id}_"
                for entry in out_dir.iterdir():
                    if entry.is_dir() and entry.name.startswith(prefix):
                        suffix = entry.name[len(prefix):]
                        if suffix.isdigit():
                            existing = max(existing, int(suffix) + 1)
            _sample_counters[bdmap_id] = existing

        sample_num = _sample_counters[bdmap_id]
        _sample_counters[bdmap_id] += 1

    return sample_num


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


def _write_sample(out_dir, bdmap_id, organ, final_volume, tumor_mask, organ_mask, affine,
                   radiomics_manifest, radiomics_json_path, manifest_flush_every, global_step):
    """
    Writes one generated sample to disk under a collision-safe
    `{bdmap_id}_{sample_num}` directory, and records it in the shared
    radiomics manifest under that same key. `final_volume`/`tumor_mask` here
    are the (1, 1, X, Y, Z) single-sample tensors returned per-item from
    synthesize_tumor / synthesize_tumor_batch.
    """
    sample_num = _next_sample_num(out_dir, bdmap_id)
    sample_key = f"{bdmap_id}_{sample_num}"

    sample_dir = out_dir / sample_key
    (sample_dir / "segmentations").mkdir(parents=True, exist_ok=True)

    save_synthesis_outputs(
        final_volume, tumor_mask, organ_mask,
        out_volume_path=str(sample_dir / "ct.nii.gz"),
        out_mask_path=str(sample_dir / "segmentations" / f"{organ}_lesion.nii.gz"),
        out_organ_path=str(sample_dir / "segmentations" / f"{organ}.nii.gz"),
        affine=affine,
    )

    return sample_key


@hydra.main(config_path='config', config_name='synthesis', version_base=None)
def generate_samples(cfg: DictConfig):
    torch.cuda.set_device(cfg.inference.gpu_idxs)
    device = torch.device(f"cuda:{cfg.inference.gpu_idxs}")

    healthy_loader, _, _ = get_healthy_loader(cfg.dataset)

    # Decide once, up front, whether this run draws radiomics from the GMM
    # bank or from a real-data CSV. radiomics_real_csv wins if both are
    # configured; an error is raised if neither is.
    gmm_bank_path, radiomics_real_csv = resolve_radiomics_source(cfg)

    mask_tester = prepare_mask_model(device, cfg)
    tumor_tester = prepare_tumor_model(device, cfg)

    tumor_norm_stats, mask_norm_stats = load_normalization_dicts(cfg.dataset.tumor_norm_stats, cfg.dataset.mask_norm_stats)


    out_dir = Path(cfg.inference.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)


    radiomics_manifest = {}
    radiomics_json_path = out_dir / "radiomics_manifest_3.json"
    manifest_flush_every = 50

    global_step = 0
    for step, batch in enumerate(tqdm(healthy_loader, desc="Batches")):
        ct = batch["image"].to(device)
        organ_mask = batch["organ_mask"].to(device)
        m_organ_mask = batch["m_organ_mask"].to(device)
        heatmap = batch["heatmap"].to(device)

        # Per-sample metadata — organ and bdmap_id can differ freely across
        # items in the batch (e.g. ["duodenum", "prostate", "colon", ...]).
        organs = list(batch["organ"])
        bdmap_ids = list(batch["bdmap_id"])

        batch_size = ct.shape[0]

        # Per-sample affines. `ct.meta["affine"]` may be a single (4,4)
        # matrix shared by the batch or a stacked (B,4,4) tensor depending on
        # the dataloader/collate behavior, so handle both.
        raw_affine = ct.meta["affine"]
        if torch.is_tensor(raw_affine):
            raw_affine = raw_affine.cpu().numpy()
        else:
            raw_affine = np.asarray(raw_affine)
        if raw_affine.ndim == 3:
            affines = [raw_affine[i] for i in range(batch_size)]
        else:
            affines = [raw_affine for _ in range(batch_size)]

        try:
            # Runs the mask DDIM sampler and the tumor latent-diffusion
            # reverse process ONCE for the whole batch (not once per
            # sample), with each sample's own organ + own sampled radiomics
            # correctly reflected in its row of the conditioning tensors.
            # Radiomics come from either the GMM bank or the real CSV,
            # per resolve_radiomics_source's decision above.
            batch_results = synthesize_tumor_batch(
                ct, organ_mask, heatmap, m_organ_mask, organs,
                mask_tester, tumor_tester, gmm_bank_path, tumor_norm_stats, mask_norm_stats,
                cond_scale=cfg.inference.get("cond_scale", 1.0),
                ddim_steps=cfg.inference.get("ddim_steps", 50),
                mask_dim_size=cfg.inference.get("mask_dim_size", 32),
                mask_threshold=cfg.inference.get("mask_threshold", 0.5),
                apply_fill_holes=cfg.inference.get("apply_fill_holes", True),
                radiomics_real_csv=radiomics_real_csv,
                
            )

            for i, (final_volume, tumor_mask, radiomics) in enumerate(batch_results):
                bdmap_id = bdmap_ids[i]
                organ = organs[i]
                affine = affines[i]

                try:
                    sample_key = _write_sample(
                        out_dir, bdmap_id, organ,
                        final_volume, tumor_mask, organ_mask[i:i + 1],
                        affine, radiomics_manifest, radiomics_json_path,
                        manifest_flush_every, global_step,
                    )

                    with _manifest_lock:
                        radiomics_manifest[sample_key] = {
                            "bdmap_id": bdmap_id,
                            "organ": organ,
                            **radiomics,
                        }
                        global_step += 1
                        if global_step % manifest_flush_every == 0:
                            save_radiomics_manifest(radiomics_manifest, radiomics_json_path)

                except Exception as e:
                    print(f"[FAILED WRITE] bdmap_id={bdmap_id} organ={organ}: {e}")
                    continue

        except Exception as e:
            print(f"[FAILED] batch={step} bdmap_ids={bdmap_ids} organs={organs}: {e}")
            torch.cuda.empty_cache()
            continue

    save_radiomics_manifest(radiomics_manifest, radiomics_json_path)

if __name__ == "__main__":
    generate_samples()