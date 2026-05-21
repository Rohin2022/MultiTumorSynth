import nibabel as nib
import numpy as np
import torch
import pprint
import torch.nn.functional as F
from pathlib import Path
from monai.transforms import FillHoles, KeepLargestConnectedComponent, Compose


def load_nifti_and_compute_metrics(cleaned_nifti_path):
    """Wrapper function to handle NIfTI file I/O safely."""
    img = nib.load(cleaned_nifti_path)
    raw_data = img.get_fdata() # Typically (X, Y, Z)
    return compute_mask_metrics(raw_data, img.header.get_zooms())

def compute_diameters(mask: np.ndarray, voxel_spacing=(1.0, 1.0, 1.0)):
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    
    zeros = {"diameter_x_mm": 0, "diameter_y_mm": 0, "diameter_z_mm": 0,
             "mean_x_mm": 0, "mean_y_mm": 0, "mean_z_mm": 0,
             "std_x_mm": 0, "std_y_mm": 0, "std_z_mm": 0,
    }
    if not mask.any():
        return zeros

    def span(axis):
        projected = np.any(mask, axis=axis)
        nonzero_idx = np.where(projected)[0]
        return nonzero_idx[-1] - nonzero_idx[0] + 1

    dx = span(axis=(1, 2))
    dy = span(axis=(0, 2))
    dz = span(axis=(0, 1))
    diameters_mm = np.array([dx, dy, dz]) * np.array(voxel_spacing)

    coords = np.argwhere(mask > 0)
    coords_mm = coords * np.array(voxel_spacing)
    means_mm = coords_mm.mean(axis=0)
    stds_mm  = coords_mm.std(axis=0)

    return {
        "diameter_x_mm": float(diameters_mm[0]),
        "diameter_y_mm": float(diameters_mm[1]),
        "diameter_z_mm": float(diameters_mm[2]),
        "mean_x_mm":     float(means_mm[0]),
        "mean_y_mm":     float(means_mm[1]),
        "mean_z_mm":     float(means_mm[2]),
        "std_x_mm":      float(stds_mm[0]),
        "std_y_mm":      float(stds_mm[1]),
        "std_z_mm":      float(stds_mm[2])
    }

def compute_mask_metrics(mask, voxel_spacing):
    sx, sy, sz = voxel_spacing
    print(voxel_spacing)
    print(mask.sum())
    voxel_vol_ml = sx * sy * sz / 1000
    diameters = compute_diameters(mask, voxel_spacing)
    volume_ml = float(mask.sum() * voxel_vol_ml)
    diameters["volume_ml"] = volume_ml
    return diameters




if __name__ == "__main__":
    cleaned_nifti_path = "checkpoints/liver_tumor_train/mask_diffusion_train_pro/debug_masks/step_3500_sample_0_CLEANED.nii.gz"
    
    if Path(cleaned_nifti_path).exists():
        metrics = load_nifti_and_compute_metrics(
            cleaned_nifti_path=cleaned_nifti_path,
        )
        pprint.pprint(metrics)
    else:
        print(f"File not found: {cleaned_nifti_path}")