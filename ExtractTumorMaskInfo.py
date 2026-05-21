from monai.transforms.io.array import LoadImage
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob

tumor_sheet = pd.read_csv("../../Data/metadata_per_tumor_ucsf_batch_1_to_6_and_merlin.csv")

COLUMNS = [
    "bdmap_id", "organ",
    "diameter_x_mm", "diameter_y_mm", "diameter_z_mm",
    "mean_x_mm", "mean_y_mm", "mean_z_mm",
    "std_x_mm", "std_y_mm", "std_z_mm",
    "volume_ml",
]

def load_mask(bdmap_id, organ):
    loader = LoadImage(image_only=False, dtype=np.float32)
    mask, meta = loader("/projects/bodymaps/Data/radiologist_annotations_merlin_ucsf_atlas_multi_cancer/" + bdmap_id + "/segmentations/" + organ + "_lesion.nii.gz")
    spacing = meta["pixdim"][1:4]
    return mask, spacing

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
    voxel_vol_ml = sx * sy * sz / 1000
    diameters = compute_diameters(mask, voxel_spacing)
    volume_ml = float(mask.sum() * voxel_vol_ml)
    return [*diameters.values(), volume_ml]

def process_one(segmentation_path):
    bdmap = segmentation_path.split("/")[-3]
    organ = "_".join(segmentation_path.split("/")[-1].split("_")[:-1])
    try:
        mask, voxel_spacing = load_mask(bdmap, organ)
        metrics = compute_mask_metrics(mask, voxel_spacing)
        return [bdmap, organ, *metrics]
    except Exception as e:
        print(f"ERROR {bdmap} {organ}: {e}")
        return None

def generate_mask_diffusion_txt(segmentation_paths, per_tumor_df, num_workers=20, output_csv="mask_metrics.csv"):
    rows = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, p): p for p in segmentation_paths}
        for future in tqdm(as_completed(futures), total=len(segmentation_paths)):
            result = future.result()
            if result is not None:
                rows.append(result)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(["bdmap_id", "organ"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
    return df

bdmaps_with_tumor_mask = glob.glob("/projects/bodymaps/Data/radiologist_annotations_merlin_ucsf_atlas_multi_cancer/*/segmentations/*")
df = generate_mask_diffusion_txt(bdmaps_with_tumor_mask, tumor_sheet, output_csv="mask_metrics.csv")