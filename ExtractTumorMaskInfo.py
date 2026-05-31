import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import glob
import os
from scipy.ndimage import label
from radiomics import featureextractor
import logging

logging.getLogger("radiomics").setLevel(logging.ERROR)

BBOX_LIMIT_MM = 128.0

COLUMNS = [
    "bdmap_id", "organ",
    "major_axis_mm", "minor_axis_mm", "least_axis_mm",
    "volume_ml",
    "sphericity", "surface_volume_ratio",
    "elongation", "flatness", "max_3d_diameter_mm",
    "num_components",
    "diameter_x_mm", "diameter_y_mm", "diameter_z_mm",
]

_extractor = None

def worker_init():
    global _extractor
    settings = {'geometryTolerance': 1e-4, 'label': 1}
    _extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    _extractor.disableAllFeatures()
    _extractor.enableFeaturesByName(shape=[
        'Elongation', 'Flatness', 'Maximum3DDiameter',
        'MeshVolume', 'Sphericity', 'SurfaceVolumeRatio',
        'MajorAxisLength', 'MinorAxisLength', 'LeastAxisLength'
    ])


def compute_bbox_mm(bin_mask: np.ndarray, spacing):
    if not bin_mask.any():
        return None
    nz        = np.argwhere(bin_mask)
    extent    = nz.max(axis=0) - nz.min(axis=0) + 1
    bbox_x_mm = float(extent[2] * spacing[0])
    bbox_y_mm = float(extent[1] * spacing[1])
    bbox_z_mm = float(extent[0] * spacing[2])
    return bbox_x_mm, bbox_y_mm, bbox_z_mm


def compute_metrics(mask_sitk, bin_mask, bbox_x_mm, bbox_y_mm, bbox_z_mm):
    zeros = {col: 0.0 for col in COLUMNS if col not in ["bdmap_id", "organ"]}
    zeros["num_components"] = 0
    zeros.update({"diameter_x_mm": bbox_x_mm, "diameter_y_mm": bbox_y_mm, "diameter_z_mm": bbox_z_mm})

    if not bin_mask.any():
        return zeros

    structure = np.ones((3, 3, 3), dtype=bool)
    _, num_components = label(bin_mask, structure=structure)

    bin_mask_sitk = sitk.GetImageFromArray(bin_mask.astype(np.uint8))
    bin_mask_sitk.CopyInformation(mask_sitk)

    try:
        features = _extractor.execute(bin_mask_sitk, bin_mask_sitk)

        elongation           = float(features.get('original_shape_Elongation', 0.0))
        flatness             = float(features.get('original_shape_Flatness', 0.0))
        max_3d_diameter      = float(features.get('original_shape_Maximum3DDiameter', 0.0))
        volume_ml            = float(features.get('original_shape_MeshVolume', 0.0)) / 1000.0
        sphericity           = float(features.get('original_shape_Sphericity', 0.0))
        surface_volume_ratio = float(features.get('original_shape_SurfaceVolumeRatio', 0.0))
        major_axis_mm        = float(features.get('original_shape_MajorAxisLength', 0.0))
        minor_axis_mm        = float(features.get('original_shape_MinorAxisLength', 0.0))
        least_axis_mm        = float(features.get('original_shape_LeastAxisLength', 0.0))

    except Exception as e:
        print(e)
        elongation, flatness, max_3d_diameter       = 0.0, 0.0, 0.0
        volume_ml, sphericity, surface_volume_ratio = 0.0, 0.0, 0.0
        major_axis_mm, minor_axis_mm, least_axis_mm = 0.0, 0.0, 0.0

    return {
        "major_axis_mm":        major_axis_mm,
        "minor_axis_mm":        minor_axis_mm,
        "least_axis_mm":        least_axis_mm,
        "volume_ml":            volume_ml,
        "sphericity":           sphericity,
        "surface_volume_ratio": surface_volume_ratio,
        "elongation":           elongation,
        "flatness":             flatness,
        "max_3d_diameter_mm":   max_3d_diameter,
        "num_components":       int(num_components),
        "diameter_x_mm":            bbox_x_mm,
        "diameter_y_mm":            bbox_y_mm,
        "diameter_z_mm":            bbox_z_mm,
    }


def process_one(segmentation_path):
    parts = segmentation_path.split("/")
    bdmap = parts[-3]
    organ = "_".join(parts[-1].split("_")[:-1])

    try:
        mask_sitk = sitk.ReadImage(segmentation_path)
        bin_mask  = sitk.GetArrayFromImage(mask_sitk) > 0

        bbox = compute_bbox_mm(bin_mask, mask_sitk.GetSpacing())
        if bbox is None:
            return None

        bbox_x_mm, bbox_y_mm, bbox_z_mm = bbox
        if max(bbox_x_mm, bbox_y_mm, bbox_z_mm) > BBOX_LIMIT_MM:
            print(f"SKIPPED  bbox=({bbox_x_mm:.1f}, {bbox_y_mm:.1f}, {bbox_z_mm:.1f}) mm  →  {segmentation_path}")
            return None

        metrics = compute_metrics(mask_sitk, bin_mask, bbox_x_mm, bbox_y_mm, bbox_z_mm)
        return [bdmap, organ] + [metrics[col] for col in COLUMNS[2:]]

    except Exception as e:
        print(f"ERROR processing {segmentation_path}: {e}")
        return None


def generate_mask_diffusion_txt(segmentation_paths, num_workers=16, output_csv="mask_metrics.csv"):
    # ── Resume: load existing CSV and skip already-processed bdmap_ids ────────
    existing_rows = []
    already_done  = set()
    if os.path.exists(output_csv):
        df_existing   = pd.read_csv(output_csv)
        existing_rows = df_existing.values.tolist()
        already_done  = set(df_existing["bdmap_id"].astype(str).unique())
        print(f"Resuming — found {len(df_existing)} rows across "
              f"{len(already_done)} bdmap_ids already in {output_csv}")

    def bdmap_from_path(p):
        return p.split("/")[-3]

    segmentation_paths = [
        p for p in segmentation_paths
        if bdmap_from_path(p) not in already_done
    ]
    print(f"{len(segmentation_paths)} paths remaining to process")
    # ─────────────────────────────────────────────────────────────────────────

    # Sort largest files first so heavy work runs early, not at the tail
    segmentation_paths = sorted(segmentation_paths, key=os.path.getsize, reverse=True)

    rows = list(existing_rows)  # seed with already-completed results

    with Pool(
        processes=num_workers,
        initializer=worker_init,
        maxtasksperchild=200,
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(process_one, segmentation_paths, chunksize=1),
            total=len(segmentation_paths),
        ):
            if result is not None:
                rows.append(result)

            if len(rows) % 50 == 0 and len(rows) > 0:
                pd.DataFrame(rows, columns=COLUMNS).to_csv(output_csv, index=False)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(["bdmap_id", "organ"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
    return df


if __name__ == '__main__':
    search_path = (
        "/projects/bodymaps/Data/"
        "radiologist_annotations_merlin_ucsf_atlas_multi_cancer/"
        "*/segmentations/*"
    )
    bdmaps_with_tumor_mask = glob.glob(search_path)

    df = generate_mask_diffusion_txt(
        bdmaps_with_tumor_mask, num_workers=44, output_csv="mask_metrics_v7.csv"
    )