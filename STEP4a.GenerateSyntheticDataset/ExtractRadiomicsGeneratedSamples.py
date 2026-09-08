import argparse
import glob
import logging
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label
from tqdm import tqdm
from radiomics import featureextractor


"""

python ExtractRadiomicsGeneratedSamples.py \
    --data-root "/scratch/rpinise1/MultiTumorSynthesis/SyntheticSamplesV3_301" \
    --output-csv radiomics_metrics_V3_301.csv \
    --num-workers 36
"""

logging.getLogger("radiomics").setLevel(logging.ERROR)

_extractor = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute PyRadiomics features and tumor metrics from segmentation masks. "
            "Only the largest connected component of each tumor mask is used."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help=(
            "Root folder containing BDMAP_ID directories. Each BDMAP_ID directory "
            "must contain ct.nii.gz and segmentations/<organ>_lesion.nii.gz."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="radiomics_metrics.csv",
        help="Output CSV path (default: radiomics_metrics.csv)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of multiprocessing workers (default: 1)",
    )
    parser.add_argument(
        "--bbox-limit-mm",
        type=float,
        default=10000.0,
        help="Skip masks whose largest-component bbox exceeds this size in any axis (default: 10000 mm)",
    )
    parser.add_argument(
        "--clip-intensities",
        action="store_true",
        help="Clip CT HU values before radiomics extraction.",
    )
    parser.add_argument(
        "--hu-clip-min",
        type=float,
        default=-1000.0,
        help="Lower HU clipping bound (default: -1000)",
    )
    parser.add_argument(
        "--hu-clip-max",
        type=float,
        default=500.0,
        help="Upper HU clipping bound (default: 500)",
    )
    return parser.parse_args()


# Parsed once in the main process. Worker processes inherit these values under fork.
ARGS = None


def worker_init(args):
    global _extractor, ARGS
    ARGS = args

    settings = {
        "geometryTolerance": 1e-4,
        "label": 1,
        "binWidth": 25,
    }

    _extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    _extractor.enableAllFeatures()


def compute_bbox_mm(bin_mask: np.ndarray, spacing):
    if not bin_mask.any():
        return None

    nz = np.argwhere(bin_mask)
    extent = nz.max(axis=0) - nz.min(axis=0) + 1

    bbox_x_mm = float(extent[2] * spacing[0])
    bbox_y_mm = float(extent[1] * spacing[1])
    bbox_z_mm = float(extent[0] * spacing[2])

    return bbox_x_mm, bbox_y_mm, bbox_z_mm


def parse_organ_name(organ_name):
    if organ_name == "gallbladder":
        return "gall_bladder"
    return organ_name


def tumor_metrics(
    ct_sitk,
    tumor_mask_sitk,
    comp_mask,
    bbox_x_mm,
    bbox_y_mm,
    bbox_z_mm,
    mean_organ,
    std_organ,
):
    """Compute radiomics and manually-derived tumor metrics for one mask."""

    comp_mask_sitk = sitk.GetImageFromArray(comp_mask.astype(np.uint8))
    comp_mask_sitk.CopyInformation(tumor_mask_sitk)

    result = {
        "diameter_x_mm": bbox_x_mm,
        "diameter_y_mm": bbox_y_mm,
        "diameter_z_mm": bbox_z_mm,
    }

    try:
        features = _extractor.execute(ct_sitk, comp_mask_sitk)

        for key, value in features.items():
            if key.startswith("diagnostics_"):
                continue

            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                result[key] = value

        mean_tumor = float(features.get("original_firstorder_Mean", 0.0))
        result["attenuation_delta"] = (mean_tumor - mean_organ) / std_organ

    except Exception as e:
        print(f"Extraction failed: {e}")
        result["attenuation_delta"] = 0.0

    return result


def process_one(segmentation_path):
    """
    Compute one row per tumor segmentation.

    If the segmentation contains multiple connected components, only the
    largest component by voxel count is used. No labeled/component mask is
    written to disk.
    """
    # Expected layout:
    #   data-root/
    #       BDMAP_ID/
    #           ct.nii.gz
    #           segmentations/
    #               <organ>_lesion.nii.gz
    parts = os.path.normpath(segmentation_path).split(os.sep)
    bdmap = parts[-3]
    filename = parts[-1]

    if not filename.endswith("_lesion.nii.gz"):
        return ("FAIL", bdmap, "", f"Unexpected segmentation filename: {filename}")

    organ = filename[:-len("_lesion.nii.gz")]
    organ_mask_name = parse_organ_name(organ)

    bdmap_dir = os.path.dirname(os.path.dirname(segmentation_path))
    ct_path = os.path.join(bdmap_dir, "ct.nii.gz")
    organ_mask_path = os.path.join(
        bdmap_dir,
        "segmentations",
        f"{organ_mask_name}.nii.gz",
    )

    try:
        tumor_mask_sitk = sitk.ReadImage(segmentation_path)
        bin_tumor_mask = sitk.GetArrayFromImage(tumor_mask_sitk) > 0
        spacing = tumor_mask_sitk.GetSpacing()

        if not bin_tumor_mask.any():
            return None

        if not os.path.exists(ct_path):
            return ("FAIL", bdmap, organ, f"CT not found: {ct_path}")

        if not os.path.exists(organ_mask_path):
            return ("FAIL", bdmap, organ, f"Organ mask not found: {organ_mask_path}")

        ct_sitk = sitk.ReadImage(ct_path)
        organ_mask_sitk = sitk.ReadImage(organ_mask_path)

        if ARGS.clip_intensities:
            ct_sitk = sitk.Clamp(
                ct_sitk,
                lowerBound=ARGS.hu_clip_min,
                upperBound=ARGS.hu_clip_max,
            )

        ct_img = sitk.GetArrayFromImage(ct_sitk)
        bin_organ_mask = sitk.GetArrayFromImage(organ_mask_sitk) > 0

        healthy_organ_voxels = ct_img[
            bin_organ_mask & (~bin_tumor_mask)
        ]

        if len(healthy_organ_voxels) > 0:
            mean_organ = float(np.mean(healthy_organ_voxels))
            std_organ = float(np.std(healthy_organ_voxels))
            std_organ = std_organ if std_organ != 0 else 1e-5
        else:
            mean_organ = 0.0
            std_organ = 1e-5

        # Find connected components, but DO NOT save the labeled mask.
        # Only the largest component by voxel count is retained.
        structure = np.ones((3, 3, 3), dtype=bool)
        labeled_mask, num_components = label(
            bin_tumor_mask,
            structure=structure,
        )

        if num_components == 0:
            return None

        component_sizes = np.bincount(labeled_mask.ravel())
        largest_component_id = int(np.argmax(component_sizes[1:]) + 1)
        largest_component_size = int(component_sizes[largest_component_id])

        comp_mask = labeled_mask == largest_component_id

        bbox = compute_bbox_mm(comp_mask, spacing)
        if bbox is None:
            return None

        bbox_x_mm, bbox_y_mm, bbox_z_mm = bbox

        if max(bbox_x_mm, bbox_y_mm, bbox_z_mm) > ARGS.bbox_limit_mm:
            print(
                f"SKIPPED bbox=({bbox_x_mm:.1f}, {bbox_y_mm:.1f}, "
                f"{bbox_z_mm:.1f}) mm -> {segmentation_path}"
            )
            return None

        metrics = tumor_metrics(
            ct_sitk,
            tumor_mask_sitk,
            comp_mask,
            bbox_x_mm,
            bbox_y_mm,
            bbox_z_mm,
            mean_organ,
            std_organ,
        )

        row = {
            "bdmap_id": bdmap,
            "organ": organ,
            "num_components": num_components,
            "largest_component_voxels": largest_component_size,
        }
        row.update(metrics)

        return row

    except Exception as e:
        print(f"ERROR processing {segmentation_path}: {bdmap} ({e})")
        return ("FAIL", bdmap, organ, str(e))


def generate_tumor_metrics(segmentation_paths, args):
    output_csv = args.output_csv
    fail_csv = output_csv.replace(".csv", "_failures.csv")

    existing_rows = []
    already_done = set()

    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        existing_rows = df_existing.to_dict("records")

        if {"bdmap_id", "organ"}.issubset(df_existing.columns):
            already_done = set(
                zip(
                    df_existing["bdmap_id"].astype(str),
                    df_existing["organ"].astype(str),
                )
            )

        print(
            f"Resuming — found {len(df_existing)} rows across "
            f"{len(already_done)} (bdmap_id, organ) pairs in {output_csv}"
        )

    existing_failures = []
    already_failed = set()

    if os.path.exists(fail_csv):
        df_failed = pd.read_csv(fail_csv)
        existing_failures = df_failed.to_dict("records")

        if {"bdmap_id", "organ"}.issubset(df_failed.columns):
            already_failed = set(
                zip(
                    df_failed["bdmap_id"].astype(str),
                    df_failed["organ"].astype(str),
                )
            )

        print(
            f"Found {len(already_failed)} previously-failed "
            f"(bdmap_id, organ) pairs in {fail_csv} — these will be skipped"
        )

    def bdmap_organ_from_path(path):
        path = os.path.normpath(path)
        bdmap = os.path.basename(os.path.dirname(os.path.dirname(path)))
        filename = os.path.basename(path)
        organ = filename[:-len("_lesion.nii.gz")]
        return bdmap, organ

    skip_set = already_done | already_failed

    segmentation_paths = [
        p for p in segmentation_paths
        if bdmap_organ_from_path(p) not in skip_set
    ]

    print(f"{len(segmentation_paths)} paths remaining to process")

    segmentation_paths = sorted(
        segmentation_paths,
        key=os.path.getsize,
        reverse=True,
    )

    rows = list(existing_rows)
    failures = list(existing_failures)

    with Pool(
        processes=args.num_workers,
        initializer=worker_init,
        initargs=(args,),
        maxtasksperchild=200,
    ) as pool:

        for result in tqdm(
            pool.imap_unordered(process_one, segmentation_paths, chunksize=1),
            total=len(segmentation_paths),
        ):
            if result is None:
                continue

            if isinstance(result, tuple) and result[0] == "FAIL":
                _, bdmap, organ, reason = result
                failures.append(
                    {
                        "bdmap_id": bdmap,
                        "organ": organ,
                        "reason": reason,
                    }
                )
            else:
                rows.append(result)

            # Save periodically so the job can resume after interruption.
            if rows:
                pd.DataFrame(rows).to_csv(output_csv, index=False)

            if failures:
                pd.DataFrame(failures).to_csv(fail_csv, index=False)

    if rows:
        df = pd.DataFrame(rows)

        base_columns = [
            "bdmap_id",
            "organ",
            "num_components",
            "largest_component_voxels",
            "diameter_x_mm",
            "diameter_y_mm",
            "diameter_z_mm",
            "attenuation_delta",
        ]

        feature_cols = [
            c for c in df.columns
            if c not in base_columns
        ]

        df = df[
            [c for c in base_columns if c in df.columns]
            + sorted(feature_cols)
        ]

        df = df.sort_values(
            ["bdmap_id", "organ"]
        ).reset_index(drop=True)

        df.to_csv(output_csv, index=False)

        print(f"Saved {len(df)} rows to {output_csv}")
    else:
        df = pd.DataFrame()
        print("No successful rows were produced.")

    if failures:
        df_fail = (
            pd.DataFrame(failures)
            .sort_values(["bdmap_id", "organ"])
            .reset_index(drop=True)
        )
        df_fail.to_csv(fail_csv, index=False)
        print(f"Saved {len(df_fail)} failures to {fail_csv}")

    return df


def main():
    global ARGS
    ARGS = parse_args()

    data_root = os.path.abspath(ARGS.data_root)

    # Discover exactly:
    #   <data-root>/<BDMAP_ID>/segmentations/<organ>_lesion.nii.gz
    segmentation_paths = glob.glob(
        os.path.join(data_root, "*", "segmentations", "*_lesion.nii.gz")
    )

    print(f"Data root: {data_root}")
    print(f"Found {len(segmentation_paths)} tumor segmentation paths")
    print(f"Using {ARGS.num_workers} workers")
    print("Using ONLY the largest connected component for each mask")
    print("No connected-component masks will be written to disk")

    generate_tumor_metrics(segmentation_paths, ARGS)


if __name__ == "__main__":
    main()