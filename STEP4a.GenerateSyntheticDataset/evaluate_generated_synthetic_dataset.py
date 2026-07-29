import hydra
from omegaconf import DictConfig

from pathlib import Path
import nibabel as nib
import numpy as np
import json
import csv
import os
from scipy.stats import pearsonr, spearmanr

from metrics import RadiomicsMetricsEvaluator


def normalize_features(features, norm_stats):
    """
    Z-score normalize radiomics features.
    """
    normalized = {}

    for k, v in features.items():

        if k not in norm_stats:
            normalized[k] = v
            continue

        mean = norm_stats[k]["mean"]
        std = norm_stats[k]["std"]

        if std > 1e-8:
            normalized[k] = (v - mean) / std
        else:
            normalized[k] = 0.0

    return normalized


def compute_errors(pred, target):
    """
    Compute MAE and percent error between normalized radiomics.
    """

    abs_errors = []
    percent_errors = []

    for k, v in pred.items():

        if k not in target:
            continue

        gt = target[k]

        abs_errors.append(abs(v - gt))

        if abs(gt) > 1e-6:
            percent_errors.append(
                abs(v - gt) / abs(gt) * 100
            )

    mae = np.mean(abs_errors) if abs_errors else None
    pct = np.mean(percent_errors) if percent_errors else None

    return mae, pct


def process_feature_group(
    prefix,
    generated_raw,
    gt_features,
    norm_stats,
    row,
):
    """
    Normalize a generated feature dict, compare it against the
    corresponding ground truth dict, and write results into `row`
    using the given prefix (e.g. "tumor" or "mask") to disambiguate
    columns between the two feature groups.

    Returns (mae, pct_error) for logging purposes.
    """

    generated = normalize_features(generated_raw, norm_stats)

    mae, pct_error = compute_errors(generated, gt_features)

    row[f"{prefix}_mean_abs_error"] = mae
    row[f"{prefix}_mean_percent_error"] = pct_error

    # Ground truth radiomics
    for k, v in gt_features.items():
        row[f"{prefix}_gt_{k}"] = v

    # Normalized generated radiomics
    for k, v in generated.items():
        row[f"{prefix}_gen_{k}"] = v

    # Feature-level errors
    for k, v in generated.items():
        if k in gt_features:
            row[f"{prefix}_abs_error_{k}"] = abs(v - gt_features[k])

    return mae, pct_error


@hydra.main(
    config_path="config",
    config_name="synthesis",
    version_base=None
)
def compute_radiomics(cfg: DictConfig):

    synth_dir = Path(
        "/scratch/rpinise1/MultiTumorSynthesis/SyntheticSamplesV1"
    )

    manifest_path = synth_dir / "radiomics_manifest_combined.json"
    out_csv = Path("radiomics_comparison.csv")

    # ----------------------------------------------------
    # Load ground truth manifest
    # ----------------------------------------------------

    with open(manifest_path, "r") as f:
        gt_manifest = json.load(f)

    print(
        f"Loaded {len(gt_manifest)} ground truth patients"
    )


    # ----------------------------------------------------
    # Load and combine normalization statistics
    # ----------------------------------------------------

    with open(cfg.dataset.tumor_norm_stats, "r") as f:
        tumor_norm_stats = json.load(f)

    with open(cfg.dataset.mask_norm_stats, "r") as f:
        mask_norm_stats = json.load(f)


    # Keep separate norm stats per feature group so that a feature
    # name collision between tumor/mask features (if any) doesn't
    # silently clobber the other group's stats.
    tumor_norm_stats_combined = dict(tumor_norm_stats)
    mask_norm_stats_combined = dict(mask_norm_stats)


    print(
        f"Loaded normalization stats: "
        f"{len(tumor_norm_stats_combined)} tumor features, "
        f"{len(mask_norm_stats_combined)} mask features"
    )


    evaluator_cache = {}


    file_exists = out_csv.exists()

    csv_file = open(
        out_csv,
        "a",
        newline=""
    )

    writer = None


    try:

        for patient_dir in sorted(synth_dir.iterdir()):

            if not patient_dir.is_dir():
                continue


            patient_id = patient_dir.name


            if patient_id not in gt_manifest:
                continue


            ct_path = patient_dir / "ct.nii.gz"
            seg_dir = patient_dir / "segmentations"


            if not ct_path.exists() or not seg_dir.exists():
                continue


            try:

                # Load CT once
                ct_img = nib.load(ct_path)

                ct = (
                    ct_img
                    .get_fdata()
                    .astype(np.float32)
                )


                spacing = ct_img.header.get_zooms()[:3]

                spacing_key = tuple(spacing)


                if spacing_key not in evaluator_cache:

                    evaluator_cache[spacing_key] = (
                        RadiomicsMetricsEvaluator(
                            spacing=spacing,
                            bin_width=25,
                        )
                    )


                evaluator = evaluator_cache[spacing_key]


                lesion_files = sorted(
                    seg_dir.glob("*_lesion.nii.gz")
                )


                for lesion_path in lesion_files:


                    # Correct nii.gz parsing
                    organ = (
                        lesion_path.name
                        .replace(".nii.gz", "")
                        .replace("_lesion", "")
                    )


                    try:

                        tumor_mask = (
                            nib.load(lesion_path)
                            .get_fdata()
                            > 0
                        ).astype(np.uint8)


                        expected = gt_manifest[patient_id]


                        if organ != expected["organ"]:
                            print(
                                f"Organ mismatch {patient_id}: "
                                f"generated={organ}, "
                                f"expected={expected['organ']}"
                            )


                        row = {

                            "patient_id": patient_id,

                            "organ_gt": expected["organ"],

                            "organ_generated": organ,
                        }


                        # -----------------------------
                        # Compute all radiomics features once.
                        # The evaluator doesn't distinguish shape vs
                        # intensity features -- it just returns
                        # whatever PyRadiomics extracts. We split the
                        # result into "tumor" and "mask" groups based
                        # on which keys are present in each GT dict.
                        # -----------------------------

                        all_raw = evaluator.compute_radiomics(
                            ct,
                            tumor_mask,
                        )

                        tumor_gt = expected["tumor_radiomics"]
                        mask_gt = expected["mask_radiomics"]

                        tumor_raw = {
                            k: v for k, v in all_raw.items()
                            if k in tumor_gt
                        }

                        mask_raw = {
                            k: v for k, v in all_raw.items()
                            if k in mask_gt
                        }


                        tumor_mae, tumor_pct = process_feature_group(
                            "tumor",
                            tumor_raw,
                            tumor_gt,
                            tumor_norm_stats_combined,
                            row,
                        )

                        mask_mae, mask_pct = process_feature_group(
                            "mask",
                            mask_raw,
                            mask_gt,
                            mask_norm_stats_combined,
                            row,
                        )


                        print(
                            f"{patient_id} {organ}: "
                            f"Tumor MAE={tumor_mae:.4f}, "
                            f"Tumor Mean % Error={tumor_pct:.2f}% | "
                            f"Mask MAE={mask_mae:.4f}, "
                            f"Mask Mean % Error={mask_pct:.2f}%"
                        )


                        # Initialize writer
                        if writer is None:

                            writer = csv.DictWriter(
                                csv_file,
                                fieldnames=list(row.keys())
                            )


                            if not file_exists:

                                writer.writeheader()
                                csv_file.flush()


                        writer.writerow(row)


                        # Save immediately
                        csv_file.flush()

                        os.fsync(
                            csv_file.fileno()
                        )


                        print(
                            f"Processed {patient_id} {organ}"
                        )


                    except Exception as e:

                        print(
                            f"Failed lesion "
                            f"{patient_id} {organ}: {e}"
                        )


            except Exception as e:

                print(
                    f"Failed {patient_id}: {e}"
                )


    finally:

        csv_file.close()


    print(
        f"Saved comparisons to {out_csv}"
    )



if __name__ == "__main__":

    compute_radiomics()