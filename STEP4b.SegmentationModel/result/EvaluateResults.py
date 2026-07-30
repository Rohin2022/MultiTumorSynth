"""
python EvaluateResults.py \
    --pred_root /projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/result/baseline_model_turkish/abdomenatlas/PRETRAIN_UCSF_133K_and_Merlin_w0_many_cancers_100_epch \
    --gt_root /projects/bodymaps/Data/radiologist_annotations_merlin_ucsf_atlas_multi_cancer \
    --label_names /projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/preprocessing/label_names.yaml \
    --gt_subdir segmentations \
    --pred_subdir predictions \
    --output dice_results.csv \
    --workers 8


python EvaluateResults.py \
    --pred_root /projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/result/finetuned_model_turkish/abdomenatlas/mask_only_model_name \
    --gt_root /projects/bodymaps/Data/radiologist_annotations_merlin_ucsf_atlas_multi_cancer \
    --label_names /projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/preprocessing/label_names.yaml \
    --gt_subdir segmentations \
    --pred_subdir predictions \
    --output dice_results.csv \
    --workers 8

"""

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_label_names(path):
    """Loads a flat list of names from a yaml file and returns only the
    lesion entries (those already ending in '_lesion'), used as-is to build
    filenames. Plain organ entries (e.g. 'bladder') are dropped here since
    we only score lesions."""
    with open(path, "r") as f:
        names = yaml.load(f, Loader=yaml.SafeLoader)
    if not isinstance(names, list):
        raise ValueError(
            f"Expected label_names.yaml to contain a flat list of names, got {type(names)}"
        )
    lesion_names = sorted(n for n in names if n.endswith("_lesion"))
    dropped = sorted(n for n in names if not n.endswith("_lesion"))
    if dropped:
        logger.info(f"Ignoring {len(dropped)} non-lesion entries from label_names.yaml: {dropped}")
    return lesion_names


def dice_score(pred_bin, gt_bin):
    """Binary Dice = 2*|A∩B| / (|A|+|B|). Returns None if both masks are empty
    (nothing to score -- avoids reporting a degenerate 0/0 as 0)."""
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()
    if pred_sum == 0 and gt_sum == 0:
        return None
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    return (2.0 * intersection) / (pred_sum + gt_sum)


def load_mask(path):
    """Loads a NIfTI file and returns a boolean array (nonzero = foreground)."""
    img = nib.load(path)
    data = img.get_fdata(caching="unchanged")
    return data > 0.5, img.shape


def find_bdmap_dirs(pred_root):
    return sorted(
        d for d in os.listdir(pred_root)
        if os.path.isdir(os.path.join(pred_root, d)) and d.startswith("BDMAP")
    )


def eval_one_case(args_tuple):
    """Computes per-organ Dice for a single BDMAP_ID. Returns a dict of
    organ -> dice (or None if a file was missing / unscoreable), plus the
    BDMAP_ID itself and a list of warning strings."""
    bdmap_id, pred_root, gt_root, pred_subdir, gt_subdir, organs = args_tuple

    pred_dir = os.path.join(pred_root, bdmap_id, pred_subdir)
    gt_dir = os.path.join(gt_root, bdmap_id, gt_subdir)

    result = {"BDMAP_ID": bdmap_id}
    warnings_list = []

    for organ in organs:
        fname = f"{organ}.nii.gz"  # organ name already ends in '_lesion'
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        if not os.path.exists(pred_path):
            result[organ] = None
            warnings_list.append(f"[{bdmap_id}] missing prediction: {pred_path}")
            continue
        if not os.path.exists(gt_path):
            result[organ] = None
            warnings_list.append(f"[{bdmap_id}] missing ground truth: {gt_path}")
            continue

        try:
            pred_bin, pred_shape = load_mask(pred_path)
            gt_bin, gt_shape = load_mask(gt_path)
        except Exception as e:
            result[organ] = None
            warnings_list.append(f"[{bdmap_id}] failed to load {organ}: {e}")
            continue

        if pred_shape != gt_shape:
            result[organ] = None
            warnings_list.append(
                f"[{bdmap_id}] shape mismatch for {organ}: pred={pred_shape} gt={gt_shape}"
            )
            continue

        result[organ] = dice_score(pred_bin, gt_bin)

    return result, warnings_list


def main():
    parser = argparse.ArgumentParser(description="Evaluate lesion Dice scores against ground truth.")
    parser.add_argument("--pred_root", type=str, required=True,
                         help="Root folder containing BDMAP_ID subfolders with predictions.")
    parser.add_argument("--gt_root", type=str, required=True,
                         help="Root folder containing BDMAP_ID subfolders with ground truth.")
    parser.add_argument("--label_names", type=str, required=True,
                         help="Path to label_names.yaml (flat list of organ names).")
    parser.add_argument("--pred_subdir", type=str, default="predictions",
                         help="Subfolder inside each BDMAP_ID dir holding prediction masks (default: predictions).")
    parser.add_argument("--gt_subdir", type=str, default="segmentations",
                         help="Subfolder inside each BDMAP_ID dir holding ground truth masks (default: segmentations).")
    parser.add_argument("--output", type=str, default="dice_results.csv",
                         help="Path to write the per-case CSV of results.")
    parser.add_argument("--workers", type=int, default=1,
                         help="Number of parallel processes (default: 1, sequential).")
    parser.add_argument("--ids_subset", type=str, default=None,
                         help="Optional path to a text file with one BDMAP_ID per line, to restrict evaluation.")
    args = parser.parse_args()

    organs = load_label_names(args.label_names)
    logger.info(f"Scoring {len(organs)} organs from {args.label_names}: {organs}")

    bdmap_ids = find_bdmap_dirs(args.pred_root)
    if args.ids_subset:
        with open(args.ids_subset, "r") as f:
            keep = set(line.strip() for line in f if line.strip())
        bdmap_ids = [b for b in bdmap_ids if b in keep]

    if not bdmap_ids:
        logger.error(f"No BDMAP_* folders found in {args.pred_root}")
        sys.exit(1)

    logger.info(f"Found {len(bdmap_ids)} cases to evaluate.")

    task_args = [
        (bid, args.pred_root, args.gt_root, args.pred_subdir, args.gt_subdir, organs)
        for bid in bdmap_ids
    ]

    all_results = []
    all_warnings = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(eval_one_case, t): t[0] for t in task_args}
            iterator = as_completed(futures)
            if HAS_TQDM:
                iterator = tqdm(iterator, total=len(futures), desc="Evaluating")
            for fut in iterator:
                result, warns = fut.result()
                all_results.append(result)
                all_warnings.extend(warns)
    else:
        iterator = task_args
        if HAS_TQDM:
            iterator = tqdm(iterator, desc="Evaluating")
        for t in iterator:
            result, warns = eval_one_case(t)
            all_results.append(result)
            all_warnings.extend(warns)

    # sort results by BDMAP_ID for a stable CSV
    all_results.sort(key=lambda r: r["BDMAP_ID"])

    # write per-case CSV
    fieldnames = ["BDMAP_ID"] + organs
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})
    logger.info(f"Wrote per-case results to {args.output}")

    # summary stats: mean dice per organ, ignoring None (missing/unscoreable) entries
    print("\n=== Per-organ mean Dice (lesions only, excluding missing/unscoreable cases) ===")
    summary_rows = []
    for organ in organs:
        vals = [r[organ] for r in all_results if r.get(organ) is not None]
        if vals:
            mean_d = float(np.mean(vals))
            std_d = float(np.std(vals))
            n = len(vals)
            print(f"{organ:20s} mean={mean_d:.4f}  std={std_d:.4f}  n={n}/{len(all_results)}")
        else:
            mean_d, std_d, n = float("nan"), float("nan"), 0
            print(f"{organ:20s} no valid cases")
        summary_rows.append({"organ": organ, "mean_dice": mean_d, "std_dice": std_d, "n_cases": n})

    all_vals = [r[o] for r in all_results for o in organs if r.get(o) is not None]
    if all_vals:
        print(f"\n{'OVERALL':20s} mean={np.mean(all_vals):.4f}  std={np.std(all_vals):.4f}  n={len(all_vals)}")

    summary_path = os.path.splitext(args.output)[0] + "_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["organ", "mean_dice", "std_dice", "n_cases"])
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info(f"Wrote summary to {summary_path}")

    if all_warnings:
        warn_path = os.path.splitext(args.output)[0] + "_warnings.txt"
        with open(warn_path, "w") as f:
            f.write("\n".join(all_warnings))
        logger.warning(f"{len(all_warnings)} warnings (missing files / mismatches) written to {warn_path}")


if __name__ == "__main__":
    main()