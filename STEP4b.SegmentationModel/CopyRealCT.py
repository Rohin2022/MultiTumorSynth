

#!/usr/bin/env python3

import os
import shutil
import argparse
import pandas as pd


"""
python CopyRealCT.py \
    --ct-source /projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro/ \
    --mask-source /projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/ \
    --dest /scratch/rpinise1/MultiTumorSynthesis/CopiedAAProWithMasks \
    --csv /projects/bodymaps/Rohin/TumorSynthesis/STEP4b.SegmentationModel/cross_eval/ucsf_133k_train_mask_ids.csv
"""


def get_organ_names(organ):
    organ = str(organ).lower()

    if organ == "gallbladder":
        return ["gallbladder", "gall_bladder"]

    return [organ]


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Copy BDMAP CTs and selected organ segmentations"
    )

    parser.add_argument(
        "--ct-source",
        required=True,
        help="Source directory containing BDMAP CT folders"
    )

    parser.add_argument(
        "--mask-source",
        required=True,
        help="Source directory containing BDMAP segmentation folders"
    )

    parser.add_argument(
        "--dest",
        required=True,
        help="Destination directory"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="CSV containing BDMAP IDs and organs"
    )

    parser.add_argument(
        "--id-column",
        default="bdmap",
        help="BDMAP ID column name"
    )

    parser.add_argument(
        "--organ-column",
        default="organ",
        help="Organ column name"
    )

    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    df = pd.read_csv(args.csv)

    for col in [args.id_column, args.organ_column]:
        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}'. Available columns: {list(df.columns)}"
            )

    # Group organs per BDMAP ID
    cases = (
        df.groupby(args.id_column)[args.organ_column]
        .apply(list)
        .to_dict()
    )

    copied = 0
    skipped = 0
    missing = 0

    for bdmap_id, organs in cases.items():

        bdmap_id = str(bdmap_id)

        dst_case = os.path.join(args.dest, bdmap_id)

        if os.path.exists(dst_case):
            print(f"SKIP {bdmap_id} (already exists)")
            skipped += 1
            continue

        ct_path = os.path.join(
            args.ct_source,
            bdmap_id,
            "ct.nii.gz"
        )

        mask_dir = os.path.join(
            args.mask_source,
            bdmap_id,
            "segmentations"
        )

        if not os.path.exists(ct_path):
            print(f"MISSING CT {bdmap_id}")
            missing += 1
            continue

        print(f"COPY {bdmap_id}")

        # Copy CT
        copy_file(
            ct_path,
            os.path.join(dst_case, "ct.nii.gz")
        )

        # Copy only requested masks
        for organ in set(organs):
            for name in get_organ_names(organ):

                for suffix in [
                    ".nii.gz",
                    "_lesion.nii.gz"
                ]:
                    mask = os.path.join(
                        mask_dir,
                        name + suffix
                    )

                    if os.path.exists(mask):
                        copy_file(
                            mask,
                            os.path.join(
                                dst_case,
                                "segmentations",
                                os.path.basename(mask)
                            )
                        )

        copied += 1

    print("\nSummary")
    print("----------------")
    print(f"Copied:  {copied}")
    print(f"Skipped: {skipped}")
    print(f"Missing: {missing}")


if __name__ == "__main__":
    main()