#!/usr/bin/env python3

import os
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


"""
python MakeAtlasMeta.py \
    --mask-source /projects/bodymaps/Data/mask_only/AbdomenAtlasPro/AbdomenAtlasPro/ \
    --organs gallbladder bladder spleen uterus prostate esophagus colon stomach duodenum \
    --output /projects/bodymaps/Rohin/TumorSynthesis/atlas_meta.csv \
    --workers 8

Produces a CSV with columns:
    BDMAP ID, number of gallbladder instances, number of bladder instances, ...
where each column is 1 if that organ's tumor mask FILE exists for the case, else 0.
A single case can have multiple organs' masks present (e.g. spleen + colon tumors on the same CT) --
each organ is checked independently.
"""


def get_organ_names(organ):
    organ = str(organ).lower()
    if organ == "gallbladder":
        return ["gallbladder", "gall_bladder"]
    return [organ]


def check_case(bdmap_id, mask_source, organs):
    """Checks tumor-mask file presence for one case across all requested organs."""
    mask_dir = os.path.join(mask_source, bdmap_id, "segmentations")
    row = {'BDMAP ID': bdmap_id}

    for organ in organs:
        present = False
        for name in get_organ_names(organ):
            lesion_path = os.path.join(mask_dir, name + "_lesion.nii.gz")
            if os.path.exists(lesion_path):
                present = True
                break
        row[f'number of {organ} instances'] = int(present)

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Build an atlas_meta CSV (BDMAP ID + per-organ tumor mask presence) from mask folders"
    )

    parser.add_argument("--mask-source", required=True,
                        help="Directory containing one subfolder per BDMAP case (segmentations/ subfolder)")
    parser.add_argument("--organs", nargs='+', required=True,
                        help="Organ names to check tumor-mask presence for (e.g. gallbladder bladder spleen ...)")
    parser.add_argument("--output", required=True,
                        help="Path to write the resulting atlas_meta CSV")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes")

    args = parser.parse_args()

    case_ids = sorted([
        d for d in os.listdir(args.mask_source)
        if os.path.isdir(os.path.join(args.mask_source, d)) and d.startswith('BDMAP')
    ])
    print(f'Found {len(case_ids)} candidate cases in {args.mask_source}')

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(check_case, bdmap_id, args.mask_source, args.organs): bdmap_id
            for bdmap_id in case_ids
        }
        done = 0
        for future in as_completed(futures):
            bdmap_id = futures[future]
            try:
                rows.append(future.result())
            except Exception as e:
                print(f'ERROR processing {bdmap_id}: {e}')
            done += 1
            if done % 500 == 0:
                print(f'Processed {done}/{len(case_ids)}')

    out_df = pd.DataFrame(rows).sort_values('BDMAP ID').reset_index(drop=True)
    out_df.to_csv(args.output, index=False)

    print(f'\nWrote {len(out_df)} rows to {args.output}')
    for organ in args.organs:
        col = f'number of {organ} instances'
        n_positive = (out_df[col] > 0).sum()
        print(f'  {organ}: {n_positive} cases with a tumor mask present')


if __name__ == "__main__":
    main()