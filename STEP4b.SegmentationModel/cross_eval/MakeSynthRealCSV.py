#!/usr/bin/env python3

import os
import argparse
import pandas as pd


"""
python MakeSynthRealCSV.py \
    --real-ct-source /scratch/rpinise1/MultiTumorSynthesis/CopiedAAProWithMasks/ \
    --synthetic-csv /projects/bodymaps/Rohin/TumorSynthesis/STEP4a.GenerateSyntheticDataset/cross_eval/abdomen_atlas_pro/healthy_dataset.csv \
    --synthetic-id-column bdmap_id \
    --output is_synthetic.csv
"""


def main():
    parser = argparse.ArgumentParser(
        description="Build a BDMAP_ID,is_synthetic CSV from a real-CT source dir and a synthetic-IDs CSV"
    )

    parser.add_argument("--real-ct-source", required=True,
                        help="Directory containing one subfolder per real BDMAP case (e.g. AbdomenAtlasPro root)")
    parser.add_argument("--synthetic-csv", required=True,
                        help="CSV containing synthetic BDMAP IDs")
    parser.add_argument("--synthetic-id-column", default="bdmap_id",
                        help="Column name for the BDMAP ID in the synthetic CSV")
    parser.add_argument("--output", required=True,
                        help="Path to write the resulting BDMAP_ID,is_synthetic CSV")

    args = parser.parse_args()

    # Real IDs: one subdirectory per case
    real_ids = {
        d for d in os.listdir(args.real_ct_source)
        if os.path.isdir(os.path.join(args.real_ct_source, d)) and d.startswith('BDMAP')
    }
    print(f'Found {len(real_ids)} real cases in {args.real_ct_source}')

    # Synthetic IDs from the provided CSV
    synth_df = pd.read_csv(args.synthetic_csv)
    if args.synthetic_id_column not in synth_df.columns:
        raise ValueError(
            f"Missing column '{args.synthetic_id_column}'. Available columns: {list(synth_df.columns)}"
        )
    synth_ids = set(synth_df[args.synthetic_id_column].astype(str))
    print(f'Found {len(synth_ids)} synthetic cases in {args.synthetic_csv}')

    overlap = real_ids & synth_ids
    if overlap:
        print(f'WARNING: {len(overlap)} IDs appear in both real and synthetic sources. '
              f'Marking them as synthetic. Example: {list(overlap)[:5]}')
        real_ids -= overlap

    rows = (
        [{'BDMAP_ID': i, 'is_synthetic': 1} for i in sorted(synth_ids)] +
        [{'BDMAP_ID': i, 'is_synthetic': 0} for i in sorted(real_ids)]
    )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)

    print(f'\nWrote {len(out_df)} rows to {args.output}')
    print(f'  synthetic: {(out_df.is_synthetic == 1).sum()}')
    print(f'  real:      {(out_df.is_synthetic == 0).sum()}')


if __name__ == "__main__":
    main()