"""
Scan a root directory structured as:

    root/
      BDMAP_00000001/
        ct.nii.gz
        segmentations/
          liver.nii.gz
          kidney_left.nii.gz
          ...
      BDMAP_00000002/
        segmentations/
          ...
      ...

and collect the set of unique label names (segmentation file stems) found
across all cases. Also reports per-label frequency (how many cases have it),
which is useful for deciding what belongs in label_names.yaml.

Usage:
    python find_unique_labels.py --root /path/to/root
    python find_unique_labels.py --root /path/to/root --out label_names.yaml
    python find_unique_labels.py --root /path/to/root --seg_dirname predictions
    python find_unique_labels.py --root /path/to/root --min_count 5
"""

import os
import argparse
import yaml
from collections import Counter
from tqdm import tqdm


def strip_ext(fname):
    # handles .nii.gz and .nii and .npz
    if fname.endswith('.nii.gz'):
        return fname[:-len('.nii.gz')]
    return os.path.splitext(fname)[0]


def find_unique_labels(root, seg_dirname='segmentations', id_prefix='BDMAP'):
    case_names = [d for d in os.listdir(root)
                  if id_prefix in d and os.path.isdir(os.path.join(root, d))]

    label_counter = Counter()
    cases_with_no_seg_dir = []
    cases_scanned = 0

    for case in tqdm(case_names, desc="Scanning cases"):
        seg_dir = os.path.join(root, case, seg_dirname)
        if not os.path.isdir(seg_dir):
            cases_with_no_seg_dir.append(case)
            continue

        cases_scanned += 1
        for fname in os.listdir(seg_dir):
            if fname.endswith('.nii.gz') or fname.endswith('.nii') or fname.endswith('.npz'):
                label_counter[strip_ext(fname)] += 1

    return label_counter, case_names, cases_with_no_seg_dir, cases_scanned


def main():
    parser = argparse.ArgumentParser(description="Find unique segmentation label names under a BDMAP-style root directory.")
    parser.add_argument("--root", type=str, required=True,
                         help="Root directory containing BDMAP_* case folders.")
    parser.add_argument("--seg_dirname", type=str, default="segmentations",
                         help="Name of the subfolder inside each case containing label files (default: segmentations). "
                              "Use 'predictions' if that's where your masks live.")
    parser.add_argument("--id_prefix", type=str, default="BDMAP",
                         help="Substring used to identify case folders (default: BDMAP).")
    parser.add_argument("--out", type=str, default=None,
                         help="Optional path to write the unique label names as a YAML list (e.g. label_names.yaml).")
    parser.add_argument("--min_count", type=int, default=1,
                         help="Only include labels that appear in at least this many cases (default: 1, i.e. all).")
    parser.add_argument("--sort_by", choices=["name", "count"], default="count",
                         help="How to sort the printed/report output (default: count, descending).")

    args = parser.parse_args()

    label_counter, case_names, missing_seg_dir, cases_scanned = find_unique_labels(
        args.root, seg_dirname=args.seg_dirname, id_prefix=args.id_prefix
    )

    print(f"\nTotal case folders found: {len(case_names)}")
    print(f"Cases with a '{args.seg_dirname}/' folder: {cases_scanned}")
    if missing_seg_dir:
        print(f"Cases missing '{args.seg_dirname}/' folder: {len(missing_seg_dir)}")
        for c in missing_seg_dir[:10]:
            print(f"  - {c}")
        if len(missing_seg_dir) > 10:
            print(f"  ... and {len(missing_seg_dir) - 10} more")

    filtered = {k: v for k, v in label_counter.items() if v >= args.min_count}

    if args.sort_by == "count":
        items = sorted(filtered.items(), key=lambda x: (-x[1], x[0]))
    else:
        items = sorted(filtered.items(), key=lambda x: x[0])

    print(f"\nUnique labels found (min_count={args.min_count}): {len(items)}\n")
    print(f"{'label':40s} count   coverage")
    print("-" * 65)
    for name, count in items:
        pct = 100.0 * count / max(cases_scanned, 1)
        print(f"{name:40s} {count:6d}   {pct:6.1f}%")

    if args.out:
        label_names = sorted(filtered.keys())
        with open(args.out, "w", encoding="utf-8") as f:
            yaml.dump(label_names, f)
        print(f"\nWrote {len(label_names)} label names to {args.out}")


if __name__ == "__main__":
    main()