#!/usr/bin/env python3
import os
import glob
import numpy as np
from multiprocessing import Pool, cpu_count
import argparse
from tqdm import tqdm
import logging

def check_file(file_path):
    """
    Load the file and check for NaNs, Infs, or values outside the range [-50, 50].
    For .npz files, each stored array is checked.
    For .npy files, the single array is checked.
    If values outside the range are found, the min and max values are also reported.
    Returns a list of issues found (or an empty list if none).
    """
    issues = []
    try:
        if file_path.endswith('.npz'):
            data = np.load(file_path)
            for key in data.files:
                arr = data[key]
                if np.isnan(arr).any():
                    issues.append(f"File '{file_path}', array '{key}' contains NaN values.")
                if np.isinf(arr).any():
                    issues.append(f"File '{file_path}', array '{key}' contains Inf values.")
                if (arr > 50).any() or (arr < -50).any():
                    # Using nanmin/nanmax so that any existing NaNs won't break the computation
                    min_val = np.nanmin(arr)
                    max_val = np.nanmax(arr)
                    issues.append(f"File '{file_path}', array '{key}' contains values outside [-50, 50] (min: {min_val}, max: {max_val}).")
            data.close()
        elif file_path.endswith('.npy'):
            arr = np.load(file_path)
            if np.isnan(arr).any():
                issues.append(f"File '{file_path}' contains NaN values.")
            if np.isinf(arr).any():
                issues.append(f"File '{file_path}' contains Inf values.")
            if (arr > 50).any() or (arr < -50).any():
                min_val = np.nanmin(arr)
                max_val = np.nanmax(arr)
                issues.append(f"File '{file_path}' contains values outside [-50, 50] (min: {min_val}, max: {max_val}).")
    except Exception as e:
        issues.append(f"Error loading {file_path}: {e}")
    return issues

def check_files(folder, workers, logger):
    # Gather both .npz and .npy files from the folder.
    npz_files = glob.glob(os.path.join(folder, '*.npz'))
    npy_files = glob.glob(os.path.join(folder, '*.npy'))
    all_files = npz_files + npy_files
    all_files = [f for f in all_files if ('gt' not in f)]
    msg = f"Found {len(all_files)} files (.npz and .npy) to check."
    print(msg)
    logger.info(msg)

    issues_found = False
    with Pool(processes=workers) as pool:
        for issues in tqdm(pool.imap_unordered(check_file, all_files), total=len(all_files)):
            if issues:
                issues_found = True
                for issue in issues:
                    tqdm.write(issue)   # Print immediately without breaking the progress bar.
                    logger.info(issue)    # Log the issue in real time.

    if not issues_found:
        msg = "No issues found in any files."
        print(msg)
        logger.info(msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Check .npz and .npy files for NaNs, Infs, or values outside the range [-50, 50], reporting min and max values when applicable."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing .npz and .npy files to check."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes to use (default: number of CPU cores)."
    )
    parser.add_argument(
        "-l", "--logfile",
        type=str,
        default="check_files.log",
        help="Path to the log file (default: check_files.log)."
    )
    args = parser.parse_args()

    # Setup logging to file and console.
    logger = logging.getLogger("check_files")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(args.logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    check_files(args.folder, args.workers, logger)