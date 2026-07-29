#!/usr/bin/env python3

import os
import argparse
import nibabel as nib
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

LOG_FILENAME = "refine.log"

def has_required_files(parent_folder: str, patient: str) -> bool:
    """
    Returns True if the patient has at least one complete set:
      1) The entire liver set (liver.nii.gz + 8 subsegments)
         OR
      2) The entire pancreas set (pancreas.nii.gz + head/body/tail).
    Otherwise, returns False (skips the patient).
    """
    patient_seg_path = os.path.join(parent_folder, patient, "segmentations")
    if not os.path.isdir(patient_seg_path):
        return False

    # Liver set
    liver_files = ["liver.nii.gz"] + [f"liver_segment_{i}.nii.gz" for i in range(1, 9)]
    # Pancreas set
    pancreas_files = [
        "pancreas.nii.gz",
        "pancreas_head.nii.gz",
        "pancreas_body.nii.gz",
        "pancreas_tail.nii.gz",
    ]

    def all_exist(files):
        return all(os.path.isfile(os.path.join(patient_seg_path, f)) for f in files)

    # Does the patient have a complete liver set?
    liver_ok = all_exist(liver_files)
    # Does the patient have a complete pancreas set?
    pancreas_ok = all_exist(pancreas_files)

    # Patient is valid if it has at least one full set
    return (liver_ok or pancreas_ok)


def refine_subsegment(subsegment_path, organ_path):
    """
    Load subsegment and organ NIfTI files, compute bitwise overlap,
    and overwrite the original subsegment file with the refined mask.
    Prints progress with flush=True for real-time updates.
    """
    if not os.path.isfile(subsegment_path):
        print(f"[WARNING] Missing subsegment file: {subsegment_path}", flush=True)
        return
    if not os.path.isfile(organ_path):
        print(f"[WARNING] Missing organ file: {organ_path}; cannot refine {subsegment_path}", flush=True)
        return

    try:
        sub_img = nib.load(subsegment_path)
        sub_data = sub_img.get_fdata(dtype=np.float32).astype(np.uint8)

        organ_img = nib.load(organ_path)
        organ_data = organ_img.get_fdata(dtype=np.float32).astype(np.uint8)

        # Bitwise AND to refine
        refined_data = sub_data & organ_data

        refined_img = nib.Nifti1Image(refined_data, sub_img.affine, sub_img.header)
        nib.save(refined_img, subsegment_path)

        print(f"[INFO] Refined {os.path.basename(subsegment_path)} with {os.path.basename(organ_path)}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed refining {subsegment_path} with {organ_path} – {e}", flush=True)


def refine_one_patient(args):
    """
    Process exactly one patient's folder:
      - The folder is expected at: parent_folder/patient/segmentations
      - Identify and refine:
          * Liver set, if present
          * Pancreas set, if present
        (We do not require both sets to exist; we refine whichever exists.)
    Returns 'patient' if processed, or None if skipped.
    """
    parent_folder, patient = args
    patient_seg_path = os.path.join(parent_folder, patient, "segmentations")

    if not os.path.isdir(patient_seg_path):
        print(f"[WARNING] Skipping '{patient}': segmentations folder not found.", flush=True)
        return None

    # Liver
    liver_path = os.path.join(patient_seg_path, "liver.nii.gz")
    liver_segments = [os.path.join(patient_seg_path, f"liver_segment_{i}.nii.gz") for i in range(1, 9)]

    # Pancreas
    pancreas_path = os.path.join(patient_seg_path, "pancreas.nii.gz")
    pancreas_subsegments = [
        os.path.join(patient_seg_path, "pancreas_head.nii.gz"),
        os.path.join(patient_seg_path, "pancreas_body.nii.gz"),
        os.path.join(patient_seg_path, "pancreas_tail.nii.gz"),
    ]

    # Refine the liver set if it looks present
    if os.path.isfile(liver_path) and all(os.path.isfile(s) for s in liver_segments):
        for seg_path in liver_segments:
            refine_subsegment(seg_path, liver_path)

    # Refine the pancreas set if it looks present
    if os.path.isfile(pancreas_path) and all(os.path.isfile(s) for s in pancreas_subsegments):
        for seg_path in pancreas_subsegments:
            refine_subsegment(seg_path, pancreas_path)

    print(f"[DONE] Finished refining patient '{patient}'", flush=True)
    return patient


def refine_segments_in_folder(parent_folder, workers, parts, part_index, restart):
    """
    1) Gather all patient subdirectories under parent_folder.
    2) Filter out:
       - already refined (unless --restart)
       - missing required sets (both liver & pancreas are absent)
    3) Split them into 'parts' and process only the subset for 'part_index'.
    4) Parallelize the refinement with Pool(workers).
    5) Log each completed patient in refine.log.
    6) Show a tqdm progress bar for the overall process.
    """
    all_patients = sorted([
        d for d in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, d))
    ])

    # Load log if not restarting
    refined_patients = set()
    if not restart and os.path.exists(LOG_FILENAME):
        with open(LOG_FILENAME, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    refined_patients.add(line)

    # Filter out:
    # (1) Already-refined patients (unless --restart)
    # (2) Patients missing both sets
    pending_patients = []
    for p in all_patients:
        if p in refined_patients:
            continue
        if has_required_files(parent_folder, p):
            pending_patients.append(p)
        else:
            print(f"[INFO] Skipping '{p}': missing both (full) liver and pancreas sets.", flush=True)

    total_pending = len(pending_patients)
    if total_pending == 0:
        print("[INFO] No pending patients to refine. Exiting.")
        return

    # Partition into 'parts'
    part_size = (total_pending + parts - 1) // parts  # ceiling division
    start_idx = part_index * part_size
    end_idx = min(start_idx + part_size, total_pending)

    if start_idx >= total_pending:
        print(f"[WARNING] Part index {part_index} is out of range. Nothing to process.")
        return

    subset_patients = pending_patients[start_idx:end_idx]
    print(f"[INFO] Processing {len(subset_patients)} patients in part {part_index}/{parts-1}")

    task_args = [(parent_folder, p) for p in subset_patients]

    if workers < 1:
        workers = 1

    # Process in parallel, logging each completed patient
    with open(LOG_FILENAME, "a") as log_file:
        with Pool(processes=workers) as pool:
            for result in tqdm(pool.imap_unordered(refine_one_patient, task_args),
                               total=len(task_args), desc="Refining Patients"):
                if result is not None:
                    log_file.write(result + "\n")
                    log_file.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Refine sub-segments of liver/pancreas by overlapping with the organ mask, in parallel, "
                    "with partitioning and logging."
    )
    parser.add_argument("parent_folder",
                        help="Path to the folder containing patient subfolders")
    parser.add_argument("--workers", "-w",
                        type=int, default=1,
                        help="Number of multiprocessing workers (default: 1)")
    parser.add_argument("--parts", type=int, default=1,
                        help="Total number of parts to split patients into (default: 20)")
    parser.add_argument("--part", type=int, default=0,
                        help="Which 0-based part to process (default: 0)")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore the log file and refine all patients in this part, from scratch.")

    args = parser.parse_args()

    refine_segments_in_folder(
        parent_folder=args.parent_folder,
        workers=args.workers,
        parts=args.parts,
        part_index=args.part,
        restart=args.restart
    )


if __name__ == "__main__":
    main()