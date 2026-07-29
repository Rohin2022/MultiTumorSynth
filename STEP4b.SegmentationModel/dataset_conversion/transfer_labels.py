#!/usr/bin/env python3
import os
import shutil
import argparse
import nibabel as nib
from multiprocessing import Pool
from tqdm import tqdm

def process_patient(args):
    """
    Process one patient: if the patient's source segmentations folder exists and
    the destination segmentations folder exists, load reference shapes from
    destination files. Then, for each segmentation file in the source that meets
    the naming criteria, load its shape and copy it over only if its shape matches
    one of the reference shapes.
    """
    patient, source, destin = args
    source_patient_dir = os.path.join(source, patient, "segmentations")
    destin_patient_dir = os.path.join(destin, patient, "segmentations")
    
    if not os.path.isdir(source_patient_dir):
        print(f"Skipping patient '{patient}': 'segmentations' folder not found.")
        return
    if not os.path.isdir(destin_patient_dir):
        print(f"Skipping patient '{patient}': destination 'segmentations' folder not found.")
        return
    
    # Gather reference shapes from destination nifti files
    ref_shapes = []
    flag = False
    for f in os.listdir(destin_patient_dir):
        if f.endswith(".nii.gz"):
            ref_path = os.path.join(destin_patient_dir, f)
            try:
                img = nib.load(ref_path)
                ref_shapes.append(img.shape)
                flag = True
            except Exception as e:
                print(f"Error loading {ref_path}: {e}")
            if flag:
                break
    
    if not ref_shapes:
        print(f"Skipping patient '{patient}': no reference nifti files found in destination for shape comparison.")
        return
    
    # Loop through segmentation files in the source's predictions folder
    for filename in os.listdir(source_patient_dir):
        if filename.endswith(".nii.gz"):
            src_file = os.path.join(source_patient_dir, filename)
            dst_file = os.path.join(destin_patient_dir, filename)
            
            # Only copy if the destination file doesn't already exist
            if os.path.exists(dst_file):
                continue
            
            try:
                src_img = nib.load(src_file)
                src_shape = src_img.shape
            except Exception as e:
                print(f"Error loading {src_file}: {e}")
                continue
            
            # Check if the source file shape matches any reference shape
            if any(src_shape == ref for ref in ref_shapes):
                shutil.copy2(src_file, dst_file)
                print(f"Copied {src_file} to {dst_file}")
            else:
                print(f"Skipping {src_file}: shape {src_shape} does not match any reference shape.")

def copy_segmentations(source, destin):
    """
    List all patients in the source folder, then use a process pool with tqdm
    to process each patient concurrently.
    """
    patients = os.listdir(source)
    # Prepare argument tuples for each patient
    args_list = [(patient, source, destin) for patient in patients]
    
    with Pool() as pool:
        # Use imap_unordered for concurrent processing with a progress bar
        list(tqdm(pool.imap_unordered(process_patient, args_list), total=len(args_list)))

def main():
    parser = argparse.ArgumentParser(
        description="Copy segmentation files (.nii.gz) from source to destin if not already present, "
                    "using multiprocessing, tqdm, and a shape check with nibabel."
    )
    parser.add_argument("source", help="Path to the source folder")
    parser.add_argument("destin", help="Path to the destination folder")
    
    args = parser.parse_args()
    copy_segmentations(args.source, args.destin)

if __name__ == "__main__":
    main()