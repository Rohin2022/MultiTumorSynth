#!/usr/bin/env python3
import os
import shutil
import argparse
import nibabel as nib
from multiprocessing import Pool
from tqdm import tqdm

essential_organs = ['kidney','pancrea','bladder','gall','spleen','prostate','adrenal','esophagus','stomach','duodenum','colon']#'liver',

def load_ref_shapes(destin_patient_dir: str):
    """Collect shapes from any readable .nii.gz under destination segmentations."""
    ref_shapes = []
    if not os.path.isdir(destin_patient_dir):
        return ref_shapes
    for f in os.listdir(destin_patient_dir):
        if f.endswith(".nii.gz"):
            ref_path = os.path.join(destin_patient_dir, f)
            try:
                img = nib.load(ref_path)
                ref_shapes.append(img.shape)
            except Exception:
                # unreadable ref -> ignore
                pass
    return ref_shapes

def process_patient(args) -> int:
    """
    Returns:
      1 if at least one file was copied for this patient, else 0.
    """
    patient, source, destin, ct_mode = args

    destin_patient_dir = os.path.join(destin, patient, "segmentations")
    # Must have destination segmentations for reference shape check
    if not os.path.isdir(destin_patient_dir):
        return (0, None)

    ref_shapes = load_ref_shapes(destin_patient_dir)
    if not ref_shapes:
        # no reference shapes -> skip silently
        return (0, None)

    copied_any = 0

    if ct_mode:
        # Copy only CT: src/{ID}/ct.nii.gz -> dst/{ID}/ct.nii.gz
        src_ct = os.path.join(source, patient, "ct.nii.gz")
        dst_ct = os.path.join(destin, patient, "ct.nii.gz")

        if not os.path.isfile(src_ct):
            return (0, None)  # skip silently

        # If destination CT already exists, don't overwrite
        if os.path.exists(dst_ct):
            return (0, None)

        # Load CT and shape-check against any reference label shape
        try:
            ct_img = nib.load(src_ct)
            ct_shape = ct_img.shape
        except Exception as e:
            # skip silently if empty/unreadable
            if "Empty file" in str(e) or "file does not exist" in str(e).lower():
                return (0, None)
            raise RuntimeError(
                f"Failed to load CT for patient '{patient}': {src_ct} ({e})"
            )

        if not any(ct_shape == ref for ref in ref_shapes):
            #raise RuntimeError(
            #    f"CT shape mismatch for patient '{patient}': {src_ct} has shape {ct_shape}, "
            #    f"no match in destination reference shapes {set(ref_shapes)}"
            #)
            patient_dest_top = os.path.join(destin, patient)
            try:
                shutil.rmtree(patient_dest_top)
            except FileNotFoundError:
                pass

            print(
                f"Excluded patient '{patient}' due to CT shape mismatch: {src_ct} has shape {ct_shape}, "
                f"no match in destination reference shapes {set(ref_shapes)}"
            )
            return (0, patient)


        # Ensure patient dir exists in destination (top-level)
        os.makedirs(os.path.join(destin, patient), exist_ok=True)
        shutil.copy2(src_ct, dst_ct)
        copied_any = 1

    else:
        # Copy labels from source: src/{ID}/segmentations/*.nii.gz -> dst/{ID}/segmentations/*.nii.gz
        source_patient_dir = os.path.join(source, patient, "segmentations")
        if not os.path.isdir(source_patient_dir):
            return (0, None)

        for filename in os.listdir(source_patient_dir):
            if not filename.endswith(".nii.gz"):
                continue

            src_file = os.path.join(source_patient_dir, filename)
            dst_file = os.path.join(destin_patient_dir, filename)

            # Do not overwrite
            if os.path.exists(dst_file):
                continue

            try:
                src_img = nib.load(src_file)
                src_shape = src_img.shape
            except Exception as e:
                # skip silently for empty/unreadable
                if "Empty file" in str(e) or "file does not exist" in str(e).lower():
                    continue
                raise RuntimeError(
                    f"Failed to load source NIfTI for patient '{patient}': {src_file} ({e})"
                )

            if not any(src_shape == ref for ref in ref_shapes):
                is_essential = (any(org in src_file for org in essential_organs) and ('segmentations/_' not in src_file))
                if is_essential:
                    # Exclude this patient from destination entirely
                    patient_dest_top = os.path.join(destin, patient)
                    try:
                        shutil.rmtree(patient_dest_top)
                    except FileNotFoundError:
                        pass

                    print(
                        f"Excluded patient '{patient}' due to essential shape mismatch: {src_file} has shape {src_shape}, "
                        f"no match in destination reference shapes {set(ref_shapes)}"
                    )
                    return (0, patient)  # <-- key change (report excluded)
                else:
                    print(f"Skipping: Shape mismatch for patient '{patient}': {src_file} has shape {src_shape}, "
                        f"no match in destination reference shapes {set(ref_shapes)}")
                    continue

            shutil.copy2(src_file, dst_file)
            copied_any = 1

    return (copied_any, None)


def copy_segmentations(source: str, destin: str, ct_mode: bool) -> None:
    """
    Process each patient concurrently.
    At the end, prints how many distinct patients had at least one file copied.
    """
    if not os.path.isdir(source):
        print("0 patients transferred.")
        return

    patients = os.listdir(source)
    args_list = [(patient, source, destin, ct_mode) for patient in patients]

    transferred_count = 0
    excluded_patients = []

    with Pool() as pool:
        for copied_flag, excluded_id in tqdm(
            pool.imap_unordered(process_patient, args_list),
            total=len(args_list),
            desc=("Transferring CTs" if ct_mode else "Transferring labels")
        ):
            transferred_count += int(bool(copied_flag))
            if excluded_id is not None:
                excluded_patients.append(excluded_id)

    print(f"{transferred_count} patients transferred.")

    # Print excluded summary
    excluded_patients = sorted(set(excluded_patients))
    print(f"{len(excluded_patients)} patients excluded.")
    if excluded_patients:
        print("Excluded BDMAP IDs:")
        for pid in excluded_patients:
            print(f"  - {pid}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy segmentation files (.nii.gz) from source to destination if missing, "
            "with strict shape match (raises on mismatch), skipping empty files. "
            "Use --ct to copy only CTs (ct.nii.gz) with shape check against existing labels."
        )
    )
    parser.add_argument("source", help="Path to the source folder")
    parser.add_argument("destin", help="Path to the destination folder")
    parser.add_argument("--ct", action="store_true", help="Copy only CT (ct.nii.gz) instead of labels")
    args = parser.parse_args()

    copy_segmentations(args.source, args.destin, ct_mode=args.ct)


if __name__ == "__main__":
    main()