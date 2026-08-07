import SimpleITK as sitk
import numpy as np
import os
import shutil
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
import yaml

"""
    python nii2npz.py --src_path /scratch/rpinise1/MultiTumorSynthesis/Synthetic_RSuperProcessed_VV2_COMBINED --tgt_path /scratch/rpinise1/MultiTumorSynthesis/Synthetic_RSuperProcessed_VV2_COMBINED_NPZ --parts 6 --current_part 0

"""

lab_name_list = []

#raise ValueError('Converting new files to npz will change the R-Super training set, breaking reproducibility and creating leakage of external test sets. If you wish to add more masks to the radiologist processed folder, you fist need to make training set Atlas a mandatory input in training. The masks that were processed by submission time are in /projects/bodymaps/Pedro/foundational/MedFormer/133K_dataset/133k_dataset_processed_masks_May_30_2025.csv')

def pad(img, lab):
    z, y, x = img.shape
    # pad if the image size is smaller than training size
    if z < 128:
        diff = int(math.ceil((128. - z) / 2)) 
        img = np.pad(img, ((diff, diff), (0, 0), (0, 0)))
        lab = np.pad(lab, ((0, 0), (diff, diff), (0, 0), (0, 0)))
    if y < 128:
        diff = int(math.ceil((128. - y) / 2)) 
        img = np.pad(img, ((0, 0), (diff, diff), (0, 0)))
        lab = np.pad(lab, ((0, 0), (0, 0), (diff, diff), (0, 0)))
    if x < 128:
        diff = int(math.ceil((128. - x) / 2)) 
        img = np.pad(img, ((0, 0), (0, 0), (diff, diff)))
        lab = np.pad(lab, ((0, 0), (0, 0), (0, 0), (diff, diff)))

    return img, lab


def _verify_npz(path, expected_shape):
    try:
        with np.load(path) as data:
            arr = data[data.files[0]]
            return arr.shape == expected_shape
    except Exception as e:
        print(f"[verify-failed] {path}: {e}")
        return False


def process_file(file_info):
    try:
        global lab_name_list
        name, source_path, target_path, modality, write_n_delete = file_info
        img = sitk.ReadImage(os.path.join(source_path, name))
        img = sitk.GetArrayFromImage(img).astype(np.float32)

        lab = []
        create_bkg = False
        for i,file in enumerate(list(sorted(lab_name_list)),0):
            if file == 'background':
                #check if file exists
                if not os.path.exists(os.path.join(source_path, name.replace('.nii.gz', ''), file+'.nii.gz')):
                    create_bkg = True
                    bkg_index = i
                    continue
            pth = os.path.join(source_path, name.replace('.nii.gz', ''), file+'.nii.gz')
            item = sitk.ReadImage(pth)
            item = sitk.GetArrayFromImage(item).astype(np.int16)
            lab.append(item)

        if create_bkg:
            if len(lab) > 0:
                lab_arr = np.stack(lab, axis=0)                # (C,Z,Y,X) without slices
                fg = (lab_arr > 0).any(axis=0).astype(np.int16)  # (Z,Y,X)
                background = (fg == 0).astype(np.int16)
            else:
                background = np.ones_like(img, dtype=np.int16)
            # insert at the recorded background index
            lab.insert(bkg_index, background)

        # --- add the slice channel (if present) ---
        # slices.nii.gz is expected in:  {source_path}/{case_name}/slices.nii.gz
        slices_pth = os.path.join(source_path, name.replace('.nii.gz', ''), 'slices.nii.gz')

        if os.path.exists(slices_pth):
            item = sitk.ReadImage(slices_pth)
            item = sitk.GetArrayFromImage(item).astype(np.int16)
            lab.append(item)
        else:
            print(f'Missing slices for {slices_pth}')
            lab.append(np.zeros_like(img, dtype=np.int16))

        #try:
        #    lab = np.stack(lab, axis=0)  # Makes label multi-channel
        #except:
        #    print(f"Error processing {name}")
        #    return None
        

        lab = np.stack(lab, axis=0)
                        
        # Clip intensities
        if modality == 'ct':
            img = np.clip(img, -991, 500)
        else:
            percentile_2 = np.percentile(img, 2, axis=None)
            percentile_98 = np.percentile(img, 98, axis=None)
            img = np.clip(img, percentile_2, percentile_98)

        # Normalize image
        mean = np.mean(img)
        std = np.std(img)
        #guard against zero std
        if std == 0:
            std = 1.0
        img = (img - mean) / std

        # Pad image and label
        img, lab = pad(img, lab)

        #separate the slices
        slices=lab[-1]
        lab=lab[:-1]

        # Save as .npy files
        img, lab = img.astype(np.float32), lab.astype(np.int8)
        slices = slices.astype(np.int16)
        base = name.replace('.nii.gz', '')
        img_npz = os.path.join(target_path, f"{base}.npz")
        gt_npz = os.path.join(target_path, f"{base}_gt.npz")
        slice_npz = os.path.join(target_path, f"{base}_slice.npz")
        np.savez_compressed(img_npz, img)
        np.savez_compressed(gt_npz, lab)
        np.savez_compressed(slice_npz, slices)

        if write_n_delete:
            ok = (
                _verify_npz(img_npz, img.shape)
                and _verify_npz(gt_npz, lab.shape)
                and _verify_npz(slice_npz, slices.shape)
            )
            if ok:
                src_nii = os.path.join(source_path, name)
                src_case_dir = os.path.join(source_path, base)
                try:
                    if os.path.isfile(src_nii):
                        os.remove(src_nii)
                    if os.path.isdir(src_case_dir):
                        shutil.rmtree(src_case_dir)
                except Exception as e:
                    print(f"[delete-failed] {name}: {e}")
            else:
                print(f"[verify-failed] keeping source for {name}")
        return name  # Return processed file name for logging
    except Exception as e:
        print(f"[error] {name}: {e}")
        return None

# Main function
def main():
    parser = argparse.ArgumentParser(description="Convert Nifti files to NPZ format for MedFormer.")
    parser.add_argument("--src_path", type=str, default="/projects/bodymaps/Data/UFO_27k_medformer/",
                        help="Source path for the Nifti files.")
    parser.add_argument("--tgt_path", type=str, default="/projects/bodymaps/Data/UFO_27k_medformerNpz/",
                        help="Target path for the NPZ files.")
    parser.add_argument("--parts", type=int, default=1,
                        help="Number of parts to split the dataset into (default 1, meaning no split).")
    parser.add_argument("--current_part", type=int, default=0,
                        help="The index (0-based) of the current part to process.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite already processed cases.")
    parser.add_argument("--write_n_delete", action="store_true",
                        help="Delete source nii.gz files after successfully writing and verifying the target npz files.")
    args = parser.parse_args()

    source_path = args.src_path
    target_path = args.tgt_path
    parts = args.parts
    current_part = args.current_part
    dataset_list = [
        ('abdomenatlas', 'ct'),
    ]

    global lab_name_list
    with open(os.path.join(source_path, 'list', 'label_names.yaml'), 'r', encoding='utf-8') as f:
        lab_name_list = yaml.safe_load(f)

    os.makedirs(os.path.join(target_path), exist_ok=True)

    for dataset, modality in dataset_list:
        names = sorted([name for name in os.listdir(os.path.join(source_path)) if '.nii.gz' in name])
        
        # Filter out files that have already been processed if overwrite is not enabled
        if not args.overwrite:
            names = [
                name for name in names
                if not (
                    os.path.exists(os.path.join(target_path, f"{name.replace('.nii.gz', '')}.npz")) and
                    os.path.exists(os.path.join(target_path, f"{name.replace('.nii.gz', '')}_gt.npz"))
                )
            ]
            print(f"After filtering, {len(names)} cases remain.")
        
        # Split the dataset if requested
        if parts > 1:
            splits = np.array_split(names, parts)
            if current_part < 0 or current_part >= len(splits):
                raise ValueError(f"current_part must be between 0 and {len(splits)-1}")
            names = splits[current_part].tolist()
            print(f"Processing part {current_part+1}/{parts} with {len(names)} cases.")
        else:
            print(f"Processing {len(names)} cases.")

        # After splitting, filter out files that have already been processed
        if not args.overwrite:
            names = [
                name for name in names
                if not (
                    os.path.exists(os.path.join(target_path, f"{name.replace('.nii.gz', '')}.npz")) and
                    os.path.exists(os.path.join(target_path, f"{name.replace('.nii.gz', '')}_gt.npz"))
                )
            ]
            print(f"After filtering, {len(names)} cases remain.")

        file_info_list = [(name, source_path, target_path, modality, args.write_n_delete) for name in names]

        # Process in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=8) as executor:  # Adjust `max_workers` as per hardware
            for result in tqdm(executor.map(process_file, file_info_list), total=len(file_info_list), desc=f"Processing {dataset}"):
                pass
    
    os.makedirs(os.path.join(target_path, 'list'), exist_ok=True)
    #shutil.copy(os.path.join(source_path, 'list', 'dataset.yaml'), os.path.join(target_path, 'list', 'dataset.yaml'))
    shutil.copy(os.path.join(source_path, 'list', 'label_names.yaml'), os.path.join(target_path, 'list', 'label_names.yaml'))


if __name__ == "__main__":
    main()