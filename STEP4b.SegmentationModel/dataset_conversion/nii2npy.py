import SimpleITK as sitk
import numpy as np
import os
import shutil
import math
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

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

lab_name_list_old = ['kidney_right',
                    'kidney_left',
                    'kidney_lesion',
                    'pancreas',
                    'pancreas_head',
                    'pancreas_body',
                    'pancreas_tail',
                    'pancreatic_lesion',
                    'liver',
                    'liver_segment_1',
                    'liver_segment_2',
                    'liver_segment_3',
                    'liver_segment_4',
                    'liver_segment_5',
                    'liver_segment_6',
                    'liver_segment_7',
                    'liver_segment_8',
                    'liver_lesion',
                    'spleen',
                    'colon',
                    'stomach',
                    'duodenum',
                    'common_bile_duct',
                    'intestine',
                    'aorta',
                    'postcava',
                    'adrenal_gland_left',
                    'adrenal_gland_right',
                    'gall_bladder',
                    'bladder',
                    'celiac_trunk',
                    'esophagus',
                    'hepatic_vessel',
                    'portal_vein_and_splenic_vein',
                    'lung_left',
                    'lung_right',
                    'prostate',
                    'rectum',
                    'femur_left',
                    'femur_right',
                    'superior_mesenteric_artery',
                    'veins']
labels_UFO = {'background': 0,
        'aorta': 1,
        'gall_bladder': 2,
        'kidney_left': 3,
        'kidney_right': 4,
        'postcava': 5,
        'spleen': 6,
        'stomach': 7,
        'adrenal_gland_left': 8,
        'adrenal_gland_right': 9,
        'bladder': 10,
        'celiac_trunk': 11,
        'colon': 12,
        'duodenum': 13,
        'esophagus': 14,
        'femur_left': 15,
        'femur_right': 16,
        'hepatic_vessel': 17,
        'intestine': 18,
        'lung_left': 19,
        'lung_right': 20,
        'portal_vein_and_splenic_vein': 21,
        'prostate': 22,
        'rectum': 23,
        'liver_segment_1': 24,
        'liver_segment_2': 25,
        'liver_segment_3': 26,
        'liver_segment_4': 27,
        'liver_segment_5': 28,
        'liver_segment_6': 29,
        'liver_segment_7': 30,
        'liver_segment_8': 31,
        'pancreas_head': 32,
        'pancreas_body': 33,
        'pancreas_tail': 34,
        }
lab_name_list=list(labels_UFO.keys())

def process_file(file_info):
    name, source_path, target_path, modality = file_info
    img = sitk.ReadImage(os.path.join(source_path, name))
    img = sitk.GetArrayFromImage(img).astype(np.float32)

    lab = []
    for file in list(sorted(lab_name_list)):
        pth = os.path.join(source_path, name.replace('.nii.gz', ''), file+'.nii.gz')
        item = sitk.ReadImage(pth)
        item = sitk.GetArrayFromImage(item).astype(np.int8)
        lab.append(item)
    lab = np.stack(lab, axis=0)  # Makes label multi-channel
                     
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
    img -= mean
    img /= std

    # Pad image and label
    img, lab = pad(img, lab)

    # Save as .npy files
    img, lab = img.astype(np.float32), lab.astype(np.int8)
    # packbits---divides number of ch by 8
    lab = np.packbits(lab, axis=0)
    np.save(os.path.join(target_path, f"{name.replace('.nii.gz', '')}.npy"), img)
    np.save(os.path.join(target_path, f"{name.replace('.nii.gz', '')}_gt.npy"), lab)
    return name  # Return processed file name for logging

# Main function
def main():
    dataset_list = [
        ('abdomenatlas', 'ct'),
    ]

    source_path = '/mnt/ccvl15/psalvad2/UFO_medformer/'
    target_path = '/mnt/ccvl15/psalvad2/UFO_medformer_npy/'

    os.makedirs(os.path.join(target_path), exist_ok=True)

    for dataset, modality in dataset_list:
        names = [name for name in os.listdir(os.path.join(source_path)) if '.nii.gz' in name]
        file_info_list = [(name, source_path, target_path, modality) for name in names]

        # Process in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=16) as executor:  # Adjust `max_workers` as per hardware
            for result in tqdm(executor.map(process_file, file_info_list), total=len(file_info_list), desc=f"Processing {dataset}"):
                pass
    
    os.makedirs(os.path.join(target_path, 'list'), exist_ok=True)
    shutil.copy(os.path.join(source_path, 'list', 'dataset.yaml'), os.path.join(target_path, 'list', 'dataset.yaml'))

if __name__ == "__main__":
    main()