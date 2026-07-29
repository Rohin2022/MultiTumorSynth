import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground, reorient_image
import os
import random
import yaml
import copy
import numpy as np
import pdb
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(16)  # Set the number of threads (adjust to your hardware)

def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):

    imImage = reorient_image(imImage, 'RAI')
    for key in imLabel.keys():
        imLabel[key] = reorient_image(imLabel[key], 'RAI')

    spacing = imImage.GetSpacing()

    mx = []
    for key in imLabel.keys():
        mx.append(sitk.GetArrayFromImage(imLabel[key]).astype(np.uint8).max())
    mx = np.max(mx)

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))


    re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    im_size = re_img_xy.GetSize()
    im_spacing = re_img_xy.GetSpacing()
    re_lab_xy = {}
    for key in imLabel.keys():
        re_lab_xy[key]=ResampleLabelToRef(imLabel[key], re_img_xy, interp=sitk.sitkNearestNeighbor)
        assert re_lab_xy[key].GetSize() == im_size
        assert re_lab_xy[key].GetSpacing() == im_spacing
        
    re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = {}
    for key in imLabel.keys():
        re_lab_xyz[key]=ResampleLabelToRef(re_lab_xy[key], re_img_xyz, interp=sitk.sitkNearestNeighbor)
    
    if np.random.uniform() < 0.25:
        pass
    else:
        if mx == 0:
            pass
        else:
            re_img_xyz, re_lab_xyz = CropForeground(re_img_xyz, re_lab_xyz, context_size=[20, 30, 30])

    sitk.WriteImage(re_img_xyz, '%s/%s.nii.gz'%(save_path, name))
    for key in re_lab_xyz.keys():
        os.makedirs('%s/%s'%(save_path, name), exist_ok=True)
        sitk.WriteImage(re_lab_xyz[key], '%s/%s/%s.nii.gz'%(save_path, name, key))

# Define the processing function
def process_case(name):
    try:
        # Define paths for the output files
        output_ct_path = os.path.join(tgt_path, f"{name}.nii.gz")
        output_label_dir = os.path.join(tgt_path, name)

        # Check if the output CT and all labels already exist
        if os.path.exists(output_ct_path) and all(
            os.path.exists(os.path.join(output_label_dir, f"{lab_name}.nii.gz")) for lab_name in lab_name_list
        ):
            print(f"Skipping {name}: All outputs already exist.")
            return

        # Load the CT image
        img_name = os.path.join(src_path, name, 'ct.nii.gz')
        itk_img = sitk.ReadImage(img_name)

        # Prepare the label dictionary
        lab_dict = {}
        for lab_name in lab_name_list:
            pth = os.path.join(label_path, name, 'segmentations', f"{lab_name}.nii.gz")
            if not os.path.exists(pth):
                pth = os.path.join(label_path, name, 'predictions', f"{lab_name}.nii.gz")
            if not os.path.exists(pth):
                print(f"File {pth} does not exist")
                # Create a zero label
                l = sitk.Image(itk_img.GetSize(), sitk.sitkUInt8)
                l.SetSpacing(itk_img.GetSpacing())  # Match spacing
                l.SetOrigin(itk_img.GetOrigin())    # Match origin
                l.SetDirection(itk_img.GetDirection())  # Match orientation
            else:
                l = sitk.ReadImage(pth)
            lab_dict[lab_name] = l

        # Resample the image and labels
        ResampleImage(itk_img, lab_dict, tgt_path, name, (1.0, 1.0, 1.0))
        print(f"{name} processed successfully.")

    except Exception as e:
        print(f"Error processing {name}: {e}")


if __name__ == '__main__':
    src_path = '/mnt/realccvl15/zzhou82/data/AbdomenAtlas/image_only/AbdomenAtlas1.1Mini/AbdomenAtlas1.1Mini/'
    label_path = '/mnt/realccvl15/zzhou82/data/AbdomenAtlas/mask_only/AbdomenAtlas3.0Mini/AbdomenAtlas3.0Mini/'
    tgt_path = '/mnt/ccvl15/pedro/atlas_300_medformer/'
    cases = '/mnt/sdc/pedro/UCSF/foundational/data_code/atlas_ids_300.csv'
    workers = 8
    lab_name_list = ['kidney_right',
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
    


    name_list = pd.read_csv(cases)["BDMAP_ID"].tolist()

    #name_list = os.listdir(src_path)

    os.makedirs(tgt_path+"/list/", exist_ok=True)
    with open(tgt_path+"/list/dataset.yaml", "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)

    os.chdir(src_path)
    
    with ProcessPoolExecutor(max_workers=workers) as executor:  # Adjust `max_workers` as per your hardware
        # Wrap `executor.map` with `tqdm` to show a progress bar
        for _ in tqdm(executor.map(process_case, name_list), total=len(name_list), desc="Processing Cases"):
            pass


