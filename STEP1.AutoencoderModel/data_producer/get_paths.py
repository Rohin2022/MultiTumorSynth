import os
import json
from tqdm import tqdm

def generate_manifest():
    # Paths matching your Hydra config structure
    data_root = '/projects/bodymaps/Data'
    img_subpath = 'image_only/AbdomenAtlasPro/AbdomenAtlasPro/'
    seg_subpath = 'mask_only/AbdomenAtlasPro/AbdomenAtlasPro/'
    txt_list_path = os.path.join('../', 'cross_eval/recon/AbdomenAtlasProTrain.txt')
    
    organ_names = [
        'spleen', 'bladder', 'gall_bladder', 'esophagus', 'stomach', 'duodenum', 'colon', 'prostate', 'uterus',
        'spleen_lesion', 'bladder_lesion', 'gallbladder_lesion', 'esophagus_lesion', 'stomach_lesion', 'duodenum_lesion',
        'colon_lesion', 'prostate_lesion', 'uterus_lesion'
    ]

    manifest = []

    print("Scanning dataset directories to build a local manifest...")
    with open(txt_list_path, 'r') as f:
        for line in tqdm(f):
            name = line.strip()
            if not name:
                continue
            
            ct_path = os.path.join(data_root, img_subpath, name, 'ct.nii.gz')
            seg_dir = os.path.join(data_root, seg_subpath, name, 'segmentations')
            
            # Check what actually exists inside the segmentations folder ONCE
            available_segs = []
            if os.path.exists(seg_dir):
                files_in_dir = set(os.listdir(seg_dir))
                for organ in organ_names:
                    filename = f"{organ}.nii.gz"
                    if filename in files_in_dir:
                        available_segs.append({
                            "file_path": os.path.join(seg_dir, filename),
                            "organ_name": organ
                        })
            
            manifest.append({
                "name": name,
                "image": ct_path,
                "segmentations": available_segs
            })

    # Save it to your project space
    output_path = os.path.join('dataset_manifest.json')
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Success! Manifest written with {len(manifest)} cases to {output_path}")

if __name__ == "__main__":
    generate_manifest()