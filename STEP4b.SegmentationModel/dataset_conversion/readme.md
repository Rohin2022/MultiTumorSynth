
## Data preparation



**1-Dataset format.** Assemble your datasets in the format below. We consider that you have a dataset of CT-Mask pairs (e.g., [MSD](http://medicaldecathlon.com), [AbdomenAtlas 2.0](https://github.com/MrGiovanni/RadGPT/)) and a dataset of CT-Report pairs (e.g., [AbdomenAtlas 2.0](https://github.com/MrGiovanni/RadGPT/), [CT-Rate](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE), [Merlin](https://stanfordaimi.azurewebsites.net/datasets/60b9c7ff-877b-48ce-96c3-0194c8205c40)). In this case, you will need organ segmentation masks for both (see [organ_masks](../organ_masks/README.md) to create them). Organize both in the format below, in different paths (e.g., dataset_masks and dataset_reports). *Repeat steps 2, 3 and 4 (below) for each of the datasets.* We will call the outputs dataset_masks_npz and dataset_reports_npz.

<details>
<summary style="margin-left: 25px;">Dataset format.</summary>
<div style="margin-left: 25px;">

```
/path/to/dataset/
├── BDMAP_0000001
|    ├── ct.nii.gz
│    └── segmentations
│          ├── liver_lesion.nii.gz
│          ├── kidney_lesion.nii.gz
│          ├── pancreatic_lesion.nii.gz
│          ├── aorta.nii.gz
│          ├── gall_bladder.nii.gz
│          ├── kidney_left.nii.gz
│          ├── kidney_right.nii.gz
│          ├── liver.nii.gz
│          ├── pancreas.nii.gz
│          └──...
├── BDMAP_0000002
|    ├── ct.nii.gz
│    └── segmentations
│          ├── liver_lesion.nii.gz
│          ├── kidney_lesion.nii.gz
│          ├── pancreatic_lesion.nii.gz
│          ├── aorta.nii.gz
│          ├── gall_bladder.nii.gz
│          ├── kidney_left.nii.gz
│          ├── kidney_right.nii.gz
│          ├── liver.nii.gz
│          ├── pancreas.nii.gz
│          └──...
...
```
</div>
</details>



Name the tumors you want to predict in the format: {organ}_lesion.nii.gz, and the corresponding organs as {organ}.nii.gz. Exception: for pancreas, name it pancreatic_lesion.nii.gz, and name the organ masks as pancreas.nii.gz. Do not keep lesion masks in the dataset annotated with reports---if you keep empty lesion masks in the dataset annotated with reports, the code will understand that the dataset has no lesion!


**2-Convert to npz.** Convert from nii.gz to npz. This is the standard format for MedFormer and nnU-Net preprocessed.
```bash
cd dataset_conversion
python abdomenatlas_3d.py --src_path /path/to/dataset/ --label_path /path/to/dataset/ --tgt_path /path/to/dataset_b/ --workers 16
python nii2npz.py --src_path /path/to/dataset_b/ --tgt_path /path/to/dataset_npz/
cd ..
```