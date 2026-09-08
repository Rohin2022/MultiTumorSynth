from monai.transforms import MapTransform
from monai.utils.enums import PostFix
from monai.data.image_reader import ImageReader
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.transforms.io.array import LoadImage, SaveImage
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils.enums import TransformBackends
from monai.transforms.transform import Transform, MapTransform
from monai.config import DtypeLike, KeysCollection
from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset, MetaTensor
from torch.utils.data import Subset
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    SelectItemsd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    CropForegroundd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    CastToTyped,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)
from monai.data import PersistentDataset
import collections.abc
import math
import pickle
import shutil
import sys
import pandas as pd
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
import h5py
import os


import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..")





DEFAULT_POST_FIX = PostFix.meta()


class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(
            reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(
                f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(
            None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(
            meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def register(self, reader: ImageReader):
        self._loader.register(reader)

    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError(
                        "loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(
                        f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        # post_label_pth = d['post_label']
        # with h5py.File(post_label_pth, 'r') as hf:
        #     data = hf['post_label'][()]
        # d['post_label'] = data[0]
        return d


class CombineMasksToTernaryd(MapTransform):
    """
    Combines organ_mask and tumor_mask into a single ternary label:
        0 = background
        1 = organ (organ_mask=1, tumor_mask=0)
        2 = tumor (tumor_mask=1, overrides organ)

    Output key is configurable (default: 'label').
    Input masks are removed from the dict after combination.
    """

    def __init__(
        self,
        organ_key: str = "organ_mask",
        tumor_key: str = "tumor_mask",
        output_key: str = "label",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=[organ_key, tumor_key],
                         allow_missing_keys=allow_missing_keys)
        self.organ_key = organ_key
        self.tumor_key = tumor_key
        self.output_key = output_key

    def __call__(self, data):
        d = dict(data)
        organ = d[self.organ_key]  # shape: (1, H, W, D), values {0, 1}
        tumor = d[self.tumor_key]  # shape: (1, H, W, D), values {0, 1}

        # Binarize both masks defensively (in case of interpolation artifacts)
        organ = (organ > 0.5).to(torch.int64)
        tumor = (tumor > 0.5).to(torch.int64)

        # Build ternary: start with organ=1, then overlay tumor=2
        label = organ.clone()           # 0 or 1
        label[tumor == 1] = 2           # tumor overrides organ

        d[self.output_key] = label

        # Clean up source keys so downstream transforms don't see them
        d.pop(self.organ_key, None)
        d.pop(self.tumor_key, None)

        return d

class AugDataset(torch.utils.data.Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        return self.transform(self.base[idx])


TUMOR_COLUMNS = ['attenuation_delta',
                 'original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_Mean',
                 'original_firstorder_MeanAbsoluteDeviation',
                 'original_firstorder_Median',
                 'original_firstorder_Minimum',
                 'original_firstorder_Range',
                 'original_firstorder_RobustMeanAbsoluteDeviation',
                 'original_firstorder_RootMeanSquared',
                 'original_firstorder_Skewness',
                 'original_firstorder_TotalEnergy',
                 'original_firstorder_Uniformity',
                 'original_firstorder_Variance',
                 'original_glcm_Autocorrelation',
                 'original_glcm_ClusterProminence',
                 'original_glcm_ClusterShade',
                 'original_glcm_ClusterTendency',
                 'original_glcm_Contrast',
                 'original_glcm_Correlation',
                 'original_glcm_DifferenceAverage',
                 'original_glcm_DifferenceEntropy',
                 'original_glcm_DifferenceVariance',
                 'original_glcm_Id',
                 'original_glcm_Idm',
                 'original_glcm_Idmn',
                 'original_glcm_Idn',
                 'original_glcm_Imc1',
                 'original_glcm_Imc2',
                 'original_glcm_InverseVariance',
                 'original_glcm_JointAverage',
                 'original_glcm_JointEnergy',
                 'original_glcm_JointEntropy',
                 'original_glcm_MCC',
                 'original_glcm_MaximumProbability',
                 'original_glcm_SumAverage',
                 'original_glcm_SumEntropy',
                 'original_glcm_SumSquares',
                 'original_gldm_DependenceEntropy',
                 'original_gldm_DependenceNonUniformity',
                 'original_gldm_DependenceNonUniformityNormalized',
                 'original_gldm_DependenceVariance',
                 'original_gldm_GrayLevelNonUniformity',
                 'original_gldm_GrayLevelVariance',
                 'original_gldm_HighGrayLevelEmphasis',
                 'original_gldm_LargeDependenceEmphasis',
                 'original_gldm_LargeDependenceHighGrayLevelEmphasis',
                 'original_gldm_LargeDependenceLowGrayLevelEmphasis',
                 'original_gldm_LowGrayLevelEmphasis',
                 'original_gldm_SmallDependenceEmphasis',
                 'original_gldm_SmallDependenceHighGrayLevelEmphasis',
                 'original_gldm_SmallDependenceLowGrayLevelEmphasis',
                 'original_glrlm_GrayLevelNonUniformity',
                 'original_glrlm_GrayLevelNonUniformityNormalized',
                 'original_glrlm_GrayLevelVariance',
                 'original_glrlm_HighGrayLevelRunEmphasis',
                 'original_glrlm_LongRunEmphasis',
                 'original_glrlm_LongRunHighGrayLevelEmphasis',
                 'original_glrlm_LongRunLowGrayLevelEmphasis',
                 'original_glrlm_LowGrayLevelRunEmphasis',
                 'original_glrlm_RunEntropy',
                 'original_glrlm_RunLengthNonUniformity',
                 'original_glrlm_RunLengthNonUniformityNormalized',
                 'original_glrlm_RunPercentage',
                 'original_glrlm_RunVariance',
                 'original_glrlm_ShortRunEmphasis',
                 'original_glrlm_ShortRunHighGrayLevelEmphasis',
                 'original_glrlm_ShortRunLowGrayLevelEmphasis',
                 'original_glszm_GrayLevelNonUniformity',
                 'original_glszm_GrayLevelNonUniformityNormalized',
                 'original_glszm_GrayLevelVariance',
                 'original_glszm_HighGrayLevelZoneEmphasis',
                 'original_glszm_LargeAreaEmphasis',
                 'original_glszm_LargeAreaHighGrayLevelEmphasis',
                 'original_glszm_LargeAreaLowGrayLevelEmphasis',
                 'original_glszm_LowGrayLevelZoneEmphasis',
                 'original_glszm_SizeZoneNonUniformity',
                 'original_glszm_SizeZoneNonUniformityNormalized',
                 'original_glszm_SmallAreaEmphasis',
                 'original_glszm_SmallAreaHighGrayLevelEmphasis',
                 'original_glszm_SmallAreaLowGrayLevelEmphasis',
                 'original_glszm_ZoneEntropy',
                 'original_glszm_ZonePercentage',
                 'original_glszm_ZoneVariance',
                 'original_ngtdm_Busyness',
                 'original_ngtdm_Coarseness',
                 'original_ngtdm_Complexity',
                 'original_ngtdm_Contrast',
                 'original_ngtdm_Strength'
                 ]


METADATA_COLUMNS = ["image","label","organ", "bdmap_id"]

SELECT_COLUMNS = METADATA_COLUMNS + TUMOR_COLUMNS


class SelectTumorComponentd(MapTransform):
    def __init__(self, keys, component_id_key="component_id", allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.component_id_key = component_id_key

    def __call__(self, data):
        d = dict(data)
        component_id = int(d[self.component_id_key])
        for key in self.key_iterator(d):
            mask = d[key]

            if isinstance(mask, MetaTensor):
                # keep it a MetaTensor so downstream transforms (EnsureChannelFirstd, etc.)
                # can still see original_channel_dim / affine / etc.
                out = (mask.as_tensor() == component_id).to(mask.dtype)
                d[key] = MetaTensor(out, meta=mask.meta)

            elif isinstance(mask, torch.Tensor):
                d[key] = (mask == component_id).to(mask.dtype)

            else:
                arr = np.asarray(mask)
                d[key] = (arr == component_id).astype(arr.dtype)

        return d


 
import datetime

def _log_cache_event(name, data):
    """Prints whenever a deterministic transform actually executes (i.e. cache miss)."""
    ct0 = data.get("ct0_bdmap", "?")
    ct1 = data.get("ct1_bdmap", "?")
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[CACHE-BUILD {ts}] Running '{name}' for pair ({ct0} -> {ct1}) "
          f"— this should print exactly once per pair per persistent cache.",
          flush=True)


class CacheRunLogger(Transform):
    """
    Wraps a deterministic transform and logs every time it is actually
    executed. Must subclass Transform (not just be a plain callable) or
    PersistentDataset._pre_transform will bail out on the first wrapped
    transform in the Compose list and cache nothing (see prior debugging).

    Use this to confirm: deterministic transforms run once per item when
    building/populating the persistent cache, and never again afterward
    (subsequent epochs should only exercise the stochastic crop stage).
    """
    def __init__(self, transform):
        self.transform = transform
        self.name = transform.__class__.__name__

    def __call__(self, data):
        if isinstance(data, list):
            for d in data:
                _log_cache_event(self.name, d)
            return [self.transform(d) for d in data]
        _log_cache_event(self.name, data)
        return self.transform(data)

def get_loader(args):
    train_transforms_deterministic = Compose(
        [
            CacheRunLogger(LoadImageh5d(keys=["image", "tumor_mask", "organ_mask"])),
            SelectTumorComponentd(keys=["tumor_mask"]),
            EnsureChannelFirstd(keys=["image", "tumor_mask", "organ_mask"]),
            CombineMasksToTernaryd(
                organ_key="organ_mask", tumor_key="tumor_mask", output_key="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(
                keys=["image", "label"],
                source_key="label",
                select_fn=lambda x: x > 0,
                margin=64,
                allow_smaller=False,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=(
                args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            ToTensord(keys=["image", "label"]),
            CastToTyped(keys=["label"], dtype=np.uint8),
            SelectItemsd(keys=SELECT_COLUMNS)
        ]
    )



    train_transforms_stochastic = Compose(
        [
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y,
                              args.roi_z),  # 192, 192, 64
                ratios=[0, 1, 1],
                num_classes=3,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=-1,
            ),  # 9
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            # RandShiftIntensityd(
            #     keys=["image"],
            #     offsets=0.10,
            #     prob=0.20,
            # ),
        ]
    )

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),  # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # RandCropByPosNegLabeld_select(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
            #     pos=2,
            #     neg=1,
            #     num_samples=args.num_samples,
            #     image_key="image",
            #     image_threshold=0,
            # ),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y,
                              args.roi_z),  # 192, 192, 64
                ratios=[0, 0, 1],
                num_classes=3,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            ToTensord(keys=["image", "label"]),
            # KeepOnlyTensorsd(keys=["image", "label"])

        ]
    )

    # breakpoint()

    # breakpoint()
    if args.phase == 'train':
        train_input = pd.read_csv(os.path.join(
            args.tumor_csv_path, args.dataset_list, f'{args.tumor_datafile}'))

        train_input.dropna(inplace=True)
        train_input = train_input[train_input["original_shape_LeastAxisLength"]>0.0]

        # print(train_input.columns)
        train_input["tumor_mask"] = train_input.apply(
            lambda row: os.path.join(args.segmentations_root_path, str(
                row["bdmap_id"]), "segmentations", f"{row['organ']}_lesion_labeled.nii.gz"),
            axis=1
        )

        def parseOrganName(organName):
            if (organName == "gallbladder"):
                return 'gall_bladder'
            return organName

        train_input["organ_mask"] = train_input.apply(
            lambda row: os.path.join(args.organ_segmentations_root_path, str(
                row["bdmap_id"]), "segmentations", f"{parseOrganName(row['organ'])}.nii.gz"),
            axis=1
        )

        train_input["image"] = train_input.apply(
            lambda row: os.path.join(args.data_root_path, str(
                row["bdmap_id"]), f"ct.nii.gz"),
            axis=1
        )

        organ_mapping = {
            'spleen': 0,
            'bladder': 1,
            'gallbladder': 2,
            'esophagus': 3,
            'stomach': 4,
            'duodenum': 5,
            'colon': 6,
            'prostate': 7,
            'uterus': 8
        }

        # 1. Drop invalid rows first
        train_input = train_input[train_input["organ"].isin(
            list(organ_mapping.keys()))]
        train_input = train_input[train_input["original_shape_MeshVolume"] > 0.0]

        # 2. CALCULATE WEIGHTS FIRST (While original_shape_MeshVolume is still in true mL)
        vol_cutoff = float(train_input['original_shape_MeshVolume'].quantile(0.98))
        train_input['capped_volume'] = np.clip(
            train_input['original_shape_MeshVolume'], a_min=0, a_max=vol_cutoff)
        train_input['volume_bin'] = pd.cut(
            train_input['capped_volume'], bins=5, labels=False)

        organ_counts = train_input['organ'].value_counts()
        volume_counts = train_input['volume_bin'].value_counts()

        def compute_dual_weight(row):
            f_organ = organ_counts[row['organ']]
            f_vol = volume_counts[row['volume_bin']]
            return (1.0 / np.sqrt(f_organ)) * (1.0 / np.sqrt(f_vol))

        train_input['sample_weight'] = train_input.apply(
            compute_dual_weight, axis=1)

        # 3. NOW MAP ORGAN STRINGS TO INTEGERS
        train_input["organ"] = train_input.apply(
            lambda row: organ_mapping[row["organ"]], axis=1)

        # 4. NOW NORMALIZE NUMERIC FEATURES FOR THE MODEL
        import json
        from pandas.api.types import is_numeric_dtype

        stats_file = f"dataset_norm_stats_{args.results_folder_postfix}.json"
        #exclude_cols = ['volume_bin', 'sample_weight', 'organ',
        #                'tumor_mask', 'organ_mask', 'capped_volume', 'column_task', 'image', 'label',
        #                'diameter_x_mm', 'diameter_y_mm', 'diameter_z_mm', 'volume_ml',
        #                'sphericity', 'surface_volume_ratio', 'elongation', 'flatness', 'max_3d_diameter_mm']

        

        if os.path.exists(stats_file):
            print("Loading existing normalization statistics...")
            with open(stats_file, "r") as f:
                normalization_stats = json.load(f)
        else:
            print("Generating new normalization statistics...")
            normalization_stats = {}
            for key in train_input.columns:
                if key not in TUMOR_COLUMNS: # EXCLUDE ANHTHING THAT is NOT in tumor columns
                    continue

                normalization_stats[key] = {
                    "mean": float(train_input[key].mean()),
                    "std": float(train_input[key].std() + 1e-6)
                }

            with open(stats_file, "w") as f:
                json.dump(normalization_stats, f, indent=4)
            print("Saved new normalization data.")

        # Apply the normalization (using loaded or newly generated stats)
        for key, stats in normalization_stats.items():
            if key in train_input.columns:
                train_input[key] = (train_input[key] -
                                    stats["mean"]) / stats["std"]
                train_input[key] = train_input[key].clip(lower=-3,upper=3)

        # 5. CONVERT TO DICTIONARY RECORDS FOR MONAI
        data_dicts_train = train_input.to_dict("records")

        import nibabel as nib
        from tqdm import tqdm
        data_dicts_train_final = []
        for i, record in tqdm(enumerate(data_dicts_train), total=len(data_dicts_train)):
            tumor_mask = nib.load(record["tumor_mask"])
            organ_mask = nib.load(record["organ_mask"])
            if (tumor_mask.shape != organ_mask.shape):
                print(f"Shape Mismatch: {record['bdmap_id']}")
                print(tumor_mask.shape)
                print(organ_mask.shape)
                print(f"SKIPPING")
                print("==================")

            else:
                data_dicts_train_final.append(record)

        del data_dicts_train
        #data_dicts_train_final = data_dicts_train_final[:10] # TO REMOVE

        print('train len {}'.format(len(data_dicts_train_final)))

        if args.persistent_cache:
            print("SETTING UP PERSISTENT CACHE")
            os.makedirs(args.persistent_cache_dir,exist_ok=True)
            cached_dataset = PersistentDataset(
                data=data_dicts_train_final,
                transform=train_transforms_deterministic,
                cache_dir=args.persistent_cache_dir,
            )

            train_dataset = AugDataset(cached_dataset, transform=train_transforms_stochastic)
        else:
            train_dataset = Dataset(
                data=data_dicts_train_final, transform=Compose([train_transforms_deterministic, train_transforms_stochastic]))
        train_sampler = DistributedSampler(
            dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
        # breakpoint()
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers,
                                  collate_fn=list_data_collate, sampler=train_sampler, pin_memory=args.pin_memory if args.pin_memory else True, persistent_workers=True)
        return train_loader, train_sampler, len(train_dataset)
        # return train_loader

    if args.phase == 'validation':
        # validation dict part
        val_img = []
        val_lbl = []
        val_name = []
        for item in args.dataset_list:
            for line in open(os.path.join(args.data_txt_path,  item, 'real_huge_train_0.txt')):
                name = line.strip().split()[1].split('.')[0]
                val_img.append(args.data_root_path + line.strip().split()[0])
                val_lbl.append(args.data_root_path + line.strip().split()[1])
                val_name.append(name)
        data_dicts_val = [{'image': image, 'label': label, 'name': name}
                          for image, label, name in zip(val_img, val_lbl, val_name)]
        print('val len {}'.format(len(data_dicts_val)))

        if args.cache_dataset:
            val_dataset = CacheDataset(
                data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            val_dataset = Dataset(data=data_dicts_val,
                                  transform=val_transforms)
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return val_loader, val_transforms, len(val_dataset)
        # return val_loader


def get_key(name):
    # input: name
    # output: the corresponding key
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]
    else:
        template_key = name[0:2]
    return template_key


if __name__ == "__main__":
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()
