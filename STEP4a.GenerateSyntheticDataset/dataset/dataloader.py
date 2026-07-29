import datetime
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
from scipy.ndimage import distance_transform_edt
import nibabel as nib
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CopyItemsd,
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


class SkipBadSamplesDataset(torch.utils.data.Dataset):
    """Wraps a dataset; on __getitem__ failure, logs and retries with a
    different random index instead of propagating the exception."""

    def __init__(self, base, max_retries=10):
        self.base = base
        self.max_retries = max_retries

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        for attempt in range(self.max_retries):
            try:
                return self.base[idx]
            except Exception as e:
                print(f"[SKIP-RUNTIME] idx={idx} attempt={attempt}: {e}")
                idx = np.random.randint(0, len(self.base))
        raise RuntimeError(
            f"Failed to load a valid sample after {self.max_retries} retries "
            f"starting from idx={idx}"
        )

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


class GenerateRandomTumorHeatmapd(MapTransform):
    """
    Picks a random point within a binary organ mask and generates a 3D
    Gaussian heatmap centered on that point.
    """

    def __init__(self, organ_key="organ_mask", out_key="heatmap", sigma=8.0, allow_missing_keys=False):
        super().__init__([organ_key], allow_missing_keys)
        self.organ_key = organ_key
        self.out_key = out_key
        self.sigma = sigma  # Controls how "wide" the target region is

    def __call__(self, data):
        d = dict(data)
        mask = d[self.organ_key]

        # Ensure it's a tensor for fast math
        mask_tensor = mask if isinstance(
            mask, torch.Tensor) else torch.tensor(mask)

        # Assuming shape is [Channel, X, Y, Z]
        binary_mask = (mask_tensor[0] > 0).float()
        indices = torch.nonzero(binary_mask)

        if len(indices) == 0:
            # Fallback if no organ is present (blank heatmap)
            heatmap = torch.zeros_like(mask_tensor)
        else:
            # 1. Pick a random voxel inside the organ mask
            rand_idx = torch.randint(0, len(indices), (1,)).item()
            center_point = indices[rand_idx].float()

            # 2. Generate 3D grid
            X, Y, Z = binary_mask.shape
            x_grid, y_grid, z_grid = torch.meshgrid(
                torch.arange(X, device=mask_tensor.device),
                torch.arange(Y, device=mask_tensor.device),
                torch.arange(Z, device=mask_tensor.device),
                indexing='ij'
            )

            # 3. Calculate Gaussian distance
            dist_sq = (x_grid - center_point[0])**2 + (y_grid -
                                                       center_point[1])**2 + (z_grid - center_point[2])**2
            heatmap = torch.exp(-dist_sq / (2 * self.sigma**2))

            # Add channel dimension back -> [1, X, Y, Z]
            heatmap = heatmap.unsqueeze(0)

        d[self.out_key] = heatmap
        return d


class ComputeTSDFd(MapTransform):
    """
    Computes the Truncated Signed Distance Function (TSDF) for binary masks.
    Inside the mask is negative, outside is positive, boundary is 0.
    Output is normalized between [-1, 1].
    """

    def __init__(self, keys, truncation_distance=5.0, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.truncation_distance = truncation_distance

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            mask = d[key]

            # Convert to numpy for fast EDT computation on the CPU DataLoader
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = mask

            # Initialize output tensor
            tsdf_out = np.zeros_like(mask_np, dtype=np.float32)

            # Process each channel independently (usually [C, H, W, D])
            for c in range(mask_np.shape[0]):
                binary_mask = mask_np[c] > 0.5

                # 1. Distance from outside to the boundary (0 inside the mask)
                outside_dist = distance_transform_edt(1 - binary_mask)

                # 2. Distance from inside to the boundary (0 outside the mask)
                inside_dist = distance_transform_edt(binary_mask)

                # 3. Create SDF (positive outside, negative inside)
                sdf = outside_dist - inside_dist

                # 4. Truncate at margins and normalize to [-1, 1] range
                tsdf = np.clip(sdf, -self.truncation_distance,
                               self.truncation_distance)
                tsdf = tsdf / self.truncation_distance

                tsdf_out[c] = tsdf

            # Return tensor in same device/format it arrived in
            d[key] = torch.from_numpy(tsdf_out) if isinstance(
                mask, torch.Tensor) else tsdf_out

        return d


class AugDataset(torch.utils.data.Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.transform(self.base[idx])


METADATA_COLUMNS = ["image", "organ_mask", "organ", "bdmap_id"]

SELECT_COLUMNS = METADATA_COLUMNS


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


def get_healthy_loader(args):

    # ---------- DETERMINISTIC (cached) ----------
    # Everything that should be computed ONCE and persisted to disk cache.
    # Includes both the tumor/image prep AND the mask/heatmap prep.
    train_transforms_deterministic = Compose(
        [
            CacheRunLogger(LoadImageh5d(keys=["image", "organ_mask"])),
            EnsureChannelFirstd(keys=["image", "organ_mask"]),
            Orientationd(keys=["image", "organ_mask"], axcodes="RAS"),
            Spacingd(
                keys=["image", "organ_mask"],
                pixdim=(args.space_x, args.space_y, args.space_z),  # 1,1,1
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
            SpatialPadd(
                keys=["image", "organ_mask"],
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),  # 128,128,128
                mode='constant',
            ),
            ToTensord(keys=["image", "organ_mask"]),
            CastToTyped(keys=["organ_mask"], dtype=np.uint8),
        ]
    )

    train_transforms_stochastic = Compose(
        [
            # 1. Crop the fine-res pair at 128^3 @ 1mm — this fixes the physical patch location
            RandCropByLabelClassesd(
                keys=["image", "organ_mask"],
                label_key="organ_mask",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),  # 128,128,128
                ratios=[1, 10000],
                num_classes=2,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=-1,
            ),

            # 2. Derive m_organ_mask by downsampling the CROPPED organ_mask to 32^3.
            #    This guarantees m_organ_mask covers the exact same physical patch
            #    as organ_mask/image, just at coarser (4mm) resolution.
            CopyItemsd(keys=["organ_mask"], times=1, names=["m_organ_mask"]),
            Resized(
                keys=["m_organ_mask"],
                spatial_size=(args.m_roi_x, args.m_roi_y, args.m_roi_z),  # 32,32,32
                mode="nearest",  # preserve binary/label mask semantics
            ),

            # 3. Generate heatmap from the coarse mask — now guaranteed aligned
            GenerateRandomTumorHeatmapd(
                organ_key="m_organ_mask",
                out_key="heatmap",
                sigma=8.0,
            ),

            # 4. TSDF on the coarse mask
            ComputeTSDFd(keys=["m_organ_mask"]),

            ToTensord(keys=["m_organ_mask", "heatmap"]),
        ]
    )

    train_input = pd.read_csv(os.path.join(
        args.tumor_csv_path, args.dataset_list, f'{args.tumor_datafile}'))



    train_input.dropna(inplace=True)
    # train_input = train_input[train_input["original_shape_LeastAxisLength"]>0.0]

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

    #train_input = filter_valid_samples(train_input)

        
    # 3. NOW MAP ORGAN STRINGS TO INTEGERS
    # train_input["organ"] = train_input.apply(
    #    lambda row: organ_mapping[row["organ"]], axis=1)

    # 5. CONVERT TO DICTIONARY RECORDS FOR MONAI
    data_dicts_train_final = train_input.to_dict("records")

    print('train len {}'.format(len(data_dicts_train_final)))

    if args.persistent_cache:
        print("SETTING UP PERSISTENT CACHE")
        os.makedirs(args.persistent_cache_dir, exist_ok=True)
        cached_dataset = PersistentDataset(
            data=data_dicts_train_final,
            transform=train_transforms_deterministic,
            cache_dir=args.persistent_cache_dir,
        )

        train_dataset = AugDataset(
            cached_dataset, transform=train_transforms_stochastic)
    else:
        train_dataset = Dataset(
            data=data_dicts_train_final, transform=Compose([train_transforms_deterministic, train_transforms_stochastic]))

    train_dataset = SkipBadSamplesDataset(train_dataset)

    train_sampler = DistributedSampler(
        dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
    # breakpoint()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers,
                              collate_fn=list_data_collate, sampler=train_sampler, pin_memory=True, persistent_workers=True)
    return train_loader, train_sampler, len(train_dataset)
    # return train_loader


if __name__ == "__main__":
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()
