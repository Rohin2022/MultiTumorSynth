from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstD,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
)
import sys
from copy import copy, deepcopy
import h5py, os
import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..") 

from torch.utils.data import Subset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

class UniformDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey
    
    def dataset_split(self, data, datasetkey):
        self.data_dic = {}
        for key in datasetkey:
            self.data_dic[key] = []
        for img in data:
            key = get_key(img['name'])
            self.data_dic[key].append(img)
        
        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the dataset {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(datasetkey)
    
    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i
    
    def __getitem__(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class UniformCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.datasetkey = datasetkey
        self.data_statis()
    
    def data_statis(self):
        data_num_dic = {}
        for key in self.datasetkey:
            data_num_dic[key] = 0
        for img in self.data:
            key = get_key(img['name'])
            data_num_dic[key] += 1

        self.data_num = []
        for key, item in data_num_dic.items():
            assert item != 0, f'the dataset {key} has no data'
            self.data_num.append(item)
        
        self.datasetlen = len(self.datasetkey)
    
    def index_uniform(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        data_index = np.random.randint(self.data_num[set_index], size=1)[0]
        post_index = int(sum(self.data_num[:set_index]) + data_index)
        return post_index

    def __getitem__(self, index):
        post_index = self.index_uniform(index)
        return self._transform(post_index)
class LoadImaged_BodyMap(MapTransform):
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
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            try:
                loaded_data = self._loader(d[key], reader)
                
                # Check for the corrupted file condition immediately
                if not self._loader.image_only and not isinstance(loaded_data, (tuple, list)):
                    raise RuntimeError(f"Corrupted or invalid NIfTI file: {d[key]}")
                    
            except Exception as e:
                # Log it, then RAISE so the Dataset wrapper can catch it
                print(f"\n[WARNING] Skipping bad file: {d.get('name', d[key])} | Error: {e}")
                raise RuntimeError("Triggering dataset skip")
                
            if self._loader.image_only:
                d[key] = loaded_data
            else:
                d[key] = loaded_data[0]
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists.")
                d[meta_key] = loaded_data[1]
                
        return d

    def label_transfer(self, lbl_dir, shape):
        organ_lbl = np.zeros(shape)
        
        if os.path.exists(lbl_dir + 'liver' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'liver' + '.nii.gz')
            organ_lbl[array > 0] = 1
        if os.path.exists(lbl_dir + 'pancreas' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'pancreas' + '.nii.gz')
            organ_lbl[array > 0] = 2
        if os.path.exists(lbl_dir + 'kidney_left' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'kidney_left' + '.nii.gz')
            organ_lbl[array > 0] = 3
        if os.path.exists(lbl_dir + 'kidney_right' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'kidney_right' + '.nii.gz')
            organ_lbl[array > 0] = 3
        if os.path.exists(lbl_dir + 'liver_tumor' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'liver_tumor' + '.nii.gz')
            organ_lbl[array > 0] = 4
        if os.path.exists(lbl_dir + 'pancreas_tumor' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'pancreas_tumor' + '.nii.gz')
            organ_lbl[array > 0] = 5
        if os.path.exists(lbl_dir + 'pancreas_tumor' + '.nii.gz'):
            array, mata_infomation = self._loader(lbl_dir + 'kidney_tumor' + '.nii.gz')
            organ_lbl[array > 0] = 6

        return organ_lbl, mata_infomation

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
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            try:
                loaded = self._loader(d[key], reader)
            except Exception as e:
                print(f"Failed to load {d.get('name', d[key])}: {e}")
                raise  # re-raise so DataLoader skips this sample properly

            if self._loader.image_only:
                d[key] = loaded
            else:
                if not isinstance(loaded, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = loaded[0]
                if not isinstance(loaded[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = loaded[1]
        return d

import random
from torch.utils.data import Dataset as TorchDataset

class SafeDataset(TorchDataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # If a file is corrupt, try up to 10 random alternative images
        for _ in range(10):
            try:
                return self.dataset[index]
            except Exception:
                # The transform failed. Pick a new random index and try again.
                index = random.randint(0, len(self.dataset) - 1)
        
        # If 10 files in a row fail, your whole dataset is likely unreadable
        raise RuntimeError("Failed to load 10 consecutive images. Check your data path.")

    def __getattr__(self, attr):
        # Pass missing attributes/methods down to the underlying MONAI dataset.
        # This prevents breaking code that looks for `dataset.transform`, `dataset.data`, etc.
        return getattr(self.dataset, attr)
            
def get_loader(args):
    train_transforms = Compose(
        [
            LoadImaged_BodyMap(keys=["image"]),

            EnsureChannelFirstD(keys=["image"]),

            Orientationd(keys=["image"], axcodes="RAS"),

            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode="bilinear",
            ),

            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),

            # removed SpatialPadd + RandCropByPosNegLabeld dependency
            # replace with mask-free cropping
            SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode=["minimum"]),

            RandSpatialCropd(
                keys=["image"],
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
                random_size=False,
            ),

            RandRotate90d(
                keys=["image"],
                prob=0.10,
                max_k=3,
            ),

            ToTensord(keys=["image"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image"]),

            EnsureChannelFirstD(keys=["image"]),

            Orientationd(keys=["image"], axcodes="RAS"),

            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode="bilinear",
            ),

            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),

            RandSpatialCropd(
                keys=["image"],
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
                random_size=False,
            ),

            ToTensord(keys=["image"]),
        ]
    )

    if args.phase == 'train':        
        train_img=[]
        train_lbl=[]
        train_name=[]
        for line in open(os.path.join(args.data_txt_path,  args.dataset_list+'.txt')):
            name = line.strip().split('\t')[0]
            train_img.append(os.path.join(args.data_root_path, name))
            train_lbl.append(os.path.join(args.data_root_path, name + '/segmentations/'))
            train_name.append(name)
        data_dicts_train = [{'image': image, 'name': name}
                    for image, label, name in zip(train_img, train_lbl, train_name)]
        print('train len {}'.format(len(data_dicts_train)))
        # data_dicts_train=data_dicts_train[:10]
        # breakpoint()
    

        if args.uniform_sample:
            train_dataset = UniformDataset(data=data_dicts_train, transform=train_transforms, datasetkey=args.datasetkey)
        else:
            train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)

        train_dataset = SafeDataset(train_dataset)

        train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                                    collate_fn=list_data_collate, sampler=train_sampler, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        return train_loader, train_sampler, len(train_dataset)    
    
    if args.phase == 'validation':
        val_img = []
        val_lbl = []
        val_name = []
        for item in args.dataset_list:
            for line in open(os.path.join(args.data_txt_path,  item, 'real_tumor_val_0.txt')):
                name = line.strip().split()[1].split('.')[0]
                val_img.append(os.path.join(args.data_root_path, line.strip().split()[0]))
                val_lbl.append(os.path.join(args.data_root_path, line.strip().split()[1]))
                val_name.append(name)
        data_dicts_val = [{'image': image, 'name': name}
                    for image, label, name in zip(val_img, val_lbl, val_name)]
        print('val len {}'.format(len(data_dicts_val)))
    
        if args.cache_dataset:
            val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return val_loader, val_transforms, len(val_dataset)
    

def get_key(name):
    ## input: name
    ## output: the corresponding key
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