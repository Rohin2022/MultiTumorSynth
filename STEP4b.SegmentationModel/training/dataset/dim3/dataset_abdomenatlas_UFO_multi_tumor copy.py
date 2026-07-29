import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import yaml
import math
import random
import pdb
from training import augmentation
import os
import yaml
import time
import sys
import pandas as pd
import json
import copy
import shutil

import  importlib
from pathlib import Path
#python dataset_abdomenatlas.py --dataset abdomenatlas --model medformer --dimension 3d --batch_size 2 --crop_on_tumor --save_destination /fastwork/psalvador/JHU/data/atlas_300_medformer_augmented_npy_augmented_multich_crop_on_tumor/ --crop_on_tumor --multi_ch_tumor --workers_overwrite 10

from filelock import FileLock, Timeout
# ---------------------------------------------------------


def get_sample_weight(labels,proportions,class_names,balancer=None,loading_augmented=False):
    weights = []
    tumors = []
    eps = 1e-4
    if balancer is not None:
        tumor_prop = 1-proportions['healthy'] 
        if loading_augmented:
            #read yaml with tumor proportions
            #with open(os.path.join(self.save_destination, 'tumor_proportions.yaml'), 'w') as f:
            with open(os.path.join(balancer.save_destination, 'tumor_proportions.yaml'), 'r') as f:
                proportions = yaml.load(f, Loader=yaml.SafeLoader)
                if proportions is None:
                    raise ValueError('Tumor proportions could not be loaded from yaml file!')
        else:
            #get proportions from balancer
            proportions = balancer.tumor_proportions
        for k,v in proportions.items():
            proportions[k] = v * tumor_prop 
        
    for i, c in enumerate(class_names):
        if c in proportions.keys():
            if labels[i].sum() > 0: #positive sample for class
                weights.append(1.0 / (eps + proportions[c]))
                tumors.append(c) # keep track of which tumors are present in the labels
            else: #negative sample for class
                weights.append(1.0 / (eps + (1-proportions[c])))
        else:
            # If the class is not in proportions, assign a default weight
            weights.append(1.0)
    # Normalize weights to sum to 1*number of classes
    weights = torch.tensor(weights)
    weights = weights / weights.sum() # Normalize to sum to 1
    weights = weights*len(class_names) # Scale by number of classes to keep the relative weights
    
    #print
    #print('Sample tumors:',tumors,' ; sample weights:', weights, flush=True)
    return weights

def get_class_proportions(meta,sample_list,lesion_class_names):
    """
    Get class weights based on the sample list and the meta information.
    This function will return a weight for each class in the sample list.
    :param meta: pandas dataframe with the meta information
    :param sample_list: list of sample names to consider
    :param lesion_classes: list of lesion classes to consider
    :return: list of weights for each class in the sample list
    """
    # Get the meta information for the samples in the sample list
    if isinstance(meta, str):
        meta = pd.read_csv(meta)
    
    # For each sample in sample_list, we add one row to the meta dataframe, accept duplicates!
    tmp = []
    for sample in sample_list:
        # If the sample is not in the meta, add it with zeros for all classes
        tmp.append(meta[meta['BDMAP ID'] == sample]) # get the row for the sample
    meta = pd.concat(tmp, ignore_index=True) # concatenate all the rows for the sample list
    if len(meta) == 0:
        raise ValueError("No samples found in the meta file for the provided sample list.")
    
    print('Lesion class names:', lesion_class_names, flush=True)
    #organs
    organs_lesion_classes = {n.replace('_lesion','').replace('_','').replace('adrenal','adrenal gland'): n for n in lesion_class_names} # remove lesion suffix to get organs

    # Get the counts for each class
    cols = [f'number of {organ} lesion instances' for organ in list(organs_lesion_classes.keys())] # get the columns for each organ lesion instance
    
    #print cols missing from meta
    for col in cols:
        if col not in meta.columns:
            lesion_cols_meta = sorted([col for col in meta.columns if 'number of' in col.lower() and 'lesion instances' in col.lower()])
            raise ValueError(f"Column '{col}' not found in the meta dataframe. Lesion columns found in meta: {lesion_cols_meta}. Please check the meta file or the lesion class names provided.")
    
    # Get the counts for each class in the sample list
    meta = meta[cols]
    
    #make int
    meta = meta.fillna(0).astype(int) # fill NaN with 0 and convert to int
    
    #binarize
    meta = (meta >= 1).astype(int) # convert to binary, 1 if there is at least one instance, 0 otherwise
    
    #sum
    counts = meta.sum(axis=0)  # Sum across all samples for each class
    
    #create a dict of class proportions
    total = len(meta)  # Total number of samples in the sample list
    assert total == len(sample_list), f"Total number of samples in the meta ({total}) does not match the sample list ({len(sample_list)})"
    proportions = {}
    for i, organ in enumerate(list(organs_lesion_classes.keys())):
        proportions[organs_lesion_classes[organ]] = counts[f'number of {organ} lesion instances'] / total if total > 0 else 0  # Avoid division by zero
    #now calculate how many samples have no lesion
    meta['no_lesion'] = (meta[cols].sum(axis=1) == 0).astype(int)
    no_lesion_count = meta['no_lesion'].sum() # how many samples have no lesions
    # Add the no lesion count to the proportions
    proportions['healthy'] = no_lesion_count / total if total > 0 else 0  # Proportion of healthy samples
    
    #print the proportions for debugging
    print('Class proportions:', proportions, flush=True)
    print('Number of samples:', total, flush=True)
    
    return proportions

def balance_classes(class1, class2):
    """
    Balances two lists of strings by repeating the smaller one until its length
    matches the larger one and then shuffling both lists.
    
    Parameters:
        class1 (list of str): The first class.
        class2 (list of str): The second class.
        
    Returns:
        tuple: A tuple (balanced_class1, balanced_class2) with both lists balanced.
    """
    # Determine which list is smaller
    if len(class1) < len(class2):
        # Compute how many times to repeat class1 to match class2's size
        times = len(class2) // len(class1)
        remainder = len(class2) % len(class1)
        balanced_class1 = class1 * times + class1[:remainder]
        balanced_class2 = class2[:]  # Make a copy of class2
    elif len(class2) < len(class1):
        times = len(class1) // len(class2)
        remainder = len(class1) % len(class2)
        balanced_class2 = class2 * times + class2[:remainder]
        balanced_class1 = class1[:]  # Make a copy of class1
    else:
        # If they are already equal in size, just copy them
        balanced_class1, balanced_class2 = class1[:], class2[:]
    
    # Shuffle both lists in place
    random.shuffle(balanced_class1)
    random.shuffle(balanced_class2)
    
    return balanced_class1, balanced_class2
    
#THIS is the 

def clean_ufo(reports,annotated_tumors):
    """
    This function gets a list of reports and removes cases of no interest:
    - We get the healthy patients
    - We get, for each tumor we have annotated organs, all reports that have known tumor size
    - We remove, for organs that have rignr and left (adrenal glands, kidneys), the reports that have unknown sub-segment (not right or left)
    Then, we print the number of useful cases per tumor
    """
    
    
    interest = {}
    
    for organ in annotated_tumors:
        interest[organ] = reports[reports['Standardized Organ'] == organ]
        interest[organ] = interest[organ][interest[organ]['Tumor Size (mm)'] != 'u']
        interest[organ] = interest[organ][interest[organ]['Tumor Size (mm)'] != 'multiple']
        interest[organ] = interest[organ][interest[organ]['Unknow Tumor Size'] == 'no']
        if organ in ['kidney','adrenal_gland','lung','breast','femur']:
            interest[organ] = interest[organ][interest[organ]['Standardized Location'].str.contains('right') | interest[organ]['Standardized Location'].str.contains('left')]
        print('Number of useful cases for %s: %s'%(organ, interest[organ]['BDMAP_ID'].nunique()))

    interest['healthy'] = reports[reports['no lesion'] == True]
    print('Number of healthy cases:', interest['healthy']['BDMAP_ID'].nunique())
    #concat
    tumors_per_type = {}
    for k,v in interest.items():
        tumors_per_type[k]=v['BDMAP_ID'].unique().tolist()
    interest = pd.concat(interest.values())
    interest = interest.drop_duplicates()
    print('Total number of useful cases:', interest['BDMAP_ID'].nunique())
    ids_of_interest = interest['BDMAP_ID'].unique().tolist()
    return interest, ids_of_interest,tumors_per_type
    
def extract_id(image_path):
    """
    Extract the base ID from an image file path by removing the directory,
    extension, and a trailing '_gt' if present.
    """
    file_name = os.path.basename(image_path)
    base = os.path.splitext(file_name)[0]  # removes .npy or .npz
    if base.endswith('_gt'):
        base = base[:-3]  # remove the '_gt'
    return base

def generate_sanity_one_per_tumor(img_list, tumors_per_type):
    """
    Generates a mapping from tumor type to a single image path (from img_list).
    
    For every tumor type (key in tumors_per_type), this function shuffles the associated
    list of BDMAP_IDs for randomness. Then it searches through img_list to find the first
    image whose extracted base ID exactly matches one of these shuffled IDs. It returns a
    dictionary mapping tumor type -> image path.
    
    Parameters:
        img_list (list): List of full image file paths (e.g., self.img_list).
        tumors_per_type (dict): Dictionary with tumor types as keys and lists of BDMAP_ID strings as values.
        
    Returns:
        dict: Mapping of tumor type (key) to the corresponding image path (value) where a match is found.
    """
    sanity_mapping = {}
    
    # Iterate over each tumor type
    for tumor, id_list in tumors_per_type.items():
        # Shuffle IDs to introduce randomness
        shuffled_ids = id_list.copy()
        random.shuffle(shuffled_ids)
        
        candidate_image = None
        # For each id in the shuffled list, search for an exact match in img_list
        for bid in shuffled_ids:
            for image_path in img_list:
                if extract_id(image_path) == bid:
                    candidate_image = image_path
                    break
            if candidate_image is not None:
                break  # A match was found for this tumor type
        
        # If a match was found, add it to our mapping.
        if candidate_image is not None:
            sanity_mapping[tumor] = candidate_image
    
    return sanity_mapping

class AbdomenAtlasDataset(Dataset):
    def __init__(self, args, mode='train', k_fold=10, k=0, seed=0, all_train=False,
                crop_on_tumor=True,
                 save_destination=None,  
                 load_augmented=False,
                 gigantic_length=False,
                 samples_per_epoch=3000,
                 save_augmented=False,
                 tumor_classes= ['adrenal gland', 'bladder', 'colon', 'duodenum',
                    'esophagus', 'gallbladder','prostate','spleen','stomach','uterus'],#we will only crop on these tumors
                 balance_supervision=True,
                 debug=False):    
        
        self.current_epoch = 0
        self.db_count = 0
        self.mode = mode
        self.args = args
        self.load_augmented = load_augmented   
        self.save_counter = 0 
        self.save_destination = save_destination
        self.gigantic_length=gigantic_length
        self.save_augmented = save_augmented
        self.samples_per_epoch = samples_per_epoch
        self.reports = pd.read_csv(args.reports)
        self.main_pid  = os.getpid()
        self.data_root = args.data_root
        print('Reports loaded from:', args.reports, flush=True, file=sys.stderr)
        print('Number of reports:', len(self.reports), flush=True, file=sys.stderr)
        self.zero_masks={}
        assert mode in ['train', 'test']
        self.counter=0

        with open(os.path.join(args.data_root, 'list', 'dataset.yaml'), 'r') as f:
            atlas_name_list = yaml.load(f, Loader=yaml.SafeLoader)
           #print('Number of Atlas Images:', len(atlas_name_list), flush=True, file=sys.stderr)

        with open(os.path.join(args.UFO_root, 'list', 'dataset.yaml'), 'r') as f:
            img_name_list_UFO = yaml.load(f, Loader=yaml.SafeLoader)
           #print('UFO root:', args.UFO_root, flush=True, file=sys.stderr)
           #print('Number of UFO Images:', len(img_name_list_UFO), flush=True, file=sys.stderr)

        #from reports, get only those in the dataset
        
        ids = [case.replace('_0000.nii.gz','').replace('.nii.gz','') for case in img_name_list_UFO]

        if args.ucsf_ids is not None:
            cases = pd.read_csv(args.ucsf_ids)
            cases = cases['BDMAP_ID'].tolist()
            ids = [case for case in ids if case in cases]
            #filter out img_name_list_UFO
            img_name_list_UFO = [case for case in img_name_list_UFO if case in cases]
            
        print()
        print('NUMBER OF SELECTED UFO IDs:', len(ids), flush=True, file=sys.stderr)
        print('NUMBER OF SELECTED UFO IMAGES:', len(img_name_list_UFO), flush=True, file=sys.stderr)
        print()

        #concatenate the two lists 
        img_name_list = atlas_name_list + img_name_list_UFO
        random.Random(seed).shuffle(img_name_list)
        
        if args.load_clip:
            #get only images for which we have embeddings
            embed_list = os.listdir(args.clip_source)
            img_name_list = [case for case in img_name_list if case.replace('_0000.nii.gz','').replace('.nii.gz','') in embed_list]
            img_name_list_UFO = [case for case in img_name_list_UFO if case.replace('_0000.nii.gz','').replace('.nii.gz','') in embed_list]
            atlas_name_list = [case for case in atlas_name_list if case.replace('_0000.nii.gz','').replace('.nii.gz','') in embed_list]
            
        print('NUMBER OF SELECTED IMAGES AFTER EMBEDDING FILTER:', len(img_name_list), flush=True, file=sys.stderr)
        print('NUMBER OF SELECTED UFO IMAGES AFTER EMBEDDING FILTER:', len(img_name_list_UFO), flush=True, file=sys.stderr)

        self.tumor_annotated_seg = {}


        if not all_train:
            length = len(img_name_list)
            test_name_list = img_name_list[:min(200, length//10)]
            train_name_list = list(set(img_name_list) - set(test_name_list))
        else:
            train_name_list = img_name_list
            test_name_list = None
        
        if mode == 'train':
            img_name_list = train_name_list
        else:
            img_name_list = test_name_list
            
        #update ids according to img_name_list
        ids = [case.replace('_0000.nii.gz','').replace('.nii.gz','') for case in img_name_list if case in img_name_list_UFO]

        self.reports = self.reports[self.reports['BDMAP_ID'].isin(ids)]
        self.reports, ids, tumors_per_type = clean_ufo(self.reports,tumor_classes)
        
        #use ids to filter img_name_list and img_name_list_UFO
        img_name_list_UFO = [case for case in img_name_list \
            if case.replace('_0000.nii.gz','').replace('.nii.gz','') in ids]
        img_name_list = [case for case in img_name_list \
            if ((case in img_name_list_UFO) or (case in atlas_name_list))]
        img_name_list_UFO = list(set(img_name_list_UFO))
        img_name_list = list(set(img_name_list))
        assert len(img_name_list_UFO) == len(ids), f'Number of UFO images {len(img_name_list_UFO)} does not match number of ids {len(ids)}'
        assert len(img_name_list) >= len(ids), f'Number of images {len(img_name_list)} is less than number of ids {len(ids)}'
        
            
        

        #print(img_name_list)
        print('Start loading %s data'%self.mode)
        if args.balance_pos_neg and mode == 'train':
            ufo_meta = pd.read_csv(args.UFO_meta)
            ufo_healthy = ufo_meta[ufo_meta['no lesion']==1]['BDMAP ID'].tolist()
            ufo_disease = ufo_meta[ufo_meta['no lesion']==0]['BDMAP ID'].tolist()
            #get only the cases in self.img_name_list_ufo
            ufo_healthy = [i for i in ufo_healthy if i in img_name_list]
            ufo_disease = [i for i in ufo_disease if i in img_name_list]
            assert len(ufo_healthy) > 0
            assert len(ufo_disease) > 0
            print('ufo healthy cases:', len(ufo_healthy))
            print('ufo disease cases:', len(ufo_disease))
            ufo_healthy, ufo_disease = balance_classes(ufo_healthy, ufo_disease)
            print('After balancing ufo, healthy cases:', len(ufo_healthy))
            print('After balancing ufo, disease cases:', len(ufo_disease))
            
            
            atlas_meta = pd.read_csv(args.atlas_meta)
            cols = [col for col in atlas_meta.columns if 'number of' in col.lower() or 'instances' in col.lower()]
            # Filter the rows where all selected columns are 0
            atlas_healthy = atlas_meta[(atlas_meta[cols] == 0).all(axis=1)]
            atlas_diasease = atlas_meta[(atlas_meta[cols] > 0).any(axis=1)]
            #get only the cases in img_name_list
            atlas_healthy = [i for i in atlas_healthy['BDMAP ID'].tolist() if i in img_name_list]
            atlas_diasease = [i for i in atlas_diasease['BDMAP ID'].tolist() if i in img_name_list]
            assert len(atlas_healthy) > 0, 'No healthy cases found in atlas metadata!'
            assert len(atlas_diasease) > 0, 'No disease cases found in atlas metadata!'
            print('Atlas healthy cases:', len(atlas_healthy))
            atlas_healthy, atlas_diasease = balance_classes(atlas_healthy, atlas_diasease)
            print('After balancing Atlas, healthy cases:', len(atlas_healthy))
            print('After balancing Atlas, disease cases:', len(atlas_diasease))
            
            # Combine ufo and Atlas
            img_name_list = ufo_healthy + ufo_disease + atlas_healthy + atlas_diasease
            print('After balancing ufo and Atlas, total image name list length:', len(img_name_list))
            
            
        if mode == 'train' and balance_supervision is True:
            #get the atlas and ufo items following class balancing
            atlas_name_list = [x for x in img_name_list if x in atlas_name_list] # filter to only those in the atlas
            img_name_list_UFO = [x for x in img_name_list if x in img_name_list_UFO] # filter to only those in the UFO dataset
            if len(atlas_name_list)>len(img_name_list_UFO):
                diff = len(atlas_name_list) - len(img_name_list_UFO)
                #randomly select some from ufo
                sampled_items = random.choices(img_name_list_UFO, k=diff)
                img_name_list_UFO = img_name_list_UFO + sampled_items
            elif len(img_name_list_UFO)>len(atlas_name_list):
                #randomly select some from atlas
                diff = len(img_name_list_UFO) - len(atlas_name_list)
                sampled_items = random.choices(atlas_name_list, k=diff)
                atlas_name_list = atlas_name_list + sampled_items
            img_name_list = atlas_name_list + img_name_list_UFO # combine again after balancing

        self.img_list = []
        self.lab_list = []
        self.spacing_list = []

        self.UFO_paths=[]
        self.Atlas_paths=[]

        for name in img_name_list:
            img_name = name + '.npy'
            lab_name = name + '_gt.npy'

            if name in atlas_name_list:
                img_path = os.path.join(args.data_root, img_name)
                lab_path = os.path.join(args.data_root, lab_name)

                #npy or npz?
                if not os.path.exists(img_path):
                    img_path = img_path.replace('.npy','.npz')
                    lab_path = lab_path.replace('.npy','.npz')
                if not os.path.exists(img_path):
                    raise ValueError('Image %s not found in npy nor npz'%img_path)

                self.Atlas_paths.append(img_path)
                self.tumor_annotated_seg[img_path] = True
            elif name in img_name_list_UFO:
                img_path = os.path.join(args.UFO_root, img_name)
                lab_path = os.path.join(args.UFO_root, lab_name)

                #npy or npz?
                if not os.path.exists(img_path):
                    img_path = img_path.replace('.npy','.npz')
                    lab_path = lab_path.replace('.npy','.npz')
                if not os.path.exists(img_path):
                    raise ValueError('Image %s not found in npy nor npz'%img_path)

                self.UFO_paths.append(img_path)
                self.tumor_annotated_seg[img_path] = False
            else:
                raise ValueError('Image %s not in any of the two lists'%name)

            spacing = np.array((1.0, 1.0, 1.0)).tolist()
            self.spacing_list.append(spacing[::-1])  # itk axis order is inverse of numpy axis order

            self.img_list.append(img_path)
            self.lab_list.append(lab_path)

        self.crop_on_tumor = crop_on_tumor
        
        with open(os.path.join(args.data_root, 'list', 'label_names.yaml'), 'r') as f:
            classes = yaml.load(f, Loader=yaml.SafeLoader)
            #sort--we sorted when saving in nii2npy.py
            classes = sorted(classes)
            print('Classes list loaded from %s'%f, flush=True, file=sys.stderr)
            #print('Classes:', classes, flush=True, file=sys.stderr)
            print('Number of Classes:', len(classes), flush=True, file=sys.stderr)

        with open(os.path.join(args.UFO_root, 'list', 'label_names.yaml'), 'r') as f:
            classes_UFO = yaml.load(f, Loader=yaml.SafeLoader)
            #sort--we sorted when saving in nii2npy.py
            classes_UFO = sorted(classes_UFO)

        self.classes = classes
        self.classes_UFO = classes_UFO
        self.num_classes = len(classes)
        self.num_classes_UFO = len(classes_UFO)

        #print('Classes:')
        #for i, c in enumerate(classes):
        #    print(i, c)
        #print('Classes UFO:')
        #for i, c in enumerate(classes_UFO):
            #print(i, c)
            #raise ValueError('Classes UFO are:', classes_UFO)

        lesion_classes = []
        lession_class_names = []
        for i, c in enumerate(classes):
            if 'lesion' in c.lower() and any(tumor.replace(' ','_').replace('_gland','') in c.lower() for tumor in tumor_classes):
                lesion_classes.append(i)
                lession_class_names.append(c)
        self.lesion_classes = lesion_classes
        print('Lesion classes:', lesion_classes)
        print('Lesion class names:', lession_class_names)

        self.saved_count = 0  # Reset the saved count on instantiation
        #print('Load done, length of dataset:', len(self.img_list))

        #check if all ids are in the reports
        # Convert IDs from the DataFrame to a set for faster lookup
        report_ids = set(self.reports['BDMAP_ID'].values)
        # Find IDs not present in the DataFrame
        missing_ids = [id for id in ids if id not in report_ids]
        # Raise an error if there are missing IDs
        if missing_ids:
            raise ValueError(f"IDs not in reports: {missing_ids}. Length of reports: {len(self.reports)}, number of missing ids: {len(missing_ids)}")

        if args.balanced_cropper:
            self.cropper = augmentation.choose_organ_class_match_tumor(class_names=self.classes, lesion_classes=lession_class_names)
            self.cropper_UFO = augmentation.choose_organ_class_match_tumor(class_names=self.classes_UFO, 
                                                                           lesion_classes=lession_class_names,
                                                                           reports = True)
        self.balancing_crops = args.balanced_cropper
        
        if args.class_weights:
            meta = args.atlas_meta
            meta_ufo = args.ufo_meta
            meta = pd.read_csv(meta)
            meta_ufo = pd.read_csv(meta_ufo)
            meta = pd.concat([meta, meta_ufo], ignore_index=True) # combine JHH and UFO meta
            self.class_proportions = get_class_proportions(
                meta=meta, 
                sample_list=img_name_list,
                lesion_class_names=lession_class_names
            )
        else:
            self.class_proportions = None
        
        
        
        #get one ct scan per tumor type for debugging
        
        #self.debug_img_list = generate_sanity_one_per_tumor(self.img_list, tumors_per_type)
        self.debug_img_list_UFO = generate_sanity_one_per_tumor(self.UFO_paths, tumors_per_type)
        self.debug_img_list_Atlas = generate_sanity_one_per_tumor(self.Atlas_paths, tumors_per_type)
        self.crop_registry = os.path.join(self.save_destination,'list','crop_registry.db')
        self.LOCKFILE = str(Path(self.crop_registry)) + ".lock"
        
        if debug and self.mode == 'train':
            self.sanity_path = "./DatasetSanityMultiTumorOnePerTumor"
            self.non_tumor_crop_chance = 0
            self.debug()
        self.sanity_path = "./DatasetSanityMultiTumor"
        self.non_tumor_crop_chance = 0.2
        
        
        if args.model_genesis_pretrain:
            # 1) Find MedFormer/ (it’s four levels up from this file):
            medformer_root = Path(__file__).resolve().parents[3]
            if not medformer_root.joinpath("baselines").is_dir():
                raise ImportError(f"Cannot find baselines/ under {medformer_root}")
            # 2) Ensure Python will search there
            if str(medformer_root) not in sys.path:
                sys.path.insert(0, str(medformer_root))
            # 3) Import the utils module and bind your method
            mg = importlib.import_module("baselines.model_genesis.utils")
            self.generate_pair = mg.generate_one_pair
        else:
            self.generate_pair = None
        
    def debug(self):
        if os.path.exists(self.sanity_path):
            #remove
            os.system('rm -r %s'%self.sanity_path)
            

        gb_val = self.debug_img_list_Atlas.pop("gallbladder")
        self.debug_img_list_Atlas = {"gallbladder": gb_val, **self.debug_img_list_Atlas}
        
        gb_val = self.debug_img_list_UFO.pop("gallbladder")
        self.debug_img_list_UFO = {"gallbladder": gb_val, **self.debug_img_list_UFO}
            
        print('Debugging satrted')
        for tumor, img_name in self.debug_img_list_Atlas.items():
            print()
            print()
            print('Debugging Atlas image:', img_name, 'Tumor type:', tumor)
            _ = self.__getitem__(idx=None, name=img_name)
        for tumor, img_name in self.debug_img_list_UFO.items():
            print()
            print()
            print('Debugging UFO image:', img_name, 'Tumor type:', tumor)
            _ = self.__getitem__(idx=None, name=img_name)
        print('Debugging finished')
        

    def read_report(self, idx):
        id = self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
        if id not in self.reports['BDMAP_ID'].values:
            #print('ID is: ',id)
            raise ValueError('ID is not in the reports:', id, 'Length of reports:', len(self.reports))
            return None #no tumor
        else:
            tumors=self.reports[self.reports['BDMAP_ID']==id]
            #tumors=tumors.to_dict(orient='records')
            return tumors

    def __len__(self):
        if self.mode == 'train':
            if self.gigantic_length:
                return len(self.img_list) * 100000
            else:
                return len(self.img_list)
        else:
            return len(self.img_list)
            

    def __getitem__(self, idx, name=None, BDMAP_ID=None):
        if name is not None:
            idx = self.img_list.index(name)#for debugging, let's you request a sample by name
        if BDMAP_ID is not None:
            name = os.path.join(self.data_root, BDMAP_ID+'.npy')
            if not os.path.exists(name):
                name = os.path.join(self.data_root, BDMAP_ID+'.npz')
            if not os.path.exists(name):
                raise ValueError('Image %s not found in npy nor npz'%name)
            idx = self.img_list.index(name)
        
        #print('Loading:', self.img_list[idx], self.lab_list[idx])
        self.current_sample = self.img_list[idx]
        if self.load_augmented:
            return self.load_augmented_data(idx)
            try:
                return self.load_augmented_data(idx)#loads and returns data already augmented and pre-saved
            except:
                #change index to another one at random
                idx = np.random.randint(len(self.img_list))
                try:
                    return self.load_augmented_data(idx)
                except:
                    print('FAILED TO LOAD AUGMENTED DATA:', self.img_list[idx], self.lab_list[idx])
            #    #print('FAILED TO LOAD AUGMENTED DATA:', self.img_list[idx], self.lab_list[idx])
            #    pass

        try:
            np_img = np.load(self.img_list[idx], mmap_mode='r', allow_pickle=False)
            if '.npz' in self.img_list[idx]:
                np_img = np_img['arr_0']
        except:
            print('Error loading:', self.img_list[idx])
            try:
                np_img = np.load(self.img_list[idx])
                if '.npz' in self.img_list[idx]:
                    np_img = np_img['arr_0']
            except:
                raise ValueError('Error loading:', self.img_list[idx])
        try:
            np_lab = np.load(self.lab_list[idx], mmap_mode='r', allow_pickle=False)
            if '.npz' in self.lab_list[idx]:
                np_lab = np_lab['arr_0']
        except:
            print('Error loading:', self.lab_list[idx])
            try:
                np_lab = np.load(self.lab_list[idx])
                if '.npz' in self.lab_list[idx]:
                    np_lab = np_lab['arr_0']
            except:
                raise ValueError('Error loading:', self.lab_list[idx])

        if self.img_list[idx] in self.UFO_paths:
            classes = self.classes_UFO
        else:
            classes = self.classes

        if np_lab.shape[0] != len(classes):
            # 4. Unpack the bits along the same axis.
            try:
                np_lab = np.unpackbits(np_lab, axis=0)
            except:
                raise ValueError('Unpack bits failed for sample:', self.img_list[idx])
            assert np_lab.shape[0] < len(classes) +10
            np_lab = np_lab[:len(classes)]
            ##print('Label unpacked:', np_lab.shape, flush=True, file=sys.stderr)


        if self.mode == 'train':
            d, h, w = self.args.training_size
            #np_img, np_lab = augmentation.np_crop_3d(np_img, np_lab, [d+40, h+40, w+40], mode='random')

            tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0)
            tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)
            ##print('Time to load data:', time.time() - start, flush=True, file=sys.stderr)
            aug_start = time.time()

            del np_img, np_lab
            tensor_img, tensor_lab = tensor_img.contiguous(), tensor_lab.contiguous()
            #pad with zeros if the image is smaller than the training patch size + a little margin
            tensor_img, tensor_lab = augmentation.pad_volume_pair(tensor_img, tensor_lab, d+40, h+40, w+40)
            
            #print('Starting crop:', flush=True, file=sys.stderr)
            tensor_img, tensor_lab, tumor_dict, selected_tumor, selected_organ = self.crop(tensor_img, tensor_lab, idx, d, h, w)
            #print('Cropped.', flush=True, file=sys.stderr)'
            print('Selected organ:', selected_organ, flush=True, file=sys.stderr)
            selected_organ = canonical_organ(selected_organ)
            print('Selected organ canonical:', selected_organ, flush=True, file=sys.stderr)
            
            if not self.save_augmented:
                #this augmentation is online.
                if np.random.random() < 0.3:
                    tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3])
                if np.random.random() < 0.3:
                    tensor_img = augmentation.brightness_additive(tensor_img, std=0.1)
                if np.random.random() < 0.3:
                    tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5])
                if np.random.random() < 0.3:
                    tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3])
                if np.random.random() < 0.3:
                    tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.5])
                if np.random.random() < 0.3:
                    std = np.random.random() * 0.2 
                    tensor_img = augmentation.gaussian_noise(tensor_img, std=std)
        
        else:
            tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0)#.float()
            tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)#.to(torch.uint8)
            #assert type is int8
            assert tensor_lab.dtype == torch.int8
            assert tensor_img.dtype == torch.float32
            del np_img, np_lab

        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)

        assert tensor_img.shape[1:] == tensor_lab.shape[1:]
        
        #if the item is from the UFO dataset, we convert the labels to the atlas format--negative classes are set to 0, unknown classes are SET TO NAN.
        if self.img_list[idx] in self.UFO_paths:
            #convert to atlas format
            tensor_lab, unk_channels, unk_channels_tensor = self.assign_labels(tensor_lab,idx)
            tumor_volumes_in_crop,tumor_diameters=self.estimate_tumor_volume(idx,tumor_segment_crop=selected_tumor)
            chosen_segment_mask=self.get_chosen_segment_mask(tensor_lab, selected_tumor)
        else:
            unk_channels_tensor = torch.zeros(tensor_lab.shape).type_as(tensor_lab)
            unk_channels = {}
            tumor_volumes_in_crop=[0,0,0,0,0,0,0,0,0,0]
            tumor_diameters=torch.zeros((10,3)).float()
            chosen_segment_mask = torch.zeros(tensor_lab.shape).type_as(tensor_lab)#it is important to define this as 0--or it will cause loss problems!

        dta={'tumor_in_crop':selected_tumor,
             'unknown_per_voxel':unk_channels}
        

        if self.save_augmented:
            self.save(tensor_img, tensor_lab, idx, tumor_dict, dta, unk_channels_tensor=unk_channels_tensor, 
                      tumor_volumes_in_crop=tumor_volumes_in_crop,chosen_segment_mask=chosen_segment_mask,tumor_diameters=tumor_diameters,
                      selected_organ=selected_organ)
        ##print('Time to augment data:', time.time() - aug_start, flush=True, file=sys.stderr)
        
        if self.args.load_clip:
            embedding = self.load_clip(idx, selected_organ)
        
        if self.mode == 'train':
            if self.class_proportions: 
                sample_weights = get_sample_weight(tensor_lab,self.class_proportions,self.classes, balancer=self.cropper if self.balancing_crops else None) 
            else:
                sample_weights = torch.ones_like(tensor_lab)
            ##print('Shapes:', tensor_img.shape, tensor_lab.shape)
            self.SanityAssertOutput(tensor_lab, unk_channels_tensor,torch.tensor(tumor_volumes_in_crop).float(),chosen_segment_mask.float())
            
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float(), torch.from_numpy(tensor_lab).float()
            
            if not self.args.load_clip:
                return tensor_img, tensor_lab, unk_channels_tensor,torch.tensor(tumor_volumes_in_crop).float(),chosen_segment_mask.float(),tumor_diameters.type_as(tensor_img), sample_weights
            else:
                return tensor_img, tensor_lab, unk_channels_tensor,torch.tensor(tumor_volumes_in_crop).float(),chosen_segment_mask.float(),tumor_diameters.type_as(tensor_img), sample_weights, embedding
        else:
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float(), torch.from_numpy(tensor_lab).float()
            return tensor_img, tensor_lab, np.array(self.spacing_list[idx])
        
    def load_clip(self, idx, selected_organ):
        """
        Loads the CLIP embedding for the given organ from the specified sample.
        """
        if not self.balancing_crops:
            raise ValueError('Balancing crops is not enabled, but load_clip is set to True.')
        selected_organ = canonical_organ(selected_organ)
        file = selected_organ.replace('adrenal', 'adrenal_glands').replace('random','full_report').replace('gall_bladder','gallbladder')
        source = self.args.clip_source
        id = self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
        pth = os.path.join(source, id, file+'.npy')
        if not os.path.exists(pth):
            pth = os.path.join(source, id, 'full_report.npy')
        if not os.path.exists(pth):
            raise ValueError(f'Missing clip embedding at {pth}')
        clip = np.load(pth)
        clip = torch.from_numpy(clip).float()
        return clip
        
        
    def clean_subseg_list(self, tumor_segments):
        #split tumor segments that have /
        tmp=[]
        for segment in tumor_segments:
            if pd.isna(segment) or segment == 'u':
                continue
            else:
                sublist=segment.split(' / ')
                if sublist not in tmp:
                    tmp.append(sublist)
        tumor_segments = tmp
        tumor_segments_flat = list(set([item for sublist in tmp for item in sublist]))
        return tumor_segments, tumor_segments_flat
    
    def get_tumor_segment_labels(self, idx):
        """
        This function reads the LLM output for a given report, and its most importat outputs are subseg_with_only_known_sizes and organs_with_only_known_sizes_n_segments.
        These outputs represent organ/organ subsegments that contain tumors but do not contain tumors with unknown size.  
        """
        tumors=self.read_report(idx)
        if tumors is None:
            #no tumor, just do random crop
            retur = {'tumor_segments':[],
                    'tumor_segments_flat':[],
                    'tumor_organs':[],
                    'organs_with_unk_tumor_segment':[],
                    'organs_with_unk_tumor_size':[],
                    'organs_with_only_known_sizes_n_segments':[],
                    'subseg_with_only_known_sizes':[],
                    'subseg_with_unk_tumor_size':[],
                    'subsegs_in_organs_with_unk':[]}
            #print('No tumor found for:', self.img_list[idx], flush=True, file=sys.stderr)
            return retur,tumors
        else:
            #tumor is present
            tumor_segments = tumors['Standardized Location'].tolist()
            tumor_sizes = tumors['Tumor Size (mm)'].tolist()
            tumor_organs = tumors['Standardized Organ'].tolist()
            
            #add organ names to segments
            tmp=[]
            for i,s in enumerate(tumor_segments,0):
                if pd.isna(s) or s == 'u':
                    tmp.append(s)
                else:
                    tmp.append(tumor_organs[i]+'_'+s)
            tumor_segments = tmp


            #check which organs have tumors with unknown size or segment
            organs_with_unk_tumor_segment = []
            organs_with_unk_tumor_size = []
            #and which subsegments have unknown size
            subseg_with_unk_tumor_size = []
            for i in list(range(len(tumor_organs))):
                if pd.isna(tumor_sizes[i]) or tumor_sizes[i] == 'u' or tumor_sizes[i] == 'multiple':
                    organs_with_unk_tumor_size.append(tumor_organs[i])
                    subseg_with_unk_tumor_size.append(tumor_segments[i])
                if pd.isna(tumor_segments[i]) or tumor_segments[i] == 'u':
                    organs_with_unk_tumor_segment.append(tumor_organs[i])

            #check which segments are in an organ with some unknown tumor size or segment
            subsegs_in_organs_with_unk = []
            for i in list(range(len(tumor_organs))):
                #check if the organ is not in the list of organs with unknown tumor segment
                if tumor_organs[i] in organs_with_unk_tumor_segment or tumor_organs[i] in organs_with_unk_tumor_size:
                    subsegs_in_organs_with_unk.append(tumor_segments[i])

            

            tumor_segments, tumor_segments_flat = self.clean_subseg_list(tumor_segments)
            subseg_with_unk_tumor_size, subseg_with_unk_tumor_size_flat = self.clean_subseg_list(subseg_with_unk_tumor_size)
            subsegs_in_organs_with_unk, subsegs_in_organs_with_unk_flat = self.clean_subseg_list(subsegs_in_organs_with_unk)

            tumor_organs = list(set(organ for organ in tumor_organs if not pd.isna(organ) and organ != 'u'))
            organs_with_unk_tumor_segment = list(set(organ for organ in organs_with_unk_tumor_segment if not pd.isna(organ) and organ != 'u'))
            organs_with_unk_tumor_size = list(set(organ for organ in organs_with_unk_tumor_size if not pd.isna(organ) and organ != 'u'))

            #subsegments with only known sizes
            subseg_with_only_known_sizes = list(set(tumor_segments_flat) - set(subseg_with_unk_tumor_size_flat) - set(subsegs_in_organs_with_unk_flat))
            #organs with only known sizes and locations of tumors
            organs_with_only_known_sizes_n_segments = list(set(tumor_organs) - set(organs_with_unk_tumor_segment) - set(organs_with_unk_tumor_size))
            organs_with_only_known_sizes = list(set(tumor_organs) - set(organs_with_unk_tumor_size))

            #for subseg_with_only_known_sizes, you must check tumor_segments, and consider segments that come in pairs
            #check if some sub-segment is in more than one item in the list, if so, merge the items
            tmp=[]
            for segment in subseg_with_only_known_sizes:
                #get all items that contain the segment in the list tumor_segments
                items = [item for item in tumor_segments if segment in item]
                #flatten
                items = list(set([item for sublist in items for item in sublist]))
                #items represent a list of sub-segments that share tumors with segment
                #check if any of them is in the list of prohibted segments
                if any(item in subseg_with_unk_tumor_size_flat for item in items) or \
                   any(item in subsegs_in_organs_with_unk_flat for item in items):
                    continue
                else:
                    tmp.append(items)
            subseg_with_only_known_sizes=tmp
                

            #create a big dict with the variables here
            retur = {'tumor_segments':tumor_segments,
                    'tumor_segments_flat':tumor_segments_flat,
                    'tumor_organs':tumor_organs,
                    'organs_with_unk_tumor_segment':organs_with_unk_tumor_segment,
                    'organs_with_unk_tumor_size':organs_with_unk_tumor_size,
                    'organs_with_only_known_sizes_n_segments':organs_with_only_known_sizes_n_segments,
                    'organs_with_only_known_sizes':organs_with_only_known_sizes,
                    'subseg_with_only_known_sizes':subseg_with_only_known_sizes,
                    'subseg_with_unk_tumor_size':subseg_with_unk_tumor_size,
                    'subsegs_in_organs_with_unk':subsegs_in_organs_with_unk}
            #print('Retur:', retur, flush=True, file=sys.stderr)
            #raise ValueError('You must change the handling of this function output everywhere it is used')
            #print('Tumor Dict:', tumors[['Standardized Location','Tumor Size (mm)','Standardized Organ']])
            #print('subseg_with_only_known_sizes:', retur['subseg_with_only_known_sizes'], flush=True, file=sys.stderr)
            #print('organs_with_only_known_sizes_n_segments:', retur['organs_with_only_known_sizes_n_segments'], flush=True, file=sys.stderr)
            #check if tumor_organs is not nan
            
            #if isinstance(retur['tumor_organs'],str):#if nan it is a normal case
            #    #print('XXXXXXXX Tumor Found for:', self.img_list[idx],f'tumor is: {tumors[["Standardized Location","Tumor Size (mm)","Standardized Organ"]]}', flush=True, file=sys.stderr)
            return retur,tumors

    def get_random_tumor_seg_mask(self, tensor_lab, tumor_segment, exclude=None):
        #print('get_random_tumor_seg_mask - Selected tumor segment:', tumor_segment, flush=True, file=sys.stderr)
        #get the mask for a given segment/organ or segment list
        
        
        if not isinstance(tumor_segment, list):
            tumor_segment = [tumor_segment]

        if len(tumor_segment)==1 and tumor_segment[0] == 'pancreas':
            #pancreas is a special case, we have pancreas labels but they are not in the atlas format
            #we assign all pancreas labels to 1
            tumor_segment = ['pancreas head','pancreas body','pancreas tail']
        if len(tumor_segment)==1 and tumor_segment[0] == 'liver':
            #liver is a special case, we have liver labels but they are not in the atlas format
            #we assign all liver labels to 1
            tumor_segment = ['liver segment 1','liver segment 2','liver segment 3','liver segment 4',
                             'liver segment 5','liver segment 6','liver segment 7','liver segment 8']
        
        #get the labels of the tumor segment
        segment_labels=[seg.replace(' ','_').replace('pancreas','pancreatic').replace('gallbladder','gall_bladder').replace('adrenal gland','adrenal_gland').replace('uterus','prostate') for seg in tumor_segment]
        

        #print('Segment labels are:', segment_labels, flush=True, file=sys.stderr)
        for label in segment_labels:
            if label not in self.classes_UFO:
                raise ValueError('Label %s not in classes_UFO'%label)

        if len(tensor_lab.shape) == 4:
            tensor_lab = tensor_lab.unsqueeze(0)
        assert len(tensor_lab.shape) == 5, f'Label tensor must have 5 dimensions, but got {len(tensor_lab.shape)} dimensions and shape {tensor_lab.shape}'
        if tensor_lab.shape[1] == len(self.classes_UFO):
            classes = self.classes_UFO
        elif tensor_lab.shape[1] == len(self.classes):
            classes = self.classes
        else:
            raise ValueError('Label tensor must have %s channels, but got %s channels'%(len(self.classes_UFO), tensor_lab.shape[1]))
        
        tumor_segment_labels = []
        for i,clss in enumerate(classes,0):
            if clss in segment_labels:
                tumor_segment_labels.append(i)

        #print('Label indices of tumor segment are:', tumor_segment_labels, flush=True, file=sys.stderr)
        
        #get the tumor segment mask
        tumor_segment_mask=[]
        #print('The shape of tensor_lab is:', tensor_lab.shape, flush=True, file=sys.stderr)
        for i in range(tensor_lab.shape[1]):
            if i in tumor_segment_labels:
                tumor_segment_mask.append(tensor_lab[:,i])
        tumor_segment_mask=torch.stack(tumor_segment_mask,axis=0)
        tumor_segment_mask=tumor_segment_mask.sum(0)
        #binarize
        tumor_segment_mask[tumor_segment_mask>0]=1
        #assert tumor_segment_mask.sum().item()!=0.0, f'problem in case {self.current_sample}, tumor segment mask is empty, crop is in {tumor_segment}'
        return tumor_segment_mask

    def get_chosen_segment_mask(self, tensor_lab, tumor_segment):
        if tumor_segment == 'random':
            return torch.zeros_like(tensor_lab).type_as(tensor_lab)
        
        #print('Chosen segment:', tumor_segment, flush=True, file=sys.stderr)
        segment_mask = self.get_random_tumor_seg_mask(tensor_lab, tumor_segment).squeeze(0)
        #apply it to the lesion classes
        segment_mask_lesion_ch = []
        
        lesion_like_name = tumor_segment.replace(' ','_').replace('_right','').replace('_left','').replace('_gland','').replace('gall_bladder','gallbladder')
        lesion_like_name+='_lesion'
        #print('Lesion like name:', lesion_like_name, flush=True, file=sys.stderr)
        added=False
        #print('Segment is:', tumor_segment, flush=True, file=sys.stderr)
        for c in self.classes:
            #print('Classes are:', self.classes, flush=True, file=sys.stderr)
            if (any('segment' in item for item in tumor_segment) or any('liver' in item for item in tumor_segment)) and 'liver_lesion' in c:
                segment_mask_lesion_ch.append(segment_mask)
                #print('Segment added to class:', c, flush=True, file=sys.stderr)
                added=True
            elif (any('head' in item for item in tumor_segment) or any('body' in item for item in tumor_segment) or any('tail' in item for item in tumor_segment) or any('pancreas' in item for item in tumor_segment))\
                  and 'pancreatic_lesion' in c:
                segment_mask_lesion_ch.append(segment_mask)
                added=True
                #print('Segment added to class:', c, flush=True, file=sys.stderr)
            elif  any('kidney' in item for item in tumor_segment) and 'kidney_lesion' in c:
                segment_mask_lesion_ch.append(segment_mask)
                added=True
                #print('Segment added to class:', c, flush=True, file=sys.stderr)
            elif (('gallbladder' in tumor_segment) or ('gall_bladder' in tumor_segment)) and c == 'gallbladder_lesion':
                segment_mask_lesion_ch.append(segment_mask)
                added=True
            elif 'lesion' in c and lesion_like_name==c:
                segment_mask_lesion_ch.append(segment_mask)
                added=True
                #print('Segment added to class:', c, flush=True, file=sys.stderr)
            else:
                segment_mask_lesion_ch.append(torch.zeros_like(tensor_lab[0]).type_as(tensor_lab))
                #print('zero mask added for:', c, flush=True, file=sys.stderr)
        segment_mask_lesion_ch = torch.stack(segment_mask_lesion_ch,axis=0)
        assert segment_mask_lesion_ch.sum().item()!=0.0, f'problem in case {self.current_sample}, chosen segment mask is empty, crop is in {tumor_segment}'
        #assert only one channel is zero:
        for i in range(segment_mask_lesion_ch.shape[0]):
            # For sample i, lo has shape (num_lesion_channels, ...spatial dimensions...)
            lo = segment_mask_lesion_ch[i]
            # Sum over all dimensions except the channel, regardless of the number of spatial dims.
            lo_sum = lo.sum(dim=(-1,-2,-3))
            # Create a boolean mask for channels with any nonzero value.
            active_mask = lo_sum > 0
            active_count = active_mask.sum().item()
            if active_count > 1:  # If more than one lesion channel is active
                # Prepare the names of the lesion channels that are active.
                active_names = [self.classes[j] for j in range(len(self.classes)) if active_mask[j]]
                raise ValueError(
                    f"Error: For sample index {i}, more than one lesion channel has active elements. "
                    f"Active lesion channels: {active_names}"
                    f"lo.sum(dim=(-1,-2,-3)): {lo.sum(dim=(-1,-2,-3))}"
                )
        assert (segment_mask_lesion_ch.sum((-1,-2,-3))>0).float().sum()<=1, 'Only one channel should be non-zero in the chosen segment mask!'
        if not added:
            raise ValueError('No segment added to lesion class:', c, flush=True, file=sys.stderr)
        return segment_mask_lesion_ch
           
    def random_crop(self, tensor_img, tensor_lab, d, h, w):
        tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, [d+40, h+40, w+40], mode='random')
        if self.args.aug_device == 'gpu':
            tensor_img = tensor_img.cuda(self.args.proc_idx).float()
            tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
        if np.random.random() < 0.4:
            tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='center')
        else:
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, [d, h, w], mode='random')
        return tensor_img, tensor_lab
    
    def random_crop_on_tumor(self, tensor_img, tensor_lab, d, h, w, tumor_case=None,
                             tumor_prob=None, foreground_prob=None, background_prob=None,
                             ufo=False):
        if ufo:
            clss = self.classes_UFO
        else:
            clss = self.classes
        
        if self.balancing_crops:
            if ufo:
                cropper = self.cropper_UFO
                lesion_classes = [] # for ufo, our labels here do not contain lesion classes, it only has the ufo classes (organs)
            else:
                cropper = self.cropper
                lesion_classes = self.lesion_classes
        if ufo:
            lesion_classes = [] # for ufo, our labels here do not contain lesion classes, it only has the ufo classes (organs)
        else:
            lesion_classes = self.lesion_classes
        if np.random.random() < 0.4:
            #crop large, then rotate and crop small
            assert len(tensor_lab.shape) == 5
            if tumor_case is None:
                tumor_case = tensor_lab[:,lesion_classes].sum()>0
            if self.balancing_crops:
                tensor_img, tensor_lab, crop_organ = cropper(tensor_img, tensor_lab, d+40, h+40, w+40,tumor_case,
                                                      tumor_prob=tumor_prob, foreground_prob=foreground_prob, 
                                                      background_prob=background_prob,return_crop_organ=True)
            else:
                tensor_img, tensor_lab, crop_organ = augmentation.random_crop_on_tumor(tensor_img, tensor_lab, lesion_classes, 
                                                                           d+40, h+40, w+40,tumor_case,
                                                                           tumor_prob=tumor_prob, 
                                                                           foreground_prob=foreground_prob, 
                                                                           background_prob=background_prob,
                                                                           return_crop_organ=True,
                                                                           class_names=clss)
            if self.args.aug_device == 'gpu':
                tensor_img = tensor_img.cuda(self.args.proc_idx).float()
                tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
            tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='center')
            ##print('Shape of tensor after rotate tumor crop:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)

        else:
            #just crop on tumor
            assert len(tensor_lab.shape) == 5
            if tumor_case is None:
                tumor_case = tensor_lab[:,lesion_classes].sum()>0
            if self.balancing_crops:
                tensor_img, tensor_lab, crop_organ = cropper(tensor_img, tensor_lab, d, h, w,tumor_case,
                                                      tumor_prob=tumor_prob, foreground_prob=foreground_prob,
                                                      background_prob=background_prob, return_crop_organ=True)
            else:
                tensor_img, tensor_lab, crop_organ = augmentation.random_crop_on_tumor(tensor_img, tensor_lab, 
                                                                           lesion_classes, d, h, w,tumor_case,
                                                                           tumor_prob=tumor_prob, 
                                                                           foreground_prob=foreground_prob,
                                                                           background_prob=background_prob,
                                                                           return_crop_organ=True,
                                                                           class_names=clss)
            if self.args.aug_device == 'gpu':
                tensor_img = tensor_img.cuda(self.args.proc_idx).float()
                tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
            ##print('Shape of tensor after tumor crop:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)
        ##print('Crop on tumor successful for:', self.img_list[idx], flush=True, file=sys.stderr)
        return tensor_img, tensor_lab, crop_organ
    
    def random_crop_on_segment(self, tensor_img, tensor_lab, tumor_segment_mask, d, h, w):
        
        if np.random.random() < 0.4:
            #crop large, then rotate and crop small
            assert len(tensor_lab.shape) == 5
            #crop large on segment
            out = augmentation.crop_foreground_3d(tensor_ct=tensor_img, tensor_lab=tensor_lab, foreground=tumor_segment_mask, crop_size=[d+40, h+40, w+40],
                                                  rand=False)
            if isinstance(out, tuple):
                tensor_img, tensor_lab, tumor_segment_mask = out
            else:
                return out
            if self.args.aug_device == 'gpu':
                tensor_img = tensor_img.cuda(self.args.proc_idx).float()
                tensor_lab = tensor_lab.cuda(self.args.proc_idx).long()
            tensor_img, tensor_lab, tumor_segment_mask = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate, foreground=tumor_segment_mask)
            out =  augmentation.crop_foreground_3d(tensor_ct=tensor_img, tensor_lab=tensor_lab, foreground=tumor_segment_mask, crop_size=[d, h, w],
                                                  rand=True)
            
        else:
            out = augmentation.crop_foreground_3d(tensor_ct=tensor_img, tensor_lab=tensor_lab, foreground=tumor_segment_mask, crop_size=[d, h, w],
                                                  rand=True)
        if isinstance(out, tuple):
            tensor_img, tensor_lab, tumor_segment_mask = out
            return tensor_img, tensor_lab
        else:
            return out
            
    def organ_to_subsegment(self,tumor_segment, segments):
        bilateral = ['kidney','adrenal','lung','femur','breast']
        if 'pancrea' in tumor_segment:
            sub_segs=[f for f in segments['subseg_with_only_known_sizes'] if 'pancrea' in f]
            if len(sub_segs)>0:
                tumor_segment=random.choice(sub_segs)
            return tumor_segment
        elif 'liver' in tumor_segment:
            sub_segs=[f for f in segments['subseg_with_only_known_sizes'] if 'segment' in f]
            if len(sub_segs)>0:
                tumor_segment=random.choice(sub_segs)
            return tumor_segment
        elif any(b in tumor_segment for b in bilateral):
            sub_segs = []
            for sublist in segments['subseg_with_only_known_sizes']:
                for f in sublist:
                    if not isinstance(f, str):
                        raise ValueError('Subsegment is not a string:', f)
                    if tumor_segment in f:
                        sub_segs.append(f)
            if len(sub_segs)>0:
                tumor_segment=random.choice(sub_segs)
                return tumor_segment
            else: 
                #print('No subsegment found for:', tumor_segment,'; available subsegments:', segments['subseg_with_only_known_sizes'])
                return None #we cannot crop on kidney, we need to know if it is left or right. None signals a problem and asks for other organ.
        else:
            return tumor_segment #no subsegment, return organ
        
    def crop(self, tensor_img, tensor_lab, idx, d, h, w):
        if self.tumor_annotated_seg[self.img_list[idx]]:
            #for data with per-voxel tumor annotations
            tensor_img, tensor_lab, crop_organ = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w)
            #print('This is an image with per-voxel annotations:'+self.img_list[idx], flush=True, file=sys.stderr)
            return tensor_img, tensor_lab, None, None, crop_organ
        
        else:
            if not self.crop_on_tumor:
                raise ValueError('You should set crop on tumor for UFO data, as the tumor crops here are necessary for the report loss. And it makes little sense to not crop on tumor for Atlas and crop on tumor for UFO.')
            #print('This is an image with tumor annotations from reports:'+self.img_list[idx], flush=True, file=sys.stderr)
            #data without per-voxel tumor annotations, just reports mentioning tumors
            segments,tumor_dict=self.get_tumor_segment_labels(idx)

            segment_options=segments['organs_with_only_known_sizes']
            #print('Segment options:', segment_options, flush=True, file=sys.stderr)
            if len(segment_options)==0:
                #no tumor, crop like a per-voxel annotated case.
                tensor_img, tensor_lab, crop_organ = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w, tumor_case=False, ufo=True)
                #print('No segment options, random crop', flush=True, file=sys.stderr)
                return tensor_img, tensor_lab, tumor_dict, 'random', crop_organ
            
            #crop around the tumor organ/subsegment 
            # 80% chance of cropping on tumor:
            if np.random.random() < self.non_tumor_crop_chance:
                #foreground crop or background crop
                tensor_img, tensor_lab, crop_organ = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w, tumor_case=False,
                                                                   tumor_prob=0, foreground_prob=0.5, background_prob=0.5,
                                                                   ufo=True)
                #print('Random crop by chance', flush=True, file=sys.stderr)
                return tensor_img, tensor_lab, tumor_dict, 'random', crop_organ
                
            else:
                out = None
                
                while len(segment_options)>=1 and (not isinstance(out, tuple) or tumor_segment_mask.sum().item()==0.0):
                    out = None
                    #tumor crop
                    #randomly pick an organ
                    if not self.args.balanced_cropper:
                        tumor_organ = random.choice(segment_options)
                    else:
                        #get class indices
                        tumor_organ = self.cropper_UFO.choose_tumor_class(segment_options, update_EMA=False)
                        
                    #print(f'Chosen organ to crop on: {tumor_organ}', flush=True, file=sys.stderr)
                        
                    #try to get its segment
                    tumor_segment = self.organ_to_subsegment(tumor_organ, segments)
                    #print('Chosen segment to crop on:', tumor_segment, flush=True, file=sys.stderr)
                    
                    if tumor_segment is None:
                        #this indicates an impossible crop, like crop on kidney, but we do not know if it is left or right
                        #then, we remove the organ from the segment_options and continue
                        segment_options = [seg for seg in segment_options if seg not in [tumor_organ]]
                        continue
                    #print('Chosen segment a:', tumor_segment, flush=True, file=sys.stderr)
                    
                    #get the mask for the tumor segment
                    tumor_segment_mask=self.get_random_tumor_seg_mask(tensor_lab, tumor_segment)
                    if tumor_segment_mask.sum().item()==0.0:
                        #if we cannot segment the organ segment containing the tumor, we try another organ
                        #print('Zero mask for:', tumor_segment, flush=True, file=sys.stderr)
                        self.zero_masks[self.current_sample]=tumor_segment
                        #save as yaml
                        with open('zero_masks.yaml', 'w') as f:
                            yaml.dump(self.zero_masks, f)
                        segment_options = [seg for seg in segment_options if seg not in [tumor_organ]]
                        continue #try next segment
                    
                    out = self.random_crop_on_segment(tensor_img, tensor_lab, tumor_segment_mask, d, h, w)
                    
                    #suceeded_crop = False
                    if isinstance(out, tuple):
                        #we got a successful crop. 
                        tensor_img, tensor_lab = out
                        if tensor_lab.sum().item()==0.0:
                            raise ValueError('Crop produced a zero mask for:', tumor_segment, flush=True, file=sys.stderr)
                        if self.args.balanced_cropper:
                            self.cropper_UFO.update_crop_proportions_EMA(self.cropper_UFO.tumor_proportions, tumor_segment)
                        print('>>>>>>>>> Cropped around tumor segment:', tumor_segment, flush=True, file=sys.stderr)
                        crop_organ = tumor_organ
                        return tensor_img, tensor_lab, tumor_dict, tumor_segment, tumor_organ
                    else:
                        #this is a failed crop, we try another segment
                        print('Failed crop for:', tumor_segment,'out is:',out, flush=True, file=sys.stderr)
                        if True:#debug
                            import csv
                            # Append the failure details to the CSV.
                            with open('failed_crops_multi_tumor.csv', 'a', newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                # Write a row with image ID, tumor_segment, and out. Convert 'out' to string if needed.
                                writer.writerow([self.img_list[idx], tumor_segment, str(out)])
                        segment_options = [seg for seg in segment_options if seg not in [tumor_segment]]
                        continue

                if len(segment_options)==0:
                    #if we cannot crop in any tumor, we fall back to random crop
                    tensor_img, tensor_lab, crop_organ = self.random_crop_on_tumor(tensor_img, tensor_lab, d, h, w, tumor_case=False,
                                                                tumor_prob=0, foreground_prob=0.5, background_prob=0.5,
                                                                   ufo=True)
                    print(f'----------- Tumor crop failed for {self.img_list[idx]}', flush=True, file=sys.stderr)
                    return tensor_img, tensor_lab, tumor_dict, 'random', crop_organ
                
                raise ValueError('You should not be here, this is a bug')


    def save(self, tensor_img, tensor_lab, idx, tumor_dict=None, dta=None, unk_channels_tensor=None,
            tumor_volumes_in_crop=None, chosen_segment_mask=None,tumor_diameters=None,
            selected_organ=None):
        """
        Saves the augmented image/label pair to disk if a destination was specified.
        Uses numpy .npy format and keeps the original naming scheme.
        """
        os.makedirs(self.save_destination, exist_ok=True)

        # Keep the same filenames as the original
        base_img_name = os.path.basename(self.img_list[idx])   # e.g. "xxx.npy"
        base_lab_name = os.path.basename(self.lab_list[idx])   # e.g. "xxx_gt.npy"

        img_filename = os.path.join(self.save_destination, base_img_name)
        lab_filename = os.path.join(self.save_destination, base_lab_name)

        np_img = tensor_img.cpu().numpy()
        np_lab = tensor_lab.cpu().numpy().astype(np.bool_)  

        #print('Number of labels:',np_lab.shape[0], flush=True, file=sys.stderr)
        np_lab = np.packbits(np_lab, axis=0) #from bool to uint8 - reduce the channels dimension by 8. Each voxel is saved a a byte anyway. This reduce the size of the file by 8.
        ##print('Shape of label after packing:', np_lab.shape)
        
        img_filename = img_filename.replace('.npz','.npy')
        lab_filename = lab_filename.replace('.npz','.npy')

        # Save as .npy
        np.save(img_filename, np_img)
        np.save(lab_filename, np_lab)
        #print('Saved:',img_filename, flush=True, file=sys.stderr)
        #print('Saved:',lab_filename, flush=True, file=sys.stderr)


        if unk_channels_tensor is not None:
            unk_ch = unk_channels_tensor.cpu().numpy().astype(np.bool_)
            unk_channels_tensor = np.packbits(unk_ch, axis=0)
            np.save(lab_filename.replace('.npy','_unk.npy'), unk_channels_tensor)

        if chosen_segment_mask is not None:
            chosen_segment_mask = chosen_segment_mask.cpu().numpy().astype(np.bool_)
            chosen_segment_mask = np.packbits(chosen_segment_mask, axis=0)
            np.save(lab_filename.replace('.npy','_chosen_tumor_segment.npy'), chosen_segment_mask)

        if tumor_dict is not None:
            tumor_dict.to_csv(os.path.join(self.save_destination, img_filename.replace('.npy','.csv')), index=False)
        if dta is not None:
            with open(os.path.join(self.save_destination, img_filename.replace('.npy','.json')), "w") as f:
                json.dump(dta, f)
        if tumor_volumes_in_crop is not None:
            with open(os.path.join(self.save_destination, img_filename.replace('.npy','_tumor_volumes.json')), "w") as f:
                json.dump(tumor_volumes_in_crop, f)
        
        if tumor_diameters is not None:
            tumor_diameters = tumor_diameters.cpu().numpy().tolist()
            with open(os.path.join(self.save_destination, img_filename.replace('.npy','_tumor_diameters.json')), "w") as f:
                json.dump(tumor_diameters, f)

        if self.cropper is not None:
            #update the sql registry with selected_organ
            id = self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
            self.safe_upsert(id, selected_organ)

        self.save_counter += 1
        
    
    def safe_upsert(self, key: str, value: str, retries: int = 3) -> None:
        """
        Atomically “upsert” a single key/value pair into the YAML store.

        The record lives in  <self.yaml_dir>/<KEY>_crop.yaml
            └── content:  - <value>

        Concurrent writers are serialised with a .lock file next to the YAML.
        """
        print('Attempting to upsert:', key, value, flush=True, file=sys.stderr)
        yaml_dir   = Path(self.save_destination)          # make sure you set this attr!

        yml_path   = yaml_dir / f"{key}_crop.yaml"
        lock_path  = yml_path.with_suffix(".lock")
        lock       = FileLock(lock_path, timeout=30)

        for attempt in range(retries):
            try:
                with lock:                        # one writer at a time
                    # 1. write to a temp file first
                    tmp = yml_path.with_suffix(".tmp")
                    with tmp.open("w") as f:
                        yaml.safe_dump([value], f, default_flow_style=False)

                    # 2. atomic replace → readers never see a partial file
                    os.replace(tmp, yml_path)
                print(f"[yaml‑upsert] {key} → {value}", flush=True, file=sys.stderr)
                return                            # ☑ success
            except Timeout:
                # another writer is busy, wait a little then retry
                time.sleep(0.2 * (attempt + 1))

        # all retries exhausted
        raise RuntimeError(f"Could not write {key=} after {retries} attempts")
                    
    def load_augmented_data(self, idx):
        # We'll assume the user has already run the dataset once to save the augmented data.
        if self.save_destination is None:
            raise ValueError("load_augmented=True but save_destination=None. Cannot load augmented data.")
        
        #print('Loading augmented data for:', self.img_list[idx], flush=True, file=sys.stderr)

        start = time.time()

        # Derive the filenames from the original naming scheme
        base_img_name = os.path.basename(self.img_list[idx])    # e.g. "xxx.npy"
        base_lab_name = os.path.basename(self.lab_list[idx])    # e.g. "xxx_gt.npy"
        
        # Replace npz by npy
        base_img_name = base_img_name.replace('.npz', '.npy')
        base_lab_name = base_lab_name.replace('.npz', '.npy')

        aug_img_path = os.path.join(self.save_destination, base_img_name)
        aug_lab_path = os.path.join(self.save_destination, base_lab_name)
        
        # Load the augmented data
        np_img = np.load(aug_img_path, allow_pickle=False)  # shape as saved
        tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0).float()
        ##print('Time to load augmented image:', time.time() - start, flush=True, file=sys.stderr)
        start = time.time()
        ##print shapes
        ##print('Shape:', np_img.shape, np_lab.shape)

        # Convert to torch
        # The code expects image to be float32 and label int8 (for checking).
        np_lab = np.load(aug_lab_path, allow_pickle=False)  # uint8

        # 4. Unpack the bits along the same axis.
        if np_lab.shape[0] != len(self.classes):
            #print('Shape before unpack:', np_lab.shape, flush=True, file=sys.stderr)
            start_unpack = time.time()
            # 4. Unpack the bits along the same axis.
            np_lab = np.unpackbits(np_lab, axis=0)
            assert np_lab.shape[0] < self.num_classes +10
            assert np_lab.shape[0] >= self.num_classes
            np_lab = np_lab[:self.num_classes]
            #print('Label unpacked:', np_lab.shape, flush=True, file=sys.stderr)
            ##print('Time to unpack:', time.time() - start_unpack, flush=True, file=sys.stderr)

        tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)

        ##print('Time to load augmented label:', time.time() - start, flush=True, file=sys.stderr)
        aug_start = time.time()

        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)

        
        if self.mode == 'train':
            #this augmentation is online.
            if np.random.random() < 0.3:
                tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3])
            if np.random.random() < 0.3:
                tensor_img = augmentation.brightness_additive(tensor_img, std=0.1)
            if np.random.random() < 0.3:
                tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5])
            if np.random.random() < 0.3:
                tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3])
            if np.random.random() < 0.3:
                tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.5])
            if np.random.random() < 0.3:
                std = np.random.random() * 0.2 
                tensor_img = augmentation.gaussian_noise(tensor_img, std=std)
            ##print('Applied augmentation online!')
        
        ##print('Augmentation deactivated!')

        # You can still call save_sanity_check if desired
        #self.save_sanity_check(tensor_img, tensor_lab, idx)

        ##print('Time augmenting data:', time.time() - aug_start, flush=True, file=sys.stderr)

        tensor_img = tensor_img.squeeze(0)

        ##print('Shapes:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)

        if self.mode == 'train':
            if self.img_list[idx] not in self.UFO_paths:
                #annotated per-voxel, no unknnown voxel
                unk_channels_list=torch.zeros(tensor_lab.shape).type_as(tensor_lab)
                tumor_volumes_in_crop=[0,0,0,0,0,0,0,0,0,0]
                tumor_diameters=torch.zeros((10,3)).float()
                chosen_segment_mask=torch.zeros_like(tensor_lab).type_as(tensor_lab)
            else:
                #try loading unk_channels_list if saved
                unk_pth=aug_lab_path.replace('_gt.npy','_gt_unk.npy')
                if os.path.exists(unk_pth):
                    unk_channels_tensor = np.load(unk_pth, allow_pickle=False)
                    if unk_channels_tensor.shape[0] != len(self.classes):
                        unk_channels_tensor = np.unpackbits(unk_channels_tensor, axis=0)
                        unk_channels_tensor = unk_channels_tensor[:len(self.classes)]
                    unk_channels_list = torch.from_numpy(unk_channels_tensor)
                    #print(f'----------------UNK WAS LOADED FROM {unk_pth}', flush=True, file=sys.stderr)
                else:
                    unk_channels_list=self.define_unknown_voxels(tensor_lab,idx)
                    #print('----------------UNK WAS CREATED', flush=True, file=sys.stderr)
                #load the json file
                with open(os.path.join(self.save_destination, base_img_name.replace('.npy','.json')), "r") as f:
                    dta=json.load(f)
                tumor_volumes_in_crop,tumor_diameters=self.estimate_tumor_volume(idx,tumor_segment_crop=dta['tumor_in_crop'])
                if os.path.exists(aug_lab_path.replace('.npy','_chosen_tumor_segment.npy')):
                    chosen_segment_mask = np.load(aug_lab_path.replace('.npy','_chosen_tumor_segment.npy'), allow_pickle=False)
                    if chosen_segment_mask.shape[0] != len(self.classes):
                        chosen_segment_mask = np.unpackbits(chosen_segment_mask, axis=0)
                        chosen_segment_mask = chosen_segment_mask[:self.num_classes]
                    chosen_segment_mask = torch.from_numpy(chosen_segment_mask)
                else:
                    chosen_segment_mask=self.get_chosen_segment_mask(tensor_lab, dta['tumor_in_crop'])
            #print('LOADED AUGMENTED DATA', tensor_lab.shape, 'From:', os.path.join(self.save_destination, base_lab_name), flush=True, file=sys.stderr)
            
            if self.class_proportions: 
                sample_weights = get_sample_weight(tensor_lab,self.class_proportions,self.classes, balancer=self.cropper if self.balancing_crops else None) 
            else:
                sample_weights = torch.ones_like(tensor_lab)
            
            self.SanityAssertOutput(tensor_lab, unk_channels_list, torch.tensor(tumor_volumes_in_crop).float(), chosen_segment_mask.float())
            
            
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float(), torch.from_numpy(tensor_lab).float()
            
            
            if self.args.load_clip:
                if self.cropper is None:
                    raise ValueError('We are only using CLIP for the custom cropper mode.')
                #get the organ from the crop registry
                id = self.img_list[idx][self.img_list[idx].find('BDMAP_'):self.img_list[idx].find('BDMAP_')+len('BDMAP_00001111')]
                yml_path = Path(self.save_destination) / f"{id}_crop.yaml"
                with yml_path.open("r") as f:
                    data = yaml.safe_load(f) or []
                selected_organ = data[0]
                embedding = self.load_clip(idx, selected_organ)
                return tensor_img, tensor_lab, unk_channels_list, torch.tensor(tumor_volumes_in_crop).float(), chosen_segment_mask.float(), tumor_diameters.type_as(tensor_img), sample_weights, embedding
            else:
                return tensor_img, tensor_lab, unk_channels_list, torch.tensor(tumor_volumes_in_crop).float(), chosen_segment_mask.float(), tumor_diameters.type_as(tensor_img), sample_weights
        else:
            #print('LOADED AUGMENTED DATA', tensor_lab.shape, 'From:', os.path.join(self.save_destination, base_lab_name), flush=True, file=sys.stderr)
            
            if self.generate_pair is not None:
                tensor_img, tensor_lab = self.generate_pair(tensor_img.cpu().numpy())
                tensor_img, tensor_lab = torch.from_numpy(tensor_img).float(), torch.from_numpy(tensor_lab).float()
            return tensor_img, tensor_lab, np.array(self.spacing_list[idx])

    def save_sanity_check(self, img, lab, idx):
        """Save the image and labels to NIfTI format for sanity checking."""
        if self.saved_count < 10:
            save_dir = './SanityCheck'
            os.makedirs(save_dir, exist_ok=True)

            img_folder = os.path.join(save_dir, f'img{self.saved_count + 1}')
            os.makedirs(img_folder, exist_ok=True)

            # Save the image
            img_nifti = sitk.GetImageFromArray(img.squeeze().cpu().numpy())
            ##print shape
            ##print('Shape:', img.squeeze().cpu().numpy().shape)
            img_nifti.SetSpacing(self.spacing_list[idx])
            sitk.WriteImage(img_nifti, os.path.join(img_folder, 'CT.nii.gz'))

            # Save the labels
            for i, cls in enumerate(self.classes):
                label_array = (lab[i].squeeze().cpu().numpy()).astype(np.int8)
                if label_array.max() > 0:  # Save only if the label exists
                    label_nifti = sitk.GetImageFromArray(label_array)
                    label_nifti.SetSpacing(self.spacing_list[idx])
                    sitk.WriteImage(label_nifti, os.path.join(img_folder, f'{cls}.nii.gz'))

            self.saved_count += 1
    
    def assign_labels(self, tensor_lab, idx):
        """
        UFO data is not annotated per-voxel for some classes, making classes and classes_UFO missmatch. This function adds zero channels for missing classes, and creates a unk_channels dict explining which class is UNKNOWN.
        Some classes are missing but we know they are truly zero, so we do not add them to unk_channels.
        If the missing class is not a lesion (e.g., an organ we do not have pseudo-annotations for) we assign it to unk_channels.
        If it is a lesion, we check tumor_dict. tumor_dict, extracted from the report, explains which organ/segments present tumors. We check if these organs/segments are present in tensor_lab (cropped).
        Tumor labels with a corresponding tumor segment in the crop -> assign to unk_channels (we do not know where the tumor is).
        Tumor labels withour corresponding tumor segment in the crop -> assign label 0 (negative for tumor in the crop).
        """
        clss_to_idx = {clss: i for i, clss in enumerate(self.classes)}
        clss_UFO_to_idx = {clss: i for i, clss in enumerate(self.classes_UFO)}
        all_data,tumor_dict=self.get_tumor_segment_labels(idx)
        tumor_segments=all_data['tumor_segments']
        
        #get the organ subsegments for 
        #print('Tumor segments:', tumor_segments, flush=True, file=sys.stderr)
        #print('Tumor organs:', all_data['tumor_organs'], flush=True, file=sys.stderr)
        for tumor_organ in all_data['tumor_organs']:
            if isinstance(tumor_organ,str) and tumor_organ=='liver':
                if not any('segment' in item for item in tumor_segments):
                    if 'liver' not in tumor_segments:
                        tumor_segments.append('liver')
            elif isinstance(tumor_organ,str) and tumor_organ=='pancreas':
                if not any('head' in item for item in tumor_segments) and not any('body' in item for item in tumor_segments) and not any('tail' in item for item in tumor_segments):
                    if 'pancreas' not in tumor_segments:
                        tumor_segments.append('pancreas')         
            elif isinstance(tumor_organ,str) and tumor_organ in ['kidney','adrenal gland','breast','lung']:
                flag = False
                for seg in tumor_segments:
                    if isinstance(seg,list):
                        for s in seg:
                            if 'right' in s or 'left' in s:
                                flag = True
                    else:
                        if 'right' in seg or 'left' in seg:
                            flag = True
                if not flag:
                    tumor_segments.append(tumor_organ+'_right')
                    tumor_segments.append(tumor_organ+'_left')
                    #raise ValueError('We do not know if the tumor is in the left or right kidney/adrenal gland/breast/lung. We cannot crop on this organ.')
            else:
                tumor_segments.append(tumor_organ)#if the organ has no segment, we add the organ itself.

        #flatten the list of lists
        tmp=[]
        for item in tumor_segments:
            if isinstance(item, list):
                for subitem in item:
                    tmp.append(subitem)
            else:
                if item == 'pancreas':
                    for it in ['pancreas head','pancreas body','pancreas tail']:
                        tmp.append(it)
                elif item == 'liver':
                    for it in ['liver segment 1','liver segment 2','liver segment 3','liver segment 4',
                               'liver segment 5','liver segment 6','liver segment 7','liver segment 8']:
                        tmp.append(it)
                else:
                    tmp.append(item)
        tumor_segments=tmp

        tumor_segments=list(set(tumor_segments))
        #convert to standard label names:
        tumor_segments=[seg.replace(' ','_').replace('pancreas','pancreatic').replace('gallbladder','gall_bladder').replace('adrenal gland','adrenal_gland') for seg in tumor_segments]
        #assert these are in classes:
        for seg in tumor_segments:
            if seg!='uterus' and seg not in self.classes_UFO:
                raise ValueError('Segment not in classes:',seg)
        
        #tumor_segments represents all organ/subsegments with tumors in the whole ct
        #which lesion classes to add unk? check which of the tumor_segments are in the crop.
        zeros = torch.zeros((tensor_lab.shape[-3],tensor_lab.shape[-2],tensor_lab.shape[-1])).type_as(tensor_lab)
        unk_segments={}
        #this variable will create a mask of the segments in the crop that have tumors in unknown locations (report annotation)
        
        unk_lesions=[]
        for seg in tumor_segments:
            seg_idx=clss_UFO_to_idx[seg.replace('uterus','prostate')]
            #print('Segment being added to unk:',seg,'Its UFO index is:',seg_idx, flush=True, file=sys.stderr)
            if tensor_lab[seg_idx].max()>0:#organ sub-segment with tumor inside the crop
                #there is a tumor segment in the crop
                #what is the organ of the tumor segment?
                if 'uterus' in seg:
                    organ = 'uterus'
                else:
                    organ = None
                    for org in self.classes_UFO:
                        if org in seg:
                            organ = org
                    if organ is None:
                        raise ValueError('Organ not in organs:',seg)
                x = zeros.clone()
                x[tensor_lab[seg_idx]>0] = 1
                unk_segments[organ] = x
                organ=organ.replace('_head','').replace('_body','').replace('_tail','').replace('pancreas','pancreatic')
                unk_lesions.append(organ)
            else:
                #raise ValueError(f'Segment not in crop: {seg}')
                print(f'Segment not in crop:',seg, flush=True, file=sys.stderr)
            
                
        unk_lesions=list(set(unk_lesions))
        #print('unk lesions:', unk_lesions, flush=True, file=sys.stderr)

        unk_channels={}
        unk_channels_list=[]
        label=[]
        #print('Shape of tensor_lab before assigning labels:', tensor_lab.shape, flush=True, file=sys.stderr)
        assert len(tensor_lab.shape) == 4
        for j,clss in enumerate(self.classes,0):
            #print('Class:',clss,flush=True, file=sys.stderr)
            if clss in self.classes_UFO:
                label.append(tensor_lab[clss_UFO_to_idx[clss]])
                unk_channels_list.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
            else:
                if 'lesion' not in clss.lower():
                    if clss=='liver':
                        #join all liver segments
                        l=torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0])
                        for i in [1,2,3,4,5,6,7,8]:
                            l=torch.logical_or(l,tensor_lab[clss_UFO_to_idx['liver_segment_%i'%i]])
                        label.append(l)
                        unk_channels_list.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))#this channel is knwon, assign zero to unk_channels_list
                    elif clss=='pancreas':
                        #join all pancreas segments
                        l=torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0])
                        for i in ['head','body','tail']:
                            l=torch.logical_or(l,tensor_lab[clss_UFO_to_idx['pancreas_%s'%i]])
                        label.append(l)
                        unk_channels_list.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))#this channel is knwon, assign zero to unk_channels_list
                    else:
                        label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                        unk_channels[clss]=j
                        unk_channels_list.append(torch.ones(tensor_lab[0].shape).type_as(tensor_lab[0]))#no pixel is known for this channel, assign 1 to unk_channels_list

                else:
                    #lesion class
                    #print('Lesion class:',clss,flush=True, file=sys.stderr)
                    #check if there is a tumorous segment for this lesion in the crop
                    tumor_present=False
                    for organ in unk_lesions:
                        if 'bladder' in organ:
                            #as the word bladder is inside the word gallbladder, we must be careful here
                            if (('gallbladder' in organ) or ('gall_bladder' in organ)) and clss=='gallbladder_lesion':
                                    label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                                    unk_channels[clss]=j
                                    unk_channels_list.append(unk_segments[organ])#make only the pixels with unknown tumor location be 1, background pixels are 0
                                    tumor_present=True
                                    break
                            else:
                                if organ in clss and clss == 'bladder_lesion':
                                    label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                                    unk_channels[clss]=j
                                    unk_channels_list.append(unk_segments[organ])#make only the pixels with unknown tumor location be 1, background pixels are 0
                                    tumor_present=True
                                    break
                        else:
                            if 'adrenal' in organ and clss=='adrenal_lesion':
                                label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                                unk_channels[clss]=j
                                unk_channels_list.append(unk_segments[organ])#make only the pixels with unknown tumor location be 1, background pixels are 0
                                tumor_present=True
                                break
                            if ('uterus' in organ) and clss=='uterus_lesion':
                                label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                                unk_channels[clss]=j
                                unk_channels_list.append(unk_segments['uterus'])#remember the uterus class was annotated as prostate
                                tumor_present=True
                                break
                            if organ in clss:
                                label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                                unk_channels[clss]=j
                                unk_channels_list.append(unk_segments[organ])#make only the pixels with unknown tumor location be 1, background pixels are 0
                                tumor_present=True
                                break
                        
                            
                    #if not:
                    #assign label 0
                    if not tumor_present:
                        #negative for the tumor
                        label.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
                        unk_channels_list.append(torch.zeros(tensor_lab[0].shape).type_as(tensor_lab[0]))
        
         
    
    
        label=torch.stack(label,dim=0)
        unk_channels_list=torch.stack(unk_channels_list,0)
        
        
        if len(unk_lesions)>0:
            assert unk_channels_list.sum()>0, 'unk_channels_list should have some non-zero voxels if there are tumors in the crop'
        assert len(label.shape) == 4
        #print('Shape of tensor_lab after assigning labels:', label.shape, flush=True, file=sys.stderr)
        #print('Unk channels:', unk_channels, flush=True, file=sys.stderr)
        assert unk_channels_list.sum().item()!=0
        return label,unk_channels,unk_channels_list.type_as(label)
    
    def define_unknown_voxels(self, label, idx):
        """
        Defines the unknown voxels in the image. Unlike assign_labels, this function assumes your labels (tensor_lab) are already in the final format, with the correct number of channels, in the order of self.classes.
        unk_channels is a dictionary with the classes that are unknown.
        """

        #we must first re-create the tensor_lab (input of the assign_labels function), from label, the output of the assign_labels function.
        clss_to_idx = {clss: i for i, clss in enumerate(self.classes)}
        clss_UFO_to_idx = {clss: i for i, clss in enumerate(self.classes_UFO)}
        tensor_lab = []
        for j,clss in enumerate(self.classes_UFO,0):
            ##print('j:',j,flush=True, file=sys.stderr)
            ##print('clss:',clss,flush=True, file=sys.stderr)
            if clss=='background':
                #add zeros as placeholder
                tensor_lab.append(torch.zeros(label[0].shape).type_as(label[0]))
                bkg=j
            else:
                tensor_lab.append(label[clss_to_idx[clss]])
        tensor_lab=torch.stack(tensor_lab,dim=0)
        #add to background the opposite of all other classes
        tensor_lab[bkg]=(tensor_lab.sum(dim=0)>0).type_as(tensor_lab[0])
        

        #now we can use the assign_labels function
        #to define the unknown voxels
    
        #convert to atlas format
        label_out,unk_channels,unk_channels_list=self.assign_labels(tensor_lab,idx)
        #sanity check: see if label_out matches label
        assert (torch.equal(label_out,label))
        
        return unk_channels_list


    def estimate_tumor_volume(self, idx, tumor_segment_crop):
        """
        Estimates tumor volume from reports. For the segment in the crop.
        Always returns a list of 10 items, padding with 0.
        """
        _,tumor_dict=self.get_tumor_segment_labels(idx)
        #print('Tumor dict:', tumor_dict)
        #print all column names in tumor_dict
        #print(tumor_dict.columns)
        #print('Sizes:',tumor_dict['Tumor Size (mm)'])
        #print('Cropped on tumor segment:', tumor_segment_crop)
        if tumor_segment_crop is None or tumor_segment_crop=='random':
            return [0,0,0,0,0,0,0,0,0,0], torch.zeros((10,3)).float() #CT not cropped around a tumor segment
        
        if isinstance(tumor_segment_crop, list):
            pass
        elif isinstance(tumor_segment_crop, str):
            tumor_segment_crop=[tumor_segment_crop]
        else:
            raise ValueError('tumor_segment_crop must be a list or a string.')
        
        #is our tumor_segment_crop organ or segment:
        
        if 'segment' in "".join(tumor_segment_crop) or 'head' in "".join(tumor_segment_crop) or 'body' in "".join(tumor_segment_crop) or 'tail' in "".join(tumor_segment_crop) or 'left' in "".join(tumor_segment_crop) or 'right' in "".join(tumor_segment_crop):
            tpe='segment'
            col='Standardized Location'
        else:
            tpe='organ'
            col='Standardized Organ'
        
        tumors_in_crop=[]
        for row in tumor_dict.iterrows():
            location=row[1][col]
            #print('Location:',location)
            if not isinstance(location, str) or location.lower()=='u':
                continue
            if '/' in location:
                location=location.split(' / ')
            if not isinstance(location, list):
                location=[location]
            in_crop=True
            for loc in location:
                if loc not in tumor_segment_crop:
                    in_crop=False
                    break
            if in_crop:
                tumors_in_crop.append(row[1]['Tumor Size (mm)'])

            #print('Tumors in crop:', tumors_in_crop)#list of strings with sizes

            #print('Tumor dict:',tumor_dict[['Standardized Organ','Standardized Location','Tumor Size (mm)']])
                
        #estimate volumes for each tumor size
        volumes=[]
        diameters=[]
        for size in tumors_in_crop:
            if 'x' not in size:
                #single diameter provided. Use ball.
                diameter=float(size)
                volume=(4/3) * math.pi * ((diameter/2) ** 3)#sphere. volume in mm3 (voxels)
                volumes.append(volume)
                diameters.append([diameter,diameter,diameter])
            else:
                #2 or 3 diameterts, use ellipsoid
                sizes=size.split(' x ')
                sizes=[float(s) for s in sizes]
                if len(sizes)==2:
                    #assume 3rd axis is the average of the other two
                    sizes.append(sum(sizes)/2)
                elif len(sizes) > 3:                
                    # more than 3 numbers, take top 3
                    sizes = sorted(sizes, reverse=True)[:3]
                #ellipsoid volume
                volume=(4/3) * math.pi * ((sizes[0]/2) * (sizes[1]/2) * (sizes[2]/2))
                volumes.append(volume)
                diameters.append(sizes)

        #print('Estimated volumes:',volumes)

        for i in range(len(volumes),10):
            volumes.append(0)
            diameters.append([0,0,0])
            
        return volumes,torch.tensor(diameters).float()
    
    def SanityAssertOutput(self, tensor_lab, unk_channels_tensor,tumor_volumes_in_crop,chosen_segment_mask):
                                #tensor_lab, unk_channels_list, torch.tensor(tumor_volumes_in_crop).float(), chosen_segment_mask.float()
        classes=sorted(self.classes)
        #assert shapes
        assert len(tensor_lab.shape)==4 , 'tensor_lab must have 4 dimensions'
        assert tensor_lab.shape[0]==len(classes), 'Number of classes in tensor_lab (%i) does not match number of classes (%i)'%(tensor_lab.shape,len(classes))
        assert unk_channels_tensor.shape[0]==len(classes), f'Number of classes in unk_channels_tensor ({unk_channels_tensor.shape}) does not match number of classes ({len(classes)})'
        assert chosen_segment_mask.shape[0]==len(classes), f'Number of classes in chosen_segment_mask ({chosen_segment_mask.shape}) does not match number of classes ({len(classes)})'
        assert (tensor_lab.shape==unk_channels_tensor.shape) and (tensor_lab.shape==chosen_segment_mask.shape), f'tensor_lab, unk_channels_tensor and chosen_segment_mask must have the same shape. tensor_lab: %s, unk_channels_tensor: %s, tumor_volumes_in_crop: %s'%(tensor_lab.shape,unk_channels_tensor.shape,tumor_volumes_in_crop.shape)

        
        #save examples
        sample=self.current_sample
        sample=sample[sample.rfind('BDMAP_'):sample.rfind('.')]
        if self.counter<50:
            debug_save_labels(tensor_lab,sample+'_y',self.classes,out_dir=self.sanity_path)
            debug_save_labels(chosen_segment_mask,sample+'_chosen_segment_mask',self.classes,out_dir=self.sanity_path)
            debug_save_labels(unk_channels_tensor,sample+'_unk_voxels',self.classes,out_dir=self.sanity_path)
            self.counter+=1

        #assert that unk_channels_tensor and chosen_segment_mask are 0 for all non lesion classes
        missing_classes=set(classes)-set(self.classes_UFO)-{'liver','pancreas'}
        missing_classes=list(missing_classes)
        #print('Missing classes:', missing_classes,flush=True, file=sys.stderr)
        unk_cls=[]
        known_cls=[]
        for i,clss in enumerate(classes):
            if 'lesion' in clss.lower() or clss in missing_classes: 
                unk_cls.append(i)
            else:
                known_cls.append(i)
        if not unk_channels_tensor[known_cls].sum().item()==0:
            for i,clss in enumerate(classes,0):
                if i in unk_cls:
                    continue
                else:
                    if unk_channels_tensor[i].sum().item()!=0:
                        print('Class with unk voxels:',clss,'Sample is:',sample)
                        
        assert unk_channels_tensor[known_cls].sum().item()==0
        assert chosen_segment_mask[known_cls].sum().item()==0

        #print('Assertions passed!',flush=True, file=sys.stderr)



        

        

def npy_to_nii(npy_path, nii_path, spacing=(1.0, 1.0, 1.0),labels=None):
    """
    Reads a .npy file, converts it to a SimpleITK image, 
    sets spacing, and saves as .nii.gz.

    :param npy_path:    Path to the input .npy file.
    :param nii_path:    Path to the output .nii.gz file.
    :param spacing:     Tuple or list specifying the (z, y, x) spacing. 
                        Default is (1.0, 1.0, 1.0).
    """
    # Load the NumPy array
    array = np.load(npy_path)
    #print('Shape of array:', array.shape)
    #squeeze
    array = array.squeeze()
    #print('Shape after squeeze:',array.shape)

    if labels is not None:
        #load yaml labels
        with open(labels, 'r') as f:
            labels = yaml.load(f, Loader=yaml.SafeLoader)
        #print('Yaml loaded')
        #sort
        labels = sorted(labels)
       #print('Labels:',labels)
       #print('Shape of array:',array.shape)
        if len(array.shape) == 4:
            #label
            if array.shape[0] < len(labels):
                #unpack
                array = np.unpackbits(array, axis=0)
                array = array[:len(labels)]
            os.makedirs(nii_path.replace('.nii.gz',''), exist_ok=True)
            for label in labels:
                #save each label
                sitk_image = sitk.GetImageFromArray(array[labels.index(label)])
                sitk_image.SetSpacing(spacing)
                sitk.WriteImage(sitk_image, os.path.join(nii_path.replace('.nii.gz',''),label+'.nii.gz'))
               #print('Saved:', os.path.join(nii_path.replace('.nii.gz',''),label+'.nii.gz'))
    else:
       #print('No labels provided, saving as a single volume')
        # Convert NumPy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(array)

        # Optionally set image spacing (if known)
        sitk_image.SetSpacing(spacing)

        # Write to .nii.gz
        sitk.WriteImage(sitk_image, nii_path)


def debug_save_labels(labels: torch.Tensor,
                      name='',
                      label_names = '/projects/bodymaps/Pedro/data/atlas_300_medformer_npy/list/label_names.yaml',
                      out_dir: str = "./DatasetSanityMultiTumor",
                      batch_idx: int = 0):
    """
    Saves each channel of the specified batch index in `labels` as a .nii.gz file.
    
    Args:
        labels (torch.Tensor): A tensor of shape (B, C, H, W, D).
        label_names_yaml (str): Path to a YAML file containing a list of label names.
                                The list will be sorted alphabetically and used
                                to name the channels.
        out_dir (str): Output directory to save the .nii.gz files. Defaults to "LossSanity".
        batch_idx (int): Which batch element to save. Defaults to 0.
    """
    import nibabel as nib
    # 1. Create output folder if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # 2. Load and sort label names
    if not isinstance(label_names, list):
        with open(label_names, "r") as f:
            label_names = yaml.safe_load(f)  # e.g. ["liver", "kidney", "pancreas", ...]
        
    label_names_sorted = sorted(label_names)  # sort alphabetically
    
    # 3. Basic shape check
    if len(labels.shape)==4:
        labels = labels.unsqueeze(0)
    assert len(labels.shape) == 5
    B, C, H, W, D = labels.shape
    assert batch_idx < B, f"batch_idx={batch_idx} is out of range for B={B}."
    assert C == len(label_names_sorted), (
        f"Number of channels (C={C}) does not match the number of label names "
        f"(={len(label_names_sorted)})."
    )
    
    # 4. Extract just the batch element we want
    #    This will have shape (C, H, W, D).
    label_slice = labels[batch_idx]
    
    # 5. Loop over channels, save each one as a nii.gz
    for c in range(C):
        # Move channel c to CPU numpy for saving
        channel_data = label_slice[c].detach().cpu().numpy()
        
        # Build a simple identity affine; if you have real metadata, replace it
        affine = np.eye(4, dtype=np.float32)
        
        # Convert to float32 (or int16, float64, etc.)
        channel_data = channel_data.astype(np.float32)
        
        # Create a NIfTI image
        nifti_img = nib.Nifti1Image(channel_data, affine)
        
        # Derive a filename from the label name
        channel_label_name = label_names_sorted[c]
        try:
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        except:
            #remove the folder if it exists and create it again
            try:
                shutil.rmtree(os.path.join(out_dir, name))
                os.makedirs(os.path.join(out_dir, name), exist_ok=True)
            except:
                pass
        
        out_path = os.path.join(out_dir, f"{name}/{channel_label_name}.nii.gz")
        # Save
        nib.save(nifti_img, out_path)
        
    #print(f"Saved to {out_path}")

def canonical_organ(tumor_name):
        """
        Convert a tumor class name to an organ class name:
        - Remove '_lesion'
        - Substitutes some known patterns (like 'pancreatic' -> 'pancreas')
        - Randomly chooses a sided version for kidney, adrenal_gland, lung, or femur.
        """
        if isinstance(tumor_name, list):
            tumor_name = tumor_name[0] #right and left
            assert ('right' in tumor_name) or ('left' in tumor_name), 'Tumor name must be a list with right and left'
        base = tumor_name.replace('_lesion', '')
        tumor_name = tumor_name.replace('_right', '').replace('_left', '')
        lower_base = base.lower()
        if 'pancrea' in lower_base:
            return 'pancreas'
        elif 'kidney' in lower_base:
            return 'kidney'
        elif 'adrenal' in lower_base:
            return 'adrenal'
        elif 'lung' in lower_base:
            return 'lung'
        elif 'femur' in lower_base:
            return 'femur'
        elif 'gall' in lower_base:
            return 'gall_bladder'#added 28 apr 2025
        else:
            return base