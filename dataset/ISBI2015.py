import os, sys
import argparse
import torch
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler
import json
from monai.data import Dataset, DataLoader, CacheDataset
import numpy as np
from tqdm import tqdm
import glob
#import SimpleITK as sitk
import matplotlib.pyplot as plt
from monai import transforms
from monai.config import print_config


from dataset.utils import transform


class ISBI2015(CacheDataset):
    def __init__(self, root_dir, trans, folds_path, split, num_fold):
        if not os.path.isdir(root_dir):
            raise ValueError("Root directory root_dir must be a directory.")
        self.split = split
        self.num_fold = num_fold
        data = self._make_dataset_from_json(folds_path)

        transforms_comp = transforms.Compose([
            transform.AppendRootDirD(['images', 'mask'], root_dir),
            trans,
            ])
        super().__init__(data, transforms_comp)#,num_workers=num_workers)

    
    def _make_dataset_from_json(self, folds_path):
        with open(folds_path) as fp:
            path=json.load(fp)
            data = list()
            
            if self.split == 'testing':
                data = path[f'fold{self.num_fold}']['test']
                print("DATA: ", data)
            elif self.split == 'validation':
                data = path[f'fold{self.num_fold}']['val']
            elif self.split == 'training':
                data = path[f'fold{self.num_fold}']['train']
            else: 
                raise ValueError(
                        f"Unsupported split: {self.split}, "
                        "available options are ['training', 'validation', 'testing']."
                    )
            #if platform.system() != 'Windows':
            #    for sample in data:
            #        for key in sample.keys():
            #            sample[key] = sample[key].replace('\\', '/')
            return data

def get_loader(args):
    #DEFINIRE LE TRASFORMAZIONI QUI
    basics_1 = transforms.Compose([
        #transform.RegistrationD(keys=["image", "mask"]), #COMPUTE VOLUME STARTING FROM ORIGINAL MRI
        transforms.LoadImageD(keys=["images", "mask"]),
        transforms.AddChannelD(['mask']), #AddChanneld deprecated
        #transforms.OrientationD(keys=["image", "mask"], axcodes= 'SAR') #SAR
        ])
    
    train_transforms = transforms.Compose([
            basics_1,
            transforms.CropForegroundd(keys=["images", "mask"], source_key="images"),
            transforms.RandSpatialCropd(keys=["images", "mask"], roi_size=args.resize ,random_size=False),
            transforms.SpatialPadd(keys=["images", "mask"], spatial_size=args.resize),
            transforms.RandFlipd(keys=["images", "mask"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["images", "mask"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["images", "mask"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="images", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="images", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["images", "mask"],),
        ])
    val_transforms = transforms.Compose([   
            basics_1,
            transforms.CropForegroundd(keys=["images", "mask"], source_key="images"),
            transforms.NormalizeIntensityd(keys="images", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["images", "mask"]),
            ])

    #DATASET
    dataset = {}
    dataset["train"] = ISBI2015(root_dir = args.root_dir, folds_path = args.folds_path, split = 'training', num_fold = args.num_fold, trans = train_transforms)
    dataset["val"] = ISBI2015(root_dir = args.root_dir, folds_path = args.folds_path, split = 'validation', num_fold = args.num_fold, trans = val_transforms)
    dataset["test"] = ISBI2015(root_dir = args.root_dir, folds_path = args.folds_path, split = 'testing', num_fold = args.num_fold, trans = val_transforms)

    #SAMPLER
    samplers = {}
    samplers["train"] = RandomSampler(dataset["train"])
    samplers["val"] = SequentialSampler(dataset["val"])
    samplers["test"] = SequentialSampler(dataset["test"])

    #BATCH SAMPLER
    batch_sampler = {}
    batch_sampler['train'] = BatchSampler(samplers["train"], args.batch_size, drop_last=True)
    batch_sampler['val'] = BatchSampler(samplers["val"], 1, drop_last=True)
    batch_sampler['test'] = BatchSampler(samplers["test"], 1, drop_last=False)
    
    #print("LEN DATASET TRAIN: ", len(dataset["train"]))
    #DATALOADER
    loaders = {}
    loaders["train"] = DataLoader(dataset["train"], batch_sampler = batch_sampler['train'], num_workers=1, pin_memory=True, persistent_workers=True)
    loaders["val"] = DataLoader(dataset["val"], batch_sampler = batch_sampler['val'], num_workers=1, pin_memory=True, persistent_workers=True)
    loaders["test"] = DataLoader(dataset["test"], batch_sampler = batch_sampler['test'], num_workers=1, pin_memory=True, persistent_workers=True)

    return loaders, samplers

import matplotlib.pyplot as plt
if __name__ == '__main__':
    args={}
    args['resize'] = 96
    args['root_dir'] = '/storage/data_4T/alessia_medical_datasets/ISBI_2015/ISBI2015'
    args['folds_path'] = '2-dataset_crossvalidation_5fold.json'
    args['num_fold'] = 0
    args['batch_size'] = 1
    args = argparse.Namespace(**args)

    # Dataset e Loader
    loaders, samplers = get_loader(args)

    # Dataset e Loader
    loaders, samplers = get_loader(args)

    # Get samples
    tmp = next(iter(loaders["train"]))
    print(tmp["images"].size()) #torch.Size([1, 3, 96, 96, 96])
    print(tmp["mask"].size()) #torch.Size([1, 1, 96, 96, 96])
    #print("TMP KEYS:", tmp.keys()) 

    plt.imsave('img'+str(80)+'.png', tmp['images'][0].numpy()[0,80,:,:]) #0 FLAIR, 1 T1, 2 T2
    plt.imsave('msks'+str(80)+'.png', tmp['mask'][0].numpy()[0,80,:,:])
    
