
from cmath import nan
import torch
import copy
from monai.transforms import Randomizable, MapTransform, Transform
from monai.config import KeysCollection
import numpy as np
from skimage.transform import resize
import os, sys
from scipy.io import loadmat
#import SimpleITK as sitk
import scipy.ndimage as ndimage

import math
path_mni_template = "/home/alessiarondinella/fsl/data/standard/MNI152_T1_1mm.nii.gz"

class AppendRootDirD(MapTransform):
    def __init__(self, keys: KeysCollection, root_dir):
        super().__init__(keys)
        self.root_dir = root_dir
    
    def __call__(self, data):
        
        d = copy.deepcopy(data)
        #print("DATA: ", d['data']) # dict con chiavi images e mask per ogni tp (1 PAZIENTE intero)
        for tp in range(len(d['data'])):
            #print("DATA KEYS: ", d['data'][tp])# dict con chiavi images e mask per ogni tp
            #print("KEYS:", self.keys)
            for k in self.keys:
                if(k == 'images'):
                    #print(d['data'][tp][k]) FLAIR, T1, T2
                    for scan in range(len(d['data'][tp][k])):
                        #print(d['data'][tp][k][scan])
                        d['data'][tp][k][scan] = os.path.join(self.root_dir, d['data'][tp][k][scan])
                else:
                    d['data'][tp][k] = os.path.join(self.root_dir, d['data'][tp][k])
                        
        d['images'] = d['data'][0]['images']
        d['mask'] = d['data'][0]['mask']
        del d['data']
        #print("CHIAVI PRIMA TRASFORMAZIONE:", d.keys()) #'id', 'images', 'mask'
        return d
    
class AdjustDirD(MapTransform):
    def __init__(self, keys: KeysCollection, root_dir):
        super().__init__(keys)
        self.root_dir = root_dir
    
    def __call__(self, data):
        
        d = copy.deepcopy(data)
        #print("INITIAL D: ", d)
        for k in self.keys:
            #print("DATA KEYS: ", d)
            #print("Data K:", d[k])
            if(k == 'image'):
                for j in range(len(data[k])):
                    #print("IMG: ", data[k][j][0])
                    d[k][j] = os.path.join(self.root_dir, data[k][j][0])
            else:
                #print("MASK: ", data[k])
                d[k] = os.path.join(self.root_dir, data[k][0])
            
        #print("FINAL D: ", d)
        return d

class RegistrationD(MapTransform):
    def __init__(self, keys: KeysCollection):#, root_dir):
        super().__init__(keys)
        #self.root_dir = root_dir
    
    def __call__(self, data):
        
        d = copy.deepcopy(data)
        #print("INITIAL D: ", d)
        for k in self.keys:

            if(k == 'image'):
                for j in range(len(d[k])):
                    path_lst = d[k][j].split("/")[:-1]
                    path_matrix= '/'.join([str(c) for c in path_lst])
                    #FLIRT
                    os.system(f"flirt -ref {path_mni_template} -in {d[k][j]} -out {d[k][j]} -omat {os.path.join(path_matrix, 'matrix.mat')} -dof 6")
                    #BET
                    os.system(f"bet {d[k][j]} {d[k][j]}")
            else:
                os.system(f"flirt -in {d[k]} -ref {path_mni_template} -applyxfm -init {os.path.join(path_matrix, 'matrix.mat')} -out {d[k]}")   
        #print("FINAL D: ", d)
        return d



