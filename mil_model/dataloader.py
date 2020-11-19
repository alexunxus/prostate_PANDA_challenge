import numpy as np
import os
import json
import pandas as pd
import time
import random
from sklearn.model_selection import train_test_split

# iaa
from imgaug import augmenters as iaa
import imgaug as ia
import math
ia.seed(math.floor(time.time()))

# albumentations
import albumentations
import cv2
import skimage.io

# torch
import torch
from torchvision import transforms

# customized library
from .util import concat_tiles, binary_search
from hephaestus.data.openslide_wrapper_v2 import Slide_OSread

class PathoAugmentation(object):

    ''' Customized augmentation method:
        discrete augmentation: do these augmentation to each 6*6 tiles in the image respectively
        complete augmentation: do these augmentation to the whole image at one time
    '''
    discrete_aug = albumentations.Compose([
                                           albumentations.Transpose(p=0.5),
                                           albumentations.VerticalFlip(p=0.5),
                                           albumentations.HorizontalFlip(p=0.5),
                                           albumentations.transforms.Rotate(limit=15, border_mode=cv2.BORDER_WRAP, p=0.5)])
    complete_aug = albumentations.Compose([albumentations.augmentations.transforms.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15, p=0.5)]) #albumentations.augmentations.transforms.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15, p=0.5)


def get_resnet_preproc_fn():
    return  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])

def get_train_test(json_dir, ratio=0.1):
    ''' Arg: json_dir: string the path of the json file
             ratio: the test set ratio
        return: 
            X_train: a list of slide_name string with (1-ratio)% of all the training images
            X_test:  a list of slide_name string with ratio%     of all the training images
    '''
    file_list = [f.split('.json')[0] for f in os.listdir(json_dir) if f.endswith('.json')]
    X_train, X_test = train_test_split(file_list, test_size=ratio, shuffle=False, random_state=87)
    return X_train, X_test

# Dataloader for concated tiles
class TileDataset:
    def __init__(self, 
                 img_names, 
                 img_dir, 
                 json_dir, 
                 label_path, 
                 patch_size, 
                 resize_ratio=1,
                 tile_size=6, 
                 dup = 4, 
                 is_test=False, 
                 aug=None, 
                 preproc=None):
        '''
        Arg: img_names: string, the slide names from train_test split
             img_dir: path string, the directory of the slide tiff files
             json_dir: path string, the directory of the json files
             patch_size: int
             tile_size: int, each item from this dataset will be rendered as a concated 3*tile_size*tile_size image
             dup: int, will get $dup concatenated tiles from one slide to save slide reading time
             aug: augmentation function
             preproc: torch transform object, default: resnet_preproc
        '''
        self.json_dir     = json_dir
        self.img_dir      = img_dir
        self.label_path   = label_path
        self.patch_size   = patch_size
        self.resize_ratio = resize_ratio
        self.tile_size    = tile_size
        self.dup          = dup
        self.is_test      = is_test
        self.aug          = aug
        self.preproc      = preproc
        
        # the following 3 list should be kept in the same order!
        self.img_names   = img_names
        self.isup_scores = None
        self.coords      = None
        
        self._get_isup()
        self._get_coord()
        
        self.cur_pos = 0
    
    def _get_isup(self):
        print("=============Get ISUP scores=================")
        t1 = time.time()
        df = pd.read_csv(self.label_path)
        
        names = df['image_id'].to_numpy()
        grades = df['isup_grade'].to_numpy()
        li = list(zip(names, grades))
        
        self.isup_scores = [ binary_search(li, name, 0, len(li)) for name in self.img_names]
        print(f"Loading ISUP scores takes {time.time()-t1} seconds")
    
    def _get_coord(self):
        # get the coordinated prefetched by /foreground/preprocess.py, stored at /foreground/data_256/*.json
        print("==========Collecting coordinates=============")
        t1 = time.time()
        self.coords = []
        for img_name in self.img_names:
            with open(os.path.join(self.json_dir, img_name+'.json')) as f:
                js = json.load(f)
            tmp_list = (js[img_name][0]['coord_list'])
            self.coords.append(tmp_list)
        print(f"Loading coordinates takes {time.time()-t1} seconds")
        print("=========Collecting data finished============")
    
    def _get_frame(self, tiff_file):
        if self.resize_ratio not in [1, 4]:
            raise ValueError(f"Unsupported resize ratio {self.resize_ratio}")
        level = 0 if self.resize_ratio== 1 else 1
        return skimage.io.MultiImage(tiff_file)[level]
        
    
    def __getitem__(self, idx):
        '''
        This function will fectch idx-th bags of image from the coordinate list, get the first
        tile size * tile size images from the coordinate list, concatenate them, and performing augmentation, 
        preprocessing. In addition, the label will also be generated as a torch tensor
        
            grade 3 --> [1, 1, 1, 0, 0]
            grade 5 --> [1, 1, 1, 1, 1]
            grade 0 --> [0, 0, 0, 0, 0]
        
        Ultimately, the preprocessed image will be returned altogether with the groundtruth label
        
        Return:
            Training time:
                imgs: a tensor
                label: a tensor
            Testing time:
                imgs: list of tensor with length self.dup --> will be averaged after passing to model
                label: ONE tensor
        '''
        self.cur_pos = idx
        
        if self.resize_ratio == 1:
            # get image(openslide version -- slower)
            this_slide = Slide_OSread(os.path.join(self.img_dir, self.img_names[idx]+".tiff"), show_info=False)
        elif self.resize_ratio == 4:
            # get image(skimage version -- faster)
            this_slide = self._get_frame(os.path.join(self.img_dir, self.img_names[idx]+".tiff"))

        # cat tile is of type "np.float32", had been normalized to 1
        # test time augmentation: find 4 images and will average their predicted values
        if not self.is_test:
            img = torch.from_numpy(self._get_one_cat_tile(idx, this_slide).transpose((2, 0, 1))).float()
            if self.preproc is not None:
                img = self.preproc(img)
        else:
            cat_tiles = [self._get_one_cat_tile(idx, this_slide)for i in range(self.dup)]
            img = [torch.from_numpy(np.transpose(cat_tile, (2, 0, 1))).float() for cat_tile in cat_tiles]
            if self.preproc is not None:
                img = [self.preproc(im) for im in img]
            img = torch.stack(img)

        # get label
        label = np.zeros(5)
        for i in range(0, self.isup_scores[idx]):
            label[i] = 1
        label = torch.from_numpy(label).float()
        
        self.cur_pos = (self.cur_pos+1)%len(self)
        return img, label
    
    def _get_one_cat_tile(self, idx, this_slide):
        '''
        Argument: 
            this_slide: 
                1. openslide object(slower) OR
                2. skimage(faster)
            idx: int, position of the idx-th tuple in coordinate list
        Return: cat_tile: an numpy array sized 1536*1536*3 with type float32, normalized to 0-1

        No matter the value of patch size, the acquired image patches will be resized to 256*256 and
        be concatenated to a 1536*1536*3 image
        If the patch size is 256, or 512, then get patch from openslide object
        If the patch size is 1024, then get patch from the first image pyramid of skimage
        '''
        ps, ts = self.patch_size, self.tile_size
        dst_sz = self.patch_size //self.resize_ratio
        
        chosen_indexs = np.random.choice(len(self.coords[idx]), min(ts**2, len(self.coords[idx])), replace=False)
        foreground_coord = [self.coords[idx][chosen_index] for chosen_index in chosen_indexs]
        
        # this_slide is slide osread, get patch by Slide_OSread method
        if isinstance(this_slide, Slide_OSread):
            tiles = [this_slide.get_patch_with_resize(coord=(x*ps, y*ps), 
                                                      src_sz=(ps, ps), 
                                                      dst_sz=(dst_sz, dst_sz)) for x, y in foreground_coord]
        
        # this_slide is skimage multiimage object(essentially np.ndarray), get patch directly by subindexing
        elif isinstance(this_slide, np.ndarray):
            tiles = [this_slide[y*dst_sz:(y+1)*dst_sz, x*dst_sz:(x+1)*dst_sz] for x, y in foreground_coord]
        else:
            raise ValueError(f"Unknown slide type: {type(this_slide)}")
        
        
        # discrete augmentation will be performed image-wise before concatenation
        # complete augmentation will be performed to the concatenated big image 
        if self.aug is not None:
            tiles = [self.aug.discrete_aug(image=tile)['image'] for tile in tiles]
            cat_tile = self.aug.complete_aug(image=concat_tiles(tiles, patch_size=dst_sz, tile_sz=self.tile_size))['image']
            cat_tile = cat_tile.astype(np.float32)/255.
        else:
            cat_tile = concat_tiles(tiles, patch_size=dst_sz, tile_sz=self.tile_size).astype(np.float32)/255.
        return cat_tile
    
    def __next__(self):
        return self.__getitem__(self.cur_pos)
    
    def __len__(self):
        return len(self.img_names)