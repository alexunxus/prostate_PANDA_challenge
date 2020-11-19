# standard libraries
import os
import numpy as np
import math
from PIL import Image # transformation in preprocessing
import pandas as pd # reading label csv
import random # for shuffling dataset
from random import shuffle
from sklearn.model_selection import train_test_split

# torch packages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# customized libraries
from hephaestus.data.openslide_wrapper_v2 import Slide_OSread
class InferenceDataset:
    def __init__(self, slide_dir, slide_name, mask_dir, patch_sz, preproc=None):
        self.cur_pos      = 0
        self.slide_name   = slide_name
        self.patch_sz     = patch_sz
        self.resize_ratio = 1
        self.preproc      = preproc
        self.no_mask      = False
        
        if patch_sz > 512:
            self.this_slide   = skimage.io.MultiImage(os.path.join(slide_dir, slide_name))[1]
            self.resize_ratio = 4
            self.patch_sz     = self.patch_sz // 4
            patch_sz          = patch_sz // 4
        else:
            self.this_slide = skimage.io.MultiImage(os.path.join(slide_dir, slide_name))[0]
        mask_name = os.path.join(mask_dir, slide_name.split(".tiff")[0]+"_mask.tiff")
        if os.path.isfile(mask_name):
            self.this_mask = skimage.io.MultiImage(mask_name)[1 if self.resize_ratio == 4 else 0]
        else:
            self.no_mask=True

        H, W = self.this_slide.shape[:2]
        w = W//patch_sz
        h = H//patch_sz
        w = (W//patch_sz+1) if W%patch_sz != 0 else (W//patch_sz)
        h = (H//patch_sz+1) if H%patch_sz != 0 else (H//patch_sz)
        if self.no_mask:
            self.obj_mask = np.ones((h,w))
        else:
            self.obj_mask = np.mean(cv2.resize(src=self.this_mask, dsize=(w, h), 
                                               interpolation=cv2.INTER_LINEAR), 
                                    axis=-1)
        plt.imshow(self.obj_mask)
        plt.show()
        assert self.obj_mask.shape == (h, w)
        self.obj_list = list(zip(*(np.where(self.obj_mask != 0)[::-1])))
        
        if not self.no_mask:
            del self.this_mask
        
    def __getitem__(self, idx):
        self.cur_pos = idx % len(self)
        x, y = self.obj_list[self.cur_pos]
        ps = self.patch_sz
        img = self.this_slide[y*ps:(y+1)*ps, x*ps:(x+1)*ps]
        if img.shape[:2] != (ps, ps):
            img = np.pad(img, ((0, ps-img.shape[0]), (0, ps-img.shape[1]), (0, 0)), 
                         mode='constant', 
                         constant_values=(0, 255))
        self.cur_pos = (self.cur_pos+1)%len(self)
        return img, x, y
        
    def __next__(self):
        return self.__getitem__(self.cur_pos)
    
    def __len__(self):
        return len(self.obj_list)

def get_train_test(npy_path, isup_path, train_ratio=0.9):
    df = pd.read_csv(isup_path)
    df.sort_values("image_id")
    
    Xs = [f for f in os.listdir(npy_path) if f.endswith('.npy')]
    Ys = np.array(df['isup_grade'])
    
    Xs.sort()
    
    X_train, X_test, y_train, y_test = train_test_split(Xs, Ys, test_size=1-train_ratio, random_state=42)
    return X_train, X_test, y_train, y_test

class Dataset:
    def __init__(self, npy_path, feature_maps, isup_scores, target_size, shuffle=False):
        # prostate dataset x_max = 62, y_max = 110 --> padding to 64, 128
        self.npy_path = npy_path
        self.target_size = target_size
        
        self.feature_maps = feature_maps
        self.isup_scores = isup_scores/5.
        assert len(self.isup_scores) == len(self.feature_maps)
        
        self.cur_pos = 0
        
        if shuffle:
            self.shuffle()
    
    def __getitem__(self, index):
        ''' get image and label, the type of image should be torch.float
        '''
        image = np.load(os.path.join(self.npy_path, self.feature_maps[index]))
        label = self.isup_scores[index]
        assert image.shape[1] <= self.target_size[1] and image.shape[2] <= self.target_size[2]
        image = np.pad(image, 
                       ((0, 0),
                       (0, self.target_size[1]-image.shape[1]),
                       (0, self.target_size[2]-image.shape[2])), 'constant', constant_values=(0, 0))
        image = torch.from_numpy(image.astype(np.float32))

        self.cur_pos = (self.cur_pos+1) % len(self)
        return image, label
    
    def __next__(self):
        return self.__getitem__(self.cur_pos)
    
    def __len__(self):
        return len(self.feature_maps)
    
    def shuffle(self):
        tmp = list(zip(self.feature_maps, self.isup_scores))
        shuffle(tmp)
        self.featuer_maps, self.isup_scores = zip(*tmp)


'''
class InferenceDataset:
    def __init__(self, slide_dir, slide_name, mask_dir, patch_sz, preproc=None):
        self.cur_pos    = 0
        self.slide_name = slide_name
        self.patch_sz   = patch_sz
        self.preproc    = preproc
        self.no_mask    = False
        
        self.this_slide = Slide_OSread(os.path.join(slide_dir, slide_name), show_info=False)
        mask_name = slide_name.split(".tiff")[0]+"_mask.tiff"
        if os.path.isfile(os.path.join(mask_dir, mask_name)):
            self.this_mask  = Slide_OSread(os.path.join(mask_dir , mask_name), show_info=False)
        else:
            self.no_mask=True

        W, H = self.this_slide.get_size()
        w = (W//patch_sz+1) if W%patch_sz != 0 else (W//patch_sz)
        h = (H//patch_sz+1) if H%patch_sz != 0 else (H//patch_sz)
    
        if self.no_mask:
            self.obj_mask = np.ones((h,w))
        else:
            self.obj_mask = np.mean(self.this_mask.get_patch_with_resize(coord=(0, 0), src_sz=(W, H), dst_sz=(w, h)), axis=-1)
        assert self.obj_mask.shape == (h, w)
        self.obj_list = list(zip(*(np.where(self.obj_mask != 0)[::-1])))
        
        if not self.no_mask:
            del self.this_mask
        
    def __getitem__(self, idx):
        self.cur_pos = idx % len(self)
        x, y = self.obj_list[self.cur_pos]
        img = self.this_slide.get_patch_at_level((x*self.patch_sz, y*self.patch_sz), 
                                                (self.patch_sz, self.patch_sz)
                                                )
        img = Image.fromarray(img)
        if self.preproc is not None:
            img = self.preproc(img)
        self.cur_pos = (self.cur_pos+1)%len(self)
        return img, x, y
        
    def __next__(self):
        return self.__getitem__(self.cur_pos)
    
    def __len__(self):
        return len(self.obj_list)
    
    def get_img_size(self):
        return 3, int(self.patch_sz//4), int(self.patch_sz//4)
    
    def thumbnail_size(self):
        return self.obj_mask.shape
'''
