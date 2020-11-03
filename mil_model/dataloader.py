import numpy as np
import os
import json
import pandas as pd
import time

from .util import concat_tiles
from hephaestus.data.openslide_wrapper_v2 import Slide_OSread


# Dataloader for concated tiles
class TileDataset:
    def __init__(self, img_dir, json_dir, label_path, patch_size, tile_size=6, aug=None, preproc=None):
        self.json_dir = json_dir
        self.img_dir  = img_dir
        self.label_path = label_path
        self.patch_size = patch_size
        self.tile_size = tile_size
        self.aug = aug
        self.preproc = preproc
        
        # get foreground coordinates and isup score 
        self.img_names = [f.split('.json')[0] for f in os.listdir(json_dir) if f.endswith('.json')]
        self.isup_scores = None
        self.coords = []
        self._get_isup()
        self._get_coord()
        
        self.cur_pos = 0
    
    def _get_isup(self):
        print("=============Get ISUP scores=================")
        t1 = time.time()
        df = pd.read_csv(self.label_path)
        self.isup_scores = [ df.loc[df["image_id"] == name, 'isup_grade'].item() for name in self.img_names]
        print(f"Loading ISUP scores takes {time.time()-t1} second")
    
    def _get_coord(self):
        print("==========Collecting coordinates=============")
        t1 = time.time()
        for img_name in self.img_names:
            with open(os.path.join(self.json_dir, img_name+'.json')) as f:
                js = json.load(f)
            tmp_list = (js[img_name][0]['coord_list'])
            shuffle(tmp_list)
            self.coords.append(tmp_list)
        print(f"Loading coordinates takes {time.time()-t1} second")
        print("=========Collecting data finished============")
        
    def __getitem__(self, idx):
        self.cur_pos = idx
        ps, ts = self.patch_size, self.tile_size
        
        # get image
        print(f"Getting image {self.img_names[idx]}")
        this_slide = Slide_OSread(os.path.join(self.img_dir, self.img_names[idx]+".tiff"), show_info=False)
        
        print(f"coordinates: {self.coords[idx][:min(ts**2, len(self.coords[idx]))]}")
        foreground_coord = self.coords[idx][:min(ts**2, len(self.coords[idx]))]
        tiles = [this_slide.get_patch_at_level((x*ps, y*ps), (ps, ps)) for x, y in foreground_coord]
        cat_tile = concat_tiles(tiles, patch_size=ps, tile_sz=6)
        
        del this_slide
        img = torch.from_numpy(np.transpose(cat_tile, (2, 0, 1))).float()
        
        # get label
        label = np.zeros(5)
        for i in range(1, self.isup_scores[idx]):
            label[i] = 1
        label = torch.from_numpy(label).float()
        
        self.cur_pos = (self.cur_pos+1)%len(self)
        return img, label
    
    def __next__(self):
        return self.__getitem__(self.cur_pos)
    
    def __len__(self):
        return len(self.img_names)