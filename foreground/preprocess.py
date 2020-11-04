import os
import numpy as np
import pandas as pd # reading label csv
from tqdm import tqdm # only used in jupyter notebook
from skimage import color
import json # save foreground info into a json file
from scipy.ndimage.morphology import binary_dilation
from scipy import ndimage
from joblib import Parallel, delayed

from hephaestus.data.openslide_wrapper_v2 import Slide_OSread

SLIDE_DIR = '/mnt/extension/experiment/prostate-gleason/train_images/'
MASK_DIR  = '/mnt/extension/experiment/prostate-gleason/train_label_masks/'
MIL_PATCH_SIZE=512
FOREGROUND_DIR = f'/workspace/prostate_isup/foreground/data_{MIL_PATCH_SIZE}/'

def file_without_mask(img_dir, mask_dir):
    img_files = [f.split('.tiff')[0] for f in os.listdir(img_dir)]
    mask_files = [f.split('_mask')[0] for f in os.listdir(mask_dir)]
    problem_files = [f for f in img_files if f not in mask_files]
    return problem_files

def find_non_background_patches(path, patch_size, is_mask=False, orig_path=None, threshold=0.05):
    '''
    Arg:
        path: image path string
        is_mask: if the path is mask path or not
        orig_path: string, if the path is mask path, user should render the original image path, using
                   the original image for background filtering
        patch_size: int, normally 256
        threshold: float, threshold for hsv saturation filtering
    Return: a list of (x, y) left upper corner of patches that contains > 20% non-background pixels
        This function will first apply scipy mask expansion to a resized foreground mask and judge if these
    patches belong to foregound using the saturation of the mean value of the patch.
    '''
    this_slide = Slide_OSread(path, show_info=False)
    W, H = this_slide.get_size()
    w = W//patch_size if W % patch_size else W//patch_size+1
    h = H//patch_size if H % patch_size else H//patch_size+1
    
    obj_coord_list = []
    
    if is_mask:
        assert orig_path is not None
        thumbnail = this_slide.get_patch_with_resize((0, 0), (W, H), (w, h)).sum(axis=-1).squeeze()
        struct2 = ndimage.generate_binary_structure(2, 2)
        thumbnail = binary_dilation(thumbnail, structure=struct2).astype(thumbnail.dtype)
        hs, ws = np.where(thumbnail > 0)[:2]
        original_slide = Slide_OSread(orig_path, show_info=False)
        for i in range(len(hs)):
            h, w = hs[i], ws[i]
            image_patch = original_slide.get_patch_at_level((w*patch_size, h*patch_size), (patch_size, patch_size))
            image_mean = np.mean(image_patch, axis=(0, 1)).reshape(1, 1, 3)
            
            if color.rgb2hsv(image_mean)[0, 0, 1] >= threshold:
                obj_coord_list.append((w, h))   
    else:
        for hh in range(h):
            for ww in range(w):
                image_patch = this_slide.get_patch_at_level((ww*patch_size, hh*patch_size), (patch_size, patch_size))
                image_mean = np.mean(image_patch, axis=(0, 1)).reshape(1, 1, 3)
                if color.rgb2hsv(image_mean)[0, 0, 1] >= threshold:
                    obj_coord_list.append((ww, hh))
    return obj_coord_list

def save_foreground_info(save_path, slide_name, coord_list, from_mask=0):
    if os.path.isfile(save_path):
        print(f"File {save_path} exits, overwritting...")
    mil_dict={}
    mil_dict[slide_name] = []
    mil_dict[slide_name].append({"coord_list": coord_list, "from_mask": from_mask})
    with open(save_path, 'w') as fd:
        json.dump(mil_dict, fd)

def process_file_list(target_file, from_mask):
    if os.path.isfile(os.path.join(FOREGROUND_DIR, target_file+".json")):
        print(f"File {target_file} had been worked before, skipping...")
        return
    if not from_mask:
        c = find_non_background_patches(os.path.join(SLIDE_DIR, target_file+".tiff"), is_mask=False, patch_size = MIL_PATCH_SIZE)
    else:
        c = find_non_background_patches(os.path.join(MASK_DIR, mask_file+"_mask.tiff"), is_mask=True, 
                                        orig_path=os.path.join(SLIDE_DIR, mask_file+".tiff"), patch_size = MIL_PATCH_SIZE)
    save_foreground_info(os.path.join(FOREGROUND_DIR, target_file+".json"), slide_name=target_file, coord_list=c, from_mask=from_mask)
    
if __name__ == "__main__":
    img_files  = [f.split('.tiff')[0] for f in os.listdir(SLIDE_DIR)]
    mask_files = [f.split('_mask')[0] for f in os.listdir(MASK_DIR )]
    problem_files = file_without_mask(img_dir=SLIDE_DIR, mask_dir=MASK_DIR)

    broken_masks = [
        '4a2ca53f240932e46eaf8959cb3f490a',
        'aaa5732cd49bffddf0d2b7d36fbb0a83',
        'e4215cfc8c41ec04a55431cc413688a9',
        '3790f55cad63053e956fb73027179707',
    ]
    problem_files.extend(broken_masks)
    
    print("Check foreground directory exist?")
    if not os.path.isdir(FOREGROUND_DIR):
        os.makedirs(FOREGROUND_DIR)
    
    print("Processing problematic files...")
    Parallel(n_jobs=5)(delayed(process_file_list) (problem_file, 0) for problem_file in tqdm(problem_files))
    
    print("Processing normal files...")
    Parallel(n_jobs=10)(delayed(process_file_list) (mask_file, 0) for mask_file in tqdm(mask_files))
    
    