import os
import numpy as np
from PIL import Image
import time
from tqdm import tqdm

# torch
import torch
from torchvision import transforms

# customized libraries
from prostate_model.dataloader import InferenceDataset, InfLoader
from hephaestus.models.pyt_resnet.resnet import resnet50
from hephaestus.data.openslide_wrapper_v2 import Slide_OSread

#### Global variables #####
SLIDE_DIR = '/mnt/extension/experiment/prostate-gleason/train_images/'
MASK_DIR  = '/mnt/extension/experiment/prostate-gleason/train_label_masks/'
PATCH_SIZE = 224*4
TRAIN_MAP_DIR = '/workspace/prostate_isup/train_map/'

image_size = (PATCH_SIZE>>2)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

Testtransform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

def inference_slide(slide_name, model, save_path):
    # check if result file exist
    result_file = os.path.join(save_path, slide_name.split('.tiff')[0]+".npy")
    if os.path.isfile(result_file):
        print(f"Skipping: File {slide_name} had been inferenced before.")
        return
    
    inf_dataset = InferenceDataset(slide_dir=SLIDE_DIR, 
                                   slide_name=slide_name, 
                                   mask_dir=MASK_DIR, 
                                   patch_sz=(PATCH_SIZE), 
                                   preproc=Testtransform)
    inf_loader = InfLoader(inf_dataset, batch_size=32)

    feature_map = np.zeros((*inf_dataset.thumbnail_size()[:2], 1000))
    with torch.no_grad():
        for imgs, xs, ys in inf_loader:
            imgs = imgs.cuda()
            out = model(imgs)
            feature_map[ys, xs] = out.cpu()
    
    # save feature_map
    with open(result_file, 'wb') as f:
        np.save(f, feature_map)
    
    del inf_dataset
    del inf_loader

def main():
    # load model
    model = resnet50(pretrained=False).cuda()
    pretrained_dict = torch.load('/workspace/prostate_isup/NLSTv2/model_300_.pth')  
    model_dict = model.state_dict()
    new_dict = {}
    for k, v in pretrained_dict.items():
        # in multiGPU-mode: key will be added a prefiex 'module-
        if k[7:] in model_dict:
            if v.size() == model_dict[k[7:]].size():
                print(k[7:])
                new_dict.update({k[7:]:v})

    model_dict.update(new_dict) 
    model.load_state_dict(model_dict)   
    #print(pretrained_dict['module.bn1.weight'])
    #print(new_dict['bn1.weight'])
    #print(model.bn1.weight) 
    
    # prepare featuremap path
    if not os.path.isdir(TRAIN_MAP_DIR):
        os.makedirs(TRAIN_MAP_DIR)
    for slide_name in tqdm(os.listdir(SLIDE_DIR)):
        inference_slide(slide_name, model, save_path=TRAIN_MAP_DIR)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    main()