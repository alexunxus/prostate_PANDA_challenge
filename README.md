# Prostate PANDA challenge ISUP score grading
## Introduction

Prostate cancer is the most common malignant cancer type in Western countries. Thought the prognosis of prostate cancer is generally better than other types of malignancies, the disease burden can still impose great impact on health system. The main prognostic factor of prostate cancer depends on the grading of prostate biopsy, i.e. Gleason score and ISUP score. In this project we design a multiple-instance learning model based on efficient nets or residual nets and optimize the quadratic weighted kappa score of the model, featuring good inter-rater reliability. 

## Data Summary
The PANDA dataset can be acquired from the Kaggle challenge website:  
https://www.kaggle.com/c/prostate-cancer-grade-assessment

The dataset is compsed of 10616 tiff images saved in 1x, 4x resolutions. About half of them are from Karolinska(51%) medical center, and the other half are provided by Radbound (49%) medical center. The ISUP score distribution of the dataset is moderately imbalanced. 

## Preparation for Training

Firstly, go to the folder `./foreground/` to generate the foreground boxes. In `./foreground/preprocess.py` , the desired patch size of foreground box can be customized by modifying `MIL_PATCH_SIZE`,
    MIL_PATCH_SIZE= 784
  
Additionally, the foreground box information will be stored in folder 
    FOREGROUND_DIR = f'/workspace/prostate_isup/foreground/data_{MIL_PATCH_SIZE}/'
. For example, in `data_512/71abdfb52804bed1259c90f7b414e178.json`, it is a dictionary
    {"71abdfb52804bed1259c90f7b414e178": [{"coord_list": [[1, 1], [2, 1], [3, 1], [4, 1], [1, 2], [2, 2], [3, 2], [4, 2], [2, 3], [3, 3], [4, 3], [5, 3], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [36, 4], [37, 4], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [8, 5], [9, 5], [10, 5], [12, 5], [14, 5], [20, 5], [21, 5], [22, 5], [23, 5], [24, 5], [35, 5], [36, 5], [37, 5], [4, 6], [5, 6], [6, 6], [7, 6], [8, 6], [9, 6], [10, 6], [11, 6], [12, 6], [13, 6], [14, 6], [15, 6], [16, 6], [17, 6], [18, 6], [19, 6], [20, 6], [21, 6], [22, 6], [23, 6], [24, 6], [25, 6], [31, 6], [32, 6], [33, 6], [34, 6], [35, 6], [5, 7], [6, 7], [7, 7], [8, 7], [9, 7], [10, 7], [11, 7], [12, 7], [13, 7], [14, 7], [15, 7], [16, 7], [17, 7], [18, 7], [19, 7], [20, 7], [21, 7], [22, 7], [23, 7], [24, 7], [25, 7], [26, 7], [29, 7], [30, 7], [31, 7], [32, 7], [33, 7], [34, 7], [35, 7], [36, 7], [38, 7], [8, 8], [9, 8], [11, 8], [12, 8], [15, 8], [16, 8], [17, 8], [18, 8], [19, 8], [20, 8], [21, 8], [24, 8], [25, 8], [26, 8], [27, 8], [28, 8], [29, 8], [30, 8], [31, 8], [35, 8], [36, 8], [37, 8], [24, 9], [25, 9], [26, 9], [27, 9], [28, 9], [29, 9], [30, 9], [34, 9], [27, 10]], "from_mask": 0}]}
  
, which means the square patch with left-upper corner `w=1*512, h=1*512` and `side=512` is a foreground patch.  

The time used in getting the foreground box from the 10616 tiff images is about 20 minutes. 

## Multiple instance learning

In this project, we use a concatenated tile pooling image(e.g. 36-tiled 256-sized image) to represent the image acqured from one tiff file. Let's take 36-tile, patch size 1024(under 4x resolution is 256) images. These image patches are randomly selected from the coordinate list according to the foreground json file. The selected image can be in 1x or 4x resolution, however, the **4x resolution** is highly recommended since the image can be acquired faster in 4x image pyramid. Discrete augmentation is implemented to every patches before concatenation. Complete augmnetation such as hue/saturation variation is administered to the concatenated 36 tiles image afterward. 

## Test time augmentation

We get 4 patterns of patches from each test slide and 


## Efficient net and ResNet backbone