from yacs.config import CfgNode as CN

_C = CN() # Node, lv0
_C.SYSTEM = CN() # None, lv1
_C.SYSTEM.DEVICES = [5]

_C.SOURCE = CN()
_C.SOURCE.RESULT_DIR  = "" # Full path to store the result
_C.SOURCE.DEBUG       = False
_C.SOURCE.TENSORBOARD = True

_C.DATASET = CN()
_C.DATASET.JSON_DIR        = './foreground/data_1024/'
_C.DATASET.IMG_DIR         = '/mnt/extension/experiment/prostate-gleason/train_images/'
_C.DATASET.LABEL_FILE      = '/mnt/extension/experiment/prostate-gleason/train.csv'
_C.DATASET.TRAIN_RATIO     = 0.90 # if VALID_DIR is [], then it works

_C.DATASET.NUM_GRADE       = 5
_C.DATASET.PATCH_SIZE      = 1024
_C.DATASET.TILE_SIZE       = 8

_C.MODEL = CN()
_C.MODEL.BACKBONE          = 'enet-b0' #'baseline', 'R-50-xt', 'R-50-st', 'enet-b0', 'enet-b1'
_C.MODEL.BATCH_SIZE        = 4
_C.MODEL.EPOCHS            = 30
_C.MODEL.LEARNING_RATE     = 3e-4
_C.MODEL.USE_PRETRAIN      = True
_C.MODEL.NORM_USE          = "bn" # bn, gn
_C.MODEL.OPTIMIZER         = "Adam" #"SGD" # SGD, Adam
_C.MODEL.CRITERION         = "BCE"
_C.MODEL.CHECKPOINT_PATH   = '/workspace/prostate_isup/checkpoint/'
_C.MODEL.RESUME_FROM       = ''
_C.MODEL.PATIENCE          = 8

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()