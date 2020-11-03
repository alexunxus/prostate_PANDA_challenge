import os
import numpy as np 
import pandas as pd
from tqdm import tqdm

# torch library
import torch
from torch.utils.data import DataLoader

# customized libraries
from mil_model.dataloader import TileDataset, PathoAugmentation, get_train_test, PathoAugmentation
from mil_model.config import get_cfg_defaults
from mil_model.resnet_model import BaselineResNet50, build_optimizer
from mil_model.loss import get_bceloss

if __name__ == "__main__":
    # get global variables
    cfg = get_cfg_defaults()

    # assign GPU
    device = ','.join(str(i) for i in cfg.SYSTEM.DEVICES)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    # prepare train, test dataloader
    print("============Spliting datasets================")
    x_train, x_test = get_train_test(json_dir = cfg.DATASET.JSON_DIR, ratio=1-cfg.DATASET.TRAIN_RATIO)

    train_dataset = TileDataset(img_names=x_train, 
                                img_dir=cfg.DATASET.IMG_DIR, 
                                json_dir=cfg.DATASET.JSON_DIR, 
                                label_path=cfg.DATASET.LABEL_FILE, 
                                patch_size=cfg.DATASET.PATCH_SIZE, 
                                tile_size=cfg.DATASET.TILE_SIZE, 
                                aug=PathoAugmentation, 
                                preproc=None)
    test_dataset  = TileDataset(img_names=x_test, 
                                img_dir=cfg.DATASET.IMG_DIR, 
                                json_dir=cfg.DATASET.JSON_DIR, 
                                label_path=cfg.DATASET.LABEL_FILE, 
                                patch_size=cfg.DATASET.PATCH_SIZE, 
                                tile_size=cfg.DATASET.TILE_SIZE, 
                                aug=None, 
                                preproc=None)
                        
    train_loader = DataLoader(train_dataset, batch_size=cfg.MODEL.BATCH_SIZE, shuffle=True , num_workers=2)
    test_loader  = DataLoader(test_dataset , batch_size=cfg.MODEL.BATCH_SIZE, shuffle=False, num_workers=2)

    # prepare resnet50
    print("==============Building model=================")
    model = BaselineResNet50(num_grade=cfg.DATASET.NUM_GRADE).cuda()

    # prepare optimizer
    optimizer = build_optimizer(type=cfg.MODEL.OPTIMIZER, model=model, lr=cfg.MODEL.LEARNING_RATE)

    # criterion
    criterion = get_bceloss()

    # prepare for checkpoint info
    if not os.path.isdir(cfg.MODEL.CHECKPOINT_PATH):
        os.makedirs(cfg.MODEL.CHECKPOINT_PATH)
    checkpoint_prefix = f"{cfg.MODEL.BACKBONE}_{cfg.DATASET.TILE_SIZE}_"
    
    # prepare training and testing loss
    train_losses = []
    test_losses  = []
    kappa_values = []
    best_loss = 1000
    best_idx  = 0

    # training pipeline
    print("==============Start training=================")
    for epoch in range(cfg.MODEL.EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        num_data = 0
        pbar = tqdm(enumerate(train_loader, 0))
        for i, data in pbar:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs).squeeze()

            # compute loss and backpropagation
            loss = criterion(outputs, labels)
            loss.backward()

            print(model[0].weight.grad)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            num_data += labels.shape[0]
            pbar.set_postfix_str(f"[{epoch}/{cfg.MODEL.EPOCHS}] [{i}/{len(train_loader)}] training loss={(running_loss)/num_data}")
            train_losses.append(running_loss/len(train_dataset))
        
        # compute test loss
        test_loss = 0.0
        num_data = 0
        predictions = []
        groundtruth = []
        for features, labels in test_loader:
            features = features.cuda()
            labels   = labels.cuda()
            with torch.no_grad():
                outputs = model(features).squeeze()
                test_loss += criterion(labels, outputs).item()
                num_data += outputs.shape[0]
                predictions.append(outputs.cpu())
                groundtruth.append(labels.cpu())
        test_losses.append(test_loss/num_data)
        # compute quadratic kappa value:
        kappa = kappa_metric(groundtruth, predictions)
        kappa_values.append(kappa)
        print(f"[{epoch}/{cfg.MODEL.EPOCHS}]training loss = {train_losses[-1]}, testing loss={test_losses[-1]}, kappa={kappa}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"best.pth"))
    
    print('======Saving training curves=======')
    loss_dict = {}
    loss_dict["train"] = train_losses
    loss_dict["test"]  = test_losses
    loss_dict["kappa"] = kappa_values
    df = pd.DataFrame(loss_dict)
    df.to_csv(os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"loss.csv"))
    
    print('Finished Training')

