import os
import numpy as np 
import pandas as pd

# torch library
import torch

# customized libraries
from prostate_model.dataloader import Dataset, XDataLoader, get_train_test
from prostate_model.config import get_cfg_defaults
from prostate_model.model import build_resnet50, build_optimizer, build_baseline
from prostate_model.loss import mean_square_error

if __name__ == "__main__":
    # get global variables
    cfg = get_cfg_defaults()

    # assign GPU
    device = ','.join(str(i) for i in cfg.SYSTEM.DEVICES)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    # prepare train, test dataloader
    x_train, x_test, y_train, y_test = get_train_test(npy_path=cfg.DATASET.TRAIN_DIR, 
                                                      isup_path=cfg.DATASET.ISUP_SCORE_FILE,
                                                      train_ratio=cfg.DATASET.TRAIN_RATIO)
    train_dataset = Dataset(npy_path=cfg.DATASET.TRAIN_DIR, 
                            feature_maps=x_train, 
                            isup_scores=y_train, 
                            target_size=cfg.DATASET.IMG_SIZE, 
                            shuffle=True)
    test_dataset  = Dataset(npy_path=cfg.DATASET.TRAIN_DIR, 
                            feature_maps=x_test, 
                            isup_scores=y_test, 
                            target_size=cfg.DATASET.IMG_SIZE, 
                            shuffle=False)
    train_loader = XDataLoader(train_dataset, batch_size=cfg.MODEL.BATCH_SIZE)
    test_loader  = XDataLoader(test_dataset , batch_size=cfg.MODEL.BATCH_SIZE)

    # prepare resnet50
    model = build_baseline().to('cuda:0')

    # prepare optimizer
    optimizer = build_optimizer(type=cfg.MODEL.OPTIMIZER, model=model, lr=cfg.MODEL.LEARNING_RATE)

    # criterion
    criterion = mean_square_error()

    # prepare for checkpoint info
    if not os.path.isdir(cfg.MODEL.CHECKPOINT_PATH):
        os.makedirs(cfg.MODEL.CHECKPOINT_PATH)
    checkpoint_prefix = f"best.pth"
    
    # prepare training and testing loss
    train_losses = []
    test_losses  = []
    best_loss = 1000
    best_idx  = 0

    # training pipeline
    for epoch in range(cfg.MODEL.EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())

            # compute loss and backpropagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print(f"[{epoch}/{cfg.MODEL.EPOCHS}] [{i}/{len(train_loader)}] training loss={(running_loss)/i}")
            train_losses.append(running_loss/len(train_loader))
        
        # compute test loss
        test_loss = 0.0
        for features, labels in test_loader:
            features = features.cuda().float()
            labels   = labels.cuda()
            with torch.no_grad():
                outputs = model(features)
                test_loss += criterion(labels, outputs).item()
        test_losses.append(test_loss/len(test_loader))
        print(f"[{epoch}/{cfg.MODEL.EPOCHS}]training loss = {train_losses[-1]}, testing loss={test_losses[-1]}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix))

    print('Finished Training')

