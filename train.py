import os
import numpy as np 
import pandas as pd
from tqdm import tqdm

# torch library
import torch
from torch.utils.data import DataLoader

# customized libraries
from prostate_model.dataloader import Dataset, get_train_test
from prostate_model.config import get_cfg_defaults
from prostate_model.model import build_resnet50, build_optimizer, build_baseline
from prostate_model.loss import mean_square_error, get_ranking_loss

if __name__ == "__main__":
    # get global variables
    cfg = get_cfg_defaults()

    # assign GPU
    device = ','.join(str(i) for i in cfg.SYSTEM.DEVICES)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    # prepare train, test dataloader
    print("=======Spliting datasets===========")
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
                        
    train_loader = DataLoader(train_dataset, batch_size=cfg.MODEL.BATCH_SIZE, shuffle=True , num_workers=2)
    test_loader  = DataLoader(test_dataset , batch_size=cfg.MODEL.BATCH_SIZE, shuffle=False, num_workers=2)

    # prepare resnet50
    print("=========Building model============")
    model = build_baseline().cuda()

    # prepare optimizer
    optimizer = build_optimizer(type=cfg.MODEL.OPTIMIZER, model=model, lr=cfg.MODEL.LEARNING_RATE)

    # criterion
    criterion = get_ranking_loss()

    # prepare for checkpoint info
    if not os.path.isdir(cfg.MODEL.CHECKPOINT_PATH):
        os.makedirs(cfg.MODEL.CHECKPOINT_PATH)
    checkpoint_prefix = f"{cfg.MODEL.BACKBONE}"
    
    # prepare training and testing loss
    train_losses = []
    test_losses  = []
    best_loss = 1000
    best_idx  = 0

    # training pipeline
    print("=========Start training============")
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
        for features, labels in test_loader:
            features = features.cuda()
            labels   = labels.cuda()
            with torch.no_grad():
                outputs = model(features).squeeze()
                test_loss += criterion(labels, outputs).item()
                num_data += outputs.shape[0]
        test_losses.append(test_loss/num_data)
        print(f"[{epoch}/{cfg.MODEL.EPOCHS}]training loss = {train_losses[-1]}, testing loss={test_losses[-1]}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"best.pth"))
    
    print('======Saving training curves=======')
    loss_dict = {}
    loss_dict["train"] = train_losses
    loss_dict["test"]  = test_losses
    df = pd.DataFrame(loss_dict)
    df.to_csv(os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"loss.csv"))
    
    print('Finished Training')

