import os
import numpy as np 
import pandas as pd
from tqdm import tqdm
import time

# torch library
import torch
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter

# customized libraries
from mil_model.dataloader import TileDataset, PathoAugmentation, get_train_test
from mil_model.config import get_cfg_defaults
from mil_model.resnet_model import BaselineResNet50, CustomModel, build_optimizer
from mil_model.loss import get_bceloss, kappa_metric, correct
from mil_model.util import shuffle_two_arrays

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
                                is_test=False,
                                aug=PathoAugmentation, 
                                preproc=None)
    test_dataset  = TileDataset(img_names=x_test, 
                                img_dir=cfg.DATASET.IMG_DIR, 
                                json_dir=cfg.DATASET.JSON_DIR, 
                                label_path=cfg.DATASET.LABEL_FILE, 
                                patch_size=cfg.DATASET.PATCH_SIZE, 
                                tile_size=cfg.DATASET.TILE_SIZE, 
                                is_test=True,
                                aug=None, 
                                preproc=None)
                        
    train_loader = DataLoader(train_dataset, batch_size=cfg.MODEL.BATCH_SIZE, shuffle=True , num_workers=2)
    test_loader  = DataLoader(test_dataset , 
                              batch_size=cfg.MODEL.BATCH_SIZE, 
                              shuffle=False, 
                              num_workers=2)

    # prepare for checkpoint info
    if not os.path.isdir(cfg.MODEL.CHECKPOINT_PATH):
        os.makedirs(cfg.MODEL.CHECKPOINT_PATH)
    checkpoint_prefix = f"{cfg.MODEL.BACKBONE}_{cfg.DATASET.TILE_SIZE}_{cfg.DATASET.PATCH_SIZE}"

    # prepare resnet50, resume from checkpoint
    print("==============Building model=================")
    if os.path.isfile(cfg.MODEL.RESUME_FROM):
        resume_path = cfg.MODEL.RESUME_FROM
    else:
        resume_path = None
    
    if cfg.MODEL.BACKBONE == 'baseline':
        model = BaselineResNet50(num_grade=cfg.DATASET.NUM_GRADE, resume_from=resume_path).cuda()
    else:
        model = CustomModel(backbone=cfg.MODEL.BACKBONE, num_grade=cfg.DATASET.NUM_GRADE, resume_from=resume_path).cuda()

    # prepare optimizer
    warmup_factor = 10
    warmup_epo = 1
    n_epochs = cfg.MODEL.EPOCHS
    optimizer = build_optimizer(type=cfg.MODEL.OPTIMIZER, model=model, lr=cfg.MODEL.LEARNING_RATE/warmup_factor)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs-warmup_epo)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

    # criterion
    criterion = get_bceloss()

    # tensorboard writer
    if cfg.SOURCE.TENSORBOARD:
        writer = SummaryWriter()
    
    # prepare training and testing loss
    # will save the model with best loss only and have the patience=10
    train_losses = []
    test_losses  = []
    train_acc = []
    test_acc  = []
    kappa_values = []
    resume_from_epoch = 0
    best_loss = 1000
    best_kappa= -10
    best_idx  = 0
    patience  = 0
    if (os.path.isfile(os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"loss.csv")) and 
        cfg.MODEL.RESUME_FROM != ''):
        csv_path = os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"loss.csv")
        df = pd.read_csv(os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"loss.csv"))
        kappa_values = list(df['kappa'])
        best_kappa = max(kappa_values)
        resume_from_epoch = best_idx = np.argmax(np.array(best_kappa))
        kappa_values = kappa_values[:best_idx+1]
        train_losses = list(df['train'])[:best_idx+1]
        test_losses  = list(df['test'])[:best_idx+1]
        if 'train acc' in df.columns:
            train_acc = list(df['train acc'])[:best_idx+1]
            test_acc  = list(df['test acc'])[:best_idx+1]
        best_loss = min(test_losses)
        print(f"Loading csv from {csv_path}, best test loss = {best_loss}, best kappa = {best_kappa}, epoch = {resume_from_epoch}")

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optimizer.zero_grad()
    optimizer.step()

    # training pipeline
    print("==============Start training=================")
    for epoch in range(0, cfg.MODEL.EPOCHS):  # loop over the dataset multiple times
        # update scheduler 
       if epoch <= resume_from_epoch:
            scheduler.step()
            optimizer.step()
            continue
        scheduler.step()
        
        total_loss = 0.0
        train_correct = 0.
        pbar = tqdm(enumerate(train_loader, 0))
        
        end_time = time.time()
        
        model.train()
        for i, data in pbar:

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            data_time = time.time()-end_time

            end_time = time.time()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimizer
            outputs = model(inputs)

            # compute loss and backpropagation
            loss = criterion(outputs, labels)
            loss.backward()

            # debug
            if cfg.SOURCE.DEBUG:
                print(model.backbone.conv1.weight.grad)

            optimizer.step()
            gpu_time = time.time()-end_time

            # print statistics
            total_loss   += loss.item()
            running_loss =  loss.item()

            train_correct += correct(outputs.detach().cpu(), labels.detach().cpu())
            
            pbar.set_postfix_str(f"[{epoch}/{cfg.MODEL.EPOCHS}] [{i+1}/{len(train_loader)}] training loss={running_loss:.4f}, data time = {data_time:.4f}, gpu time = {gpu_time:.4f}")
            end_time = time.time()
            
            # tensorboard writer
            if cfg.SOURCE.TENSORBOARD:
                writer.add_scalar("Running loss", running_loss, i)

        train_losses.append(total_loss/len(train_loader))
        train_acc.append(train_correct/len(train_dataset))
        
        # compute test loss
        test_total_loss = 0.0
        predictions = []
        groundtruth = []
        model.eval()
        for features, labels in tqdm(test_loader):
            # features is a list of 4 tensors
            # labels is a tensor
            with torch.no_grad():
                first_two_dim = features.shape[:2]
                features = features.view(-1, *features.shape[2:]).cuda()
                labels   = labels.cuda()
                outputs = model(features)
                outputs = torch.mean(outputs.view((*first_two_dim, -1)), axis=1)
                test_loss = criterion(outputs, labels).item()
                test_total_loss += test_loss

                predictions.append(outputs.cpu())
                groundtruth.append(labels.cpu())
        test_losses.append(test_total_loss/len(test_loader))
        test_acc.append(correct(predictions, groundtruth)/(len(predictions)*predictions[0].shape[0]))
        
        # compute quadratic kappa value:
        kappa = kappa_metric(groundtruth, predictions)
        kappa_values.append(kappa)
        print(f"[{epoch+1}/{cfg.MODEL.EPOCHS}] lr = {optimizer.param_groups[0]['lr']:.7f}, training loss = {train_losses[-1]:.5f}"+
              f", testing loss={test_losses[-1]:.5f}, kappa={kappa:.5f}, train acc={train_acc[-1]:.5f}, test acc = {test_acc[-1]:.5f}")
        
        if cfg.SOURCE.TENSORBOARD:
            writer.add_scalar("Training loss", train_losses[-1], epoch)
            writer.add_scalar("Training acc", train_acc[-1], epoch)
            writer.add_scalar("Testing loss", test_losses[-1], epoch)
            writer.add_scalar("Testing acc", test_acc[-1], epoch)
            writer.add_scalar("Test Kappa", kappa_values[-1], epoch)
            writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)

        if cfg.SOURCE.DEBUG:
            print("Debugging, not saving...")
            continue

        update_loss  = False
        update_kappa = False
        
        if test_losses[-1] < best_loss:
            best_loss = test_losses[-1]
            torch.save(model.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"best.pth"))
            patience = 0
            update_loss = True
        if kappa >= best_kappa:
            best_kappa = kappa
            torch.save(model.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"best_kappa.pth"))
            patience = 0
            update_kappa = True
        if not update_loss and not update_kappa:
            patience += 1
            print(f"Patience = {patience}")
            if patience >= cfg.MODEL.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print('======Saving training curves=======')
        loss_dict = {}
        loss_dict["train"]     = train_losses
        loss_dict["test"]      = test_losses
        loss_dict["kappa"]     = kappa_values
        loss_dict["train acc"] = train_acc
        loss_dict["test acc"]  = test_acc
        df = pd.DataFrame(loss_dict)
        df.to_csv(os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"loss.csv"), index=False)
    
    print('Finished Training')

