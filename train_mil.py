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
from torch import nn

# customized libraries
from mil_model.dataloader import TileDataset, PathoAugmentation, get_train_test, get_resnet_preproc_fn
from mil_model.config import get_cfg_defaults
from mil_model.resnet_model import CustomModel, build_optimizer
from mil_model.loss import get_bceloss, kappa_metric, correct
from mil_model.util import shuffle_two_arrays, Metric
from mil_model.sync_bn.sync_batchnorm.batchnorm import convert_model

if __name__ == "__main__":
    # get config variables
    cfg = get_cfg_defaults()

    # assign GPU
    device = ','.join(str(i) for i in cfg.SYSTEM.DEVICES)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    # prepare train, test dataloader
    # x_train and x_test are list of image names that are split to train and test set, respectively.
    print("============Spliting datasets================")
    x_train, x_test = get_train_test(json_dir = cfg.DATASET.JSON_DIR, ratio=1-cfg.DATASET.TRAIN_RATIO)

    train_dataset = TileDataset(img_names =x_train, 
                                img_dir   =cfg.DATASET.IMG_DIR, 
                                json_dir  =cfg.DATASET.JSON_DIR, 
                                label_path=cfg.DATASET.LABEL_FILE, 
                                patch_size=cfg.DATASET.PATCH_SIZE, 
                                resize_ratio=cfg.DATASET.RESIZE_RATIO,
                                tile_size =cfg.DATASET.TILE_SIZE, 
                                is_test   =False,
                                aug       =PathoAugmentation, 
                                preproc   =get_resnet_preproc_fn())
    test_dataset  = TileDataset(img_names =x_test, 
                                img_dir   =cfg.DATASET.IMG_DIR, 
                                json_dir   =cfg.DATASET.JSON_DIR, 
                                label_path =cfg.DATASET.LABEL_FILE, 
                                patch_size =cfg.DATASET.PATCH_SIZE, 
                                resize_ratio=cfg.DATASET.RESIZE_RATIO,
                                tile_size  =cfg.DATASET.TILE_SIZE, 
                                is_test    =True,
                                aug        =None, 
                                preproc    =get_resnet_preproc_fn())
                        
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg.MODEL.BATCH_SIZE, 
                              shuffle=True , 
                              num_workers=2,
                              drop_last=True)
    # NOTE: the batch size of test loader is 4 times smaller than traning loader
    #       since test dataset returns a list of "4" image tensor.
    # However, the batch size does not really matter if the training resources is enough to 
    # accommodate 4 times the batch size of training loader.
    test_loader  = DataLoader(test_dataset , 
                              batch_size=cfg.MODEL.BATCH_SIZE//4, 
                              shuffle=False, 
                              num_workers=2,
                              drop_last=True
                              )

    # prepare for checkpoint info
    if not os.path.isdir(cfg.MODEL.CHECKPOINT_PATH):
        os.makedirs(cfg.MODEL.CHECKPOINT_PATH)
    checkpoint_prefix = f"{cfg.MODEL.BACKBONE}_{cfg.DATASET.TILE_SIZE}_{cfg.DATASET.PATCH_SIZE}"

    # prepare resnet50, resume from checkpoint
    print("==============Building model=================")
    model = CustomModel(backbone=cfg.MODEL.BACKBONE, 
                        num_grade=cfg.DATASET.NUM_GRADE, 
                        resume_from=cfg.MODEL.RESUME_FROM,
                        norm=cfg.MODEL.NORM_USE)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs...")
        model = nn.DataParallel(model)
    model = model.cuda()

    # prepare optimizer: Adam is suggested in this case.
    warmup_epo    = 3
    n_epochs      = cfg.MODEL.EPOCHS
    optimizer = build_optimizer(type=cfg.MODEL.OPTIMIZER, 
                                model=model, 
                                lr=cfg.MODEL.LEARNING_RATE)

    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=3e-6, T_max=(n_epochs-warmup_epo))
    scheduler = GradualWarmupScheduler(optimizer, 
                                       multiplier=1.0, 
                                       total_epoch=warmup_epo,
                                       after_scheduler=base_scheduler)

    # criterion: BCE loss for multilabel tensor with shape (BATCH_SIZE, 5)
    criterion = get_bceloss()

    # prepare tensorboard writer
    if cfg.SOURCE.TENSORBOARD:
        writer = SummaryWriter()
    
    # prepare training and testing loss
    best_idx               = 0
    patience               = 0
    loss_kappa_acc_metrics = Metric()
    csv_path               = os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"loss.csv")
    best_kappa, best_loss, resume_from_epoch = loss_kappa_acc_metrics.load_metrics(csv_path, resume=cfg.MODEL.LOAD_CSV)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optimizer.zero_grad()
    optimizer.step()

    # training pipeline
    print("==============Start training=================")
    for epoch in range(0, cfg.MODEL.EPOCHS):  # loop over the dataset multiple times
        # update scheduler  
        if epoch < resume_from_epoch:
            scheduler.step()
            optimizer.step()
            continue
        scheduler.step()
        
        # ===============================================================================================
        #                                 Train for loop block
        # ===============================================================================================
        total_loss    = 0.0
        train_correct = 0.
        predictions   = []
        groundtruth   = []
        pbar = tqdm(enumerate(train_loader, 0))
        
        # tracking data time and GPU time and print them on tqdm bar.
        end_time = time.time()

        model.train()
        for i, data in pbar:

            # get the inputs; data is a list of [inputs, labels], put them in cuda
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            data_time = time.time()-end_time # data time
            end_time = time.time()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # feed forward
            outputs = model(inputs)

            # compute loss and backpropagation
            loss = criterion(outputs, labels)
            loss.backward()
            loss = loss.detach()

            optimizer.step()
            gpu_time = time.time()-end_time # gpu time

            # collect statistics
            predictions.append(outputs.detach().cpu())
            groundtruth.append(labels.detach().cpu())

            running_loss  =  loss.item()
            total_loss    += running_loss
            train_correct += correct(predictions[-1], groundtruth[-1])

            pbar.set_postfix_str(f"[{epoch}/{cfg.MODEL.EPOCHS}] [{i+1}/{len(train_loader)}] "+
                                 f"training loss={running_loss:.4f}, data time = {data_time:.4f}, gpu time = {gpu_time:.4f}")
            end_time = time.time()

        loss_kappa_acc_metrics.push_loss_acc_kappa(loss=total_loss/len(train_loader), 
                                                   acc=train_correct/len(train_dataset), 
                                                   kappa=kappa_metric(groundtruth, predictions), 
                                                   train=True,
                                                   )
        
        # ===============================================================================================
        #                                     TEST for loop block
        # ===============================================================================================
        test_total_loss = 0.0
        predictions = []
        groundtruth = []
        model.eval()
        for batch_4tensors, labels in tqdm(test_loader):
            #===============================================================================
            #    batch_4tensors is 4-tensor with shape (batch_size//4, 4, 3, 1536, 1536), 
            #    labels is a "single" tensor (batch_size, 5)
            #===============================================================================
            # 
            # The first dimension of the batch_4tensors is batch_size//4  because I take 4 images from each test slide 
            # and will ensemble the result.
            # Firstly, reshape the image tensor by combining the first two channel 
            #          (batch_size//4, 4, 3, 1536, 1536)--> (batch_size, 3, 1536, 1536)
            # After the tensor passes the model, I will resume the first two dimension of the output
            # to a (batch_size//4, 4, 5) tensor, and average the results by axis=1 --> get a shape (batch_size//4, 5)
            # output, similar to labels
            with torch.no_grad():
                first_two_dim = batch_4tensors.shape[:2]
                batch_tensors = batch_4tensors.view(-1, *batch_4tensors.shape[2:]).cuda()
                labels        = labels.cuda()
                outputs       = model(batch_tensors)
                outputs       = torch.mean(outputs.view((*first_two_dim, -1)), axis=1)
                test_loss     = criterion(outputs, labels).item()
                test_total_loss += test_loss

                predictions.append(outputs.cpu())
                groundtruth.append(labels.cpu())
        
        kappa = kappa_metric(groundtruth, predictions)
        test_epoch_loss = test_total_loss/len(test_loader)
        loss_kappa_acc_metrics.push_loss_acc_kappa(loss=test_epoch_loss,
                                                   acc=correct(predictions, groundtruth)/(len(predictions)*predictions[0].shape[0]), 
                                                   kappa=kappa, train=False,
                                                   )

        loss_kappa_acc_metrics.print_summary(epoch=epoch, total_epoch=cfg.MODEL.EPOCHS, lr=optimizer.param_groups[0]['lr'])
        
        if cfg.SOURCE.TENSORBOARD:
            loss_kappa_acc_metrics.write_to_tensorboard(writer=writer, epoch=epoch)
            writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)


        update_loss  = False
        update_kappa = False
        
        # ===============================================================================================
        #                                     Callback block
        # ===============================================================================================
        # 1. Save best weight: 
        #   If for one epoch, the test loss or kappa is better than current best kappa and best loss, then
        #   I reset the patience and save the model
        #   Otherwise, patience <- patience+1, and the model weight is not saved.
        # 2. Early stopping: 
        #   If the patience >= the patience limit, then break the epoch for loop and finish training.
        # 3. Saving training curve:
        #   For each epoch, update the loss, kappa dictionary and save them.

        if cfg.SOURCE.DEBUG:
            print("Debugging, not saving...")
            loss_kappa_acc_metrics.save_metrics(csv_path=csv_path)
            print(f"best loss={best_loss}, best kappa={best_kappa}")
        else:
            print('======Saving training curves=======')
            loss_kappa_acc_metrics.save_metrics(csv_path=csv_path)
        
        if test_epoch_loss < best_loss:
            best_loss = test_epoch_loss
            if not cfg.SOURCE.DEBUG:
                torch.save(model.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"best.pth"))
            patience = 0
            update_loss = True
        if kappa >= best_kappa:
            best_kappa = kappa
            if not cfg.SOURCE.DEBUG:
                torch.save(model.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"best_kappa.pth"))
            patience = 0
            update_kappa = True
        if not update_loss and not update_kappa:
            patience += 1
            print(f"Patience = {patience}")
            if patience >= cfg.MODEL.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print('Finished Training')

