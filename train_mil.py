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
from mil_model.dataloader import TileDataset, PathoAugmentation, get_train_test, get_resnet_preproc_fn
from mil_model.config import get_cfg_defaults
from mil_model.resnet_model import CustomModel, build_optimizer
from mil_model.loss import get_bceloss, kappa_metric, correct
from mil_model.util import shuffle_two_arrays

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
                              num_workers=2)
    # NOTE: the batch size of test loader is 4 times smaller than traning loader
    #       since test dataset returns a list of "4" image tensor.
    # However, the batch size does not really matter if the training resources is enough to 
    # accommodate 4 times the batch size of training loader.
    test_loader  = DataLoader(test_dataset , 
                              batch_size=cfg.MODEL.BATCH_SIZE//4, 
                              shuffle=False, 
                              num_workers=2)

    # prepare for checkpoint info
    if not os.path.isdir(cfg.MODEL.CHECKPOINT_PATH):
        os.makedirs(cfg.MODEL.CHECKPOINT_PATH)
    checkpoint_prefix = f"{cfg.MODEL.BACKBONE}_{cfg.DATASET.TILE_SIZE}_{cfg.DATASET.PATCH_SIZE}"

    # prepare resnet50, resume from checkpoint
    print("==============Building model=================")
    model = CustomModel(backbone=cfg.MODEL.BACKBONE, 
                        num_grade=cfg.DATASET.NUM_GRADE, 
                        resume_from=cfg.MODEL.RESUME_FROM
                        norm=cfg.MODEL.NORM_USE).cuda()

    # prepare optimizer: Adam is suggested in this case.
    warmup_epo    = 3
    n_epochs      = cfg.MODEL.EPOCHS
    optimizer = build_optimizer(type=cfg.MODEL.OPTIMIZER, 
                                model=model, 
                                lr=cfg.MODEL.LEARNING_RATE)
    
    # base_scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
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
    # will save the model with best loss only and have the patience=10
    train_losses      = []
    test_losses       = []
    train_acc         = []
    test_acc          = []
    test_kappa        = []
    train_kappa       = []
    resume_from_epoch = -1
    best_loss         = 1000
    best_kappa        = -10
    best_idx          = 0
    patience          = 0
    if (os.path.isfile(os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"loss.csv")) and 
        cfg.MODEL.LOAD_CSV):
        # if csv file exist, then first find out the epoch with best kappa(named resume_from_epoch), 
        # get the losses, kappa values within range 0~ best_epoch +1
        csv_path = os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"loss.csv")
        df = pd.read_csv(os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"loss.csv"))
        test_kappa = list(df['kappa'])
        best_kappa = max(test_kappa)
        
        best_idx = np.argmax(np.array(test_kappa))
        resume_from_epoch = best_idx + 1
        
        test_kappa = test_kappa[:best_idx+1]
        train_losses = list(df['train'])[:best_idx+1]
        test_losses  = list(df['test'])[:best_idx+1]
        if 'train acc' in df.columns:
            train_acc = list(df['train acc'])[:best_idx+1]
            test_acc  = list(df['test acc'])[:best_idx+1]
        if 'train kappa' in df.columns:
            train_kappa=list(df['train kappa'])[:best_idx+1]
        else:
            train_kappa =[0 for i in range(len(train_kappa))]
        best_loss = min(test_losses)
        print(f"Loading csv from {csv_path}, best test loss = {best_loss},"+
              f" best kappa = {best_kappa}, epoch = {resume_from_epoch}")

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
        labels        = []
        pbar = tqdm(enumerate(train_loader, 0))
        
        # tracking data time and GPU time and print them on tqdm bar.
        end_time = time.time()

        model.train()
        for i, data in pbar:

            # get the inputs; data is a list of [inputs, labels], put them in cuda
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            data_time = time.time()-end_time
            end_time = time.time()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # feed forward
            outputs = model(inputs)

            # compute loss and backpropagation
            loss = criterion(outputs, labels)
            loss.backward()

            # Collect labels and predictions
            predictions.append(outputs.cpu())
            groundtruth.append(labels.cpu())

            # debug
            if cfg.SOURCE.DEBUG:
                print(model.backbone.conv1.weight.grad)

            optimizer.step()
            gpu_time = time.time()-end_time

            # print statistics
            total_loss   += loss.item()
            running_loss =  loss.item()

            train_correct += correct(outputs.detach().cpu(), labels.detach().cpu())

            pbar.set_postfix_str(f"[{epoch}/{cfg.MODEL.EPOCHS}] [{i+1}/{len(train_loader)}] "+
                                 f"training loss={running_loss:.4f}, data time = {data_time:.4f}, gpu time = {gpu_time:.4f}")
            end_time = time.time()

        train_kappa.append(kappa_metric(groundtruth, predictions))
        train_losses.append(total_loss/len(train_loader))
        train_acc.append(train_correct/len(train_dataset))
        
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
        test_losses.append(test_total_loss/len(test_loader))
        test_acc.append(correct(predictions, groundtruth)/(len(predictions)*predictions[0].shape[0]))
        
        # compute quadratic kappa value:
        test_kappa.append(kappa_metric(groundtruth, predictions))
        print(f"[{epoch+1}/{cfg.MODEL.EPOCHS}] lr = {optimizer.param_groups[0]['lr']:.7f}, training loss = {train_losses[-1]:.5f}"+
              f", testing loss={test_losses[-1]:.5f}, test kappa={test_kappa[-1]:.5f}, train kappa={train_kappa[-1]:.5f}"+
              f", train acc={train_acc[-1]:.5f}, test acc = {test_acc[-1]:.5f}")
        
        if cfg.SOURCE.TENSORBOARD:
            writer.add_scalar("Training loss", train_losses[-1]               , epoch)
            writer.add_scalar("Training acc" , train_acc[-1]                  , epoch) # can be omitted
            writer.add_scalar("Testing loss" , test_losses[-1]                , epoch)
            writer.add_scalar("Testing acc"  , test_acc[-1]                   , epoch)   # can be omitted
            writer.add_scalar("Test Kappa"   , test_kappa[-1]                 , epoch)
            writer.add_scalar("Train kappa"  , train_kappa[-1]                , epoch)
            writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)

        if cfg.SOURCE.DEBUG:
            print("Debugging, not saving...")
            continue

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

        print('======Saving training curves=======')
        loss_dict = {}
        loss_dict["train"]      = train_losses
        loss_dict["test"]       = test_losses
        loss_dict["kappa"]      = test_kappa
        loss_dict["train kappa"] =train_kappa
        loss_dict["train acc"]  = train_acc
        loss_dict["test acc"]   = test_acc
        df = pd.DataFrame(loss_dict)
        df.to_csv(os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"loss.csv"), index=False)

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
    
    print('Finished Training')

