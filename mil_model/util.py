import numpy as np
import random # shuffling
from torch import nn
import pandas as pd
import typing
from typing import Tuple
import os

def replace_bn2gn(model):
    for name, module in model.named_children():
        if len(list(module.named_children())):
            model._modules[name] = replace_bn2gn(module)
        elif type(module) == nn.BatchNorm2d or type(module) == nn.BatchNorm1d:
            layer_new = nn.GroupNorm(num_groups=4, num_channels=module.num_features)
            model._modules[name]=layer_new
    return model

def concat_tiles(tiles, patch_size, tile_sz=6):
    '''concat tiles into (3, tile_sz*patch_size, tile_size*patch_size) image'''
    ret = np.ones((patch_size*tile_sz, patch_size*tile_sz, 3), dtype=np.uint8)*255
    if len(tiles) == 0:
        return ret
    for i in range(tile_sz**2):
        tile = tiles[i%len(tiles)]
        h, w = i//tile_sz, i%tile_sz
        ret[h*patch_size:h*patch_size + tile.shape[0], w*patch_size: w*patch_size+tile.shape[1]] = tile
    return ret

def shuffle_two_arrays(imgs, labels):
    zip_obj = list(zip(imgs, labels))
    random.shuffle(zip_obj)
    return zip(*zip_obj)

def binary_search(li, x, lo, hi):
    mid = (lo+hi)//2
    if li[mid][0] == x:
        return li[mid][1]
    if mid == hi:
        raise ValueError(f"{x} is not in list!")
    if li[mid][0] > x:
        return binary_search(li, x, lo, mid)
    else:
        return binary_search(li, x, mid+1, hi)

class Metric:
    def __init__(self):
        self.metric_dict = {
            'test_kappa':[],'train_kappa':[],'test_acc':[],'train_acc':[],'test_losses':[],'train_losses':[]
        }
    
    def load_metrics(self, csv_path: str, resume: bool) -> Tuple[float, float, int]:
        resume_from_epoch = -1
        best_kappa        = -10
        best_loss         = 1000
        if not os.path.isfile(csv_path) or not resume:
            return best_kappa, best_loss, resume_from_epoch
        # if csv file exist, then first find out the epoch with best kappa(named resume_from_epoch), 
        # get the losses, kappa values within range 0~ best_epoch +1
        df = pd.read_csv(csv_path)
        for key in self.metric_dict.keys():
            if key not in df.columns:
                print(f"Key {key} not found in {df.columns}, not loading csv")
                return best_kappa, best_loss, resume_from_epoch

        test_kappa = list(df['test_kappa'])
        
        best_idx   = np.argmax(np.array(test_kappa))
        best_kappa = test_kappa[best_idx]
        best_loss  = list(df['test_losses'])[best_idx]
        resume_from_epoch = best_idx+1

        for key in self.metric_dict.keys():
            self.metric_dict[key]= list(df[key])[:resume_from_epoch]

        print("================Loading CSV==================")
        print(f"|Loading csv from {csv_path},")
        print(f"|best test loss = {best_loss:.4f},")
        print(f"|best kappa     = {best_kappa:.4f},")
        print(f"|epoch          = {resume_from_epoch:.4f}")
        print("=============================================")
        return best_kappa, best_loss, resume_from_epoch
    
    def save_metrics(self, csv_path: str) -> None:
        '''index: train,test,kappa,train kappa,train acc,test acc'''
        # train_losses,test_losses,test_kappa,train_kappa,train_acc,test_acc
        
        df = pd.DataFrame(self.metric_dict)
        print(df)
        df.to_csv(csv_path, index=False)
    
    def push_loss_acc_kappa(self, loss, acc, kappa, train=True):
        if train:
            self.metric_dict['train_losses'].append(loss)
            self.metric_dict['train_acc'].append(acc)
            self.metric_dict['train_kappa'].append(kappa)
        else: # test/valid
            self.metric_dict['test_losses'].append(loss)
            self.metric_dict['test_acc'].append(acc)
            self.metric_dict['test_kappa'].append(kappa)
    
    def print_summary(self, epoch, total_epoch, lr):
        print(f"[{epoch+1}/{total_epoch}] lr = {lr:.7f}, ", end='')
        for key in self.metric_dict.keys():
            print(f"{key} =  {self.metric_dict[key][epoch]}, ", end='')
        print('\n')
    
    def write_to_tensorboard(self, writer, epoch):
        for key in self.metric_dict.keys():
            writer.add_scalar(key, self.metric_dict[key][-1], epoch)
    
if __name__ == '__main__':
    m = Metric()
    m.load_metrics('../checkpoint/tmp.csv')

    for i in range(10):
        train_ = [np.random.rand() for i in range(3)]
        test_  = [np.random.rand() for i in range(3)]
        m.push_loss_acc_kappa(*train_, train=True)
        m.push_loss_acc_kappa(*test_,  train=False)
    
    m.save_metrics('../checkpoint/tmp.csv')
