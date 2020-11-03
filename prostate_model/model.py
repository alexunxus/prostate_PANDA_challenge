from hephaestus.models.pyt_resnet.resnet import resnet50
from torch import optim
from torch import nn
import torch


def build_resnet50():
    return resnet50(pretrained=True)

def build_baseline():
    # input: (32 , 64 , 128 , 1000)
    intermediate = 10
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1000, 1000, 3, padding=1), # (32, 64, 1000)
        torch.nn.MaxPool2d(2, stride=2), # (32, 64, 1000)
        torch.nn.ReLU(),
        torch.nn.Conv2d(1000, 1000, 3, padding=1),
        torch.nn.MaxPool2d(2, stride=2), # (16, 32, 1000)
        torch.nn.ReLU(),
        torch.nn.Conv2d(1000, 1000, 3, padding=1),
        torch.nn.MaxPool2d(2, stride=2), # (8, 16, 1000)
        torch.nn.ReLU(),
        torch.nn.Conv2d(1000, 1000, 3, padding=1),
        torch.nn.MaxPool2d((1, 2), stride=(1, 2)), # (8, 8, 1000)
        torch.nn.ReLU(),
        torch.nn.Conv2d(1000, 1000, 3, padding=1),
        torch.nn.MaxPool2d(2, stride=2), # (4, 4, 1000)
        torch.nn.ReLU(),
        torch.nn.Conv2d(1000, 1000, 3, padding=1),
        torch.nn.MaxPool2d(2, stride=2), # (2, 2, 1000)
        torch.nn.ReLU(),
        torch.nn.Conv2d(1000, 1000, 3, padding=1),
        torch.nn.MaxPool2d(2, stride=2), # (1, 1, 1000)
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(1000, 1),
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.constant_(m.bias, 0)
    return model

def build_linear_head():
    pass

def build_optimizer(type, model, lr):
    if type=='SGD':
        return optim.SGD(model.parameters(), lr=lr)
    elif type=='Adam':
        return optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type {type}")
