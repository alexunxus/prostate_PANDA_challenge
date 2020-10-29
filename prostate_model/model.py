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
        torch.nn.MaxPool2d(2, stride=2), # (32, 64, 1000)
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, stride=2), # (16, 32, 1000)
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, stride=2), # (8, 16, 1000)
        torch.nn.ReLU(),
        torch.nn.MaxPool2d((1, 2), stride=(1, 2)), # (8, 8, 1000)
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, stride=2), # (4, 4, 1000)
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, stride=2), # (2, 2, 1000)
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, stride=2), # (1, 1, 1000)
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(1000, 1),
    )
    return model

def build_optimizer(type, model, lr):
    if type=='SGD':
        return optim.SGD(model.parameters(), lr=lr)
    elif type=='Adam':
        return optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer type {type}")
