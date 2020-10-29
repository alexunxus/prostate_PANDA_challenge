import torch
from torch import nn


def mean_square_error():
    def MSE(gt, pred):
        return torch.mean(torch.square(torch.sub(gt, pred)))
    return MSE

