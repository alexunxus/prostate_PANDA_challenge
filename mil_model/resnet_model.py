from torchvision.models import resnet50, resnext50_32x4d
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import os
from .util import replace_bn2gn

class DomainNorm(nn.Module):
    def __init__(self, channel, l2=True):
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=False)
        self.l2 = l2
        self.weight = nn.Parameter(torch.ones(1,channel,1,1))
        self.bias = nn.Parameter(torch.zeros(1,channel,1,1))
        self.weight.requires_grad = True
        self.bias.requires_grad = True
    def forward(self, x):
        x = self.normalize(x)
        if self.l2:
            return F.normalize(x, p=2, dim=1)
        return x * self.weight + self.bias

def get_backbone(string, pretrained=True):
    if string == "R-50-xt":
        return resnext50_32x4d(pretrained=pretrained)
    elif string == "R-50-st":
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        return torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=pretrained)
    elif 'enet-b' in string:
        return EfficientNet.from_pretrained(f'efficientnet-b{string[-1]}')
    elif string == 'baseline':
        return resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone type {string}!")

class CustomModel(nn.Module):
    def __init__(self, backbone, num_grade, resume_from=None, norm='bn'):
        super(CustomModel, self).__init__()
        if (resume_from and not os.path.isfile(resume_from)):
            raise ValueError(f"Path {resume_from} does not exist, cannot load weight.")
        weight_path = f'./checkpoint/.{backbone}_imagenet_weight'
        
        if resume_from is None and os.path.isfile(weight_path):
            resume_from = weight_path

        self.backbone = get_backbone(string=backbone, pretrained=False if resume_from else True)
        self.linear_side_chain = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000)
        )
        self.linear = nn.Linear(1000, num_grade)
        self.activation = nn.Sigmoid()

        if norm not in ['bn', 'gn']:
            raise ValueError(f"Unkonwn norm type {norm}")
        if norm == 'gn':
            self = replace_bn2gn(self)

        if resume_from:
            self.load_state_dict(torch.load(resume_from), strict=False)
            print(f"Resume from checkpoint {resume_from}")
        else: 
            self.init_weights()
            print(f"Saving model to {weight_path}")
            torch.save(self.state_dict(), weight_path)
        print(f"{backbone} is prepared.")

    def init_weights(self):
        tails = [m for m in self.linear_side_chain]
        tails.append(self.linear)
        for m in tails:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.BatchNorm1d:
                m.weight.data.fill_(0)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.add(x, self.linear_side_chain(x))
        x = self.linear(x)
        return self.activation(x)

def build_optimizer(type, model, lr):
    if type == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    if type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Unknown optimizer type {type}")
