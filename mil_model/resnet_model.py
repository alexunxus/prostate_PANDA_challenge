from torchvision.models import resnet50, resnext50_32x4d
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim
import os

def get_backbone(string, pretrained=True):
    if string == "R-50-xt":
        return resnext50_32x4d(pretrained=pretrained)
    elif string == "R-50-st":
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        # load pretrained models, using ResNeSt-50 as an example
        return torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=pretrained)
    elif string == 'enet-b0':
        return EfficientNet.from_pretrained('efficientnet-b0')
    elif string == 'enet-b1':
        return EfficientNet.from_pretrained('efficientnet-b1')
    elif string == 'baseline':
        return resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone type {string}!")

class CustomModel(nn.Module):
    def __init__(self, backbone, num_grade, resume_from):
        super(CustomModel, self).__init__()
        if resume_from is not None and os.path.isfile(resume_from):
            self.backbone = get_backbone(string=backbone, pretrained=False)
            self.linear = nn.Linear(1000, num_grade)
            self.activation = nn.Sigmoid()
            self.load_state_dict(torch.load(resume_from))
            print(f"Resume from checkpoint {resume_from}")
        else: 
            weight_path = f'./checkpoint/.{backbone}_imagenet_weight'
            if not os.path.isfile(weight_path):
                self.backbone = get_backbone(string=backbone, pretrained=True)
                torch.save(self.backbone.state_dict(), weight_path)
            else:
                print(f"Loading model weight from {weight_path}...")
                self.backbone = get_backbone(string=backbone, pretrained=False)
                self.backbone.load_state_dict(torch.load(weight_path))
            self.linear = nn.Linear(1000, num_grade)
            self.activation = nn.Sigmoid()

        print(f"{backbone} is prepared.")
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.linear(x)
        return self.activation(x)

def build_optimizer(type, model, lr):
    if type == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Unknown optimizer type {type}")
