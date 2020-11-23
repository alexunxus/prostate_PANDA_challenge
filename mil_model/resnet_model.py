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
            self.swap_norm_layers()

        if resume_from:
            self.load_state_dict(torch.load(resume_from), strict=False)
            print(f"Resume from checkpoint {resume_from}")
        else: 
            self.init_weights()
            print(f"Saving model to {weight_path}")
            torch.save(self.state_dict(), weight_path)
        print(f"{backbone} is prepared.")
    
    def swap_norm_layers(self):
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Get current bn layer
                bn = getattr(self.backbone, name)
                # Create new gn layer
                gn = nn.GroupNorm(4, bn.num_features)
                # Assign gn
                print('Swapping {} with {}'.format(bn, gn))
                setattr(self.backbone, name, gn)

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
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Unknown optimizer type {type}")
