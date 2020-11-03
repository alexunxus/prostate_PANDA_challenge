from torchvision.models import resnet50
import torch.nn as nn
import torch.optim

class BaselineResNet50(nn.Module):
    def __init__(self, num_grade):
        super(BaselineResNet50, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.linear = nn.Linear(1000, num_grade)
        self.activation = nn.Sigmoid()

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