from torchvision.models import resnet50
import torch.nn as nn
import torch.optim
import os

class BaselineResNet50(nn.Module):
    def __init__(self, num_grade):
        super(BaselineResNet50, self).__init__()
        weight_path = './checkpoint/.imagenet_weight'
        if not os.path.isfile(weight_path):
            self.backbone = resnet50(pretrained=True)
            torch.save(self.backbone.state_dict(), weight_path)
        else:
            print(f"Loading model weight from {weight_path}...")
            self.backbone = resnet50(pretrained=False)
            self.backbone.load_state_dict(torch.load(weight_path))
            print(f"Resnet model is prepared.")
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