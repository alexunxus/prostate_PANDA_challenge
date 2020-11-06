from torchvision.models import resnet50
import torch.nn as nn
import torch.optim
import os

class BaselineResNet50(nn.Module):
    def __init__(self, num_grade, resume_from=None):
        super(BaselineResNet50, self).__init__()
        if resume_from is not None and os.path.isfile(resume_from):
            self.backbone = resnet50(pretrained=False)
            self.linear = nn.Linear(1000, num_grade)
            self.activation = nn.Sigmoid()
            self.load_state_dict(torch.load(resume_from))
            print(f"Resume from checkpoint {resume_from}")
        else: 
            weight_path = './checkpoint/.imagenet_weight'
            if not os.path.isfile(weight_path):
                self.backbone = resnet50(pretrained=True)
                torch.save(self.backbone.state_dict(), weight_path)
            else:
                print(f"Loading model weight from {weight_path}...")
                self.backbone = resnet50(pretrained=False)
                self.backbone.load_state_dict(torch.load(weight_path))
            self.linear = nn.Linear(1000, num_grade)
            self.activation = nn.Sigmoid()

        print(f"Resnet model is prepared.")

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

class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x
    
    
def load_models(model_files):
    models = []
    for model_f in model_files:
        model_f = os.path.join(model_dir, model_f)
        backbone = 'efficientnet-b0'
        model = enetv2(backbone, out_dim=5)
        model.load_state_dict(torch.load(model_f, map_location=lambda storage, loc: storage), strict=True)
        model.eval()
        model.to(device)
        models.append(model)
        print(f'{model_f} loaded!')
    return models