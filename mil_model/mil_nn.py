import torch
import torchvision.models as models
from torch import nn

class MilResNet(torch.nn.Module):
    def __init__(self):
        super(MilResNet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
    
    def forward(self, x):
        ''' x is a dict e.g. {idx: 3*256*256 image tensor} '''
        return {idx:self.backbone(obj) for idx, obj in x.items()}


class NoisyAnd(torch.nn.Module):
    def __init__(self, a=10, dims=[1,2]):
        super(NoisyAnd, self).__init__()
        # self.output_dim = output_dim
        self.a = a
        self.b = torch.nn.Parameter(torch.tensor(0.01))
        self.dims =dims
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # h_relu = self.linear1(x).clamp(min=0)
        mean = torch.mean(x, self.dims, True)
        res = (self.sigmoid(self.a * (mean - self.b)) - self.sigmoid(-self.a * self.b)) / (
              self.sigmoid(self.a * (1 - self.b)) - self.sigmoid(-self.a * self.b))
        return res
    


class NN(torch.nn.Module):

    def __init__(self, n=512, n_mid = 1024,
                 n_out=1, dropout=0.2,
                 scoring = None,
                ):
        super(NN, self).__init__()
        self.linear1 = torch.nn.Linear(n, n_mid)
        self.non_linearity = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(n_mid, n_out)
        self.dropout = torch.nn.Dropout(dropout)
        if scoring:
            self.scoring = scoring
        else:
            self.scoring = torch.nn.Softmax() if n_out>1 else torch.nn.Sigmoid()
        
    def forward(self, x):
        z = self.linear1(x)
        z = self.non_linearity(z)
        z = self.dropout(z)
        z = self.linear2(z)
        y_pred = self.scoring(z)
        return y_pred
    

class LogisticRegression(torch.nn.Module):
    def __init__(self, n=512, n_out=1):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n, n_out)
        self.scoring = torch.nn.Softmax() if n_out>1 else torch.nn.Sigmoid()

    def forward(self, x):
        z = self.linear(x)
        y_pred = self.scoring(z)
        return y_pred

    
def regularization_loss(params,
                        reg_factor = 0.005,
                        reg_alpha = 0.5):
    params = [pp for pp in params if len(pp.shape)>1]
    l1_reg = nn.L1Loss()
    l2_reg = nn.MSELoss()
    loss_reg =0
    for pp in params:
        loss_reg+=reg_factor*((1-reg_alpha)*l1_reg(pp, target=torch.zeros_like(pp)) +\
                           reg_alpha*l2_reg(pp, target=torch.zeros_like(pp)))
    return loss_reg

class SoftMaxMeanSimple(torch.nn.Module):
    def __init__(self, n, n_inst, dim=0):
        """
        if dim==1:
            given a tensor `x` with dimensions [N * M],
            where M -- dimensionality of the featur vector
                       (number of features per instance)
                  N -- number of instances
            initialize with `AggModule(M)`
            returns:
            - weighted result: [M]
            - gate: [N]
        if dim==0:
            ...
        """
        super(SoftMaxMeanSimple, self).__init__()
        self.dim = dim
        self.gate = torch.nn.Softmax(dim=self.dim)      
        self.mdl_instance_transform = nn.Sequential(
                            nn.Linear(n, n_inst),
                            nn.LeakyReLU(),
                            nn.Linear(n_inst, n),
                            nn.LeakyReLU(),
                            )
    def forward(self, x):
        z = self.mdl_instance_transform(x)
        if self.dim==0:
            z = z.view((z.shape[0],1)).sum(1)
        elif self.dim==1:
            z = z.view((1, z.shape[1])).sum(0)
        gate_ = self.gate(z)
        res = torch.sum(x* gate_, self.dim)
        return res, gate_

    
class AttentionSoftMax(torch.nn.Module):
    def __init__(self, in_features = 3, out_features = None):
        """
        given a tensor `x` with dimensions [N * M],
        where M -- dimensionality of the featur vector
                   (number of features per instance)
              N -- number of instances
        initialize with `AggModule(M)`
        returns:
        - weighted result: [M]
        - gate: [N]
        """
        super(AttentionSoftMax, self).__init__()
        self.otherdim = ''
        if out_features is None:
            out_features = in_features
        self.layer_linear_tr = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU()
        self.layer_linear_query = nn.Linear(out_features, 1)
        
    def forward(self, x):
        keys = self.layer_linear_tr(x)
        keys = self.activation(keys)
        attention_map_raw = self.layer_linear_query(keys)[...,0]
        attention_map = nn.Softmax(dim=-1)(attention_map_raw)
        result = torch.einsum(f'{self.otherdim}i,{self.otherdim}ij->{self.otherdim}j', attention_map, x)
        return result, attention_map

class MIL_NN(torch.nn.Module):
    def __init__(self, n=512,  
                 n_mid=1024, 
                 n_classes=1, 
                 dropout=0.1,
                 agg = None,
                 scoring=None,
                ):
        super(MIL_NN, self).__init__()
        self.agg = agg if agg is not None else AttentionSoftMax(n)
        
        if n_mid == 0:
            self.bag_model = LogisticRegression(n, n_classes)
        else:
            self.bag_model = NN(n, n_mid, n_classes, dropout=dropout, scoring=scoring)
        
    def forward(self, bag_features, bag_lbls=None):
        """
        bag_feature is an aggregated vector of 512 features {1: [512], 2:[512]}
        bag_att is a gate vector of n_inst instances
        bag_lbl is a vector a labels
        figure out batches
        """
        bag_feature, bag_att, bag_keys = list(zip(*[list(self.agg(ff.float())) + [idx]
                                                    for idx, ff in (bag_features.items())]))
        bag_att = dict(zip(bag_keys, [a.detach().cpu() for a in bag_att]))
        bag_feature_stacked = torch.stack(bag_feature)
        y_pred = self.bag_model(bag_feature_stacked)
        
        '''customized: get top k objects and average'''
        y_pred = torch.topk(y_pred.veiw(-1), 10)
        y_pred = torch.mean(y_pred)
        return y_pred, bag_att, bag_keys


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # preparing model
    print("==========Preparing for MIL model============")
    test_model = torch.nn.Sequential(
        MilResNet(), 
        MIL_NN(n=1000,
               n_mid=1024, 
               n_classes=1, 
               dropout=0.1,
               scoring=AttentionSoftMax)
    )
    test_model = test_model.cuda()


    print("==========Preparing dummy tensors============")
    dummy = {i:torch.randn(1, 3, 256, 256).cuda() for i in range(100)}
    dummy_label = torch.tensor(1.0).cuda()
    out = test_model(dummy)
    print(out)
    
    print("==========Computing loss=====================")
    loss = torch.square(torch.sub(dummy_label, out[0]))
    loss.backward()