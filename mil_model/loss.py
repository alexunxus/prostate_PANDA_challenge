import torch
from torch.nn import BCELoss
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import numpy as np

def get_bceloss():
    # binary cross entropy loss
    return BCELoss()

def kappa_score(gt, pred):
    # quadratic weighted kappa
    return cohen_kappa_score(gt, pred, weights='quadratic')

def cat2num(np_array):
    '''Arg: concatenated np_array with shape (N, 5), Return: (N), the grading'''
    ret = np.zeros(np_array.shape[0], dtype=np.int32)
    for i in range(np_array.shape[0]):
        zero_array = np_array[i].nonzero()[0]
        if zero_array.shape[0] == 0:
            ret[i] = 0
        else:
            ret[i] = int(zero_array[-1]+1)
    return ret

@torch.no_grad()
def kappa_metric(gts, preds):
    '''
    gts:   [tensor[[batch, 5]...], tensor[[batch, 5]...]...] a list of tensor in batches
    preds: [tensor[[batch, 5]...], tensor[[batch, 5]...]...]
    '''
    gts   = np.concatenate([tensor.numpy()>0.5 for tensor in gts  ], axis=0).reshape((-1, 5))
    preds = np.concatenate([tensor.numpy()>0.5 for tensor in preds], axis=0).reshape((-1, 5))
    gts   = cat2num(gts)
    preds = cat2num(preds)
    k     = kappa_score(gts, preds)
    conf  = confusion_matrix(gts, preds)
    print(f"Kappa score = {k}")
    print("Confusion matrix:\n", conf)
    return k

def correct(gt, pred):
    '''
    gt    is a (B, 5) tensor or [tensor[[batch, 5]...], tensor[[batch, 5]...]...]
    label is a (B, 5) tensor or [tensor[[batch, 5]...], tensor[[batch, 5]...]...]
    '''
    if isinstance(gt, list):
        gt   = np.concatenate([tensor.numpy()>0.5 for tensor in gt  ], axis=0).reshape((-1, 5))
        pred = np.concatenate([tensor.numpy()>0.5 for tensor in pred], axis=0).reshape((-1, 5))
    if torch.is_tensor(gt):
        gt = gt.numpy()
        pred = pred.numpy()
    gt   = gt.round().astype(np.int32).sum(1)
    pred = pred.round().astype(np.int32).sum(1)
    return (gt == pred).sum()


if __name__ == '__main__':
    a = np.array([1, 3, 1, 2, 2, 1, 1, 3, 1])
    b = np.array([1, 2, 1, 1, 1, 3, 3, 1, 2])
    print(kappa_score(a, b))
    print(kappa_score(b, a))

    test_array = [torch.rand(32, 5) for i in range(10)]
    test_array = np.array([tensor.numpy()>0.5 for tensor in test_array]).reshape((-1, 5))
    print(test_array)
    test_array = cat2num(test_array)
    print(test_array)
    
    with torch.no_grad():
        preds = [torch.rand(32, 5) for i in range(10)]
        gts   = [torch.rand(32, 5) for i in range(10)]
        gts = preds
        kappa_metric(gts, preds)
        kappa_metric(gts, gts)
    
    # BECLoss test
    a = torch.rand(32, 5)
    b = torch.rand(32, 5)
    print(a, b)
    criterion = get_bceloss()

    loss = criterion(a, b)
    print(loss)

    # 
    a = torch.tensor([1, 0, 0, 0, 1, 5, 0, 0, 4, 5])
    b = torch.tensor([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    print(confusion_matrix(a, b))
    print(kappa_score(a, b))
    
    # check correctedness
    a = torch.rand(32, 5)
    b = torch.rand(32, 5)
    with torch.no_grad():
        print(correct(a, b))

