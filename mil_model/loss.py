import torch
from torch.nn import BCELoss
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def get_bceloss():
    # binary cross entropy loss
    return BCELoss()

def kappa_score(gt, pred):
    # quadratic weighted kappa
    return cohen_kappa_score(gt, pred, weights='quadratic')

def cat2num(np_array):
    '''Arg: np_array with shape (N, 5), Return: (N), the grading'''
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
    print(gts)
    print(preds)
    gts   = np.array([tensor.numpy()>0.5 for tensor in gts  ]).reshape((-1, 5))
    preds = np.array([tensor.numpy()>0.5 for tensor in preds]).reshape((-1, 5))
    gts   = cat2num(gts)
    preds = cat2num(preds)
    print(gts)
    print(preds)
    print(f"Ground truth shape = {gts.shape}")
    print(f"Prediction shape   = {preds.shape}")
    k = kappa_score(gts, preds)
    conf = confusion_matrix(gts, preds)
    print(f"Kappa score = {k}")
    print("Confusion matrix:\n", conf)
    return k

if __name__ == '__main__':
    import numpy as np
    import torch
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


