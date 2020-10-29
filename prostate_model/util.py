import numpy as np 
from sklearn.metrics import confusion_matrix

def quadratic_kappa(groundtruth, prediction, maxN):
    '''
    actuals = np.array([4, 4, 3, 4, 4, 4, 1, 1, 2, 1])
    preds   = np.array([0, 2, 1, 0, 0, 0, 1, 1, 2, 1])
    '''
    N = groundtruth.shape[0]
    O = confusion_matrix(groundtruth, prediction).astype(np.float64)
    W = np.array([[(i-j)**2 for j in range(maxN)] for i in range(maxN)])
    act_hist=np.zeros([maxN])
    for item in groundtruth: 
        act_hist[item]+=1
    pred_hist=np.zeros([maxN])
    for item in prediction: 
        pred_hist[item]+=1
    E = np.outer(act_hist, pred_hist).astype(np.float64)
    
    O /=np.sum(O)
    E /=np.sum(E)
    kappa = 1 - np.sum(O*W)/np.sum(E*W)
    return kappa

if __name__ == '__main__':
    actuals = np.array([4, 4, 3, 4, 4, 4, 1, 1, 2, 1])
    preds   = np.array([0, 2, 1, 0, 0, 0, 1, 1, 2, 1])
    kappa   = quadratic_kappa(actuals, preds, 5)
    print(kappa)