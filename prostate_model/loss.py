import torch
from torch import nn


def mean_square_error():
    def MSE(pred, gt):
        return torch.mean(torch.square(torch.sub(gt, pred)))
    return MSE

def get_ranking_loss():
    def ranking_loss(pred, gt):
        '''
        Arg: two float tensor
            gt:  [Batch] float tensor
            pred:[Batch] float tensor
           e.g.
            gt:  [0.2,  0.4,  0,    0.8,  1.0 ]
            pred:[0.03, 0.01, 0.04, 0.03, 0.08]
        Return: sum of loss of each batch size
            minimum for each element happens at gt == pred, loss = ln(1/(1+exp(0))) = -ln2 = - 0.693
        '''
        gt_is_dead = torch.ones(gt.shape[0], dtype=torch.long, requires_grad=False).cuda()
        order_lt_matrix = ( gt.unsqueeze(-1) <  gt.unsqueeze(-2) )
        order_eq_matrix = ( gt.unsqueeze(-1) == gt.unsqueeze(-2))
        lhs_dead_matrix = gt_is_dead.unsqueeze(-1) == 1
        rhs_alive_matrix= gt_is_dead.unsqueeze(-2) == 0

        true_lt_matrix = torch.logical_and(
            lhs_dead_matrix,
            torch.logical_or(
                order_lt_matrix,
                torch.logical_and(
                    order_eq_matrix,
                    rhs_alive_matrix
                )
            )
        )

        true_lt_matrix = true_lt_matrix.type(torch.float)
        prob_lt_matrix = 1.0 - torch.sigmoid(
            pred.unsqueeze(-1) - pred.unsqueeze(-2)
        )

        loss_matrix = true_lt_matrix * (-torch.log(prob_lt_matrix+1e-8))
        return torch.sum(loss_matrix)
        '''
        diff = torch.sub(gt, pred)
        Sij  = torch.sigmoid(diff)
        tij  = torch.mul((torch.sign(diff)+1), 0.5)
        loss = tij * torch.log(Sij) + (1-tij) * torch.log(1-Sij)
        return torch.sum(loss)
        '''
    return ranking_loss


if __name__ == "__main__":
    a = torch.tensor([0.2, 0.4], dtype=torch.float32)
    b = torch.tensor([0.0, 0.0], dtype= torch.float32)

    criterion = get_ranking_loss()
    print(criterion(b, a))