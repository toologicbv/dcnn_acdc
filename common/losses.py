import torch


def compute_brier_score(pred_labels, true_labels):

    t = torch.sum(torch.sum((true_labels - pred_labels)**2, dim=2), dim=1)
    loss = torch.mean(t, dim=0)
    return loss

