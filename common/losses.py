import torch


def compute_brier_score(softmax_probs, true_labels):

    t = torch.sum(torch.sum((true_labels - softmax_probs)**2, dim=2), dim=1)
    loss = torch.mean(t, dim=0)
    return loss

