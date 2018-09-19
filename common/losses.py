import torch


def compute_brier_score(softmax_probs, true_labels):

    if not isinstance(true_labels, torch.FloatTensor) and not isinstance(true_labels, torch.DoubleTensor):
        true_labels = true_labels.float()

    t = torch.sum(torch.sum((true_labels - softmax_probs)**2, dim=2), dim=1)
    loss = torch.mean(t, dim=0)
    return loss

