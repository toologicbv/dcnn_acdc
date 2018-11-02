import torch


def compute_brier_score(softmax_probs, true_labels, log_stddev=None):

    if not isinstance(true_labels, torch.FloatTensor) and not isinstance(true_labels, torch.DoubleTensor):
        true_labels = true_labels.float()
    if log_stddev is not None:
        t = torch.sum(torch.sum(torch.exp(-log_stddev) * (true_labels - softmax_probs) ** 2, dim=2), dim=1)
    else:
        t = torch.sum(torch.sum((true_labels - softmax_probs)**2, dim=2), dim=1)
    loss = torch.mean(t, dim=0)
    return loss

