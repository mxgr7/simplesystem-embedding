import torch
import torch.nn.functional as F


def cosine_bce_loss(scores, labels, scale, positive_weight=1.0, negative_weight=1.0):
    logits = scores * scale
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weights = torch.where(
        labels > 0.5,
        torch.full_like(labels, positive_weight),
        torch.full_like(labels, negative_weight),
    )
    return (losses * weights).mean()
