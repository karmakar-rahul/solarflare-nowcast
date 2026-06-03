"""
src/focal_loss.py
-----------------
Focal Loss for heavily imbalanced binary classification.

Solar flares (M/X class) occur in ~1-2% of all 1-minute GOES intervals,
so standard BCE will collapse to always predicting "no flare."

Focal loss down-weights easy negatives (the vast quiet-sun majority) and
forces the model to focus on hard positives (pre-flare signatures).

Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection"
           https://arxiv.org/abs/1708.02002

Usage:
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    loss = criterion(logits, labels)    # logits: (B,1), labels: (B,1) float
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary focal loss.

    Args:
        alpha (float): Weighting factor for the rare (positive) class.
                       0.25 is the standard default from the paper.
                       Increase toward 0.5–0.75 if recall is still too low.
        gamma (float): Focusing exponent. Higher = more focus on hard examples.
                       2.0 is the standard default.
        reduction (str): 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, 1) or (B,)  — raw model output (before sigmoid)
        targets : (B, 1) or (B,)  — binary float labels {0.0, 1.0}
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss)                         # probability of correct class

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_loss = alpha_t * (1.0 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Simple alternative: weighted BCE where positive weight is computed
    from class frequency.  Use this if focal loss feels too aggressive.

    Example:
        # If dataset has 98% negatives and 2% positives:
        pos_weight = torch.tensor([98.0 / 2.0])   # = 49.0
        criterion = WeightedBCELoss(pos_weight=pos_weight)
    """

    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)