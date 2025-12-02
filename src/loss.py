import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets, weight_map=None):
        """
        logits: (B, C, H, W) - Raw output from model (before softmax)
        targets: (B, H, W) - Ground truth masks (0 or 1)
        weight_map: (B, H, W) optional weights to emphasize pixels (already aligned)
        """
        probs = F.softmax(logits, dim=1)
        pred_flat = probs[:, 1].contiguous().view(logits.size(0), -1)
        target_flat = targets.contiguous().view(logits.size(0), -1).float()

        if weight_map is not None:
            weight_flat = weight_map.contiguous().view(logits.size(0), -1)
        else:
            weight_flat = torch.ones_like(target_flat)

        intersection = (weight_flat * pred_flat * target_flat).sum(dim=1)
        pred_sum = (weight_flat * pred_flat).sum(dim=1)
        target_sum = (weight_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred_sum + target_sum + self.smooth
        )

        return 1.0 - dice.mean()
