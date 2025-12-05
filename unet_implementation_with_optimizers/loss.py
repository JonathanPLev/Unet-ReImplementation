import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W) - Raw output from model (before softmax)
        targets: (B, H, W) - Ground truth masks (0 or 1)
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # We are interested in the foreground class (index 1)
        # targets are 0 (background) or 1 (foreground)
        
        # Flatten the tensors
        # probs[:, 1] selects the probability map for the foreground class
        pred_flat = probs[:, 1].contiguous().view(-1)
        target_flat = targets.contiguous().view(-1).float()
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice
