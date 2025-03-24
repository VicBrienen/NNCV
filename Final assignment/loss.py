import torch
import torch.nn as nn


class GeneralizedDice(torch.nn.Module):
    def __init__(self, num_classes=19, ignore_index=255, epsilon=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, pred, target):
        pred = torch.nn.functional.softmax(pred, dim=1)  # (batch, num_classes, H, W)
        valid = (target != self.ignore_index)

        # mask out the ignore_index
        target = target.masked_fill(~valid, 0)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()  # (batch, num_classes, H, W)
        target_one_hot = target_one_hot * valid.unsqueeze(1).float()

        target_sum = target_one_hot.sum(dim=(0, 2, 3))  # (num_classes,)
        weights = 1.0 / (target_sum * target_sum + self.epsilon)

        intersection = (pred * target_one_hot).sum(dim=(0, 2, 3))
        pred_sum = pred.sum(dim=(0, 2, 3))
        union = pred_sum + target_sum

        # calculate weighted dice score
        numerator = 2 * (weights * intersection).sum()
        denominator = (weights * union).sum() + self.epsilon
        generalized_dice_score = numerator / denominator

        return 1 - generalized_dice_score