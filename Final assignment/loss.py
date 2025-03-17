import torch

def dice_loss_multiclass(pred, target, num_classes, ignore_index=255, epsilon=1e-6):
    pred = torch.softmax(pred, dim=1)  # (batch, num_classes, H, W)
    target = target.long()
    mask = (target != ignore_index).unsqueeze(1).float()  # (batch, 1, H, W)
    target_clone = target.clone()
    target_clone[target_clone == ignore_index] = 0
    target_one_hot = torch.nn.functional.one_hot(target_clone, num_classes).permute(0, 3, 1, 2).float() # (batch, num_classes, H, W)
    target_one_hot = target_one_hot * mask
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice_score = (2. * intersection + epsilon) / (union + epsilon)  # (batch, num_classes)
    return 1 - dice_score.mean()