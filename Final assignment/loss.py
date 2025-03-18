import torch

def dice_loss_multiclass(pred, target, num_classes, ignore_index=255, epsilon=1e-6):
    pred = torch.softmax(pred, dim=1)  # (batch, num_classes, H, W)
    target = target.long()
    
    # create mask to ignore ignore_index pixels
    mask = (target != ignore_index).unsqueeze(1).float()  # (batch, 1, H, W)
    
    # convert target to one-hot representation
    target_clone = target.clone()
    target_clone[target_clone == ignore_index] = 0
    target_one_hot = torch.nn.functional.one_hot(target_clone, num_classes).permute(0, 3, 1, 2).float()  # (batch, num_classes, H, W)
    
    # apply mask to ignore certain pixels
    target_one_hot = target_one_hot * mask

    # compute dice components
    intersection = (pred * target_one_hot).sum(dim=(2, 3))  # (batch, num_classes)
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # (batch, num_classes)
    
    dice_score = (2. * intersection + epsilon) / (union + epsilon)  # (batch, num_classes)

    # take mean over classes first, then over batch
    return 1 - dice_score.mean(dim=1).mean()


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        
    def dice(self, pred, target):
        pass

    def mean_dice_loss(self, pred, target):
        pass
    
    def focal(self, pred, target):
        pass

    def focal_dice(self, pred, target):
        pass