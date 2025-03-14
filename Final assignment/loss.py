import torch

def dice_loss_multiclass(pred, target, num_classes, ignore_index=255, epsilon=1e-6):
    """
    Computes the Dice loss for multi-class segmentation while ignoring specified label indices.

    Args:
        pred (Tensor): Logits from the model with shape (batch, num_classes, H, W).
        target (Tensor): Ground truth labels with shape (batch, H, W).
        num_classes (int): Number of segmentation classes.
        ignore_index (int): Label value to ignore (default: 255 for Cityscapes).
        epsilon (float): Small constant for numerical stability.

    Returns:
        Tensor: The computed Dice loss (scalar).
    """
    # Convert logits to softmax probabilities
    pred = torch.softmax(pred, dim=1)  # shape: (batch, num_classes, H, W)

    # Ensure target is long before one-hot encoding
    target = target.long()

    # Create a mask for valid pixels (ignore_index should not be considered)
    mask = (target != ignore_index).unsqueeze(1).float()  # shape: (batch, 1, H, W)

    # Clone target and replace ignore_index values with a valid index (e.g., 0)
    target_clone = target.clone()
    target_clone[target_clone == ignore_index] = 0

    # Convert target to one-hot encoding with shape (batch, num_classes, H, W)
    target_one_hot = torch.nn.functional.one_hot(target_clone, num_classes).permute(0, 3, 1, 2).float()

    # Apply mask after one-hot encoding
    target_one_hot = target_one_hot * mask  # Zero out ignored pixels in one-hot encoding

    # Compute intersection and union while considering the mask
    intersection = (pred * target_one_hot).sum(dim=(2, 3))  # Sum over H, W
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # Sum predictions & target

    # Compute Dice score for each class (add epsilon to avoid division by zero)
    dice_score = (2. * intersection + epsilon) / (union + epsilon)  # shape: (batch, num_classes)

    # Mean over classes and batch
    return 1 - dice_score.mean()