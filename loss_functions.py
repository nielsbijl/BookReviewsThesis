import torch


def compute_class_weights(labels):
    class_counts = torch.bincount(labels)
    class_weights = 1. / class_counts.float()
    class_weights /= class_weights.min()

    # TODO: try more weight or remove
    # Increase weight for class 1 by 10%
    # class_weights[1] *= 1.10

    return class_weights


class MoMLoss(torch.nn.Module):
    def __init__(self):
        super(MoMLoss, self).__init__()

    def forward(self, logits, targets):
        # Flatten predictions and labels
        logits = logits.view(-1, 2)  # Assuming 2 classes
        targets = targets.view(-1)

        # Masking -100 labels
        valid_indices = (targets != -100)
        logits = logits[valid_indices]
        targets = targets[valid_indices]

        weights = compute_class_weights(targets)

        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        loss = ce_loss(logits, targets)
        weighted_loss = loss * weights[targets]
        return weighted_loss.mean()


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.): 
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Flatten predictions and labels
        logits = logits.view(-1, 2)  # Assuming 2 classes
        targets = targets.view(-1)

        intersection = torch.sum(logits[:, 1] * targets)
        union = torch.sum(logits[:, 1]) + torch.sum(targets)

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score

        return dice_loss