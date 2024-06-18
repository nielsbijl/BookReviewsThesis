import torch


def compute_class_weights(labels):
    """
    Compute weights for each class to address class imbalance.

    Args:
        labels (torch.Tensor): A tensor of class labels.

    Returns:
        torch.Tensor: A tensor of computed class weights.
    """
    class_counts = torch.bincount(labels)
    class_weights = 1. / class_counts.float()
    class_weights /= class_weights.min()
    return class_weights


class MoMLoss(torch.nn.Module):
    """
    Majority or Minority Loss function for handling data imbalance in Named Entity Recognition (NER) tasks.

    Nemoto, S., Kitada, S., & Iyatomi, H. (2024). Majority or Minority: Data Imbalance Learning Method for Named Entity Recognition. arXiv preprint arXiv:2401.11431.
    """
    def __init__(self):
        super(MoMLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Compute the forward pass for the loss function.

        Args:
            logits (torch.Tensor): Predicted logits from the model, expected to be of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels, expected to be of shape (batch_size).

        Returns:
            torch.Tensor: The mean weighted loss.
        """
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
    """
    Dice Loss function for handling data imbalance in Named Entity Recognition.
    """
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