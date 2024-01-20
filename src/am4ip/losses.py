import torch
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class DiceLoss(_Loss):
    def __init__(self, class_weights=None, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.class_weights = None

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inp = torch.softmax(inp, dim=1)  # Convert logits to probabilities
        target = torch.nn.functional.one_hot(target, num_classes=inp.shape[1]).permute(0, 3, 1, 2).float()  # Convert target to one-hot encoding
        intersection = (inp * target).sum(dim=(-2, -1))
        union = inp.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice = (2. * intersection + self.eps) / (union + self.eps)
        if self.class_weights is not None:
            # Ensure the weight tensor is on the same device as dice
            class_weights = self.class_weights.to(dice.device)
            dice = dice * class_weights
        return 1. - dice.mean()
    

class CombinedLoss(_Loss):
    def __init__(self, class_counts, eps=1e-8):
        super(CombinedLoss, self).__init__()
        self.eps = eps
        total_count = sum(class_counts)
        weights = [total_count / count for count in class_counts]
        weight_sum = sum(weights)
        self.weight = torch.tensor([weight / weight_sum for weight in weights])
        self.dice_loss = DiceLoss(self.weight, eps)
       # print("Weight", self.weight)

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Reshape weight to match inp and target_one_hot
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(inp)
        # Ensure the weight tensor is on the same device as inp and target_one_hot
        weight = weight.to(inp.device)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=inp.shape[1]).permute(0, 3, 1, 2).float()  # Convert target to one-hot encoding
        bce_loss = F.binary_cross_entropy_with_logits(inp, target_one_hot, weight=weight)
        dice_loss = self.dice_loss(inp, target)

        # Combine the two loss functions
        return 0.5* dice_loss  + 0.5 * bce_loss