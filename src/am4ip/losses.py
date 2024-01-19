import torch
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class DiceLoss(_Loss):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inp = torch.softmax(inp, dim=1)  # Convert logits to probabilities
        target = torch.nn.functional.one_hot(target, num_classes=inp.shape[1]).permute(0, 3, 1, 2).float()  # Convert target to one-hot encoding
        intersection = (inp * target).sum(dim=(-2, -1))
        union = inp.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice = (2. * intersection + self.eps) / (union + self.eps)
        return 1. - dice.mean()
    

class CombinedLoss(_Loss):
    def __init__(self, eps=1e-8, weight=None):
        super(CombinedLoss, self).__init__()
        self.eps = eps
        # self.weight = torch.tensor([1.0, 1.0, 2.0, 1.0, 2.0])
        if weight == 3:
            self.weight = torch.tensor([1.0, 1.0, 5.0])
        elif weight == 5:
            self.weight = torch.tensor([1.0, 1.0, 5.0, 1.0, 5.0])
        self.dice_loss = DiceLoss(eps)

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Reshape weight to match inp and target_one_hot
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(inp)
        # Ensure the weight tensor is on the same device as inp and target_one_hot
        weight = weight.to(inp.device)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=inp.shape[1]).permute(0, 3, 1, 2).float()  # Convert target to one-hot encoding
        bce_loss = F.binary_cross_entropy_with_logits(inp, target_one_hot, weight=weight)
        dice_loss = self.dice_loss(inp, target)
        # Combine the two loss functions
        return dice_loss + bce_loss