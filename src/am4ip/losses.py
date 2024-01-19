import torch
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss

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
    
import torch.nn.functional as F

class CombinedLoss(_Loss):
    def __init__(self, eps=1e-5):
        super(CombinedLoss, self).__init__()
        self.we
        self.dice_loss = DiceLoss(eps)

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=inp.shape[1]).permute(0, 3, 1, 2).float()
        bce_loss = F.binary_cross_entropy_with_logits(inp, target_one_hot)
        dice_loss = self.dice_loss(inp, target)
        return bce_loss + dice_loss