
import torch
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss


class TVLoss(_Loss):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AsymmetricLoss(_Loss):
    def __init__(self):
        super(AsymmetricLoss, self).__init__()

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TotalLoss(_Loss):
    def __init__(self, lambda_tv, lambda_asymm):
        super(TotalLoss, self).__init__()
        self.lambda_tv = lambda_tv
        self.lambda_asymm = lambda_asymm

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class DiceLoss(_Loss):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = (inp * target).sum(dim=(2, 3))
        union = inp.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.eps) / (union + self.eps)
        return 1. - dice.mean()