
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
