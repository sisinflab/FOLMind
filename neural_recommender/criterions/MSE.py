import torch
from .BaseCriterion import Criterion


class MSELoss(Criterion):
    def __init__(self):
        super().__init__("MSE")

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(preds, targets)
