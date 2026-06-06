import torch
from .BaseCriterion import Criterion
import torch.nn.functional as F


class HINGELoss(Criterion):
    def __init__(self, margin: float = 0.5):
        super().__init__("HINGE")
        self.MARGIN = margin

    def __call__(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> torch.Tensor:
        return F.relu(self.MARGIN - x_pos + x_neg).mean()
