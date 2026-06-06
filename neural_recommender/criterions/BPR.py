import torch
from .BaseCriterion import Criterion


class BPRLoss(Criterion):
    def __init__(self):
        super().__init__("BPR")

    def __call__(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        # -(log sigma(pos - neg)).mean()    # 1e-8 -> eps
        return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


