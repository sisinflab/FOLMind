import torch
from pydantic import BaseModel
from typing import Optional, Dict, Any


class ModelConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}
    device: str = "cpu"


class BaseNeuralModel(torch.nn.Module):
    """
    Abstract neural_recommender model. Subclasses must implement forward and score.
    """
    # FAMILY
    model_family = "standard"

    def __init__(self, verbose: bool = True, *args, **kwargs):
        super().__init__()
        self.verbose = verbose

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def score(self, *args, **kwargs):
        raise NotImplementedError

    def show_model_architecture(self):
        """
        Prints the module tree (same as `print(self)`)
        Lists every parameter name, its shape, and whether it's trainable
        Summarizes total, trainable, and frozen parameter counts
        """
        print(f"\n=== {self.__class__.__name__} ARCHITECTURE ===\n")
        # full module tree
        print(self)

        # parameter listing
        total_params = 0
        trainable_params = 0
        print("\n--- Parameters (name: shape, trainable) ---")
        for name, param in self.named_parameters():
            n = param.numel()
            total_params += n
            if param.requires_grad:
                trainable_params += n

            shape_str = str(tuple(param.shape))
            print(f"  {name:30} | {shape_str:>15} | {'yes' if param.requires_grad else 'no'}")

        # summary
        frozen_params = total_params - trainable_params
        print("\n--- Summary ---")
        print(f"  Total params     : {total_params:,}")
        print(f"  Trainable params : {trainable_params:,}")
        print(f"  Frozen params    : {frozen_params:,}")
        print("=" * 70 + "\n")
