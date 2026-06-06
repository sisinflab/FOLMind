from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any


class HyperParameterConfig(BaseModel):
    """Defines a single hyperparameter to be tuned by Optuna."""
    param_name: str = Field(..., description="Dot-notation path to the parameter, e.g., 'optimizer.lr'")
    type: Literal["categorical", "float", "int"]
    params: Dict[str, Any] = Field(...,
                                description="Parameters for the Optuna suggest method, e.g., {'low': 0.1, 'high': 1.0}")


class OptunaStudyConfig(BaseModel):
    """Configuration for the entire Optuna study."""
    study_name: str
    n_trials: int = Field(gt=0)
    direction: Literal["maximize", "minimize"]
    search_space: List[HyperParameterConfig]
