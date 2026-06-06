from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any
from .datasetLoader import DataConfig
from pydantic import Field
from .models import ModelConfig
from .OptunaConfig import OptunaStudyConfig
# Importing eval metrics using "Allowed"
from .evaluator.Evaluator_neural import AllowedMetrics


class CheckpointConfig(BaseModel):
    load_from_dir: str
    mode: Literal["fine_tune", "evaluate_only"] = Field(default='evaluate_only',
                                                        description="Mode of checkpoint loading.\n- evaluate_only can "
                                                                    "be used for cold start testing")


class OptimizerConfig(BaseModel):
    name: Literal["Adam", "AdamW", "SGD"]
    lr: float
    weight_decay: float = 0.0


class SchedulerConfig(BaseModel):
    name: Literal["ReduceLROnPlateau", "StepLR"]
    params: Dict[str, Any] = {}


class CriterionConfig(BaseModel):
    name: Literal["BPR", "HINGE", "MSE"]
    params: Dict[str, Any] = {}


class SetEval(BaseModel):
    metrics: List[AllowedMetrics] = Field(
        default_factory=lambda: ['ndcg', 'recall'],  # Default metrics
        description="List of metrics to use for evaluation."
    )
    cutoff: List[int] = Field(
        default_factory=lambda: [5, 10],  # Default cutoffs
        description="List of cutoff values for metrics."
    )
    eval_every: int = Field(
        default=1,
        description="Evaluate every N epochs and the first one."
    )
    save_recs: bool = Field(
        default=True,
        description="If True (default), saves the top-k recommendations (rank list) to a .tsv file."
    )


class EvalConfig(BaseModel):
    validation: SetEval = Field(
        default_factory=SetEval,  # Default validation config
        description="Configuration for validation set evaluation."
    )
    test: SetEval = Field(
        default_factory=SetEval,  # Default test config
        description="Configuration for test set evaluation."
    )


class EarlyStoppingConfig(BaseModel):
    patience: int
    monitor: str  # metric name to monitor for improvement


class TrainConfig(BaseModel):
    epochs: int
    batch_size: int
    eval_batch: Optional[int] = None
    evaluate_cutoff: int
    metrics: List[str]


class AppConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    criterion: CriterionConfig
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None
    train: TrainConfig
    eval: EvalConfig
    early_stopping: EarlyStoppingConfig
    optuna: Optional[OptunaStudyConfig] = None
    checkpoint: Optional[CheckpointConfig] = None
