import numpy as np
from torch.utils.data import Dataset
from typing import Literal, Optional
from pydantic import Field, model_validator, BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    dataset_dir: str = Field(description="The dataset directory to load data from. eg: 'ml-latest-small' ")

    # user-specific configurations
    user_emb_source: Literal["metaEmb", "LLMEmb"] = Field(default='metaEmb',
                                                          description="User embeddings source directory.")
    user_emb_type: Literal["last", "mean", "step", "bert", "ST"] = Field(default='last',
                                                                   description="User embedding generation pooling type.")
    user_info_type: Literal["likes", "dislikes", "mixed", "profile", "reasoning"] = (
        Field(default='likes', description="Stored user information type.")
    )

    user_emb_model: Optional[str] = Field(default='all-mpnet-base-v2',
                                          description="User embeddings source model.")

    # item-specific configurations
    item_emb_source: Optional[Literal["metaEmb", "LLMEmb"]] = Field(default=None,
                                                                    description="Item embeddings source directory. "
                                                                                "If None, uses user_emb_source.")
    item_emb_type: Optional[Literal["last", "mean", "step", "bert", "ST"]] = Field(default=None,
                                                                             description="Item embedding generation "
                                                                                         "pooling type. If None, uses "
                                                                                         "user_emb_type.")

    item_emb_model: Optional[str] = Field(default='all-mpnet-base-v2',
                                          description="Item embeddings source model.")

    dataset_type: Literal["BPR", "Pointwise"]
    neg_samples: Optional[int] = Field(
        default=1,
        description="Number of negative samples (for BPR-like datasets)."
    )
    seed: Optional[int] = Field(default=42, description='Random seed set for reproducibility.')
    binarize: Optional[bool] = Field(default=False, description='Binarize scores in ratings.csv.')
    pos_threshold: Optional[float | None] = Field(default=None,
                                                  description='If binarize=True, you can set the threshold '
                                                              '(>=) for considering a sample as positive '
                                                              '(=1) in the interaction matrix. [DEFAULT = 3]')
    rating_col: Optional[str] = Field(default='rating', description='Ratings column name in the .csv file.')

    @model_validator(mode="after")
    def check_and_default_configs(self) -> "DataConfig":
        # default item configurations to user configurations if not provided
        if self.item_emb_source is None:
            self.item_emb_source = self.user_emb_source
        if self.item_emb_type is None:
            self.item_emb_type = self.user_emb_type

        # validation for user configurations
        if self.user_emb_source == "metaEmb":
            user_allowed = {"likes", "dislikes", "mixed"}
        else:  # self.user_emb_source == "LLMEmb"
            user_allowed = {"profile", "reasoning"}

        if self.user_info_type not in user_allowed:
            raise ValueError(
                f"user_info_type='{self.user_info_type}' is invalid for user_emb_source='{self.user_emb_source}'. "
                f"Must be one of {sorted(user_allowed)}."
            )

        if self.binarize and self.pos_threshold is None:
            self.pos_threshold = 3.0
            logger.info(msg=f'Binarize is set to True has been set to pos_threshold={self.pos_threshold} '
                            f'as default (>=)')

        return self


class BaseDataset(Dataset):
    """
    Abstract dataset. Subclasses must implement __len__ and __getitem__.
    """

    def __init__(self, config: DataConfig, interactions: np.ndarray):
        self.config = config
        self.interactions = interactions

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
