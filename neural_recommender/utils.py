from importlib import import_module
from neural_recommender import AppConfig
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any
import logging
import random
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> AppConfig:
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)


def get_dataset(name: str, config, interactions):
    """Dynamically load a dataset class by name from datasetLoader/"""
    module = import_module(f"neural_recommender.datasetLoader.{name}")
    cls = getattr(module, name + "Dataset")
    return cls(config, interactions)


# Interaction matrix builder
def df_to_binary_matrix(df, num_user, num_item, user2idx, item2idx):
    mat = np.zeros((num_user, num_item), dtype=int)
    for _, r in df.iterrows():
        u = user2idx[r.userId]
        i = item2idx.get(r.itemId) # itemId
        if i is not None:
            mat[u, i] = 1
    return mat


def _summarize_matrix(mat: np.ndarray, n_skipped: int) -> None:
    """
    Helper to print out the shape of the built matrix
    and how many interactions were dropped.
    """
    logger.info(f"** Interaction matrix shape: {mat.shape}")
    logger.info(f"** Skipped {n_skipped} rows due to missing user/item mapping\n")


def build_interactions(
        df: pd.DataFrame,
        user2idx: Dict[Any, int],
        item2idx: Dict[Any, int],
        pos_threshold: float,
        binary: bool = True,
        rating_col: str = "rating"
) -> np.ndarray:
    """
    Build a [n_users x n_items] interaction matrix from a ratings DataFrame,
    and then summarize its final shape and any dropped rows.

    Args:
        df:             DataFrame with columns userId, itemId, (optional) rating_col
        user2idx:       map raw userId -> row index
        item2idx:       map raw itemId -> col index
        pos_threshold:  minimum rating to consider “positive” when binarizing
        binary:         if True, binarize with that threshold; else keep raw ratings
        rating_col:     name of rating column

    Returns:
        mat: np.ndarray of shape (len(user2idx), len(item2idx))
    """
    n_users = len(user2idx)
    n_items = len(item2idx)
    mat = np.zeros((n_users, n_items), dtype=float)

    skipped = 0
    for _, row in df.iterrows():
        u = user2idx.get(row.userId)
        i = item2idx.get(row.itemId) # itemId
        if u is None or i is None:
            skipped += 1
            continue

        r = float(row.get(rating_col, 0.0))
        if binary:
            mat[u, i] = 1.0 if r >= pos_threshold else 0.0
        else:
            mat[u, i] = r

    _summarize_matrix(mat, skipped)
    return mat


def infer_emb_dim(
        user_emb_tensor: torch.Tensor,
        item_emb_tensor: torch.Tensor
) -> tuple:
    """
    Infer embedding dimension from pretrained user/item tensors,
    and raise a warning if they disagree.
    """
    u_dim = user_emb_tensor.size(1)
    i_dim = item_emb_tensor.size(1)
    if u_dim != i_dim:
        raise Warning(f"Embedding dim mismatch: users={u_dim}, items={i_dim}")
    return u_dim, i_dim


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True # MUST SET ENV VAR
    torch.backends.cudnn.benchmark = False
    # MUST SET ENV VAR
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except Exception:
    #     pass


# store_recommendation elliot reworked function (for neural)
def store_recommendation(recommendations, path: str, sep: str = '\t'):
    """
    Writes recommendations to a file.
    Format: user_id [sep] item_id [sep] score

    Args:
        recommendations: dictionary or list containing (u, i, s) triplets
                         or a list of [user_idx, item_idx, score]
        path: output file path
        sep: separator (default tab for .tsv)
    """
    logger.info(f"Writing recommendations to: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        # if recommendations is a flat list of tuples/lists
        for user_idx, item_idx, score in recommendations:
            f.write(f"{user_idx}{sep}{item_idx}{sep}{score}\n")


class EarlyStopping:
    """
    Tracks a monitored metric and stops training when no improvement for a given patience.
    """

    def __init__(self, patience: int, monitor: str):
        self.patience = patience
        self.monitor = monitor
        self.best: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, current: float) -> bool:
        if self.best is None or current > self.best:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            logger.info(msg=f'Current counter [{self.counter}] with max patience [{self.patience}]')
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


