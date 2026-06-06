import numpy as np
import random
from tqdm import tqdm
from pydantic import Field
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity

from .BaseModelConfig import BaseModelConfig


# Configuration
class MFElliotConfig(BaseModelConfig):
    factors: int = Field(default=10, description="Number of latent factors")
    lr: float = Field(default=0.05, description="Learning rate")
    reg: float = Field(default=0.0, description="Regularization coefficient")
    epochs: int = Field(default=10, description="Number of training epochs")
    m: int = Field(default=3, description="Number of negative samples per positive")
    seed: int = Field(default=42, description="Random seed for reproducibility")


# Data Sampler
class ElliotSampler:
    def __init__(self, train_matrix: np.ndarray, m: int, seed: int):
        np.random.seed(seed)
        random.seed(seed)
        self.num_items = train_matrix.shape[1]
        self._m = m
        self.users, self.pos_items = train_matrix.nonzero()

    def get_all_samples(self):
        """Generates the full training matrix of positive and negative samples."""
        num_pos_examples = len(self.users)

        # Create positive samples [user, item, 1]
        positive_samples = np.stack([self.users, self.pos_items, np.ones(num_pos_examples)], axis=1)

        # Create negative samples [user, item, 0]
        neg_users = np.repeat(self.users, self._m)
        neg_items = np.random.randint(0, self.num_items, size=num_pos_examples * self._m)
        negative_samples = np.stack([neg_users, neg_items, np.zeros(num_pos_examples * self._m)], axis=1)

        # Combine and shuffle
        all_samples = np.vstack([positive_samples, negative_samples])
        np.random.shuffle(all_samples)

        return all_samples.astype(np.int32)


class MatrixFactorizationElliot:
    """
    A 1-to-1 replication of the Elliot framework's Matrix Factorization logic.
    - Uses NumPy for all calculations.
    - Employs manual Stochastic Gradient Descent (SGD) for updates per sample.
    - Uses a pointwise logistic loss.
    """

    def __init__(self, datadir: str, **kwargs):
        self.config = MFElliotConfig(**kwargs)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        self.category_ = "MatrixFactorization"  # For Folminder compatibility

        # Placeholders
        self.train_matrix: Optional[np.ndarray] = None
        self.pred_matrix: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None

        # Model parameters as NumPy arrays
        self._global_bias: float = 0.0
        self._user_bias: Optional[np.ndarray] = None
        self._item_bias: Optional[np.ndarray] = None
        self._user_factors: Optional[np.ndarray] = None
        self._item_factors: Optional[np.ndarray] = None

    def _initialize_parameters(self, n_users, n_items):
        """Initializes all model parameters as NumPy arrays."""
        self._global_bias = 0.0
        self._user_bias = np.zeros(n_users)
        self._item_bias = np.zeros(n_items)
        self._user_factors = np.random.normal(loc=0.0, scale=0.1, size=(n_users, self.config.factors))
        self._item_factors = np.random.normal(loc=0.0, scale=0.1, size=(n_items, self.config.factors))

    def _predict_single(self, user: int, item: int) -> float:
        """Predicts the score for a single user-item pair."""
        return self._global_bias + self._user_bias[user] + self._item_bias[item] \
            + self._user_factors[user] @ self._item_factors[item]

    def _train_step(self, user: int, item: int, rating: int):
        """Performs a single SGD update step, exactly like Elliot's MFModel."""
        # make prediction
        prediction = self._predict_single(user, item)

        # calculate error/gradient
        sigmoid = 1 / (1 + np.exp(-prediction))
        grad = rating - sigmoid

        # update parameters
        lr = self.config.lr
        reg = self.config.reg

        # store factors before update to use in item factor update
        user_factors_snapshot = self._user_factors[user].copy()

        # update user factors and bias
        self._user_factors[user] += lr * (grad * self._item_factors[item] - reg * user_factors_snapshot)
        self._user_bias[user] += lr * (grad - reg * self._user_bias[user])

        # update item factors and bias (using the pre-update user factor)
        self._item_factors[item] += lr * (grad * user_factors_snapshot - reg * self._item_factors[item])
        self._item_bias[item] += lr * (grad - reg * self._item_bias[item])

        # update global bias
        self._global_bias += lr * (grad - reg * self._global_bias)

    def fit(self, train_matrix: np.ndarray, **kwargs):
        self.train_matrix = train_matrix
        n_users, n_items = train_matrix.shape

        self._initialize_parameters(n_users, n_items)
        sampler = ElliotSampler(train_matrix, self.config.m, self.config.seed)

        print(f"Starting training with Elliot's NumPy-based SGD logic.")

        for epoch in range(1, self.config.epochs + 1):
            # generate all samples for the epoch and shuffle them
            training_samples = sampler.get_all_samples()

            with tqdm(total=len(training_samples), desc=f"Epoch {epoch}/{self.config.epochs}", leave=False) as pbar:
                for user, item, rating in training_samples:
                    self._train_step(user, item, rating)
                    pbar.update(1)

        self.similarity_matrix = cosine_similarity(self._user_factors)
        np.fill_diagonal(self.similarity_matrix, 0.0)

    def compute_predictions(self) -> np.ndarray:
        """Computes the full user-item score matrix after training."""
        if self.pred_matrix is None:
            if self._user_factors is None:
                raise RuntimeError("Model has not been trained yet. Call fit() first.")

            # Use broadcasting for efficient matrix calculation with NumPy
            self.pred_matrix = self._global_bias \
                               + self._user_bias[:, np.newaxis] \
                               + self._item_bias[np.newaxis, :] \
                               + (self._user_factors @ self._item_factors.T)

        return self.pred_matrix