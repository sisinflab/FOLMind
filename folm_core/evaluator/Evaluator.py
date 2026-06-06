from pydantic import BaseModel, field_validator
from typing import List, Literal, Dict, Tuple
import numpy as np
import importlib
import logging
import math

# supported metrics
AccuracyMetrics = Literal['ndcg', 'recall', 'precision', 'map', 'mrr', 'f1']
logger = logging.getLogger(__name__)


class EvaluatorConfig(BaseModel):
    train_matrix: np.ndarray
    test_matrix: np.ndarray
    cutoffs: List[int] = [10, 20]
    accuracy_metrics: List[AccuracyMetrics] = ['ndcg', 'recall', 'precision', 'map', 'mrr', 'f1']

    class Config:
        arbitrary_types_allowed = True

    @field_validator('cutoffs')
    def check_cutoff_positive(cls, v):
        if not all(isinstance(x, int) and x > 0 for x in v):
            raise ValueError("All cutoff values must be positive integers")
        return v


class Evaluator:
    def __init__(self, train_matrix: np.ndarray, test_matrix: np.ndarray, **kwargs):
        # use EvaluatorConfig to validate and set params
        config = EvaluatorConfig(train_matrix=train_matrix, test_matrix=test_matrix, **kwargs)

        self.train = config.train_matrix.astype(float)
        self.test = config.test_matrix.astype(float)
        self.num_users, self.num_items = self.train.shape

        # binary relevance: hit if test rating >= 1
        self.relevance = (self.test >= 1).astype(int)
        self.n_rel_per_user = self.relevance.sum(axis=1)

        self.cutoffs = config.cutoffs
        self.accuracy_metrics = config.accuracy_metrics



    def _get_metric(self, metric_name: str, pred: np.ndarray, k: int):
        """
        Dynamically imports a metric function from the correct sub-package
        ('accuracy_metrics' or 'beyond_accuracy_metrics').
        """
        func_name = f"{metric_name}_at_k"
        base_pkg = self.__class__.__module__.rsplit('.', 1)[0]
        module_name = func_name

        sub_folder = None
        if metric_name in self.accuracy_metrics:
            sub_folder = 'accuracy_metrics'

        if not sub_folder:
            raise ValueError(f"Metric '{metric_name}' is not defined in any configured metric list.")

        # construct the full, correct module path
        # e.g., folm_core.evaluator.metrics.accuracy_metrics.ndcg_at_k
        module_path = f"{base_pkg}.metrics.{sub_folder}.{module_name}"

        try:
            # import the module (e.g., the file ndcg_at_k.py)
            metrics_mod = importlib.import_module(module_path)
            # get the function from that module
            metric_fn = getattr(metrics_mod, func_name)
        except (ImportError, AttributeError) as e:
            print(f"DEBUG: Failed to import from path: {module_path}. Error: {e}")
            raise ValueError(
                f"Metric function '{func_name}' could not be loaded. "
                f"Ensure a file named '{module_name}.py' with the function '{func_name}' "
                f"exists in the 'folm_core/evaluator/metrics/{sub_folder}/' directory."
            )

        return metric_fn(pred, k, self)

    def evaluate(self, pred: np.ndarray) -> Dict[str, float]:
        res = {}
        all_metrics = self.accuracy_metrics

        for k in self.cutoffs:
            for metric in all_metrics:
                res[f'{metric}@{k}'] = self._get_metric(metric_name=metric, pred=pred, k=k)
        return res

    def get_recommendations(self, pred: np.ndarray, k: int = 50) -> List[Tuple[int, int, float]]:
        """
        Extracts the top-k items for every user from the prediction matrix.
        Returns a list of tuples: (user_idx, item_idx, score)
        Used for saving .tsv files for statistical significance.
        """
        n_users = pred.shape[0]
        recs_list = []

        # Iterate users to generate the list
        for u_idx in range(n_users):
            user_scores = pred[u_idx]

            # Get top-K indices
            if len(user_scores) <= k:
                # If we have fewer items than k, sort all
                top_k_items = np.argsort(user_scores)[::-1]
            else:
                # argpartition puts the k-th largest element at position -k
                # Everything after -k is larger. We then sort that small slice.
                unsorted_top_k = np.argpartition(user_scores, -k)[-k:]
                sorted_indices_local = np.argsort(user_scores[unsorted_top_k])[::-1]
                top_k_items = unsorted_top_k[sorted_indices_local]

            for i_idx in top_k_items:
                score = float(user_scores[i_idx])
                recs_list.append((u_idx, i_idx, score))

        return recs_list