from pydantic import BaseModel, field_validator
from typing import List, Literal
import numpy as np
import importlib

# This array defines allowed metric values
# NOTE: You have to define here allowed metric 'name' and put them in evaluator\metrics folder
# then define them in evaluator\metrics\__init__.py
AllowedMetrics = Literal['ndcg', 'recall', 'precision']


class EvaluatorConfig(BaseModel):
    train_matrix: np.ndarray
    test_matrix: np.ndarray
    cutoff: List[int] = [5, 10, 20]
    simple_metrics: List[AllowedMetrics] = ['ndcg', 'recall', 'precision']

    model_config = {
        "arbitrary_types_allowed": True
    }

    @field_validator('cutoff')
    def check_cutoff_positive(cls, v):
        if not all(isinstance(x, int) and x > 0 for x in v):
            raise ValueError("All cutoff values must be positive integers")
        return v


class Evaluator:
    def __init__(self, train_matrix: np.ndarray, test_matrix: np.ndarray,
                 cutoff: List[int] = [5, 10, 20],
                 simple_metrics: List[AllowedMetrics] = ['ndcg', 'recall', 'precision']):
        self.train = train_matrix.astype(float)
        self.test = test_matrix.astype(float)
        # binary relevance: hit if rating ≥ 1
        self.relevance = (self.test >= 1).astype(int)
        # number of relevant items per user (for recall / IDCG)
        self.n_rel_per_user = self.relevance.sum(axis=1)
        self.cut_off = cutoff
        self.simple_metrics = simple_metrics

    def _get_metric(self, metric, pred, k, evaluator):
        func_name = f"{metric}_at_k"
        # metrics package is folm_core.evaluator.metrics
        base_pkg = self.__class__.__module__.rsplit('.', 1)[0]
        metrics_mod = importlib.import_module(f"{base_pkg}.metrics")
        metric_fn = getattr(metrics_mod, func_name, None)
        if metric_fn is None:
            raise ValueError(f"Unknown metric '{metric}'")
        return metric_fn(pred, k, evaluator)

    def evaluate(self, pred: np.ndarray):
        res = {}
        for k in self.cut_off:
            for metric in self.simple_metrics:
                res[f'{metric}@{k}'] = self._get_metric(metric=metric, pred=pred, k=k, evaluator=self)
        return res

    def get_recommendations(self, pred: np.ndarray, k: int = 50):
        """
        Extracts the top-k items for every user from the prediction matrix.
        Returns a list of tuples: (user_idx, item_idx, score)
        """
        n_users = pred.shape[0]

        # argpartition for efficiency
        # argpartition puts the k-th element in position, and smaller elements before, larger after;
        # since we want largest scores, we negate or handle indices carefully
        # we use standard argsort on the top-k partition for exact ordering

        recs_list = []

        # Iterate users to keep memory usage predictable, or use batched torch operations if available

        for u_idx in range(n_users):
            user_scores = pred[u_idx]

            # get indices of top-k
            if len(user_scores) <= k:
                top_k_items = np.argsort(user_scores)[::-1]
            else:
                # partial sort
                unsorted_top_k = np.argpartition(user_scores, -k)[-k:]
                # sort just the top k
                sorted_indices_local = np.argsort(user_scores[unsorted_top_k])[::-1]
                top_k_items = unsorted_top_k[sorted_indices_local]

            for i_idx in top_k_items:
                score = float(user_scores[i_idx])
                recs_list.append((u_idx, i_idx, score))

        return recs_list
