import numpy as np


def mrr_at_k(pred: np.ndarray, k: int, evaluator) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) at k.
    The MRR is the average of the reciprocals of the ranks of the first correct recommendation.
    """
    reciprocal_ranks = []
    for u in range(pred.shape[0]):
        cands = np.where(evaluator.train[u] == 0)[0]
        if len(cands) == 0 or evaluator.n_rel_per_user[u] == 0:
            continue

        scores = pred[u, cands]
        top_k_indices = np.argsort(-scores)[:k]
        top_k_items = cands[top_k_indices]

        # find rank of first relevant item
        rank = 0
        for i, item_idx in enumerate(top_k_items):
            if evaluator.relevance[u, item_idx] == 1:
                rank = i + 1
                break

        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks, dtype=np.float64)) if reciprocal_ranks else 0.0