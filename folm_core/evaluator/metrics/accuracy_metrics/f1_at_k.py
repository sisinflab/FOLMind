import numpy as np


def f1_at_k(pred: np.ndarray, k: int, evaluator) -> float:
    """
    Calculate the F1-Score at k.
    """
    f1_scores = []
    for u in range(pred.shape[0]):
        cands = np.where(evaluator.train[u] == 0)[0]
        n_rel = int(evaluator.n_rel_per_user[u])

        if len(cands) == 0 or n_rel == 0:
            continue

        scores = pred[u, cands]
        top_k_indices = np.argsort(-scores)[:k]
        top_k_items = cands[top_k_indices]

        hits = evaluator.relevance[u, top_k_items].sum()

        precision = hits / k
        recall = hits / n_rel

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
        else:
            f1_scores.append(0.0)

    return float(np.mean(f1_scores, dtype=np.float64)) if f1_scores else 0.0