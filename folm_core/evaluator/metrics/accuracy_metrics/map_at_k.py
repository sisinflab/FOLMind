import numpy as np


def map_at_k(pred: np.ndarray, k: int, evaluator) -> float:
    """
    Calculate the Mean Average Precision (MAP) at k.
    MAP is the average of the Average Precision (AP) for each user.
    AP rewards recommendation lists that place relevant items higher up.
    """
    aps = []
    for u in range(pred.shape[0]):
        # only not seen items in training set
        cands = np.where(evaluator.train[u] == 0)[0]
        n_rel = int(evaluator.n_rel_per_user[u])

        if len(cands) == 0 or n_rel == 0:
            continue

        # get top-k recommended items
        scores = pred[u, cands]
        top_k_indices = np.argsort(-scores)[:k]
        top_k_items = cands[top_k_indices]

        # calculate AP@k
        hits = 0
        score = 0.0
        for i, item_idx in enumerate(top_k_items):
            if evaluator.relevance[u, item_idx] == 1:
                hits += 1
                score += hits / (i + 1.0)

        # normalize by number of relevant items (or k, if lower)
        ap = score / min(n_rel, k)
        aps.append(ap)

    return float(np.mean(aps, dtype=np.float64)) if aps else 0.0