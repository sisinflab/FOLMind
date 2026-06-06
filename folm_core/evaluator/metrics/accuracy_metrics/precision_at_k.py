import numpy as np

def precision_at_k(pred: np.ndarray, k: int, evaluator) -> float:
    precisions = []
    for u in range(pred.shape[0]):
        # only recommend items the user hasn't seen in train
        cands = np.where(evaluator.train[u] == 0)[0]
        if len(cands) == 0:
            continue

        scores = pred[u, cands]
        if len(cands) > k:
            idx = np.argpartition(-scores, k)[:k]
        else:
            idx = np.argsort(-scores)[:k]
        top_items = cands[idx]

        hits = evaluator.relevance[u, top_items].sum()
        precisions.append(hits / k)

    return float(np.mean(precisions, dtype=np.float64)) if precisions else 0.0
