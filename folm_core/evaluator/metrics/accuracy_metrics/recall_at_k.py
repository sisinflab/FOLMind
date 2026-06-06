import numpy as np

def recall_at_k(pred: np.ndarray, k: int, evaluator) -> float:
    recalls = []
    for u in range(pred.shape[0]):
        cands = np.where(evaluator.train[u] == 0)[0]
        if len(cands) == 0 or evaluator.n_rel_per_user[u] == 0:
            continue

        scores = pred[u, cands]
        if len(cands) > k:
            idx = np.argpartition(-scores, k)[:k]
        else:
            idx = np.argsort(-scores)[:k]
        top_items = cands[idx]

        hits = evaluator.relevance[u, top_items].sum()
        recalls.append(hits / evaluator.n_rel_per_user[u])

    return float(np.mean(recalls, dtype=np.float64)) if recalls else 0.0
