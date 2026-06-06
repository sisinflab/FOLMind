import numpy as np


def ndcg_at_k(pred: np.ndarray, k: int, evaluator) -> float:
    ndcgs = []
    # discount vector for positions 1..k
    discount = 1.0 / np.log2(np.arange(2, k + 2))

    for u in range(pred.shape[0]):
        cands = np.where(evaluator.train[u] == 0)[0]
        n_rel = int(evaluator.n_rel_per_user[u])
        if len(cands) == 0 or n_rel == 0:
            continue

        scores = pred[u, cands]

        # pick up to k candidates
        if len(cands) > k:
            idx_k = np.argpartition(-scores, k)[:k]
            part_items = cands[idx_k]
            part_scores = scores[idx_k]
            # sort these k by descending score
            order = np.argsort(-part_scores)
            top_items = part_items[order]
            n = k
        else:
            # when fewer candidates than k, sort them all
            order = np.argsort(-scores)
            top_items = cands[order]
            n = len(top_items)

        # binary gains = 1 for any relevant item
        gains = evaluator.relevance[u, top_items]
        dcg = np.dot(gains, discount[:n])

        # ideal DCG: sum of the first min(n_rel, k) discounts
        ideal_n = min(n_rel, k)
        idcg = discount[:ideal_n].sum()

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs, dtype=np.float64)) if ndcgs else 0.0
