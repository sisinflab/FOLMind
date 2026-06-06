import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Folminder:
    """
    Hybrid recommender post-processor using embedding-derived similarity.
    This class receives the full similarity ranking and applies the top_k_similar cut internally.
    """

    def __init__(self, filters: Dict[int, Dict[str, Any]], folm_config: Any, model=None):
        self.folm_config = folm_config
        self.mode = folm_config.mode
        self.model = model
        self.alpha = folm_config.alpha
        self.top_k_similar = folm_config.top_k_similar

        # store neighbors and their similarity scores, applying the top_k_similar cut
        k = self.top_k_similar
        if k != -1:
            self.similar_entities = {u: spec.get('similar_entities', [])[:k] for u, spec in filters.items()}
            self.similarity_scores = {u: spec.get('similarity_scores', [])[:k] for u, spec in filters.items()}
        else:
            self.similar_entities = {u: spec.get('similar_entities', []) for u, spec in filters.items()}
            self.similarity_scores = {u: spec.get('similarity_scores', []) for u, spec in filters.items()}

        logger.info(
            f"** ---> Folminder loaded. Will use top {k if k != -1 else 'All'} semantic neighbors "
            f"for entity type '{folm_config.entity_emb}'.")

    def apply(self, *args, **kwargs) -> np.ndarray:
        """
        Dispatch to the correct apply method based on model category and mode.
        """
        method_name = f"_apply_{self.model.category_}_{self.mode}"
        if not hasattr(self, method_name):
            raise NotImplementedError(
                f"**Folminder ---> No method found for model category '{self.model.category_}' and mode '{self.mode}'")

        method = getattr(self, method_name)
        return method(*args, **kwargs)

    def _apply_kNN_inject(self) -> np.ndarray:
        """
        For k-NN models. Injects embedding-based similarity into the
        original similarity matrix. The `alpha` parameter controls the weight.
        The neighbors used for injection are already pruned to top_k_similar.
        """
        if self.model.similarity_matrix is None:
            raise RuntimeError("Base model similarity matrix not found. Did you call .fit()?")

        sim = self.model.similarity_matrix.copy()

        # Inject similarities from embeddings (already pruned to top_k_similar in __init__)
        for entity_idx, neighs in self.similar_entities.items():
            scores = self.similarity_scores.get(entity_idx, [])
            for neigh_idx, score in zip(neighs, scores):
                if entity_idx < sim.shape[0] and neigh_idx < sim.shape[1]:
                    sim[entity_idx, neigh_idx] += self.alpha * score

        np.fill_diagonal(sim, 0.0)
        # final pruning uses the base model's k, as it's an injection into the collaborative matrix
        W = self.model.prune_to_k(S=sim)

        entity_type = self.folm_config.entity_emb
        logger.info(f"Applying kNN injection for entity type: '{entity_type}'")

        if entity_type == 'user':
            return W.dot(self.model.train_matrix)
        elif entity_type == 'item':
            return self.model.train_matrix.dot(W)
        else:
            raise ValueError(f"Unsupported entity type for kNN_inject: {entity_type}")

    def _apply_kNN_hybrid_blend(self) -> np.ndarray:
        """
        For k-NN models. Blends predictions from a collaborative k-NN and a semantic k-NN.
        The semantic k-NN is built using top_k_similar neighbors.
        The collaborative k-NN uses the base model's k.
        """
        logger.info(f"Applying 'hybrid_blend' to k-NN model for entity type '{self.folm_config.entity_emb}'.")

        # Build a semantic similarity matrix S_emb based on pre-computed similarities
        entity_type = self.folm_config.entity_emb
        num_users, num_items = self.model.train_matrix.shape
        num_entities = num_users if entity_type == 'user' else num_items
        S_emb = np.zeros((num_entities, num_entities), dtype=np.float32)

        for entity_idx, neighs in self.similar_entities.items():
            scores = self.similarity_scores.get(entity_idx, [])
            if entity_idx < S_emb.shape[0]:
                valid_neighs = [n for n in neighs if n < S_emb.shape[1]]
                valid_scores = [s for n, s in zip(neighs, scores) if n < S_emb.shape[1]]
                S_emb[entity_idx, valid_neighs] = valid_scores

        np.fill_diagonal(S_emb, 0.0)

        # Prune the semantic matrix using top_k_similar
        k_semantic = self.top_k_similar if self.top_k_similar != -1 else num_entities - 1
        W_emb = self._prune_to_k_generic(S_emb, k_semantic)

        # Identify semantic neighbors
        has_semantic = (W_emb.sum(axis=1) > 0).astype(np.float32)

        # Normalize weights to prevent score inflation
        row_sums = W_emb.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W_emb_normalized = W_emb / row_sums

        # Compute predictions from the semantic k-NN
        if entity_type == 'user':
            knn_semantic_preds = W_emb_normalized.dot(self.model.train_matrix)
        else:  # item
            knn_semantic_preds = self.model.train_matrix.dot(W_emb_normalized)

        # Get predictions from the base collaborative k-NN model
        knn_collaborative_preds = self.model.compute_predictions()

        if self.alpha == 1.0:
            logger.info("Alpha is 1.0, returning predictions from COLLABORATIVE k-NN ONLY.")
            return knn_collaborative_preds

        logger.info(f"Blending Collaborative k-NN (alpha={self.alpha}) and Semantic k-NN (alpha={1.0 - self.alpha}). "
                    f"Applying Smart Fallback for entities with missing embeddings.")

        # Calculate dynamic alpha per entity:
        # If has_semantic == 1: uses config alpha.
        # If has_semantic == 0: forces alpha to 1.0 (pure collaborative).
        effective_alpha = (self.alpha * has_semantic) + (1.0 * (1.0 - has_semantic))
        effective_beta = (1.0 - self.alpha) * has_semantic

        if entity_type == 'user':
            # Broadcast across users (rows)
            final_preds = (effective_alpha[:, np.newaxis] * knn_collaborative_preds) + \
                          (effective_beta[:, np.newaxis] * knn_semantic_preds)
        else:
            # Broadcast across items (columns)
            final_preds = (effective_alpha[np.newaxis, :] * knn_collaborative_preds) + \
                          (effective_beta[np.newaxis, :] * knn_semantic_preds)

        return final_preds

    def _apply_MatrixFactorization_hybrid_blend(self) -> np.ndarray:
        """
        For MF models. Blends MF predictions with k-NN predictions from embeddings.
        This implementation is already decoupled and uses top_k_similar correctly.
        """
        if self.folm_config.entity_emb != 'user':
            raise ValueError(
                "Hybrid blend for MatrixFactorization currently only supports user-based embeddings ('entity_emb: user').")

        mf_preds = self.model.compute_predictions()

        num_users, num_items = self.model.train_matrix.shape
        S_emb = np.zeros((num_users, num_users), dtype=np.float32)

        # pruned neighbors
        for u, neighs in self.similar_entities.items():
            scores = self.similarity_scores.get(u, [])
            if u < S_emb.shape[0]:
                valid_neighs = [n for n in neighs if n < S_emb.shape[1]]
                valid_scores = [s for n, s in zip(neighs, scores) if n < S_emb.shape[1]]
                S_emb[u, valid_neighs] = valid_scores

        np.fill_diagonal(S_emb, 0.0)

        k = self.top_k_similar if self.top_k_similar != -1 else num_users - 1
        W_emb = self._prune_to_k_generic(S_emb, k)

        # Identify which users have semantic neighbors
        has_semantic = (W_emb.sum(axis=1) > 0).astype(np.float32)

        row_sums = W_emb.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        W_emb = W_emb / row_sums

        knn_preds = W_emb.dot(self.model.train_matrix)

        logger.info(f"Blending MF and Embedding-kNN predictions (alpha={self.alpha}). "
                    f"Applying Smart Fallback for users with missing embeddings.")

        # Calculate dynamic alpha per user
        effective_alpha = (self.alpha * has_semantic) + (1.0 * (1.0 - has_semantic))
        effective_beta = (1.0 - self.alpha) * has_semantic

        # Broadcast across users (rows)
        final_preds = (effective_alpha[:, np.newaxis] * mf_preds) + \
                      (effective_beta[:, np.newaxis] * knn_preds)

        return final_preds

    @staticmethod
    def _prune_to_k_generic(S: np.ndarray, k: int) -> np.ndarray:
        """A generic function to keep only the top-k positive neighbors per row."""
        n = S.shape[0]
        W = np.zeros_like(S)
        if k < 0 or k >= n: k = n - 1
        if k == 0: return W

        for i in range(n):
            row = S[i].copy()
            pos_indices = np.where(row > 0)[0]

            if len(pos_indices) == 0:
                continue

            if len(pos_indices) > k:
                top_k_in_pos = np.argpartition(-row[pos_indices], k)[:k]
                top_idx = pos_indices[top_k_in_pos]
            else:
                top_idx = pos_indices

            W[i, top_idx] = row[top_idx]

        return W