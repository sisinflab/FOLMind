import os
import numpy as np
import pickle
import logging
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# avoid division by zero
EPS = 1e-8


class Refiner:
    """
    Refines recommendations by computing similarity rankings for entities.
    This class computes the full ranking and saves it to a reusable cache file.
    """

    def __init__(self, dataset_dir: str, recs_dir: str, model_cfg):
        self.dataset_path = os.path.join(os.getcwd(), 'data', dataset_dir)
        recs_path = os.path.join(self.dataset_path, 'recs')
        os.makedirs(recs_path, exist_ok=True)

        # configuration
        self.fc = model_cfg.folm
        self.entity_type = self.fc.entity_emb  # 'user' or 'item'

        if not self.fc.emb_source_model:
            self.fc.emb_source_model = ""
        # the save path is generic and reusable, based on the full config
        self.save_path = os.path.join(
            recs_dir,
            f'refined_similarities_{self.fc.entity_emb}_{self.fc.type_emb}_{self.fc.info_emb}_{self.fc.emb_source_model}.pkl'
        )

        # switch between LLMEmb and metaEmb embedding sources
        emb_dir_name = 'LLMEmb_output' if self.fc.info_emb == 'LLMEmb' else 'metaEmb_output'
        self.dir = os.path.join(os.getcwd(), 'data', dataset_dir, emb_dir_name)

        embedding_path = os.path.join(self.dir, f'{self.entity_type}_{self.fc.type_emb}_embeddings_{self.fc.emb_source_model}.safetensors')

        # load and filter embeddings based on their key names
        self.entity_data, self.entity_ids = self._load_and_filter_embeddings(embedding_path)

    def _load_and_filter_embeddings(self, embedding_path: str):
        """
        Loads embeddings from a safetensors file and filters them based on the
        entity type and information source ('LLMEmb' vs 'MetaEmb')
        """
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found at {embedding_path}")

        embeddings = load_file(embedding_path)
        entity_data = {}

        # define the required key suffix based on configuration
        suffix_map = {
            ('user', 'LLMEmb'): "_profile",
            ('user', 'MetaEmb'): "_likes",
        }
        required_suffix = suffix_map.get((self.entity_type, self.fc.info_emb))

        for full_key, vec in embeddings.items():
            canonical_key = None

            if self.entity_type == 'user':
                # for users, filter by suffix (e.g., '_profile' or '_likes')
                if required_suffix and full_key.startswith('user_') and full_key.endswith(required_suffix):
                    canonical_key = full_key[:-len(required_suffix)]

            elif self.entity_type == 'item':
                # for MetaEmb items, key is a simple ID.
                if self.fc.info_emb == 'MetaEmb' and full_key.isdigit():
                    canonical_key = f"{full_key}"
                # for LLMEmb items, key is '{id}'.
                elif self.fc.info_emb == 'LLMEmb' and full_key.isdigit():
                    canonical_key = f"{full_key}"

            if canonical_key and canonical_key not in entity_data:
                entity_data[canonical_key] = np.nan_to_num(vec.float().numpy())

        entity_ids = sorted(entity_data.keys())

        logger.info(
            f"Loaded {len(entity_ids)} valid '{self.entity_type}' embeddings from {embedding_path} "
            f"(Source: '{self.fc.info_emb}', Type: '{self.fc.type_emb}')."
        )

        if not entity_ids:
            logger.error(f"No embeddings were loaded. Check configuration and file: {embedding_path}")
            first_key = next(iter(embeddings.keys()), "N/A")
            logger.error(f"Sample key in file: '{first_key}'. Check if it matches expected patterns.")

        return entity_data, entity_ids

    def _build_embedding_matrix(self) -> np.ndarray:
        """Stack embeddings into a matrix (n_entities, dim)"""
        return np.stack([self.entity_data[eid] for eid in self.entity_ids])

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        """Row-normalize matrix to unit length for cosine similarity"""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms < EPS, EPS, norms)
        return matrix / norms

    def _get_top_k(self, entity_idx: int, norm_matrix: np.ndarray, k: int):
        """
        Return indices and similarity scores for top-k similar entities (excluding self).
        k = -1 means all other entities.
        """
        target = norm_matrix[entity_idx]
        scores = norm_matrix.dot(target)
        scores[entity_idx] = -np.inf
        scores = np.nan_to_num(scores, neginf=-np.inf)

        if k == -1 or k >= len(scores) - 1:
            k = len(scores) - 1

        if k == 0:
            return np.array([]), np.array([])

        top_idx_unsorted = np.argpartition(-scores, k)[:k]
        top_scores_unsorted = scores[top_idx_unsorted]

        sorted_order = np.argsort(-top_scores_unsorted)
        top_idx = top_idx_unsorted[sorted_order]
        top_scores = top_scores_unsorted[sorted_order]

        return top_idx, top_scores

    def compose_refining(self) -> dict:
        """
        Compute similarity rankings for each entity, always for ALL other entities.
        Returns dict: eid -> {'similar_entities', 'similarity_scores'}
        """
        if not self.entity_ids:
            raise ValueError(
                "Cannot compose refining because no entities were loaded. Check config and embedding files.")

        n_entities = len(self.entity_ids)
        # compute the full ranking
        k = n_entities - 1

        logger.info(f"Refiner will compute and save the full similarity ranking for each of the {n_entities} entities.")
        embedding_matrix = self._build_embedding_matrix()
        norm_embeddings = self._normalize(embedding_matrix)

        refined_data = {}
        for idx, eid in enumerate(self.entity_ids):
            idx_k, scores_k = self._get_top_k(idx, norm_embeddings, k)
            entities_k = [self.entity_ids[i] for i in idx_k]

            refined_data[eid] = {
                'similar_entities': entities_k,
                'similarity_scores': [float(s) for s in scores_k],
            }

        with open(self.save_path, 'wb') as f:
            pickle.dump(refined_data, f)
        logger.info(f"Saved FULL refined {self.entity_type} rankings to {self.save_path}")
        return refined_data
