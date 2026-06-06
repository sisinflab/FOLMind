import os
import pickle
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from safetensors import safe_open
from safetensors.numpy import save_file

logger = logging.getLogger(__name__)


class UserEmbeddingAggregator:
    """
    - MetaEmb pipeline
    Computes user embeddings by performing a weighted aggregation of item embeddings
    based on user interactions (likes/dislikes).
    """

    def __init__(self, datadir: str, predicates: list):
        cwd = os.getcwd()
        self.kg_file_path = os.path.join(cwd, 'data', datadir, 'splitting', 'train', 'knowledge_graph.tsv')
        self.entity_mapping_path = os.path.join(cwd, 'data', datadir, 'splitting', 'train', 'entity_name_mapping.pkl')
        self.metaemb_dir = os.path.join(cwd, 'data', datadir, 'metaEmb_output')
        self.predicates = predicates

    def aggregate_for_all_item_embeddings(self):
        """
        Dynamically finds all item embedding files (`item_*.safetensors`) in the
        meta-embedding directory and generates corresponding user embeddings
        (`user_*.safetensors`) if they do not already exist.
        """
        logging.info(f"Scanning for item embeddings to process in: {self.metaemb_dir}")

        try:
            all_files_in_dir = os.listdir(self.metaemb_dir)
        except FileNotFoundError:
            logging.error(f"Directory not found: {self.metaemb_dir}. Cannot aggregate user embeddings.")
            return

        # find all valid 'item_*' embedding files in the directory.
        item_files = [
            f for f in all_files_in_dir
            if f.startswith('item_') and f.endswith('.safetensors')
        ]

        if not item_files:
            logging.warning("No item embedding files found in the directory. Nothing to aggregate.")
            return

        # identify which item files are missing a corresponding user embedding file.
        files_to_process = []
        for item_filename in item_files:
            # construct the expected user embedding filename
            user_filename = item_filename.replace('item_', 'user_', 1)
            user_filepath = os.path.join(self.metaemb_dir, user_filename)

            # if the user embedding file doesn't exist, add the item file to our queue
            if not os.path.exists(user_filepath):
                files_to_process.append(item_filename)
            else:
                logging.info(f"User embedding '{user_filename}' already exists. Skipping.")

        # process the queue of item files that need user embeddings
        if not files_to_process:
            logging.info("All required user embeddings are already generated. No new aggregation needed.")
            return

        logging.info(f"Found {len(files_to_process)} item embedding(s) that need user aggregation: {files_to_process}")

        for item_filename in files_to_process:
            user_filename = item_filename.replace('item_', 'user_', 1)
            self.compute_and_save_user_embeddings(item_filename, user_filename)

    def compute_and_save_user_embeddings(self, item_emb_filename: str, user_emb_filename: str):
        """
        Orchestrates the process of loading data and computing user embeddings.
        """
        logger.info(f"Aggregating user embeddings from {item_emb_filename}...")

        # parse KG for user interactions
        likes, dislikes = self._parse_interactions()
        total_users = len(likes)

        # load item embeddings
        item_embeddings = self._load_item_embeddings(item_emb_filename)
        if not item_embeddings:
            logger.warning(f"No item embeddings found in {item_emb_filename}. Skipping user embedding generation.")
            return

        # compute item popularity
        entity2id = self._load_entity_mapping()
        like_pop, dislike_pop = self._compute_item_popularity(likes, dislikes, entity2id)

        # compute user embeddings
        user_embeddings = {}
        dim = next(iter(item_embeddings.values())).shape[0]

        for user_id, liked_ents in tqdm(likes.items(), desc="Aggregating user profiles"):
            liked_ids = [entity2id[e] for e in liked_ents if e in entity2id]
            disliked_ids = [entity2id[e] for e in dislikes.get(user_id, set()) if e in entity2id]

            if not liked_ids and not disliked_ids:
                continue

            emb_l = self._weighted_mean(liked_ids, like_pop, item_embeddings, dim, total_users)
            emb_d = self._weighted_mean(disliked_ids, dislike_pop, item_embeddings, dim, total_users)

            user_embeddings[f"{user_id}_likes"] = emb_l
            user_embeddings[f"{user_id}_dislikes"] = emb_d

        # Save
        output_path = os.path.join(self.metaemb_dir, user_emb_filename)
        save_file(user_embeddings, output_path)
        logger.info(f"Saved {len(user_embeddings)} aggregated user embeddings to {output_path}")

    def _parse_interactions(self):
        likes, dislikes = defaultdict(set), defaultdict(set)
        with open(self.kg_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3 and parts[1] in self.predicates:
                    user, pred, item = parts
                    (likes if pred == 'likes' else dislikes)[user].add(item)
        return likes, dislikes

    def _load_item_embeddings(self, filename: str) -> dict:
        path = os.path.join(self.metaemb_dir, filename)
        embeddings = {}
        with safe_open(path, framework='np') as sf:
            for key in sf.keys():
                embeddings[int(key)] = sf.get_tensor(key).astype(np.float32)
        return embeddings

    def _load_entity_mapping(self) -> dict:
        with open(self.entity_mapping_path, 'rb') as f:
            return pickle.load(f)

    def _compute_item_popularity(self, likes, dislikes, entity2id):
        like_pop, dislike_pop = defaultdict(int), defaultdict(int)
        for user_likes in likes.values():
            for entity in user_likes:
                if entity in entity2id:
                    like_pop[entity2id[entity]] += 1
        for user_dislikes in dislikes.values():
            for entity in user_dislikes:
                if entity in entity2id:
                    dislike_pop[entity2id[entity]] += 1
        return like_pop, dislike_pop

    @staticmethod
    def _weighted_mean(ids, popularity, embeddings, dim, total_users, T=0.1):
        """Computes a log-IDF softmax weighted mean of embeddings."""
        embs, pops = [], []
        for i in ids:
            if i in embeddings:
                embs.append(embeddings[i])
                pops.append(float(popularity.get(i, 1)))

        if not embs: return np.zeros(dim, dtype=np.float32)

        mats = np.vstack(embs)
        pops = np.array(pops, dtype=np.float32)

        raw_idf = np.log((total_users + 1.0) / (pops + 1.0))
        scaled = (raw_idf / T) - (raw_idf / T).max()  # numerical stability
        exp_w = np.exp(scaled)
        weights = exp_w / exp_w.sum()

        return weights @ mats
