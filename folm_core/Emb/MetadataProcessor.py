import os
import pickle
import logging
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)


class MetadataProcessor:
    """
    Handles the processing of knowledge graph data to create structured
    item descriptions suitable for embedding.
    """

    def __init__(self, datadir: str, config):
        self.config = config
        cwd = os.getcwd()
        self.kg_file_path = os.path.join(cwd, 'data', datadir, 'splitting', 'train', 'knowledge_graph.tsv')
        self.entity_file_path = os.path.join(cwd, 'data', datadir, 'splitting', 'train', 'entity_name_mapping.pkl')
        self.output_dir = os.path.join(cwd, 'data', datadir, 'metaEmb_output')
        os.makedirs(self.output_dir, exist_ok=True)

        self.predicate_separator = " [sep] "
        self.value_separator = " | "

    def _join_unique_values(self, items: pd.Series) -> str:
        """Normalizes, deduplicates, sorts, and joins KG object values."""
        cleaned = items.dropna().astype(str).str.replace("_", " ").str.strip().str.lower()
        uniques = sorted(list(dict.fromkeys(cleaned)), key=str.lower)
        return self.value_separator.join(uniques)

    def process_and_save_descriptions(self, output_filename: str) -> Dict[str, str]:
        """
        Builds, filters, and saves item descriptions from the knowledge graph.

        Returns:
            A dictionary of item names to their final, processed descriptions.
        """
        output_path = os.path.join(self.output_dir, output_filename)
        if os.path.exists(output_path):
            logger.info(f"Descriptions file already exists. Loading from {output_path}")
            with open(output_path, "rb") as f:
                return pickle.load(f)

        logger.info("Building item descriptions from knowledge graph...")
        triples = pd.read_csv(
            self.kg_file_path, sep="\t", header=None,
            names=["s", "p", "o"], dtype=str
        )

        with open(self.entity_file_path, "rb") as f:
            name_to_id: Dict[str, str] = pickle.load(f)
        item_names = sorted(name_to_id.keys())

        item_triples = triples[triples["s"].isin(item_names)].copy()
        item_triples["p_clean"] = item_triples["p"].str.strip().str.lower()
        item_triples["po"] = item_triples["p_clean"] + self.predicate_separator + item_triples["o"]

        grouped = item_triples.groupby("s")["po"].agg(self._join_unique_values).to_dict()

        all_items_descriptions = {name: grouped.get(name, "") for name in item_names}

        # filter descriptions based on predicate completeness
        filtered_descriptions = self._filter_descriptions_by_predicate(all_items_descriptions, item_triples, item_names)

        final_descriptions = {
            k: str(v).replace(self.predicate_separator, ' ')
            for k, v in filtered_descriptions.items()
        }

        with open(output_path, "wb") as f:
            pickle.dump(final_descriptions, f)
        logger.info(f"Item descriptions saved to {output_path}")

        return final_descriptions

    def _filter_descriptions_by_predicate(
            self,
            descriptions: Dict[str, str],
            item_triples: pd.DataFrame,
            item_names: List[str]
    ) -> Dict[str, str]:
        """Filters descriptions to only include predicates that meet the missing rate threshold."""
        predicate_stats = self._calculate_predicate_stats(descriptions, len(item_names))

        max_missing_rate = self.config.max_missing_rate_per_pred
        valid_predicates = {
            pred for pred, info in predicate_stats.items()
            if info['pct_missing'] <= max_missing_rate
        }
        logger.info(f"Retaining {len(valid_predicates)} predicates with max missing rate of {max_missing_rate * 100}%")

        pruned_triples = item_triples[item_triples['p_clean'].isin(valid_predicates)].copy()

        if pruned_triples.empty:
            raise ValueError("No triples survived predicate filtering. Cannot create item embeddings.")

        pruned_grouped = pruned_triples.groupby("s")["po"].agg(self._join_unique_values).to_dict()

        return {name: pruned_grouped.get(name, "") for name in item_names}

    def _calculate_predicate_stats(self, descriptions: Dict[str, str], total_items: int) -> Dict[str, Dict]:
        """Calculates statistics on predicate usage across all items."""
        predicate_counts = {p: 0 for p in set(
            part.split(self.predicate_separator)[0].strip()
            for desc in descriptions.values() if desc
            for part in desc.split(self.value_separator)
        )}

        for desc in descriptions.values():
            if not desc:
                continue
            seen_predicates = {
                part.split(self.predicate_separator)[0].strip()
                for part in desc.split(self.value_separator)
            }
            for p in seen_predicates:
                if p in predicate_counts:
                    predicate_counts[p] += 1

        stats = {}
        for pred, count in predicate_counts.items():
            stats[pred] = {
                'count': count,
                'n_missing': total_items - count,
                'pct_missing': (total_items - count) / total_items
            }
        return stats
