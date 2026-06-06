import pandas as pd
from pathlib import Path
import multiprocessing
from .config import ConfigLoader
from .loader import Loader, find_csv_files
from .query import SparqlQueryer
from .processor import TripleProcessor
from .utils import download_dataset, store_tsv, apply_value_mappings, convert_to_numeric, create_entity_name_mapping
from typing import Set, Any, Optional


class Crawler:
    def __init__(self, config_manager, config_path: str):
        multiprocessing.freeze_support()
        multiprocessing.set_start_method('spawn', force=True)
        self.config_manager = config_manager
        self.config = ConfigLoader(config_path).load()
        self.run()

    def process_local_data(self, loader, entity_processor, interaction_processor, user_processor, value_mappings,
                           queryer, valid_item_ids: Optional[Set[Any]] = None):
        print("Processing local data...")
        entities_df = loader.load_entities(queryer, valid_item_ids=valid_item_ids)
        interactions_df = loader.load_interactions(entities_df)
        users_df = loader.load_users()
        entities_df = apply_value_mappings(entities_df, value_mappings.get('entities', {}))
        interactions_df = apply_value_mappings(interactions_df, value_mappings.get('interactions', {}))
        users_df = apply_value_mappings(users_df, value_mappings.get('users', {}))
        local_triples = pd.concat([
            interaction_processor.create_relations(interactions_df),
            entity_processor.create_relations(entities_df),
            user_processor.create_relations(users_df)
        ])
        return entities_df, interactions_df, users_df, local_triples

    def query_knowledge_base(self, queryer, uris):
        print("Querying knowledge base...")
        http_uris = [u for u in uris if isinstance(u, str) and u.startswith('http')]
        return queryer.multi_hop_query(http_uris)

    def run(self):
        dataset_dir = download_dataset(self.config.dataset.url, self.config.dataset.data_root)
        self.config.dataset.data_dir = dataset_dir

        train_dir = Path(dataset_dir) / 'splitting/train'
        kg_path = train_dir / 'knowledge_graph.tsv'

        ent_name_mapping_path = train_dir / "entity_name_mapping.pkl"

        if kg_path.exists() and ent_name_mapping_path.exists():
            print(f"Knowledge graph and entity mapping already exist. Skipping processing.")
            return

        loader = Loader(self.config)
        queryer = SparqlQueryer(self.config)

        entity_processor = TripleProcessor(self.config.processing.entity_relations,
                                           self.config.predicate_mapping.get(self.config.knowledge_base, {}))
        interaction_processor = TripleProcessor(self.config.processing.interaction_relations,
                                                self.config.predicate_mapping.get(self.config.knowledge_base, {}))
        user_processor = TripleProcessor(self.config.processing.user_relations,
                                         self.config.predicate_mapping.get(self.config.knowledge_base, {}))
        value_mappings = self.config.value_mappings if hasattr(self.config, 'value_mappings') else {}

        valid_item_ids = None
        try:
            interaction_files = find_csv_files(dataset_dir, self.config.processing.interaction_keywords)
            if interaction_files:
                entire_interactions_df = pd.read_csv(interaction_files[0])
                entire_interactions_df.rename(columns={loader.detect_column(entire_interactions_df, 'entity_id'): "itemId"}, inplace=True)
                entire_interactions_df.rename(columns={self.config.columns.user_id: "userId"}, inplace=True)
                valid_item_ids = self.config_manager.dataset_config.split_data(
                    entire_interactions_df, dataset_dir, self.config.processing.interaction_keywords[0]
                )
        except (IndexError, FileNotFoundError):
            print("Warning: No interaction file found to split. The loader will search for a pre-split train file.")


        if not train_dir.exists():
            train_dir.mkdir(parents=True, exist_ok=True)

        entities_df, interactions_df, users_df, local_triples = self.process_local_data(
            loader, entity_processor,
            interaction_processor,
            user_processor, value_mappings,
            queryer,
            valid_item_ids=valid_item_ids
        )


        interacted_uris = set(local_triples['o'].dropna().unique())
        all_entity_uris = set(entities_df[self.config.columns.kb_uri].dropna().unique().tolist())
        uris_to_query = list(interacted_uris | all_entity_uris)
        queried_triples = self.query_knowledge_base(queryer, uris_to_query)

        complete_df = pd.concat([local_triples, queried_triples]).drop_duplicates().reset_index(drop=True)

        # create the URI-to-Name map
        uri_to_name_map = pd.Series(
            entities_df[self.config.columns.fallback_uri].values,
            index=entities_df[self.config.columns.kb_uri].astype(str)
        ).to_dict()

        if self.config.knowledge_base == 'wikidata':
            all_entities_in_graph = set(complete_df['s']).union(set(complete_df['o']))
            external_uris_to_lookup = {uri for uri in all_entities_in_graph if
                                       isinstance(uri, str) and uri.startswith('http') and uri not in uri_to_name_map}
            if external_uris_to_lookup:
                qid_to_label_map = queryer.batch_lookup_labels(list(external_uris_to_lookup))
                uri_to_name_map.update(qid_to_label_map)

        # create the human-readable DataFrame
        human_readable_df = complete_df.copy()
        human_readable_df['s'] = human_readable_df['s'].map(uri_to_name_map).fillna(human_readable_df['s'])
        human_readable_df['o'] = human_readable_df['o'].map(uri_to_name_map).fillna(human_readable_df['o'])

        # save the KG. The new store_tsv function will handle all cleaning.
        store_tsv(human_readable_df, str(kg_path))

        # create the consistent entity name mapping pickle file
        create_entity_name_mapping(
            entities_df,
            str(ent_name_mapping_path),
            self.config
        )

        print(f"Created knowledge graph with {len(complete_df)} triples")

        numeric_kg_path = train_dir / 'knowledge_graph_numeric.tsv'
        convert_to_numeric(str(kg_path), str(train_dir), str(numeric_kg_path))
        print(f"Created knowledge graph with numerical values instead of string")