import os
import yaml
from types import SimpleNamespace

class ConfigLoader:
    """
    Loads and validates configuration from a YAML file and converts it into a nested SimpleNamespace.
    This allows attribute-style access (e.g., config.dataset.url) instead of dictionary keys.
    """
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.raw_config = yaml.safe_load(f)

    def load(self) -> SimpleNamespace:
        """
        Transforms the raw configuration into a nested SimpleNamespace.
        """
        raw = self.raw_config
        smart_search_raw = raw.get('smart_search_service', {})
        return SimpleNamespace(
            # knowledge Base and URI configuration
            knowledge_base=raw['knowledge_base'],
            base_uri=raw['base_uri'],
            support_keyword_smart_search = list(raw.get('support_keyword_smart_search', [])),
            smart_search_service=SimpleNamespace(**smart_search_raw),

            # lookup uri config
            lookup_services={
                kb: SimpleNamespace(
                    endpoint=svc.get('endpoint'),
                    headers=svc.get('headers', {}),
                    static_params=svc.get('static_params', {}),
                    keyword_param=svc.get('keyword_param'),
                    results_key=svc.get('results_key'),
                    uri_field=svc.get('uri_field'),
                    id_field=svc.get('id_field'),
                    uri_template=svc.get('uri_template'),
                    fallback_base=svc.get('fallback_base'),
                    search_class_filter=svc.get('search_class_filter'),
                    disambiguation_property=svc.get('disambiguation_property'),
                    year_regex=svc.get('year_regex')
                ) for kb, svc in raw.get('lookup_services', {}).items()
            },

            # dataset configuration
            dataset=SimpleNamespace(
                url=raw['dataset']['url'],
                data_root=os.path.abspath(raw['dataset']['data_root'])
            ),

            # SPARQL configuration
            sparql=SimpleNamespace(
                endpoint=raw['sparql']['endpoint'],
                templates=raw['sparql']['templates'],
                processes=raw['sparql']['processes'],
                chunk_size=raw['sparql']['chunk_size'],
                max_hops=raw['sparql']['max_hops'],
                hop_categories=raw['sparql'].get('hop_categories', [])
            ),

            # data processing rules
            processing=SimpleNamespace(
                entity_keywords=raw['processing'].get('entity_keywords', []),
                interaction_keywords=raw['processing'].get('interaction_keywords', []),
                user_keywords=raw['processing'].get('user_keywords', []),
                entity_columns=raw['processing'].get('entity_columns', []),
                interaction_columns=raw['processing'].get('interaction_columns', []),
                user_columns=raw['processing'].get('user_columns', []),
                entity_relations=raw['processing'].get('entity_relations', []),
                interaction_relations=raw['processing'].get('interaction_relations', []),
                user_relations=raw['processing'].get('user_relations', []),
                column_aliases = dict(
                    raw['processing'].get('column_aliases', {
                        'entity_id': [],
                        'interaction_entity_id': [],
                        'user_id': []
                    })
                )
            ),

            # column name external_mappings
            columns=SimpleNamespace(**raw['columns']),

            # entity mapping configuration
            mapping=SimpleNamespace(
                file=raw['mapping']['file'],
                columns=raw['mapping']['columns']
            ),

            # predicate translation rules
            predicate_mapping=raw['predicate_mapping'],

            # map value in datasets
            value_mappings=raw.get('value_mappings', {})
        )
