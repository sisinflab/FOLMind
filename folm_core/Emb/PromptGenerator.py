import os
import re
import pickle
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import ujson

logger = logging.getLogger(__name__)


class PromptGenerator:
    """Generates user and item prompts based on templates and a knowledge graph."""

    def __init__(self, datadir: str, config):
        self.config = config
        cwd = os.getcwd()
        self.kg_file_path = os.path.join(cwd, 'data', datadir, 'splitting', 'train', 'knowledge_graph.tsv')
        self.rules_file_path = os.path.join(cwd, 'data', datadir, 'amie_mined', 'mined_rules_string.csv')
        self.output_dir = os.path.join(cwd, 'data', datadir, 'LLMEmb_output')
        os.makedirs(self.output_dir, exist_ok=True)

        self.user_template = self._load_template(config.user_prompt_file)
        self.item_template = self._load_template(config.item_prompt_file)

        self.triple_index = self._build_triple_index()
        self.interaction_predicates = self._load_interaction_predicates()
        self.rules = self._load_rules()

    def _load_template(self, filename: str) -> str:
        """Loads a prompt template file."""
        path = os.path.join(os.getcwd(), 'folm_core', 'Emb', 'prompts', filename)
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _build_triple_index(self):
        index = defaultdict(list)
        with open(self.kg_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    index[parts[0]].append(tuple(parts))
        return index

    def _load_interaction_predicates(self) -> list:
        # Simplified predicate loading logic
        return ['likes', 'dislikes']  # Assuming this is sufficient based on context

    def _load_rules(self) -> list:
        if not self.interaction_predicates:
            return []
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, self.interaction_predicates)) + r')\b')
        matched_rules = []
        with open(self.rules_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '=>' in line and pattern.search(line.split('=>')[1]):
                    matched_rules.append(line.strip())
        return matched_rules

    def generate_all_prompts(self):
        """Generates both user and item prompts."""
        self.generate_user_prompts()
        self.generate_item_prompts()

    def generate_user_prompts(self):
        """Generates and saves prompts for all users."""
        users = [u for u in self.triple_index if u.startswith('user_')]
        self._generate_prompts_parallel(
            entity_list=users,
            process_batch_func=self._process_user_batch,
            output_filename='user_prompts.pkl',
            description="Generating User Prompts"
        )

    def generate_item_prompts(self):
        """Generates and saves prompts for all items."""
        items = [i for i in self.triple_index if not i.startswith('user_')]
        self._generate_prompts_parallel(
            entity_list=items,
            process_batch_func=self._process_item_batch,
            output_filename='item_prompts.pkl',
            description="Generating Item Prompts"
        )

    def _generate_prompts_parallel(self, entity_list, process_batch_func, output_filename, description):
        output_path = os.path.join(self.output_dir, output_filename)

        all_prompts = self._load_existing(output_path) if self.config.resume else {}
        processed_entities = set(all_prompts.keys())

        entities_to_process = [e for e in entity_list if e not in processed_entities]

        if not entities_to_process:
            logger.info(f"All prompts for {output_filename} already generated.")
            return

        with ProcessPoolExecutor(max_workers=self.config.threads) as executor:
            chunk_size = max(1, len(entities_to_process) // (self.config.threads * 4))
            futures = [
                executor.submit(
                    process_batch_func,
                    entities_to_process[i:i + chunk_size],
                    self.triple_index,
                    self.rules,
                    self.user_template if 'user' in output_filename else self.item_template,
                    self.interaction_predicates
                )
                for i in range(0, len(entities_to_process), chunk_size)
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=description):
                all_prompts.update(future.result())

        self._atomic_save(all_prompts, output_path)
        logger.info(f"Saved {len(all_prompts)} prompts to {output_path}")

    @staticmethod
    def _process_user_batch(user_batch, triple_index, rules, template, predicates):
        batch_prompts = {}
        for user_id in user_batch:
            full_triples = {user_id: triple_index.get(user_id, [])}
            for _, _, obj in full_triples[user_id]:
                full_triples[obj] = triple_index.get(obj, [])

            user_history, items_meta, user_profile = PromptGenerator._format_user_data_for_prompt(full_triples,
                                                                                                  predicates)
            batch_prompts[user_id] = template.format(
                user_profile=user_profile, user_history=user_history, items_meta=items_meta, rules=rules
            )
        return batch_prompts

    @staticmethod
    def _process_item_batch(item_batch, triple_index, rules, template, predicates):
        batch_prompts = {}
        for item_id in item_batch:
            triples = triple_index.get(item_id, [])
            meta = defaultdict(list)
            for _, pred, obj in triples:
                meta[pred].append(obj)

            items_meta = ujson.dumps({item_id: dict(meta)})
            batch_prompts[item_id] = template.format(items_meta=items_meta, rules=rules)
        return batch_prompts

    @staticmethod
    def _format_user_data_for_prompt(data, predicates):
        user_id = next(iter(data))
        user_interactions = defaultdict(list)
        user_attributes = defaultdict(list)
        item_metadata = defaultdict(lambda: defaultdict(list))

        for entity, triples in data.items():
            if entity == user_id:
                for _, pred, obj in triples:
                    (user_interactions if pred in predicates else user_attributes)[pred].append(obj)
            else:
                for _, pred, obj in triples:
                    item_metadata[entity][pred].append(obj)

        return (
            ujson.dumps({user_id: dict(user_interactions)}),
            ujson.dumps({k: dict(v) for k, v in item_metadata.items()}),
            ujson.dumps({user_id: dict(user_attributes)})
        )

    def _atomic_save(self, data, path):
        temp_path = path + ".tmp"
        with open(temp_path, 'wb') as f:
            pickle.dump(data, f)
        os.replace(temp_path, path)

    def _load_existing(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return {}