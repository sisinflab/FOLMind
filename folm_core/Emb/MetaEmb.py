import os
import logging
import pickle
import torch
from sentence_transformers import SentenceTransformer
from safetensors.torch import save_file
from .LocalModelHandler import ModelHandler
from .MetadataProcessor import MetadataProcessor

logger = logging.getLogger(__name__)


class MetaEmb:
    """
    Generates item embeddings from processed metadata descriptions.
    It relies on a MetadataProcessor to first prepare the data.
    """

    def __init__(self, datadir: str, config, model_handler: ModelHandler = None):

        self.BERT_DEFAULT_MODEL = config.bert_default_model
        self.bert_model_basename = os.path.basename(self.BERT_DEFAULT_MODEL)
        self.config = config
        self.datadir = datadir
        self.model_basename = os.path.basename(config.embedding_model)

        cwd = os.getcwd()
        self.output_dir = os.path.join(cwd, 'data', datadir, 'metaEmb_output')
        self.entity_file_path = os.path.join(cwd, 'data', datadir, 'splitting', 'train', 'entity_name_mapping.pkl')
        os.makedirs(self.output_dir, exist_ok=True)

        self.model_handler = model_handler if model_handler is not None else self._load_model_handler()
        self.metadata_processor = MetadataProcessor(datadir, config)

    def _load_model_handler(self) -> ModelHandler:
        """Loads the local transformer model using ModelHandler."""
        logger.info(f"MetaEmb is loading model: {self.config.embedding_model}")
        return ModelHandler(
            model_id=self.config.embedding_model,
            only_last_token=self.config.only_last
        )

    def generate_all_embeddings(self):
        """
        Orchestrates the generation of all types of item embeddings
        (from local model and from a standard SentenceTransformer).
        """
        item_descriptions = self.metadata_processor.process_and_save_descriptions("item_descriptions.pkl")

        logger.info(f"Operating with MetaEmb batch size: {self.config.metaEmb_batch_size}")

        # generate embeddings using the primary local model
        self._generate_local_model_embeddings(item_descriptions)

        # generate embeddings using the fallback/default BERT model
        self._generate_bert_embeddings(item_descriptions)

    def _generate_local_model_embeddings(self, descriptions: dict):
        """Generates embeddings using the configured local ModelHandler."""
        output_path = os.path.join(self.output_dir, f'item_last_embeddings_{self.model_basename}.safetensors')
        if os.path.exists(output_path):
            logger.info("Local model item embeddings already exist. Skipping.")
            return

        logger.info("Generating embeddings with local model handler...")
        texts = list(descriptions.values())

        last_embs, mean_embs, step_embs = self.model_handler.encode(
            texts,
            batch_size=self.config.metaEmb_batch_size,
            use_tqdm=True
        )

        with open(self.entity_file_path, "rb") as f:
            name_to_id: dict[str, int] = pickle.load(f)

        # map description keys (names) to true entity IDs
        true_ids = [name_to_id.get(name) for name in descriptions.keys()]

        self._save_embeddings(last_embs, true_ids, f'item_last_embeddings_{self.model_basename}.safetensors')
        if not self.config.only_last:
            self._save_embeddings(mean_embs, true_ids, f'item_mean_embeddings_{self.model_basename}.safetensors')
            self._save_embeddings(step_embs, true_ids, f'item_step_embeddings_{self.model_basename}.safetensors')

    def _generate_bert_embeddings(self, descriptions: dict):
        """Generates embeddings using a standard SentenceTransformer model."""
        output_path = os.path.join(self.output_dir, f'item_bert_embeddings_{self.bert_model_basename}.safetensors')
        if os.path.exists(output_path):
            logger.info("BERT item embeddings already exist. Skipping.")
            return

        logger.info("Generating embeddings with SentenceTransformer (BERT)...")
        try:
            bert_model = SentenceTransformer(self.BERT_DEFAULT_MODEL).cuda()

            with open(self.entity_file_path, "rb") as f:
                name_to_id: dict[str, int] = pickle.load(f)

            item_ids = []
            texts_to_encode = []
            for name, desc in descriptions.items():
                item_id = name_to_id.get(name)
                if item_id is not None:
                    item_ids.append(item_id)
                    texts_to_encode.append(desc)

            embeddings = bert_model.encode(
                texts_to_encode,
                show_progress_bar=True,
                batch_size=self.config.batch_size
            )

            self._save_embeddings(torch.tensor(embeddings), item_ids,
                                  f'item_bert_embeddings_{self.bert_model_basename}.safetensors')

        except Exception as e:
            logger.error(f"Failed to generate BERT embeddings: {e}")

    def _save_embeddings(self, embeddings: torch.Tensor, ids: list, filename: str):
        """Saves a dictionary of embeddings to a safetensors file."""
        if not ids or embeddings.shape[0] != len(ids):
            logger.error(
                f"Mismatch between number of embeddings ({embeddings.shape[0]}) and IDs ({len(ids)}). Cannot save.")
            return

        embeddings = embeddings.detach().cpu()
        if embeddings.dtype in (torch.float16, torch.bfloat16):
            embeddings = embeddings.to(torch.float32)

        embedding_dict = {str(id_): vec for id_, vec in zip(ids, embeddings) if id_ is not None}

        out_path = os.path.join(self.output_dir, filename)
        save_file(embedding_dict, out_path)
        logger.info(f"Saved {len(embedding_dict)} embeddings to {out_path}")
