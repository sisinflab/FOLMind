import os
import logging
import pickle
import torch
from sentence_transformers import SentenceTransformer
from safetensors.numpy import save_file as save_numpy_safetensors

from .LocalModelHandler import ModelHandler
from .PromptGenerator import PromptGenerator
from .ApiResponseProcessor import ApiResponseProcessor
from .LLMEmbAPI import LLMEmbAPI
from typing import List, Any, Dict, Optional
from pydantic import BaseModel, Field, ValidationError


logger = logging.getLogger(__name__)


class LLMEmbConfig(BaseModel):
    user_prompt_file: str
    item_prompt_file: str
    text_cleaning_labels: Optional[List[str]] = Field(default=None)
    crawler_config: str
    resume: bool = False
    threads: int = 1
    save_calls_steps: int = 100
    use_scs_for_prompt_pregen: bool = False
    bert_default_model: str = 'sentence-transformers/all-mpnet-base-v2'
    use_local_model_handler: bool = False
    batch_size: int
    only_last: bool = True
    embedding_model: str | None
    huggingface_key: str | None
    api_keys: list
    encode_item: bool = True
    show_tsne_plot: bool = False
    max_missing_rate_per_pred: Optional[float] = Field(default=0.05, ge=0.01, le=1.0)  # [0, 1] -> [0%, 100%]
    metaEmb_batch_size: int | None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        try:
            return cls(**config_dict)
        except ValidationError as e:
            logger.error(f"Config validation error: {e}")
            raise


class LLMEmb:
    """
    Orchestrates the LLM-based embedding generation pipeline.
    It manages prompt generation, API calls, response processing, and the final
    embedding creation from the generated text.
    """

    def __init__(self, datadir: str, config):
        # Define the default BERT model as a constant
        BERT_DEFAULT_MODEL = config.bert_default_model

        self.datadir = datadir
        self.config = config
        self.model_basename = os.path.basename(config.embedding_model)
        self.output_dir = os.path.join(os.getcwd(), 'data', datadir, 'LLMEmb_output')
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize components specific to the LLM pipeline
        self.prompt_generator = PromptGenerator(datadir, config)
        self.api_handler = LLMEmbAPI(datadir, config)
        self.api_processor = ApiResponseProcessor(datadir, config)
        self.model_handler = ModelHandler(config.embedding_model, config.only_last)

        # Initialize the separate BERT model for comparison embeddings
        self.bert_model = SentenceTransformer(BERT_DEFAULT_MODEL)
        self.bert_model_basename = os.path.basename(BERT_DEFAULT_MODEL)

    def run_llm_api_pipeline(self):
        """
        Executes the full data acquisition pipeline:
        - generates prompts from the knowledge graph.
        - calls the LLM API to get responses.
        - processes and cleans the raw API responses into structured metadata.
        """
        logger.info("Starting LLM data acquisition pipeline...")

        logger.info("Handling API prompts - 1/3: Generating prompts...")
        self.prompt_generator.generate_all_prompts()

        logger.info("Handling API prompts - 2/3: Calling LLM API for user and item prompts...")
        self.api_handler.process_user_prompts()
        self.api_handler.process_item_prompts()

        logger.info("Handling API prompts - 3/3: Processing API responses into structured metadata...")
        self.api_processor.process_all_responses()

        logger.info("LLM data acquisition pipeline finished successfully.")

    def generate_embeddings_from_api_responses(self):
        """
        Generates and saves user and item embeddings using the structured metadata
        """
        temp_path = os.path.join(os.getcwd(), 'data', self.datadir, 'LLMEmb_output',
                                 f'user_bert_embeddings_{self.bert_model_basename}.safetensors')
        if not os.path.exists(temp_path):
            logger.info("Generating user embeddings from processed LLM responses...")

            # Process User Embeddings
            user_metadata_path = os.path.join(self.output_dir, "user_metadata.pkl")
            if os.path.exists(user_metadata_path):
                with open(user_metadata_path, "rb") as f:
                    user_metadata = pickle.load(f)
                self._generate_embeddings_from_metadata(metadata=user_metadata, entity_type="user")
            else:
                logger.warning("User metadata not found. Skipping user embedding generation.")
        else:
            logger.warning(f"{temp_path} already exists. Skipping user embedding generation. "
                           f"If you want to run any other USER embeddings generation delete only "
                           f"the {self.bert_model_basename} embeddings.")

        # Process Item Embeddings
        temp_path = os.path.join(os.getcwd(), 'data', self.datadir, 'LLMEmb_output',
                                 f'item_bert_embeddings_{self.bert_model_basename}.safetensors')
        if not os.path.exists(temp_path):
            logger.info("Generating item embeddings from processed LLM responses...")
            item_metadata_path = os.path.join(self.output_dir, "item_metadata.pkl")
            if os.path.exists(item_metadata_path):
                with open(item_metadata_path, "rb") as f:
                    item_metadata = pickle.load(f)
                self._generate_embeddings_from_metadata(metadata=item_metadata, entity_type="item")
            else:
                logger.warning("Item metadata not found. Skipping item embedding generation.")
        else:
            logger.warning(f"{temp_path} already exists. Skipping item embedding generation. "
                           f"If you want to run any other ITEM embeddings generation delete only "
                           f"the {self.bert_model_basename} embeddings.")

    def _generate_embeddings_from_metadata(self, metadata: dict, entity_type: str):
        """
        Generic function to generate embeddings for users or items from metadata.
        """
        logger.info(f"Generating {entity_type} embeddings with both main model and BERT...")

        texts_to_encode, id_map = [], {}  # For main model
        bert_texts, bert_id_map = [], {}  # For BERT model

        for entity_id, data in metadata.items():
            if entity_type == "user":
                profile, reasoning = data.get("profile"), data.get("reasoning")
                if profile:
                    texts_to_encode.append(profile.lower())
                    id_map[len(texts_to_encode) - 1] = (entity_id, "profile")
                    bert_texts.append(profile)
                    bert_id_map[len(bert_texts) - 1] = (entity_id, "profile")
                if reasoning:
                    texts_to_encode.append(reasoning.lower())
                    id_map[len(texts_to_encode) - 1] = (entity_id, "reasoning")
                    bert_texts.append(reasoning)
                    bert_id_map[len(bert_texts) - 1] = (entity_id, "reasoning")
            else:  # item
                profile = data.get("profile")
                if profile:
                    texts_to_encode.append(profile.lower())
                    id_map[len(texts_to_encode) - 1] = (entity_id, "profile")
                    bert_texts.append(profile)
                    bert_id_map[len(bert_texts) - 1] = (entity_id, "profile")

        # Generate embeddings with the primary local model
        if texts_to_encode:
            last_embs, mean_embs, one_step_embs = self.model_handler.encode(
                texts_to_encode, self.config.batch_size, use_tqdm=True
            )
            self._save_split_embeddings(last_embs, id_map, entity_type, "last")
            if not self.config.only_last:
                self._save_split_embeddings(mean_embs, id_map, entity_type, "mean")
                self._save_split_embeddings(one_step_embs, id_map, entity_type, "step")

        # Generate embeddings with the BERT model
        if bert_texts:
            bert_embs = self.bert_model.encode(
                bert_texts, show_progress_bar=True, batch_size=self.config.batch_size
            )
            self._save_split_embeddings(torch.tensor(bert_embs), bert_id_map, entity_type, "bert")

    def _save_split_embeddings(self, embeddings: torch.Tensor, id_map: dict, entity_type: str, emb_type: str):
        """Helper to save embeddings with correctly formatted filenames."""
        if embeddings.numel() == 0:
            return

        emb_dict = {}
        for i, (entity_id, field) in id_map.items():
            key = f"{entity_id}_{field}" if entity_type == 'user' else str(entity_id)
            emb_dict[key] = embeddings[i].cpu().numpy()

        if emb_type == "bert":
            filename = f"{entity_type}_bert_embeddings_{self.bert_model_basename}.safetensors"
        else:
            filename = f"{entity_type}_{emb_type}_embeddings_{self.model_basename}.safetensors"

        path = os.path.join(self.output_dir, filename)
        save_numpy_safetensors(emb_dict, path)
        logger.info(f"Saved {len(emb_dict)} embeddings to {path}")
