import os
import pickle
import logging
import torch
from sentence_transformers import SentenceTransformer
from safetensors.numpy import save_file as save_numpy_safetensors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VariantHandler:
    def __init__(self, st_model_name, datadir, hf_token: str, batch_size: int = 32):
        self.basename = os.path.basename(st_model_name)
        self.datadir = datadir
        self.output_dir = os.path.join(os.getcwd(), 'data', self.datadir, 'LLMEmb_output')
        self._type = "ST"  # generic sentence transformer
        self.batch_size = batch_size

        # INSTRUCT PROMPTS
        self.user_instruct = ("Represent the following user profile to capture the user’s preferences, "
                              "tastes, and interests for recommendation tasks: The User ")
        self.item_instruct = ("Represent the following item in a way that highlights its main attributes, themes, and "
                              "style so it can be matched to user preferences for recommendation tasks: The Item ")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {self.device}")

        # initialize the SentenceTransformer model (trust_remote_code=True added for Qwen/Snowflake compatibility)
        self.model = SentenceTransformer(st_model_name, token=hf_token, device=self.device, trust_remote_code=True)

        # set the specific embedding function with _switch_handler
        self.embed_function = self._switch_handler()

    def _switch_handler(self):
        """
        Returns the appropriate embedding function and model type based on the model name.
        """

        if "embeddinggemma" in self.basename:
            logging.info("Using Gemma-optimized embedding function.")

            def gemma_encoder(texts, entity_type=None, **kwargs):
                # manually concatenate instruction + profile
                instruction = self.user_instruct if entity_type == "user" else self.item_instruct
                formatted_texts = [f"{instruction}{text}" for text in texts]
                return self.model.encode_query(formatted_texts, **kwargs)

            return gemma_encoder

        elif "snowflake-arctic-embed" in self.basename:
            logging.info("Using Snowflake-optimized embedding function.")

            # recsys symmetry: user and item act as task-guided queries
            # Snowflake requires the 'query: ' prefix to activate instruction-following mode
            self.model.prompts["user_task"] = f"query: {self.user_instruct}"
            self.model.prompts["item_task"] = f"query: {self.item_instruct}"

            def snowflake_encoder(texts, entity_type=None, **kwargs):
                prompt_to_use = "user_task" if entity_type == "user" else "item_task"
                return self.model.encode(texts, prompt_name=prompt_to_use, **kwargs)

            return snowflake_encoder

        elif "Qwen" in self.basename:
            logging.info("Using Qwen-optimized embedding function.")

            # recsys symmetry: user and item act as task-guided queries
            # Qwen3 strictly requires "Instruct: {task}\nQuery:{text}"
            self.model.prompts["user_task"] = f"Instruct: {self.user_instruct}\nQuery:"
            self.model.prompts["item_task"] = f"Instruct: {self.item_instruct}\nQuery:"

            def qwen_encoder(texts, entity_type=None, **kwargs):
                prompt_to_use = "user_task" if entity_type == "user" else "item_task"
                return self.model.encode(texts, prompt_name=prompt_to_use, **kwargs)

            return qwen_encoder

        else:
            logging.error(f'ERROR: {self.basename} - Model not supported, please add the appropriate Handler')
            raise ValueError(f"Model {self.basename} not supported.")

    def generate_embeddings_from_api_responses(self):
        """
        Generates and saves user and item embeddings using the structured metadata
        """
        temp_path = os.path.join(self.output_dir, f'user_{self._type}_embeddings_{self.basename}.safetensors')
        if not os.path.exists(temp_path):
            logging.info(f"Variant - {self.basename} generating USER embedding..")

            # process User Embeddings
            user_metadata_path = os.path.join(self.output_dir, "user_metadata.pkl")
            if os.path.exists(user_metadata_path):
                with open(user_metadata_path, "rb") as f:
                    user_metadata = pickle.load(f)
                self._generate_embeddings_from_metadata(metadata=user_metadata, entity_type="user")
            else:
                logging.warning("User metadata not found. Skipping user embedding generation.")
        else:
            logging.warning(f"{temp_path} already exists.")

        # process Item Embeddings
        temp_path = os.path.join(self.output_dir, f'item_{self._type}_embeddings_{self.basename}.safetensors')
        if not os.path.exists(temp_path):
            logging.info(f"Variant - {self.basename} generating ITEM embedding..")
            item_metadata_path = os.path.join(self.output_dir, "item_metadata.pkl")
            if os.path.exists(item_metadata_path):
                with open(item_metadata_path, "rb") as f:
                    item_metadata = pickle.load(f)
                self._generate_embeddings_from_metadata(metadata=item_metadata, entity_type="item")
            else:
                logging.warning("Item metadata not found. Skipping item embedding generation.")
        else:
            logging.warning(f"{temp_path} already exists.")

    def _generate_embeddings_from_metadata(self, metadata: dict, entity_type: str):
        """
        Generic function to generate embeddings for users or items from metadata.
        """
        logging.info(f"Generating {entity_type} embeddings with {self.basename}")

        texts, id_map = [], {}
        logging.info(f"Generating {len(metadata.items())} embeddings..")

        # instructions are applied automatically inside self.embed_function()
        for entity_id, data in metadata.items():
            profile = data.get("profile")
            if profile:
                texts.append(profile)
                id_map[len(texts) - 1] = (entity_id, "profile")

        # generate embeddings with SentenceTransformer
        if texts:
            # strictly pass entity_type to the encoder so the correct instruction is triggered
            embs = self.embed_function(
                texts, entity_type=entity_type, show_progress_bar=True, batch_size=self.batch_size
            )
            self._save_split_embeddings(torch.tensor(embs), id_map, entity_type, self._type)

    def _save_split_embeddings(self, embeddings: torch.Tensor, id_map: dict, entity_type: str, emb_type: str):
        """ Helper to save embeddings with correctly formatted filenames. """
        if embeddings.numel() == 0:
            return

        emb_dict = {}
        for i, (entity_id, field) in id_map.items():
            key = f"{entity_id}_{field}" if entity_type == 'user' else str(entity_id)
            emb_dict[key] = embeddings[i].cpu().numpy()

        filename = f"{entity_type}_{emb_type}_embeddings_{self.basename}.safetensors"

        path = os.path.join(self.output_dir, filename)
        save_numpy_safetensors(emb_dict, path)
        logging.info(f"Saved {len(emb_dict)} embeddings to {path}")


if __name__ == "__main__":
    #  - This code is stand-alone
    # REQUIRES:
    # an HuggingFace Token
    # sentenceTransformer version depends on the model you want to use
    # NOTE: you can create another env with sentenceTransformer (new version) + pytorch CUDA + safetensors
    ##################################################################
    hf_token = ""  # !!!!!

    # Models you can quickly switch between:
    # 'Qwen/Qwen3-Embedding-0.6B'
    # 'Snowflake/snowflake-arctic-embed-l-v2.0'
    # 'google/embeddinggemma-300m'
    st_model_name = 'Qwen/Qwen3-Embedding-4B'

    datadir = "book-genome" # "ml-1m" # "ml-1mNOFOL" # "book-genomeNOFOL"
    batch_size = 256

    # init
    handler = VariantHandler(st_model_name=st_model_name, datadir=datadir, batch_size=batch_size, hf_token=hf_token)

    # call gen
    handler.generate_embeddings_from_api_responses()

    # nohup python variantHandler.py > genomeNOFOL_embedding_gen_Qwen4B.log 2>&1 &
    # nohup python variantHandler.py > ml-1mNOFOL_embedding_gen_Qwen4B.log 2>&1 &

    # nohup python variantHandler.py > ml-1m_embedding_gen_Qwen4B.log 2>&1 &
    # nohup python variantHandler.py > genome_embedding_gen_Qwen4B.log 2>&1 &