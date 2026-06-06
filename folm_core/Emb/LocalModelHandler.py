from transformers import AutoTokenizer, AutoModel
import torch
import os
import gc
import warnings
from torch import Tensor
from tqdm import tqdm
import logging
from typing import List, Tuple

warnings.simplefilter("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelHandler:
    """
    Handles the loading, storage, and execution of a local transformer model
    for generating text embeddings.
    """

    def __init__(self, model_id: str, only_last_token: bool = True):
        """
        Initializes the ModelHandler.

        Args:
            model_id (str): The identifier of the model from Hugging Face.
            only_last_token (bool): If True, computes only the last-token embeddings.
                                    Otherwise, computes mean and one-step-attention embeddings as well.
        """
        self.model_id = model_id
        self.only_last = only_last_token
        self.device = self._get_device()

        self.store_path = os.path.join(
            os.getcwd(), 'folm_core', 'Emb', 'models', self.model_id.replace('/', '_')
        )
        os.makedirs(self.store_path, exist_ok=True)

        self._ensure_model_stored()
        self.tokenizer, self.model = self._load_model_and_tokenizer()

        self.output_dtype, self.norm_eps = self._determine_output_config()
        logger.info(f"Using output_dtype={self.output_dtype}, norm_eps={self.norm_eps} [if norm]")
        self.exclude_ids = set(self.tokenizer.all_special_ids)

    def _get_device(self) -> torch.device:
        """Checks for CUDA availability and returns the appropriate torch device."""
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"CUDA available: ~{vram_gb:.2f} GB")
            return torch.device('cuda')
        else:
            logger.warning("CUDA not available; using CPU.")
            return torch.device('cpu')

    def _ensure_model_stored(self):
        """Downloads and saves the model and tokenizer if not already present."""
        if not os.listdir(self.store_path):
            logger.info(f"Downloading and storing model '{self.model_id}'..")
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModel.from_pretrained(self.model_id)
            tokenizer.save_pretrained(self.store_path)
            model.save_pretrained(self.store_path)
        else:
            logger.info(f"Model '{self.model_id}' already stored locally.")

    def _load_model_and_tokenizer(self) -> Tuple[AutoTokenizer, AutoModel]:
        """Loads the model and tokenizer from the local store path."""
        logger.info(f"Loading model from {self.store_path}..")
        device_map = {"": self.device.type} if self.device.type == "cuda" else {"": "cpu"}

        tokenizer = AutoTokenizer.from_pretrained(self.store_path)
        model = AutoModel.from_pretrained(
            self.store_path,
            device_map=device_map,
            trust_remote_code=True
        )
        model.eval()
        logger.info(f"Model from {self.store_path} loaded successfully.")
        return tokenizer, model

    def _determine_output_config(self) -> Tuple[torch.dtype, float]:
        """Determines the output dtype and normalization epsilon from the model."""
        try:
            output_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            output_dtype = torch.float32

        if output_dtype in (torch.float16, torch.bfloat16):
            norm_eps = 1e-6
        else:
            norm_eps = 1e-12

        return output_dtype, norm_eps

    def _last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Pools the hidden state of the last token."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]

    def _mean_pool(self, live_mask: Tensor, hidden_states: Tensor) -> Tensor:
        """Computes uniform mean-pooled embeddings over all 'live' (non-special, non-padding) tokens."""
        live_expanded = live_mask.unsqueeze(-1).expand_as(hidden_states).float()
        summed = torch.sum(hidden_states * live_expanded, dim=1)
        counts = live_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9)
        return summed / counts

    def _one_step_pool(self, hidden_states: Tensor, live_mask: Tensor, query: Tensor) -> Tensor:
        """Performs one-step self-attention using a query vector to score each token."""
        scores = torch.bmm(hidden_states, query.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(live_mask == 0, float("-inf"))
        weights = torch.nn.functional.softmax(scores, dim=1)
        weights_expanded = weights.unsqueeze(-1).expand_as(hidden_states).float()
        return torch.sum(hidden_states * weights_expanded, dim=1)

    def encode(
            self,
            texts: List[str],
            batch_size: int,
            max_length: int = 4096,
            use_tqdm: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes a list of texts and returns embeddings based on different pooling strategies.

        This method unifies the previous `encode` and `MetaEmb_encode` functions.

        Args:
            texts (List[str]): A list of texts to encode.
            batch_size (int): The number of texts to process in each batch.
            max_length (int): The maximum sequence length for the tokenizer.
            use_tqdm (bool): If True, displays a progress bar during encoding.

        Returns:
            A tuple containing:
            - last_tokens (torch.Tensor): Embeddings from last-token pooling.
            - mean_tokens (torch.Tensor): Embeddings from mean pooling (empty if `only_last` is True).
            - one_step_tokens (torch.Tensor): Embeddings from one-step attention (empty if `only_last` is True).
        """
        all_last, all_mean, all_one_step = [], [], []

        # Tokenize all texts at once for efficiency
        encoded_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        iterable = range(0, len(texts), batch_size)
        if use_tqdm:
            iterable = tqdm(iterable, desc="Encoding Batches")

        with torch.inference_mode():
            for i in iterable:
                batch_slice = slice(i, i + batch_size)
                batch = {k: v[batch_slice].to(self.device) for k, v in encoded_inputs.items()}

                outputs = self.model(**batch)
                hs = outputs.last_hidden_state
                mask = batch["attention_mask"]

                last_embs = self._last_token_pool(hs, mask)
                all_last.append(last_embs.cpu().to(self.output_dtype))

                if not self.only_last:
                    ids = batch["input_ids"]
                    live_mask = mask.clone()
                    for token_id in self.exclude_ids:
                        live_mask &= (ids != token_id)

                    mean_embs = self._mean_pool(live_mask, hs)
                    one_step_embs = self._one_step_pool(hs, live_mask, last_embs)

                    all_mean.append(mean_embs.cpu().to(self.output_dtype))
                    all_one_step.append(one_step_embs.cpu().to(self.output_dtype))

                # Clean up GPU memory
                del outputs, hs, last_embs
                if not self.only_last:
                    del mean_embs, one_step_embs, live_mask

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()

        last_tokens = torch.cat(all_last, dim=0)
        if not self.only_last:
            mean_tokens = torch.cat(all_mean, dim=0)
            one_step_tokens = torch.cat(all_one_step, dim=0)
            return last_tokens, mean_tokens, one_step_tokens

        return last_tokens, torch.empty(0, 0), torch.empty(0, 0)