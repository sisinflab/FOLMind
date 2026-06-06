import torch
import torch.nn as nn
from ..BaseNeuralModel import BaseNeuralModel
from typing import Optional, List


class NeuMF(BaseNeuralModel):
    """
    Classic NeuMF model adapted from Elliot Framework - enhanced to do Pairwise Training (score function)
    For further details, please refer to the `paper <https://arxiv.org/abs/1708.05031>`_

    To include the recommendation model, add it to the config file adopting the following pattern:

    SHADOW_PARAM - present in the recommendation workflow, so the user doesn't need to explicit them in the config

    model:
      name: "NeuMF"                        # class name of the model
      module: "neural_recommender.models.NeuMF"   # python import path
      cls_name: "NeuMF"                    # optional if same as name
      params:
        user_emb_tensor: torch.Tensor,  # ----> SHADOW_PARAM: param will calculate num_users and then will be deleted
                                                since there is no use in this model
        item_emb_tensor: torch.Tensor, # ----> SHADOW_PARAM: param will calculate num_items and then will be deleted
                                               since there is no use in this model
        u_emb_dim: int,     # ----> SHADOW_PARAM: param will be deleted since there is no use in this model
        i_emb_dim: int,     # ----> SHADOW_PARAM: param will be deleted since there is no use in this model
        mf_factors: 32
        mlp_factors: 32
        mlp_hidden_size: [256, 128, 64] or do not specify if you want [mf_factors * 4, mf_factors * 2, mf_factors]
        dropout: 0.35
      device: "cuda"
    """

    def __init__(
            self,
            user_emb_tensor: torch.Tensor,
            item_emb_tensor: torch.Tensor,
            u_emb_dim: int,
            i_emb_dim: int,
            mf_factors: int,
            mlp_factors: int,
            dropout: float,
            # mlp_hidden_size optional -> default: [mf_factors * 4, mf_factors * 2, mf_factors]
            mlp_hidden_size: Optional[List[int]] = None,
    ):
        super().__init__()

        # infer dimensions from the input tensors
        num_users, _ = user_emb_tensor.shape
        num_items, _ = item_emb_tensor.shape

        # delete for this instance after getting the num_users and num_items
        del user_emb_tensor, u_emb_dim
        del item_emb_tensor, i_emb_dim

        # set the MLP architecture
        # If mlp_hidden_size is not provided, create a default structure.
        if mlp_hidden_size is None:
            final_mlp_hidden_size = [mf_factors * 4, mf_factors * 2, mf_factors]
        else:
            final_mlp_hidden_size = mlp_hidden_size

        # Embedding layers
        self.user_mf_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=mf_factors)
        self.item_mf_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=mf_factors)
        self.user_mlp_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=mlp_factors)
        self.item_mlp_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=mlp_factors)

        # MLP layers
        mlp_modules = []
        mlp_input_size = mlp_factors * 2
        for n_units in final_mlp_hidden_size:
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(mlp_input_size, n_units))
            mlp_modules.append(nn.ReLU())
            mlp_input_size = n_units
        self.mlp_layers = nn.Sequential(*mlp_modules)

        # Final prediction layer
        predict_input_size = mf_factors + final_mlp_hidden_size[-1]
        self.predict_layer = nn.Linear(predict_input_size, 1)

        self._init_weights()
        # show model architecture if 'verbose = True'
        if self.verbose:
            self.show_model_architecture()

    def _init_weights(self):
        """Initializes weights using Xavier uniform distribution."""
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)

    def score(self, u_idx: torch.LongTensor, i_idx: torch.LongTensor) -> torch.Tensor:
        """
        Calculates the interaction score for user-item pairs based on classic NeuMF logic.
        Returns raw logits, suitable for a pairwise loss like BPR.
        """
        # --- GMF Path ---
        user_mf_emb = self.user_mf_embedding(u_idx)
        item_mf_emb = self.item_mf_embedding(i_idx)
        mf_vector = user_mf_emb * item_mf_emb

        # --- MLP Path ---
        user_mlp_emb = self.user_mlp_embedding(u_idx)
        item_mlp_emb = self.item_mlp_embedding(i_idx)
        mlp_input = torch.cat([user_mlp_emb, item_mlp_emb], dim=-1)
        mlp_vector = self.mlp_layers(mlp_input)

        # --- Combine Paths & Predict ---
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        # We return the raw logits before the sigmoid for use with pairwise losses
        return self.predict_layer(predict_vector).squeeze(1)

    def forward(
            self,
            u_idx: torch.LongTensor,
            pos_idx: torch.LongTensor,
            neg_idx: torch.LongTensor
    ):
        """
        Calculates scores for positive and negative items.
        It only orchestrates calls to the 'score' method.
        """
        #  flatten multi-neg if needed
        if neg_idx.dim() > 1:
            B, K = neg_idx.size()
            # repeat each user & positive K times, then flatten
            u_idx = u_idx.unsqueeze(1).expand(-1, K).reshape(-1)
            pos_idx = pos_idx.unsqueeze(1).expand(-1, K).reshape(-1)
            neg_idx = neg_idx.reshape(-1)

        # call score method for positive and negative pairs
        return self.score(u_idx, pos_idx), self.score(u_idx, neg_idx)


