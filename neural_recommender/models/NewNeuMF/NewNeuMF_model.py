import torch
import torch.nn as nn
from ..BaseNeuralModel import BaseNeuralModel


class NewNeuMF(BaseNeuralModel):
    def __init__(
            self,
            user_emb_tensor: torch.Tensor,
            item_emb_tensor: torch.Tensor,
            u_emb_dim: int,
            i_emb_dim: int,
            latent_dim: int,
            mlp_layers: list[int],
            dropout: float,
            freeze_embeddings: bool = False,
    ):
        super().__init__()
        # pretrained embeddings
        self.user_emb = nn.Embedding.from_pretrained(user_emb_tensor, freeze=freeze_embeddings)
        self.item_emb = nn.Embedding.from_pretrained(item_emb_tensor, freeze=freeze_embeddings)

        # projection + dropout
        self.user_proj = nn.Linear(u_emb_dim, latent_dim)
        self.item_proj = nn.Linear(i_emb_dim, latent_dim)
        self.proj_dropout = nn.Dropout(dropout)

        # MLP stack
        layers = []
        input_dim = latent_dim * 2
        for h in mlp_layers:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = h
        self.mlp = nn.Sequential(*layers)

        # biases
        n_users = user_emb_tensor.size(0)
        n_items = item_emb_tensor.size(0)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # init weights - except Embeddings
        self._init_weights()

        # show model architecture if 'verbose = True'
        if self.verbose:
            self.show_model_architecture()

    def _init_weights(self):
        """
        Applies Xavier initialization to Linear layers and zeros to bias embeddings.
        """
        # Initialize Linear layers with Xavier Normal
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize bias embeddings to zero
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, u_idx: torch.LongTensor, i_idx: torch.LongTensor) -> torch.Tensor:
        # lookup
        u_emb = self.user_emb(u_idx)  # (batch, emb_dim)
        i_emb = self.item_emb(i_idx)  # (batch, emb_dim)

        # project + dropout
        u = self.proj_dropout(self.user_proj(u_emb))  # (batch, latent_dim)
        i = self.proj_dropout(self.item_proj(i_emb))  # (batch, latent_dim)

        # GMF scalar
        gmf_out = (u * i).sum(dim=1, keepdim=True)  # (batch, 1)

        # MLP scalar
        mlp_vec = self.mlp(torch.cat([u, i], dim=1))  # (batch, last_hidden)
        mlp_out = mlp_vec.sum(dim=1, keepdim=True)  # (batch, 1)

        # biases
        b_u = self.user_bias(u_idx)  # (batch, 1)
        b_i = self.item_bias(i_idx)  # (batch, 1)

        # final score
        return (gmf_out + mlp_out + b_u + b_i + self.global_bias).squeeze(1)

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
        # flatten multi-neg if needed
        if neg_idx.dim() > 1:
            B, K = neg_idx.size()
            # repeat each user & positive K times, then flatten
            u_idx = u_idx.unsqueeze(1).expand(-1, K).reshape(-1)
            pos_idx = pos_idx.unsqueeze(1).expand(-1, K).reshape(-1)
            neg_idx = neg_idx.reshape(-1)

        # call score method for positive and negative pairs
        return self.score(u_idx, pos_idx), self.score(u_idx, neg_idx)
