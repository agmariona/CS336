import torch
import torch.nn as nn
from math import sqrt
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,                       # input dim
        out_features: int,                      # output dim
        device: torch.device | None = None,     # device to store params
        dtype: torch.dtype | None = None        # param dtype
    ) -> None:
        super().__init__()

        weight = torch.empty(
                [out_features, in_features],
                device=device,
                dtype=dtype
            )

        init_std = sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            weight,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )

        self.weight = nn.Parameter(weight)
        self.in_features = in_features
        self.out_features = out_features


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,                    # vocab size
        embedding_dim: int,                     # d_model
        device: torch.device | None = None,     # device to store params
        dtype: torch.dtype | None = None        # param dtype
    ) -> None:
        super().__init__()

        weight = torch.empty(
                [num_embeddings, embedding_dim],
                device=device,
                dtype=dtype
            )

        nn.init.trunc_normal_(
            weight,
            a=-3,
            b=3
        )

        self.weight = nn.Parameter(weight)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim


    def forward(
        self,
        token_ids: torch.Tensor         # [batch_size, seq_length]
    ) -> torch.Tensor:                  # [batch_size, seq_length, d_model]
        return self.weight[token_ids]
