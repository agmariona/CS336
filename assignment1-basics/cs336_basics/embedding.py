import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,                    # vocab size
        embedding_dim: int,                     # d_model
        device: torch.device | None = None,     # device to store params
        dtype: torch.dtype | None = None        # param dtype
    ) -> None:
        super().__init__()

        W = torch.empty(
                [num_embeddings, embedding_dim],
                device=device,
                dtype=dtype
            )

        nn.init.trunc_normal_(
            W,
            a=-3,
            b=3
        )

        self.W = nn.Parameter(W)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim


    def forward(
        self,
        token_ids: torch.Tensor                 # [batch_size, seq_length]
    ) -> torch.Tensor:
        return self.W[token_ids]
