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

        W = torch.empty(
                [out_features, in_features],
                device=device,
                dtype=dtype
            )

        init_std = sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            W,
            std=init_std,
            a=-3*init_std,
            b=3*init_std
        )

        self.W = nn.Parameter(W)
        self.in_features = in_features
        self.out_features = out_features


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")

